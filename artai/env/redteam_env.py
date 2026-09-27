"""Gymnasium environment: attacker agent vs scripted or human defender.

Flat masked action space: action = (type, node, arg) encoded as one int.
arg indexes into a node's vulns/local_vulns or the global credential list.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from artai.env import network as net_mod
from artai.env.catalog import load as load_catalog

ACTIONS = (
    "scan", "enumerate", "exploit", "dump_creds",
    "lateral", "privesc", "persist", "collect",
)
DEF_ACTIONS = ("pass", "isolate", "reimage", "patch", "decoy", "investigate")
MAX_ARG = 8
NODE_FEATURES = 10

WIN_REWARD = 15.0
BURNED_PENALTY = -15.0
CONTAINED_PENALTY = -5.0
DETECT_PENALTY = -2.0
FIRST_CRED_BONUS = 3.0
FIRST_ROOT_BONUS = 3.0


def encode(a_type: int, node: int, arg: int, max_nodes: int) -> int:
    return (a_type * max_nodes + node) * MAX_ARG + arg


def decode(action: int, max_nodes: int) -> tuple[int, int, int]:
    a_type, rem = divmod(action, max_nodes * MAX_ARG)
    node, arg = divmod(rem, MAX_ARG)
    return a_type, node, arg


class RedTeamEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, n_workstations=3, n_servers=3, max_steps=120,
                 defender=None, catalog=None, scenario_seed=None):
        self.n_ws, self.n_srv = n_workstations, n_servers
        self.max_steps = max_steps
        self.defender = defender
        # pinned topology for scenario-pack training; None = fresh map each reset
        self.scenario_seed = scenario_seed
        self.catalog = catalog or load_catalog()
        # same-action technique variants, indexed by arg — the hot path for
        # _technique_for (no catalog rescan per candidate)
        self._by_action: dict[str, list] = {}
        for t in self.catalog.values():
            self._by_action.setdefault(t.action, []).append(t)
        self.max_nodes = 3 + n_workstations + n_servers  # edge + dc + cj + tiers
        self.rng = np.random.default_rng()
        self.net: net_mod.Network | None = None

        self.action_space = spaces.Discrete(len(ACTIONS) * self.max_nodes * MAX_ARG)
        n_globals = 2 + 2 * len(DEF_ACTIONS) + 1
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(self.max_nodes * NODE_FEATURES + n_globals,),
            dtype=np.float32,
        )
        self.creds: list[int] = []
        self.step_count = 0
        self.last_def_action = 0
        self.def_hist = np.zeros(len(DEF_ACTIONS), dtype=np.float32)
        self.detected_last = 0.0
        self._had_owned = False
        self._got_cred = False
        self._got_root = False
        self._last_success = False
        self.won = False
        self.technique_log: list[str] = []
        self.detection_log: list[str] = []

    # ---------- gym API ----------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if self.scenario_seed is not None:
            topo_seed = self.scenario_seed  # scenario mode: seed varies luck, not map
        elif seed is not None:
            topo_seed = seed
        else:
            topo_seed = int(self.rng.integers(1 << 31))
        self.net = net_mod.generate(
            seed=topo_seed,
            n_workstations=self.n_ws, n_servers=self.n_srv,
        )
        self.creds, self.step_count = [], 0
        self.last_def_action = 0
        self.def_hist[:] = 0
        self.detected_last = 0.0
        self._had_owned = self._got_cred = self._got_root = False
        self._last_success = False
        self.won = False
        self.technique_log, self.detection_log = [], []
        return self._obs(), {"action_mask": self.action_mask()}

    def step(self, action: int):
        if self.net is None:
            raise RuntimeError("call reset() before step()")
        # out-of-space actions can't decode — negative indices used to wrap
        # onto crown-jewel collect; penalize instead of executing
        if not self.action_space.contains(action):
            return (self._obs(), -0.6, False, False,
                    {"action_mask": self.action_mask(), "techniques": [],
                     "detected": []})
        a_type, node_id, arg = decode(int(action), self.max_nodes)
        name = ACTIONS[a_type]
        node = self.net.nodes[node_id]
        reward = -0.1
        self.detected_last = 0.0
        self._last_success = False
        self._det_reward = 0.0
        self.step_count += 1

        tech = self._technique_for(name, node, arg)
        if self._legal(tech, name, node, arg):
            success = self.rng.random() < tech.success
            reward += self._apply(name, node, arg, tech, success)
            self._detection_roll(tech, node)
            reward += self._det_reward
        else:
            reward -= 0.5

        for n in self.net.nodes.values():
            n.alert = max(0.0, n.alert - 0.4)  # alerts age out over time
        if self.defender is not None:
            d_name, d_node = self.defender.act(self._defender_view(), self.rng)
            self._apply_defender(d_name, d_node)

        terminated = truncated = False
        if (
            name == "collect" and node_id == self.net.crown_jewel
            and self._last_success
        ):
            reward += WIN_REWARD
            terminated = True
            self.won = True
        elif self._had_owned and not any(
            n.owned for n in self.net.nodes.values()
        ):
            reward += BURNED_PENALTY
            terminated = True
        elif self.step_count >= self.max_steps:
            truncated = True

        mask = self.action_mask()
        if not (terminated or truncated) and mask.sum() == 0:
            terminated = True  # attacker out of moves — contained
            reward += CONTAINED_PENALTY

        info = {
            "action_mask": mask,
            "techniques": list(self.technique_log),
            "detected": list(self.detection_log),
        }
        return self._obs(), reward, terminated, truncated, info

    # ---------- action validity + effects ----------

    def _reachable(self, node_id: int) -> bool:
        if node_id == self.net.entry_point:
            return not self.net.nodes[node_id].isolated
        return any(
            self.net.nodes[m].owned and not self.net.nodes[m].isolated
            for m in self.net.links[node_id]
        )

    def _has_req(self, req: str, node, arg: int) -> bool:
        if req == "discovered":
            return node.discovered
        if req == "reachable":
            return self._reachable(node.id)
        if req == "credential":
            return node.id in self.creds
        if req == "user_access":
            return node.owned
        if req == "root":
            return node.owned and node.root
        if req == "remote_vuln":
            return node.vulns_known and arg < len(node.vulns)
        if req == "local_vuln":
            return arg < len(node.local_vulns)
        if req == "not_isolated":
            return not node.isolated
        if req == "crown_jewel":
            return node.id == self.net.crown_jewel
        if req == "creds_known":
            return node.creds_known
        return False

    def _useful(self, name: str, node, arg: int) -> bool:
        """Beyond precondition checks: is the action worth anything?"""
        if name == "scan":
            return node.owned and any(
                not self.net.nodes[m].discovered for m in self.net.links[node.id]
            )
        if name == "enumerate":
            return node.discovered and not node.vulns_known
        if name in ("exploit", "lateral"):
            return not node.owned
        if name == "dump_creds":
            return any(c not in self.creds for c in node.credentials)
        if name == "privesc":
            return not node.root
        if name == "persist":
            return not node.persistent
        return True

    def _technique_for(self, name: str, node, arg: int):
        """Technique an action would execute. arg indexes vuln lists for
        exploit/privesc, and same-action catalog variants elsewhere
        (e.g. arg=1 picks the quieter T1550 lateral over T1021)."""
        if name == "exploit":
            if not (node.vulns_known and arg < len(node.vulns)):
                return None
            t = self.catalog.get(node.vulns[arg].technique_id)
            # a vuln entry pointing at a non-exploit technique is a dead slot
            return t if t is not None and t.action == "exploit" else None
        if name == "privesc":
            if arg >= len(node.local_vulns):
                return None
            t = self.catalog.get(node.local_vulns[arg].technique_id)
            return t if t is not None and t.action == "privesc" else None
        variants = self._by_action.get(name, ())
        return variants[arg] if arg < len(variants) else None

    def _legal(self, tech, name: str, node, arg: int) -> bool:
        """The one legality predicate — shared by action_mask() and step()."""
        return (
            tech is not None
            and self._useful(name, node, arg)
            and all(self._has_req(r, node, arg) for r in tech.requires)
        )

    def _valid(self, a_type: int, node_id: int, arg: int) -> bool:
        node = self.net.nodes[node_id]
        name = ACTIONS[a_type]
        return self._legal(
            self._technique_for(name, node, arg), name, node, arg)

    def action_mask(self) -> np.ndarray:
        mask = np.zeros(self.action_space.n, dtype=np.int8)
        if self.net is None:
            return mask
        for a_type in range(len(ACTIONS)):
            for nid in self.net.nodes:
                for arg in range(MAX_ARG):
                    if self._valid(a_type, nid, arg):
                        mask[encode(a_type, nid, arg, self.max_nodes)] = 1
        return mask

    def _apply(self, name: str, node, arg: int, tech, success: bool) -> float:
        reward = -0.1 * tech.cost
        self.technique_log.append(tech.id)
        if not success:
            return reward  # attack fizzled
        self._last_success = True

        if name == "scan":
            for m in self.net.links[node.id] | {node.id}:
                if not self.net.nodes[m].discovered:
                    self.net.nodes[m].discovered = True
                    reward += 0.3  # recon yield
        elif name == "enumerate":
            node.vulns_known = node.creds_known = True
        elif name in ("exploit", "lateral"):
            node.owned = True
            reward += node.value
            self._had_owned = True
        elif name == "dump_creds":
            for c in node.credentials:
                if c not in self.creds:
                    self.creds.append(c)
            if self.creds and not self._got_cred:
                reward += FIRST_CRED_BONUS
                self._got_cred = True
        elif name == "privesc":
            node.root = True
            if not self._got_root:
                reward += FIRST_ROOT_BONUS
                self._got_root = True
        elif name == "persist":
            node.persistent = True
        return reward

    def _detection_roll(self, tech, node) -> None:
        p = tech.detectability * (2.0 if node.decoy else 1.0)
        if self.rng.random() < min(p, 1.0):
            node.alert += 1.0
            node.alert_total += 1.0
            self.detected_last = 1.0
            self.detection_log.append(tech.id)
            self._det_reward += DETECT_PENALTY

    # ---------- defender ----------

    def defender_view(self) -> dict:
        """Public view for the game layer: what a defender may legitimately
        see — alerts, never ground-truth ownership."""
        return self._defender_view()

    def apply_defender_action(self, action: str, node_id: int | None) -> bool:
        """Game-layer defender turn. Validate then delegate.

        Returns whether the action effected something real — a patch on a
        box with no vulns is wasted motion, and the debrief tracks that.
        """
        if action not in DEF_ACTIONS:
            raise ValueError(f"unknown defender action {action!r}")
        if action != "pass" and node_id not in self.net.nodes:
            raise ValueError(f"no such node {node_id!r}")
        return self._apply_defender(action, node_id)

    def _defender_view(self) -> dict:
        return {
            "alerts": {n.id: n.alert for n in self.net.nodes.values() if n.alert > 0},
            "alert_totals": {n.id: n.alert_total for n in self.net.nodes.values() if n.alert_total > 0},
            "nodes": {
                n.id: net_mod.NodeView(
                    tier=n.tier, alert=n.alert, alert_total=n.alert_total,
                    vuln_count=len(n.vulns), isolated=n.isolated,
                    decoy=n.decoy)
                for n in self.net.nodes.values()
            },
        }

    def _apply_defender(self, action: str | None, node_id: int | None) -> bool:
        idx = DEF_ACTIONS.index(action) if action in DEF_ACTIONS else 0
        self.last_def_action = idx
        self.def_hist[idx] += 1
        if node_id is None or node_id not in self.net.nodes:
            return True  # pass, or a no-op the debrief doesn't score
        node = self.net.nodes[node_id]
        if node.tier == 0 and action in ("isolate", "patch"):
            return False  # the edge is the internet — evictable, not hardenable
        if action == "isolate":
            was = not node.isolated
            node.isolated = True
            return was
        elif action == "reimage":
            hit = node.owned or node.alert > 0
            node.alert = 0.0
            node.alert_total = 0.0
            node.vulns_known = node.creds_known = False
            node.decoy = False  # fresh image — honeypot gone too
            if node.persistent:
                node.root = False
            else:
                node.owned = node.root = node.persistent = False
                self.creds = [c for c in self.creds if c != node_id]
            return hit
        elif action == "patch":
            had = bool(node.vulns or node.local_vulns)
            node.vulns = []
            node.local_vulns = []
            return had
        elif action == "decoy":
            was = not node.decoy
            node.decoy = True  # perimeter decoys are legit — allowed on tier 0
            return was
        elif action == "investigate":
            return node.alert > 0  # found real noise vs a quiet box
        return True

    # ---------- observation ----------

    def state_key(self) -> tuple:
        """Compact hashable state for tabular learners.

        Per-node status code + coarse global context. Deliberately lossy —
        tabular Q-learning needs a small reachable state space.
        """
        per_node = []
        for i in range(self.max_nodes):
            n = self.net.nodes[i]
            if not n.discovered:
                code = 0
            elif not n.vulns_known:
                code = 1
            elif not n.owned:
                code = 2
            elif not n.root:
                code = 3
            else:
                code = 4
            if n.isolated:
                code += 5
            per_node.append(code)
        return (
            tuple(per_node),
            min(len(self.creds), 4),
            self.last_def_action,
            int(self.detected_last),
            sum(n.alert > 0 for n in self.net.nodes.values()),
            self.defender_signature(),
            self.ARCHETYPES.index(self.defender_guess()[0]),
        )

    def defender_signature(self) -> tuple:
        """Coarse (dominant non-pass action, activity level) from def_hist.

        Deliberately small — raw counts would blow up the state space;
        what the policy needs is 'reimager, busy' vs 'patcher, occasional'.
        """
        total = self.def_hist.sum()
        active = self.def_hist[1:].sum()
        if not total or not active:
            return (0, 0)
        dominant = 1 + int(self.def_hist[1:].argmax())
        rate = active / total
        return (dominant, 0 if rate < 0.15 else 1 if rate < 0.45 else 2)

    ARCHETYPES = ("unknown", "quiet", "patcher", "reimager", "erratic")

    def defender_guess(self, min_samples: int = 10) -> tuple:
        """Heuristic archetype guess from def_hist -> (name, confidence).

        Confidence is the score margin over the runner-up; low-margin
        observations stay 'unknown' rather than forcing a wrong label.
        """
        total = self.def_hist.sum()
        active = self.def_hist[1:].sum()
        if total < min_samples:
            return ("unknown", 0.0)
        if not active:
            return ("quiet", 1.0)
        kinds = {i for i in range(1, len(DEF_ACTIONS))
                 if self.def_hist[i] > 0}
        pi, ri = DEF_ACTIONS.index("patch"), DEF_ACTIONS.index("reimage")
        # No scripted policy mixes these: patch+reimage together, or any
        # decoy/investigate, means a generalist (random or human).
        mixed = ({pi, ri} <= kinds or DEF_ACTIONS.index("decoy") in kinds
                 or DEF_ACTIONS.index("investigate") in kinds)
        shares = self.def_hist[1:] / active
        scores = {
            "quiet": max(0.0, 1.0 - 3.0 * (active / total)),
            "patcher": shares[pi - 1] + 0.5 * shares[
                DEF_ACTIONS.index("isolate") - 1],
            "reimager": shares[ri - 1],
            "erratic": 1.0 if mixed else 1.0 - float(shares.max()),
        }
        top, *rest = sorted(scores.items(), key=lambda kv: -kv[1])
        margin = top[1] - rest[0][1]
        return (top[0] if margin >= 0.15 else "unknown", margin)

    def _obs(self) -> np.ndarray:
        n = self.max_nodes
        obs = np.zeros(n * NODE_FEATURES + 2 + 2 * len(DEF_ACTIONS) + 1,
                       dtype=np.float32)
        for i in range(n):
            node = self.net.nodes[i]
            o = i * NODE_FEATURES
            obs[o + 0] = float(node.discovered)
            obs[o + 1] = float(node.owned)
            obs[o + 2] = float(node.root)
            obs[o + 3] = float(node.isolated)
            obs[o + 4] = min(node.alert, 5.0) / 5.0
            obs[o + 5] = node.tier / 4.0
            obs[o + 6] = node.value / 10.0
            obs[o + 7] = float(node.vulns_known)
            obs[o + 8] = len(node.services) / 6.0
            obs[o + 9] = float(node.creds_known)
        base = n * NODE_FEATURES
        obs[base] = min(len(self.creds), MAX_ARG) / MAX_ARG
        obs[base + 1] = self.step_count / self.max_steps
        obs[base + 2 + self.last_def_action] = 1.0
        h = base + 2 + len(DEF_ACTIONS)
        total = self.def_hist.sum()
        obs[h:h + len(DEF_ACTIONS)] = self.def_hist / total if total else 0
        obs[h + len(DEF_ACTIONS)] = self.detected_last
        return obs

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
                 defender=None, catalog=None):
        self.n_ws, self.n_srv = n_workstations, n_servers
        self.max_steps = max_steps
        self.defender = defender
        self.catalog = catalog or load_catalog()
        self.max_nodes = 2 + n_workstations + n_servers  # edge + dc + tiers
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
        self.technique_log: list[str] = []
        self.detection_log: list[str] = []

    # ---------- gym API ----------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.net = net_mod.generate(
            seed=seed if seed is not None else int(self.rng.integers(1 << 31)),
            n_workstations=self.n_ws, n_servers=self.n_srv,
        )
        self.creds, self.step_count = [], 0
        self.last_def_action = 0
        self.def_hist[:] = 0
        self.detected_last = 0.0
        self._had_owned = self._got_cred = self._got_root = False
        self._last_success = False
        self.technique_log, self.detection_log = [], []
        return self._obs(), {"action_mask": self.action_mask()}

    def step(self, action: int):
        assert self.net is not None
        a_type, node_id, arg = decode(int(action), self.max_nodes)
        name = ACTIONS[a_type]
        node = self.net.nodes[node_id]
        reward = -0.1
        self.detected_last = 0.0
        self._last_success = False
        self.step_count += 1

        tech = self._technique_for(name, node, arg)
        if tech is None or not self._useful(name, node, arg) or not all(
            self._has_req(r, node, arg) for r in tech.requires
        ):
            reward -= 0.5
        else:
            success = self.rng.random() < tech.success
            reward += self._apply(name, node, arg, tech, success)
            self._detection_roll(tech, node)

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
        elif self._had_owned and not any(
            n.owned for n in self.net.nodes.values()
        ):
            reward += BURNED_PENALTY
            terminated = True
        elif self.step_count >= self.max_steps:
            truncated = True

        info = {
            "action_mask": self.action_mask(),
            "techniques": list(self.technique_log),
            "detected": list(self.detection_log),
        }
        return self._obs(), reward, terminated, truncated, info

    # ---------- action validity + effects ----------

    def _reachable(self, node_id: int) -> bool:
        if node_id == self.net.entry_point:
            return True
        return any(self.net.nodes[m].owned for m in self.net.links[node_id])

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
        if name == "exploit":
            if not (node.vulns_known and arg < len(node.vulns)):
                return None
            return self.catalog.get(node.vulns[arg].technique_id)
        if name == "privesc":
            if arg >= len(node.local_vulns):
                return None
            return self.catalog.get(node.local_vulns[arg].technique_id)
        if name == "lateral":
            for t in self.catalog.values():
                if t.action == "lateral" and node.id in self.creds:
                    return t
            return None
        for t in self.catalog.values():
            if t.action == name:
                return t
        return None

    def _valid(self, a_type: int, node_id: int, arg: int) -> bool:
        node = self.net.nodes[node_id]
        tech = self._technique_for(ACTIONS[a_type], node, arg)
        return (
            tech is not None
            and self._useful(ACTIONS[a_type], node, arg)
            and all(self._has_req(r, node, arg) for r in tech.requires)
        )

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
                self.net.nodes[m].discovered = True
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
            self.detected_last = 1.0
            self.detection_log.append(tech.id)

    # ---------- defender ----------

    def _defender_view(self) -> dict:
        return {
            "alerts": {n.id: n.alert for n in self.net.nodes.values() if n.alert > 0},
            "nodes": self.net.nodes,
        }

    def _apply_defender(self, action: str | None, node_id: int | None):
        idx = DEF_ACTIONS.index(action) if action in DEF_ACTIONS else 0
        self.last_def_action = idx
        self.def_hist[idx] += 1
        if node_id is None or node_id not in self.net.nodes:
            return
        node = self.net.nodes[node_id]
        if action == "isolate":
            node.isolated = True
        elif action == "reimage":
            node.alert = 0.0
            node.vulns_known = node.creds_known = False
            if node.persistent:
                node.root = False
            else:
                node.owned = node.root = node.persistent = False
                self.creds = [c for c in self.creds if c != node_id]
        elif action == "patch":
            node.vulns = []
            node.local_vulns = []
        elif action == "decoy":
            node.decoy = True
        # investigate: reveals truth to a human defender; no env effect

    # ---------- observation ----------

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

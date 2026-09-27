"""Turn-based game loop: RL attacker vs a defender with a noisy view.

Structure per turn: the attacker executes a burst of masked actions (sim
time — real intrusions move faster than SOC shifts), then the defender
gets one action based on alerts alone. The defender never sees ownership,
only noise. That asymmetry is the game.
"""

from artai.env import catalog
from artai.env.redteam_env import ACTIONS, decode

ATTACKER_BURST = 4  # attacker steps per defender turn

_TECHNIQUES = catalog.load()

ATTACKER_BURST = 4  # attacker steps per defender turn


def run_session(attacker, env, defender_policy, eps=0.05, max_turns=40,
                train_lr=None, difficulty=None, seed=None, on_event=None):
    """Play one game. defender_policy: callable(view) -> (action, node).

    attacker: object with act(key, mask, eps); if it also has update() and
    train_lr is set, the attacker fine-tunes on this defender mid-game —
    the 'learns from your moves' hook.
    on_event(entry) fires after every log append — the TUI/replay layer's
    live feed. Returns the debrief dict.
    """
    log = []

    def emit(entry):
        log.append(entry)
        if on_event:
            on_event(entry)

    obs, info = env.reset(seed=seed)
    key = env.state_key()
    prev_defender = env.defender
    env.defender = None  # game layer owns defender timing
    if train_lr is not None and hasattr(attacker, "lr"):
        attacker.lr = train_lr
    done = False
    turn = 0

    try:
        while not done and turn < max_turns:
            turn += 1
            for _ in range(ATTACKER_BURST):
                if done:
                    break
                a = attacker.act(key, info["action_mask"], eps)
                a_type, node_id, arg = decode(a, env.max_nodes)
                n_tech = len(env.technique_log)
                obs, r, term, trunc, info = env.step(a)
                key2 = env.state_key()
                if train_lr is not None and hasattr(attacker, "update"):
                    attacker.update(key, a, r, key2, term or trunc,
                                    info["action_mask"])
                key = key2
                done = term or trunc
                emit({
                    "turn": turn, "actor": "attacker",
                    "action": ACTIONS[a_type], "node": node_id, "arg": arg,
                    "tech": (env.technique_log[-1]
                             if len(env.technique_log) > n_tech else None),
                    "reward": round(r, 2),
                    "detected": bool(env.detected_last),
                    "map": list(env.state_key()[0]),
                })
            if done:
                break

            view = env.defender_view()
            d_action, d_node = defender_policy(view)
            effected = env.apply_defender_action(d_action, d_node)
            emit({"turn": turn, "actor": "defender", "action": d_action,
                  "node": d_node, "effected": effected,
                  "map": list(env.state_key()[0])})
            # a defender turn can end the game (burned every foothold)
            if env._had_owned and not any(
                    n.owned for n in env.net.nodes.values()):
                done = True
    finally:
        env.defender = prev_defender

    report = debrief(env, log, turn, difficulty)
    report["seed"] = seed
    return report


def debrief(env, log, turns: int, difficulty=None) -> dict:
    """ATT&CK-flavored post-game report."""
    attempted = env.technique_log
    detected = env.detection_log
    owned = sum(n.owned for n in env.net.nodes.values())
    first_seen: dict[str, int] = {}
    for e in log:
        tech = e.get("tech")
        if tech in _TECHNIQUES:
            first_seen.setdefault(_TECHNIQUES[tech].tactic, e["turn"])
    kill_chain = [{"tactic": t, "turn": tn}
                  for t, tn in sorted(first_seen.items(),
                                      key=lambda kv: kv[1])]
    d_acts = [e for e in log if e["actor"] == "defender"]
    real = [e for e in d_acts if e["action"] != "pass"]
    return {
        "result": "attacker_win" if env.won else "defender_hold",
        "turns": turns,
        "nodes_owned_end": owned,
        "techniques_attempted": attempted,
        "techniques_detected": detected,
        "techniques_undetected": [t for t in attempted if t not in detected],
        "alerts_raised": sum(1 for e in log if e.get("detected")),
        "difficulty": difficulty,
        "kill_chain": kill_chain,
        "defender_efficiency": {
            "useful": sum(1 for e in real if e.get("effected")),
            "wasted": sum(1 for e in real if e.get("effected") is False),
            "passes": len(d_acts) - len(real),
        },
        "defender_actions": d_acts,
        "log": log,
    }

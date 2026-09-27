"""Defender policies for the game loop: (view, env) -> (action, node_id).

The scripted env defenders have an act(view, rng) interface — adapted here.
The human policy renders the noisy alert view and reads one line of input.
"""

import numpy as np

from artai.env.defender import DEF_NAMES, make as make_env_defender
from artai.env.redteam_env import DEF_ACTIONS


def scripted(name: str, seed: int = 0):
    """Adapter: env defender policies usable in the turn-based game."""
    d = make_env_defender(name)
    rng = np.random.default_rng(seed)
    return lambda view, env: d.act(view, rng)


def from_lines(lines):
    """Scripted human for tests: each line 'pass' or '<action> <node>'."""
    it = iter(lines)

    def policy(view, env):
        try:
            parts = next(it).strip().split()
        except StopIteration:
            return "pass", None
        if not parts or parts[0] == "pass":
            return "pass", None
        return parts[0], int(parts[1]) if len(parts) > 1 else None

    return policy


def human(out=print, inp=input):
    """Interactive CLI defender. Sees alerts, never ground truth."""
    def policy(view, env):
        alerts = view["alerts"]
        out("\n--- defender turn ---")
        out("alerts: " + (", ".join(
            f"n{nid}={a:.1f}" for nid, a in sorted(alerts.items())
        ) or "quiet"))
        out("actions: " + " ".join(DEF_ACTIONS) + "  (e.g. 'patch 4')")
        while True:
            try:
                parts = inp("defender> ").strip().split()
            except EOFError:
                return "pass", None
            if not parts or parts[0] == "pass":
                return "pass", None
            if parts[0] in DEF_ACTIONS:
                node = int(parts[1]) if len(parts) > 1 else None
                return parts[0], node
            out(f"unknown action {parts[0]!r}")

    return policy


def resolve(spec: str, script_lines=None, seed: int = 0):
    if spec == "human":
        return human()
    if spec == "script":
        return from_lines(script_lines or [])
    if spec in DEF_NAMES:
        return scripted(spec, seed=seed)
    raise ValueError(f"unknown defender {spec!r}")

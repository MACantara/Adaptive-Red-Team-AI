"""Defender policies for the game loop: (view) -> (action, node_id).

The scripted env defenders have an act(view, rng) interface — adapted here.
The human policy renders the noisy alert view and reads one line of input.
"""

import numpy as np

from artai.env.defender import DEF_NAMES, make as make_env_defender
from artai.env.redteam_env import DEF_ACTIONS


def scripted(name: str, seed: int = 0, difficulty: int | None = None):
    """Adapter: env defender policies usable in the turn-based game."""
    d = make_env_defender(name, difficulty=difficulty)
    rng = np.random.default_rng(seed)
    return lambda view: d.act(view, rng)


def from_lines(lines):
    """Scripted human for tests: each line 'pass' or '<action> <node>'.

    Validated up front — a bad line must fail fast, not mid-game."""
    parsed = []
    for i, raw in enumerate(lines):
        parts = raw.strip().split()
        if not parts or parts[0] == "pass":
            parsed.append(("pass", None))
            continue
        if parts[0] not in DEF_ACTIONS:
            raise ValueError(f"line {i + 1}: unknown action {parts[0]!r}")
        try:
            node = int(parts[1]) if len(parts) > 1 else None
        except ValueError:
            raise ValueError(f"line {i + 1}: bad node {parts[1]!r}") from None
        parsed.append((parts[0], node))
    it = iter(parsed)
    return lambda view: next(it, ("pass", None))


def parse_line(text: str):
    """'patch 4' -> ('patch', 4); 'pass' -> ('pass', None). Raises
    ValueError on malformed input — callers re-prompt."""
    parts = text.strip().split()
    if not parts or parts[0] == "pass":
        return "pass", None
    if parts[0] not in DEF_ACTIONS:
        raise ValueError(f"unknown action {parts[0]!r}")
    try:
        node = int(parts[1]) if len(parts) > 1 else None
    except ValueError:
        raise ValueError(f"bad node {parts[1]!r}") from None
    return parts[0], node


def human(out=print, inp=input):
    """Interactive CLI defender. Sees alerts, never ground truth."""
    def policy(view):
        alerts = view["alerts"]
        out("\n--- defender turn ---")
        out("alerts: " + (", ".join(
            f"n{nid}={a:.1f}" for nid, a in sorted(alerts.items())
        ) or "quiet"))
        out("actions: " + " ".join(DEF_ACTIONS) + "  (e.g. 'patch 4')")
        while True:
            try:
                return parse_line(inp("defender> "))
            except EOFError:
                return "pass", None
            except ValueError as e:
                out(str(e))
                continue

    return policy


def resolve(spec: str, script_lines=None, seed: int = 0,
            difficulty: int | None = None):
    if spec == "human":
        return human()
    if spec == "script":
        if not script_lines:
            raise ValueError("--defender script needs --script <file>")
        return from_lines(script_lines)
    if spec in DEF_NAMES:
        return scripted(spec, seed=seed, difficulty=difficulty)
    raise ValueError(f"unknown defender {spec!r}")

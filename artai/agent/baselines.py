"""Non-learning baseline attackers: random and a scripted kill-chain heuristic.

The heuristic doubles as the environment's proof-of-winnability — if it can't
reach the crown jewel, the scenario or env is broken.
"""

import numpy as np

from artai.env.redteam_env import ACTIONS, decode

_PRIORITY = (
    "collect", "privesc", "persist", "dump_creds", "lateral",
    "exploit", "enumerate", "scan",
)


class RandomAgent:
    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def act(self, obs, mask) -> int:
        valid = np.flatnonzero(mask)
        return int(valid[self.rng.integers(len(valid))])


def heuristic_action(mask, max_nodes: int) -> int:
    """Kill chain by priority: collect > privesc > creds > lateral > exploit
    > enumerate > scan. Deterministic — lowest index in the highest-priority
    available action type."""
    for name in _PRIORITY:
        t = ACTIONS.index(name)
        for a in np.flatnonzero(mask):
            if decode(int(a), max_nodes)[0] == t:
                return int(a)
    return int(np.flatnonzero(mask)[0])

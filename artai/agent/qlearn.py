"""Tabular Q-learning over env.state_key() — the default learner.

Known ceiling: a per-(scenario, defender) Q-table doesn't transfer across
topologies. Upgrade path: MaskedDQN (agent.dqn) once scenarios need
unseen-network generalization.
"""

import pickle
import random
from collections import deque
from pathlib import Path

import numpy as np


class QLearner:
    def __init__(self, n_actions: int, lr=0.3, gamma=0.95, seed=None,
                 replay_size: int = 4000, replay_k: int = 6):
        self.n_actions = n_actions
        self.lr = lr
        self.gamma = gamma
        self.table: dict[tuple, np.ndarray] = {}
        self.rng = random.Random(seed)
        self.updates = 0
        self.replay = deque(maxlen=replay_size)
        self.replay_k = replay_k

    def _row(self, key) -> np.ndarray:
        row = self.table.get(key)
        if row is None:
            row = self.table[key] = np.zeros(self.n_actions, dtype=np.float32)
        return row

    def act(self, key, mask: np.ndarray, eps: float) -> int:
        valid = np.flatnonzero(mask)
        if len(valid) == 0:
            return 0
        if self.rng.random() < eps:
            return int(valid[self.rng.randrange(len(valid))])
        q = self._row(key)
        best, best_q = int(valid[0]), -np.inf
        for a in valid:
            if q[a] > best_q:
                best, best_q = int(a), q[a]
        return best

    def _bellman(self, key, a, r, key2, done, mask2):
        q = self._row(key)
        q2 = self._row(key2)
        valid2 = np.flatnonzero(mask2)
        next_q = max((q2[x] for x in valid2), default=0.0)
        q[a] += self.lr * (r + self.gamma * (0 if done else next_q) - q[a])
        self.updates += 1

    def update(self, key, a: int, r: float, key2, done: bool, mask2: np.ndarray):
        self.replay.append((key, a, r, key2, done, mask2))
        self._bellman(key, a, r, key2, done, mask2)
        for t in self.rng.sample(
                self.replay, min(self.replay_k, len(self.replay))):
            self._bellman(*t)

    def save(self, path: Path):
        with open(path, "wb") as f:
            pickle.dump({"table": self.table, "updates": self.updates}, f)

    def load(self, path: Path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.table = data["table"]
        self.updates = data.get("updates", 0)

"""Masked DQN agent — Q-values for invalid actions are forced to -inf so the
policy can only ever choose legal moves. Learns over the same observation the
human-facing game exposes, including defender-behavior features."""

from collections import deque
import random

import numpy as np
import torch
import torch.nn as nn


class QNet(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    def __init__(self, obs_dim: int, n_actions: int, lr=3e-4, gamma=0.95,
                 buffer_size=50_000, batch_size=64, target_sync=200,
                 seed: int | None = None):
        if seed is not None:
            torch.manual_seed(seed)
            random.seed(seed)
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_sync = target_sync
        self.device = torch.device("cpu")
        self.online = QNet(obs_dim, n_actions).to(self.device)
        self.target = QNet(obs_dim, n_actions).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.opt = torch.optim.Adam(self.online.parameters(), lr=lr)
        self.buf = deque(maxlen=buffer_size)
        self.steps = 0
        self.last_loss = 0.0

    def act(self, obs: np.ndarray, mask: np.ndarray, eps: float) -> int:
        valid = np.flatnonzero(mask)
        if len(valid) == 0:
            return 0
        if random.random() < eps:
            return int(valid[random.randrange(len(valid))])
        with torch.no_grad():
            q = self.online(torch.as_tensor(obs, dtype=torch.float32))
            q = q.numpy()
        q[mask == 0] = -np.inf
        return int(q.argmax())

    def remember(self, s, a, r, s2, done, mask2):
        self.buf.append((s, a, r, s2, done, mask2))

    def train_step(self) -> float:
        if len(self.buf) < self.batch_size:
            return 0.0
        batch = random.sample(self.buf, self.batch_size)
        s = torch.tensor(np.stack([b[0] for b in batch]))
        a = torch.tensor([b[1] for b in batch])
        r = torch.tensor([b[2] for b in batch], dtype=torch.float32)
        s2 = torch.tensor(np.stack([b[3] for b in batch]))
        done = torch.tensor([b[4] for b in batch], dtype=torch.float32)
        mask2 = torch.tensor(np.stack([b[5] for b in batch]))

        q = self.online(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            q2 = self.target(s2)
            q2[mask2 == 0] = -1e9
            q2max = q2.max(1).values.clamp(min=-1e6)
            target = r + self.gamma * (1 - done) * q2max

        loss = nn.functional.smooth_l1_loss(q, target)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        self.steps += 1
        if self.steps % self.target_sync == 0:
            self.target.load_state_dict(self.online.state_dict())
        self.last_loss = float(loss.item())
        return self.last_loss

    def save(self, path):
        torch.save({"online": self.online.state_dict(),
                    "steps": self.steps}, path)

    def load(self, path):
        ckpt = torch.load(path, weights_only=True)
        self.online.load_state_dict(ckpt["online"])
        self.target.load_state_dict(ckpt["online"])
        self.steps = ckpt.get("steps", 0)

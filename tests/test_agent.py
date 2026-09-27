import numpy as np

from artai.agent.dqn import DQNAgent
from artai.agent.qlearn import QLearner
from artai.agent.baselines import RandomAgent
from artai.env.defender import NoOpDefender, make
from artai.env.redteam_env import RedTeamEnv


def _qlearn_episode(agent, env, eps):
    _, info = env.reset()
    key = env.state_key()
    won = False
    while True:
        a = agent.act(key, info["action_mask"], eps)
        _, r, term, trunc, info = env.step(a)
        key2 = env.state_key()
        agent.update(key, a, r, key2, term or trunc, info["action_mask"])
        key = key2
        won = won or (term and r > 10)
        if term or trunc:
            return won


def test_qlearn_learns_fixed_scenario():
    """Learning contract: trained on a pinned scenario vs a patterned
    defender, the policy must outscore random play on identical luck —
    paired episode rewards, not raw wins (speed and stealth count)."""
    env = RedTeamEnv(defender=make("patch_on_alert"), scenario_seed=9)
    agent = QLearner(env.action_space.n, seed=0)
    for ep in range(300):
        eps = max(0.05, 1.0 - ep / 210)
        _qlearn_episode(agent, env, eps)

    trained, baseline = [], []
    for rep in range(40):
        for tag in ("trained", "baseline"):
            _, info = env.reset(seed=1000 + rep)  # same luck for both
            key = env.state_key()
            rand = RandomAgent(seed=rep)
            total = 0.0
            while True:
                a = (agent.act(key, info["action_mask"], 0.0) if tag == "trained"
                     else rand.act(None, info["action_mask"]))
                _, r, term, trunc, info = env.step(a)
                key = env.state_key()
                total += r
                if term or trunc:
                    break
            (trained if tag == "trained" else baseline).append(total)

    diff = np.asarray(trained) - np.asarray(baseline)
    assert diff.mean() > 2.0


def test_qlearn_respects_mask_and_roundtrips(tmp_path):
    env = RedTeamEnv(defender=NoOpDefender())
    agent = QLearner(env.action_space.n, seed=0)
    env.reset(seed=3)
    key = env.state_key()
    mask = env.action_mask()
    for _ in range(50):
        a = agent.act(key, mask, eps=0.0)
        assert mask[a] == 1
    agent.update(key, a, 1.0, key, False, mask)
    p = tmp_path / "policy.pkl"
    agent.save(p)
    agent2 = QLearner(env.action_space.n, seed=9)
    agent2.load(p)
    assert agent2.table.keys() == agent.table.keys()


def test_dqn_mechanics():
    """DQN is the scale-up path: verify machinery, not convergence."""
    env = RedTeamEnv(defender=NoOpDefender())
    agent = DQNAgent(
        obs_dim=env.observation_space.shape[0],
        n_actions=env.action_space.n,
        seed=0,
    )
    obs, info = env.reset(seed=3)
    agent.remember(obs, 0, 1.0, obs, False, info["action_mask"])
    for _ in range(70):
        agent.train_step()
    a = agent.act(obs, info["action_mask"], eps=0.0)
    assert info["action_mask"][a] == 1


def test_checkpoint_roundtrip(tmp_path):
    env = RedTeamEnv()
    a1 = DQNAgent(env.observation_space.shape[0], env.action_space.n, seed=0)
    obs, info = env.reset(seed=3)
    a1.remember(obs, 0, 1.0, obs, False, info["action_mask"])
    for _ in range(70):
        a1.train_step()
    p = tmp_path / "policy.pt"
    a1.save(p)

    a2 = DQNAgent(env.observation_space.shape[0], env.action_space.n, seed=1)
    a2.load(p)
    assert a2.steps == a1.steps
    import torch
    for p1, p2 in zip(a1.online.parameters(), a2.online.parameters()):
        assert torch.equal(p1, p2)

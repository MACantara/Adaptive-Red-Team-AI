from artai.agent.dqn import DQNAgent
from artai.agent.qlearn import QLearner
from artai.eval import paired_eval, qlearn_policy, random_baseline
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

    res = paired_eval(qlearn_policy(agent), random_baseline(), env, reps=40)
    assert res["reward_diff"] > 2.0


def test_qlearn_adapts_after_defender_swap():
    """Adaptation contract: train vs defender A, hot-swap to defender B.
    After re-training, the adapted table must outscore its own frozen
    pre-swap snapshot on identical luck — 'learns from your moves',
    falsifiable."""
    env = RedTeamEnv(scenario_seed=9)
    agent = QLearner(env.action_space.n, seed=0)

    env.defender = make("patch_on_alert")
    for ep in range(300):
        _qlearn_episode(agent, env, max(0.05, 1.0 - ep / 210))

    frozen = QLearner(env.action_space.n, seed=1)
    frozen.table = {k: v.copy() for k, v in agent.table.items()}

    env.defender = make("scan_and_reimage")
    for _ in range(400):
        _qlearn_episode(agent, env, 0.2)

    res = paired_eval(qlearn_policy(agent), qlearn_policy(frozen),
                      env, reps=40)
    assert res["reward_diff"] > 2.0


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

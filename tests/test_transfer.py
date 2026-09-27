"""Transfer eval protocol: held-out seeds, one policy across fresh maps."""

from artai.agent.qlearn import QLearner
from artai.env.redteam_env import RedTeamEnv
from artai.eval import eval_transfer, qlearn_policy, random_baseline

HELD_OUT = [201, 202, 203, 204]


def test_transfer_eval_is_reproducible():
    """Same policy + same held-out seeds -> identical numbers."""
    env = RedTeamEnv(scenario_seed=0)  # only for action-space sizing
    agent = QLearner(env.action_space.n, seed=0)
    r1 = eval_transfer(qlearn_policy(agent), HELD_OUT)
    r2 = eval_transfer(qlearn_policy(agent), HELD_OUT)
    assert r1 == r2
    assert r1["seeds"] == len(HELD_OUT)


def test_transfer_baseline_is_weak():
    """Sanity floor: an untrained table shouldn't beat random play on
    unseen maps — if it does, the protocol is measuring noise."""
    env = RedTeamEnv(scenario_seed=0)
    agent = QLearner(env.action_space.n, seed=0)
    trained = eval_transfer(qlearn_policy(agent), HELD_OUT)
    baseline = eval_transfer(random_baseline(), HELD_OUT)
    assert trained["reward"] <= baseline["reward"] + 15.0


def test_shared_table_trains_across_maps():
    """The tabular transfer baseline: one table, fresh map each episode."""
    env = RedTeamEnv(scenario_seed=None)  # unpinned -> new map per episode
    agent = QLearner(env.action_space.n, seed=0)
    for _ in range(60):
        _, info = env.reset()
        key = env.state_key()
        while True:
            a = agent.act(key, info["action_mask"], 0.3)
            _, r, term, trunc, info = env.step(a)
            key2 = env.state_key()
            agent.update(key, a, r, key2, term or trunc,
                         info["action_mask"])
            key = key2
            if term or trunc:
                break
    res = eval_transfer(qlearn_policy(agent), HELD_OUT)
    assert agent.table  # it learned something, somewhere
    assert res["seeds"] == len(HELD_OUT)

import numpy as np

from artai.agent.baselines import RandomAgent, heuristic_action
from artai.env.redteam_env import ACTIONS, RedTeamEnv, decode, encode


def pick(mask, max_nodes, a_type=None, node=None):
    for a in np.flatnonzero(mask):
        t, n, _ = decode(int(a), max_nodes)
        if (a_type is None or t == a_type) and (node is None or n == node):
            return int(a)
    return None


def test_reset_returns_mask_with_legal_opening():
    env = RedTeamEnv()
    obs, info = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape
    mask = info["action_mask"]
    assert mask.sum() > 0
    # only discovery-type actions should be legal on a fresh map
    types = {decode(int(a), env.max_nodes)[0] for a in np.flatnonzero(mask)}
    assert types <= {ACTIONS.index("enumerate")}


def test_illegal_action_is_penalized_not_executed():
    env = RedTeamEnv()
    env.reset(seed=0)
    entry = env.net.entry_point
    # exploit without enumerate is masked out and penalized
    a = encode(ACTIONS.index("exploit"), entry, 0, env.max_nodes)
    _, reward, *_ = env.step(a)
    assert reward < -0.4
    assert not env.net.nodes[entry].owned


def test_scripted_entry_chain():
    env = RedTeamEnv()
    env.reset(seed=1)
    entry = env.net.entry_point

    a = pick(env.action_mask(), env.max_nodes, ACTIONS.index("enumerate"), entry)
    env.step(a)
    assert env.net.nodes[entry].vulns_known

    a = pick(env.action_mask(), env.max_nodes, ACTIONS.index("exploit"), entry)
    assert a is not None
    for _ in range(10):  # retry past exploit success rolls
        env.step(a)
        if env.net.nodes[entry].owned:
            break
    assert env.net.nodes[entry].owned

    a = pick(env.action_mask(), env.max_nodes, ACTIONS.index("scan"), entry)
    assert a is not None
    env.step(a)
    assert any(n.discovered and n.tier == 1 for n in env.net.nodes.values())


def test_episode_terminates_or_truncates():
    env = RedTeamEnv(max_steps=30)
    env.reset(seed=2)
    done = False
    for _ in range(30):
        mask = env.action_mask()
        a = int(np.flatnonzero(mask)[0])
        _, _, term, trunc, _ = env.step(a)
        done = term or trunc
        if done:
            break
    assert done


def _run_episode(env, policy_fn, seed):
    obs, info = env.reset(seed=seed)
    total, done, won = 0.0, False, False
    while not done:
        a = policy_fn(obs, info["action_mask"])
        obs, r, term, trunc, info = env.step(a)
        total += r
        won = won or (term and r > 10)  # collect lands the win reward
        done = term or trunc
    return total, won, info


def test_heuristic_completes_kill_chain():
    wins = 0
    for seed in range(10):
        env = RedTeamEnv()
        _, won, _ = _run_episode(
            env, lambda o, m: heuristic_action(m, env.max_nodes), seed
        )
        wins += won
    assert wins >= 5  # should win on most seeds; 5/10 guards against breakage


def test_env_deterministic_under_seed():
    def traj(seed):
        env = RedTeamEnv()
        obs, info = env.reset(seed=seed)
        agent = RandomAgent(seed=seed)
        out = []
        for _ in range(20):
            a = agent.act(obs, info["action_mask"])
            obs, r, t, tr, info = env.step(a)
            out.append((a, round(r, 6)))
            if t or tr:
                break
        return out

    assert traj(11) == traj(11)

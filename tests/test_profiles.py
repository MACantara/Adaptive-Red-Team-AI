"""Per-player policy persistence: namespaced paths, divergent tables."""

from artai.agent.qlearn import QLearner
from artai.env.defender import make
from artai.env.redteam_env import RedTeamEnv
from artai.profiles import policy_path


def _train(env, agent, episodes=40):
    for _ in range(episodes):
        _, info = env.reset()
        key = env.state_key()
        while True:
            a = agent.act(key, info["action_mask"], 0.2)
            _, r, term, trunc, info = env.step(a)
            key2 = env.state_key()
            agent.update(key, a, r, key2, term or trunc,
                         info["action_mask"])
            key = key2
            if term or trunc:
                break
    return agent


def test_policy_path_is_namespaced(tmp_path):
    p = policy_path("training_ground", "alice", base=tmp_path)
    assert p == tmp_path / "training_ground" / "alice.pkl"
    # hostile ids can't escape the directory
    assert policy_path("s", "../evil", base=tmp_path).parent.name == "s"


def test_two_players_divergent_tables(tmp_path):
    """Same scenario, different defenders -> per-player tables differ."""
    env = RedTeamEnv(scenario_seed=9, n_workstations=2, n_servers=2,
                     defender=make("patch_on_alert"))
    alice = _train(env, QLearner(env.action_space.n, seed=0))
    alice.save(policy_path("x", "alice", base=tmp_path))

    env.defender = make("scan_and_reimage")
    bob = _train(env, QLearner(env.action_space.n, seed=1))
    bob.save(policy_path("x", "bob", base=tmp_path))

    re_alice, re_bob = QLearner(1), QLearner(1)
    re_alice.load(tmp_path / "x" / "alice.pkl")
    re_bob.load(tmp_path / "x" / "bob.pkl")
    assert re_alice.table.keys() != re_bob.table.keys()

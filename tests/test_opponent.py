"""Opponent modeling: defender signature in the state key, archetype
classification, and archetype-conditioned learning."""

from artai.agent.baselines import heuristic_action
from artai.agent.qlearn import QLearner
from artai.env.defender import make
from artai.env.redteam_env import DEF_ACTIONS, RedTeamEnv


def _env(defender="noop"):
    return RedTeamEnv(scenario_seed=9, n_workstations=2, n_servers=2,
                      defender=make(defender))


def test_state_key_separates_defender_histories():
    """Same world state, different def_hist -> different keys."""
    env = _env()
    env.reset(seed=0)
    env.step(0)
    env.step(0)
    k0 = env.state_key()

    env.def_hist[DEF_ACTIONS.index("patch")] += 10
    k_patch = env.state_key()
    assert k_patch != k0

    env.def_hist[DEF_ACTIONS.index("patch")] = 0
    env.def_hist[DEF_ACTIONS.index("reimage")] += 10
    k_reimage = env.state_key()
    assert k_reimage != k0 and k_reimage != k_patch

    env.def_hist[:] = 0
    assert env.state_key() == k0


def test_signature_buckets():
    env = _env()
    env.reset(seed=0)
    assert env.defender_signature() == (0, 0)
    env.def_hist[DEF_ACTIONS.index("patch")] = 8
    env.def_hist[DEF_ACTIONS.index("pass")] = 20
    assert env.defender_signature() == (DEF_ACTIONS.index("patch"), 1)
    env.def_hist[DEF_ACTIONS.index("pass")] = 2
    assert env.defender_signature() == (DEF_ACTIONS.index("patch"), 2)


def test_defender_guess_identifies_scripted_defenders():
    """Heuristic attacker for ~40 steps gives each defender's policy away."""
    expected = {"noop": "quiet", "random": "erratic",
                "patch_on_alert": "patcher",
                "scan_and_reimage": "reimager"}
    correct = 0
    for name, want in expected.items():
        env = _env(name)
        _, info = env.reset(seed=0)
        for _ in range(40):
            _, _, term, trunc, info = env.step(
                heuristic_action(info["action_mask"], env.max_nodes))
            if term or trunc:
                break
        guess, conf = env.defender_guess()
        if guess == want:
            correct += 1
        else:
            print(f"{name}: guessed {guess} (conf {conf:.2f}), want {want}")
    assert correct >= 3


def test_state_key_separates_archetypes():
    """Identical world state under different defender histories lands on
    different archetype slots in the key."""
    env = _env()
    env.reset(seed=0)
    env.step(0)
    env.def_hist[DEF_ACTIONS.index("patch")] = 12
    assert env.defender_guess()[0] == "patcher"
    k_patch = env.state_key()

    env.def_hist[:] = 0
    env.def_hist[DEF_ACTIONS.index("reimage")] = 12
    assert env.defender_guess()[0] == "reimager"
    k_reimage = env.state_key()

    assert k_patch[-1] != k_reimage[-1]  # archetype slot differs
    assert k_patch != k_reimage


def test_qlearn_rows_separate_archetypes():
    """Mixed-defender training produces table rows in >=2 archetypes —
    the state space actually splits per opponent type."""
    env = _env("patch_on_alert")
    agent = QLearner(env.action_space.n, lr=0.3, seed=0)
    for ep in range(80):
        env.defender = make("patch_on_alert" if ep < 40
                            else "scan_and_reimage")
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
    archetypes = {k[-1] for k in agent.table}
    assert len(archetypes) >= 2, f"single archetype: {archetypes}"

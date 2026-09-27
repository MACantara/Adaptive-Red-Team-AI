"""Opponent modeling: defender signature in the state key, archetype
classification, and archetype-conditioned learning."""

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

import numpy as np

from artai.agent.baselines import heuristic_action
from artai.env.defender import ScanAndReimageDefender, make
from artai.env.redteam_env import ACTIONS, RedTeamEnv


def _run(env, seed):
    obs, info = env.reset(seed=seed)
    total, won, steps = 0.0, False, 0
    while True:
        a = heuristic_action(info["action_mask"], env.max_nodes)
        obs, r, term, trunc, info = env.step(a)
        total += r
        steps += 1
        won = won or (term and r > 10)
        if term or trunc:
            return total, won, steps


def test_reimage_evicts_attacker():
    env = RedTeamEnv()
    env.reset(seed=0)
    n0 = env.net.nodes[0]
    n0.owned = True
    n0.alert = 2.0
    env._apply_defender("reimage", 0)
    assert not n0.owned and not n0.root and n0.alert == 0.0


def test_reimage_keeps_persistent_foothold():
    env = RedTeamEnv()
    env.reset(seed=0)
    n0 = env.net.nodes[0]
    n0.owned = n0.root = n0.persistent = True
    env._apply_defender("reimage", 0)
    assert n0.owned and not n0.root


def test_isolate_blocks_reachability():
    env = RedTeamEnv()
    env.reset(seed=0)
    # own a workstation; its deeper neighbors are reachable only through it
    ws = env.net.nodes[1]
    ws.owned = True
    deeper = [m for m in env.net.links[1] if env.net.nodes[m].tier >= 2]
    assert deeper and all(env._reachable(m) for m in deeper)
    env._apply_defender("isolate", 1)
    assert not any(env._reachable(m) for m in deeper)


def test_tier0_immune_to_isolate_and_patch():
    """The edge is the internet — evictable via reimage, never hardenable."""
    env = RedTeamEnv()
    env.reset(seed=0)
    n0 = env.net.nodes[0]
    n0.owned = True
    vulns_before = list(n0.vulns)
    env._apply_defender("isolate", 0)
    env._apply_defender("patch", 0)
    assert not n0.isolated
    assert n0.vulns == vulns_before


def test_active_defender_hurts_heuristic():
    wins_clean, wins_defended = 0, 0
    for seed in range(12):
        _, won, _ = _run(RedTeamEnv(), seed)
        wins_clean += won
        env = RedTeamEnv(defender=ScanAndReimageDefender(probability=0.9))
        _, won, _ = _run(env, seed)
        wins_defended += won
    assert wins_defended < wins_clean


def test_make_ladder():
    for name in ("noop", "random", "patch_on_alert", "scan_and_reimage"):
        d = make(name)
        assert d.name == name

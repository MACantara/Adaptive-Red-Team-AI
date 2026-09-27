from artai.agent.qlearn import QLearner
from artai.env.redteam_env import RedTeamEnv
from artai.game.defenders import from_lines, resolve
from artai.game.session import run_session


def _env():
    return RedTeamEnv(scenario_seed=9)


def test_session_completes_end_to_end():
    env = _env()
    agent = QLearner(env.action_space.n, seed=0)
    report = run_session(agent, env, from_lines(["pass"] * 60), eps=1.0)
    assert report["result"] in ("attacker_win", "defender_hold")
    assert report["turns"] >= 1
    assert any(e["actor"] == "attacker" for e in report["log"])
    assert any(e["actor"] == "defender" for e in report["log"])


def test_scripted_defender_actions_apply():
    env = _env()
    agent = QLearner(env.action_space.n, seed=0)
    report = run_session(
        agent, env, from_lines(["patch 1", "isolate 2", "pass"] * 20),
        eps=1.0)
    acts = [e["action"] for e in report["defender_actions"]]
    assert {"patch", "isolate", "pass"} & set(acts)


def test_debrief_technique_fields():
    env = _env()
    agent = QLearner(env.action_space.n, seed=0)
    report = run_session(agent, env, from_lines(["pass"] * 60), eps=1.0)
    attempted = report["techniques_attempted"]
    detected = report["techniques_detected"]
    assert set(report["techniques_undetected"]) == set(attempted) - set(detected)
    for t in attempted:
        assert t.startswith("T")


def test_debrief_v2_kill_chain_and_efficiency():
    env = _env()
    agent = QLearner(env.action_space.n, seed=0)
    report = run_session(
        agent, env,
        from_lines(["patch 1", "investigate 1", "pass"] * 20),
        eps=1.0, difficulty=2)
    kc = report["kill_chain"]
    assert kc and all("tactic" in e and "turn" in e for e in kc)
    turns = [e["turn"] for e in kc]
    assert turns == sorted(turns)  # monotone — it's a timeline
    eff = report["defender_efficiency"]
    assert eff["useful"] + eff["wasted"] + eff["passes"] == len(
        report["defender_actions"])
    assert report["difficulty"] == 2


def test_resolve_specs():
    assert callable(resolve("random"))
    assert callable(resolve("script", ["pass"]))
    try:
        resolve("bogus")
    except ValueError:
        pass
    else:
        raise AssertionError("bad defender spec accepted")

"""TUI smoke: spectate session populates widgets; replay scrubs a debrief."""

import json

from artai.agent.qlearn import QLearner
from artai.env.redteam_env import RedTeamEnv
from artai.game.tui import RedTeamApp, ReplayApp, map_text


async def test_spectate_session_completes():
    env = RedTeamEnv(scenario_seed=9)
    agent = QLearner(env.action_space.n, seed=0)
    app = RedTeamApp(agent, env, "noop", difficulty=1, spectate=True,
                     eps=0.5, speed=0)
    async with app.run_test() as pilot:
        for _ in range(400):
            await pilot.pause(0.05)
            if getattr(app, "_report", None):
                break
        assert app._report["result"] in ("attacker_win", "defender_hold")
        rendered = str(app.query_one("#map").render())
        assert "n0" in rendered and "edge" in rendered


async def test_replay_scrubs_debrief(tmp_path):
    report = {
        "result": "attacker_win", "turns": 2,
        "kill_chain": [{"tactic": "discovery", "turn": 1}],
        "log": [
            {"turn": 1, "actor": "attacker", "action": "scan", "node": 0,
             "arg": 0, "tech": "T1046", "reward": 0.2, "detected": False,
             "map": [3, 0, 0]},
            {"turn": 1, "actor": "defender", "action": "pass", "node": None,
             "effected": True, "map": [3, 0, 0]},
        ],
    }
    path = tmp_path / "d.json"
    path.write_text(json.dumps(report))
    app = ReplayApp(path)
    async with app.run_test() as pilot:
        await pilot.pause(0.1)
        assert app.pos == 0
        app.action_step(1)
        await pilot.pause(0.05)
        assert app.pos == 1
        assert "scan" in str(app.query_one("#log").lines)


def test_map_text_renders_codes():
    t = map_text([0, 3, 9])  # unseen, owned, rooted+isolated
    s = str(t)
    assert "unseen" in s and "owned" in s and "[cut]" in s

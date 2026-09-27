"""Campaign mode: legs share one agent, table carries forward."""

import pytest

from artai.campaign import load_campaign, run_campaign


def _legs():
    return [
        {"scenario": "training_ground", "script_lines": ["pass"] * 60},
        {"scenario": "contested_dmz", "difficulty": 1,
         "script_lines": ["pass"] * 60},
        {"scenario": "contested_dmz", "difficulty": 3,
         "script_lines": ["patch 1"] * 60},
    ]


def test_campaign_runs_end_to_end():
    agent, summary = run_campaign(_legs(), eps=0.3, out=lambda *a: None)
    assert len(summary["legs"]) == 3
    assert summary["wins"] <= 3
    # leg 2+ started with the table leg 1 built — carry-over is real
    assert summary["legs"][1]["states_before"] > 0
    assert summary["legs"][2]["states_before"] >= \
        summary["legs"][1]["states_before"]
    for leg in summary["legs"]:
        assert leg["result"] in ("attacker_win", "defender_hold")
        assert leg["debrief"]["kill_chain"] is not None


def test_mismatched_leg_dims_rejected():
    legs = [{"scenario": "training_ground", "script_lines": ["pass"] * 10},
            {"scenario": "sprawl", "script_lines": ["pass"] * 10}]
    with pytest.raises(ValueError, match="map dimensions"):
        run_campaign(legs, eps=0.3, out=lambda *a: None)


def test_load_campaign_named():
    legs = load_campaign("escalation_arc")
    assert len(legs) == 3
    with pytest.raises(ValueError):
        load_campaign("nope")

import pytest

from artai.env.catalog import load, load_scenario, validate


def test_default_catalog_loads():
    cat = load()
    assert len(cat) >= 10
    for tid, t in cat.items():
        assert t.id == tid
        assert t.action
        assert 0 <= t.detectability <= 1


def test_rejects_unknown_action():
    with pytest.raises(ValueError, match="unknown action"):
        validate("T9999", {
            "name": "x", "tactic": "y", "action": "nope",
            "detectability": 0.5, "cost": 1, "success": 1.0,
        })


def test_rejects_unknown_requirement():
    with pytest.raises(ValueError, match="unknown requires"):
        validate("T9999", {
            "name": "x", "tactic": "y", "action": "scan",
            "requires": ["teleportation"],
            "detectability": 0.5, "cost": 1, "success": 1.0,
        })


def test_rejects_out_of_range_detectability():
    with pytest.raises(ValueError, match="detectability"):
        validate("T9999", {
            "name": "x", "tactic": "y", "action": "scan",
            "detectability": 1.5, "cost": 1, "success": 1.0,
        })


def test_scenario_loader():
    s = load_scenario("training_ground")
    assert s["seed"] == 9
    s = load_scenario("9")
    assert s["seed"] == 9
    s = load_scenario(None)
    assert s["seed"] is None
    with pytest.raises(ValueError, match="unknown scenario"):
        load_scenario("nowhere")

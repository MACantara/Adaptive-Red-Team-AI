"""Scripted defender policies — the training ladder.

act(view, rng) -> (action_name, node_id | None). Actions map to DEF_ACTIONS.
The view exposes only alerts and node metadata — defenders never see ground
truth ownership, matching what a human would see in the game layer.
"""

DEF_NAMES = ("noop", "random", "patch_on_alert", "scan_and_reimage")


class NoOpDefender:
    name = "noop"

    def act(self, view, rng):
        return "pass", None


class RandomDefender:
    name = "random"

    def __init__(self, probability: float = 0.3):
        self.probability = probability

    def act(self, view, rng):
        nodes = list(view["nodes"])
        if rng.random() >= self.probability or not nodes:
            return "pass", None
        action = rng.choice(["isolate", "reimage", "patch", "decoy"])
        return action, int(nodes[rng.integers(len(nodes))])


class PatchOnAlertDefender:
    """Patch triage: fixes the host with the most lifetime noise.

    Uses cumulative alerts — a SOC patches chronically suspicious boxes,
    not whatever blinked last. The edge node draws constant noise yet is
    unpatchable, which leaves the attacker working room.
    """

    name = "patch_on_alert"

    def __init__(self, probability: float = 1.0):
        self.probability = probability

    def act(self, view, rng):
        alerts = view["alert_totals"]
        if not alerts or rng.random() >= self.probability:
            return "pass", None
        target = max(alerts, key=alerts.get)
        if view["nodes"][target].vuln_count:
            return "patch", target
        return "isolate", target


class ScanAndReimageDefender:
    """CyberBattleSim-style: periodically scans, reimages the noisiest host."""

    name = "scan_and_reimage"

    def __init__(self, probability: float = 0.6):
        self.probability = probability

    def act(self, view, rng):
        alerts = view["alerts"]
        if not alerts or rng.random() >= self.probability:
            return "pass", None
        return "reimage", max(alerts, key=alerts.get)


TIER_PROB = {1: 0.3, 2: 0.6, 3: 0.9}


def make(name: str, difficulty: int | None = None, **kw):
    cls = {
        "noop": NoOpDefender,
        "random": RandomDefender,
        "patch_on_alert": PatchOnAlertDefender,
        "scan_and_reimage": ScanAndReimageDefender,
    }[name]
    if difficulty is not None:
        if difficulty not in TIER_PROB:
            raise ValueError(f"difficulty must be 1-3, got {difficulty}")
        if name != "noop":
            kw.setdefault("probability", TIER_PROB[difficulty])
    d = cls(**kw)
    d.difficulty = difficulty
    return d

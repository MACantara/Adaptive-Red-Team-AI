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

    def act(self, view, rng):
        nodes = list(view["nodes"])
        action = rng.choice(["pass", "isolate", "reimage", "patch", "decoy"])
        if action == "pass" or not nodes:
            return "pass", None
        return action, int(nodes[rng.integers(len(nodes))])


class PatchOnAlertDefender:
    name = "patch_on_alert"

    def act(self, view, rng):
        alerts = view["alerts"]
        if not alerts:
            return "pass", None
        target = max(alerts, key=alerts.get)
        if view["nodes"][target].vulns:
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


def make(name: str, **kw):
    return {
        "noop": NoOpDefender,
        "random": RandomDefender,
        "patch_on_alert": PatchOnAlertDefender,
        "scan_and_reimage": ScanAndReimageDefender,
    }[name](**kw)

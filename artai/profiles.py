"""Per-player policy persistence: policies/<scenario>/<player_id>.pkl.

The adversary you train against Alice is not the one Bob meets — each
player's defender history tunes their own table.
"""

import re
from pathlib import Path

POLICIES = Path("policies")


def policy_path(scenario: str, player_id: str,
                base: Path = POLICIES) -> Path:
    player = re.sub(r"[^A-Za-z0-9_-]", "_", player_id)
    return base / str(scenario) / f"{player}.pkl"

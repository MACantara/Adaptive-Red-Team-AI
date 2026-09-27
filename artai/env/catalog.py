"""ATT&CK technique catalog loader and validator."""

from dataclasses import dataclass
from pathlib import Path
import yaml

KNOWN_ACTIONS = {
    "scan", "enumerate", "exploit", "dump_creds",
    "lateral", "privesc", "persist", "collect",
}
KNOWN_REQUIRES = {
    "discovered", "reachable", "credential", "user_access", "root",
    "remote_vuln", "local_vuln", "not_isolated", "crown_jewel",
}

_DEFAULT = Path(__file__).resolve().parent.parent / "kb" / "techniques.yaml"


@dataclass(frozen=True)
class Technique:
    id: str
    name: str
    tactic: str
    action: str
    requires: tuple[str, ...]
    detectability: float
    cost: float
    success: float


def validate(tech_id: str, raw: dict) -> Technique:
    try:
        t = Technique(
            id=tech_id,
            name=str(raw["name"]),
            tactic=str(raw["tactic"]),
            action=str(raw["action"]),
            requires=tuple(raw.get("requires", ())),
            detectability=float(raw["detectability"]),
            cost=float(raw["cost"]),
            success=float(raw["success"]),
        )
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(f"technique {tech_id}: malformed entry ({e})") from e

    if t.action not in KNOWN_ACTIONS:
        raise ValueError(f"technique {tech_id}: unknown action {t.action!r}")
    unknown = set(t.requires) - KNOWN_REQUIRES
    if unknown:
        raise ValueError(f"technique {tech_id}: unknown requires {sorted(unknown)}")
    for f, v in (("detectability", t.detectability), ("success", t.success)):
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"technique {tech_id}: {f}={v} out of [0,1]")
    return t


def load(path: Path | None = None) -> dict[str, Technique]:
    raw = yaml.safe_load((path or _DEFAULT).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("catalog must be a mapping of technique id -> entry")
    return {tid: validate(tid, entry) for tid, entry in raw.items()}


_SCENARIOS = Path(__file__).resolve().parent.parent / "kb" / "scenarios.yaml"


def load_scenario(spec: str | int | None, path: Path | None = None) -> dict:
    """Resolve a scenario: name from kb/scenarios.yaml, int seed, or None."""
    if spec is None:
        return {"seed": None, "n_workstations": 3, "n_servers": 3}
    if isinstance(spec, int) or (isinstance(spec, str) and spec.isdigit()):
        return {"seed": int(spec), "n_workstations": 3, "n_servers": 3}
    raw = yaml.safe_load((path or _SCENARIOS).read_text(encoding="utf-8"))
    if spec not in raw:
        raise ValueError(
            f"unknown scenario {spec!r} — have {sorted(raw)}")
    entry = dict(raw[spec])
    return {
        "seed": entry.get("seed"),
        "n_workstations": int(entry.get("n_workstations", 3)),
        "n_servers": int(entry.get("n_servers", 3)),
    }

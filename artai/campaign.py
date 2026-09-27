"""Campaign mode: ordered legs over scenario packs, table carry-over.

  python -m artai.campaign --name escalation_arc --player-id alice --learn

Each leg is a fresh pinned arena at escalating defender pressure. The
agent's Q-table survives between legs — the campaign measures adaptation
under changing opposition, not per-map memorization.
"""

import argparse
import json
import sys
from pathlib import Path

import yaml

from artai.agent.qlearn import QLearner
from artai.env.catalog import load_scenario
from artai.env.defender import DEF_NAMES
from artai.env.redteam_env import RedTeamEnv
from artai.game import defenders as gdef
from artai.game.session import run_session
from artai.profiles import policy_path

_CAMPAIGNS = Path(__file__).resolve().parent / "kb" / "campaigns.yaml"


def load_campaign(name: str, path: Path | None = None) -> list[dict]:
    raw = yaml.safe_load((path or _CAMPAIGNS).read_text(encoding="utf-8"))
    if name not in raw:
        raise ValueError(f"unknown campaign {name!r} — have {sorted(raw)}")
    legs = raw[name]["legs"]
    if not legs:
        raise ValueError(f"campaign {name!r} has no legs")
    return legs


def run_campaign(legs: list[dict], defender_override: str | None = None,
                 policy: Path | None = None, learn: bool = False,
                 eps: float = 0.05, out=print):
    """Play every leg with one shared agent. Returns (agent, summary)."""
    agent: QLearner | None = None
    leg_reports = []
    for i, leg in enumerate(legs):
        scen = load_scenario(leg["scenario"])
        env = RedTeamEnv(scenario_seed=scen["seed"],
                         n_workstations=scen["n_workstations"],
                         n_servers=scen["n_servers"])
        if agent is None:
            agent = QLearner(env.action_space.n, seed=0)
            if policy and policy.exists():
                agent.load(policy)
                out(f"loaded {policy} ({len(agent.table)} states)")
        elif env.action_space.n != agent.n_actions:
            raise ValueError(
                f"leg {i} ({leg['scenario']}) action space "
                f"{env.action_space.n} != {agent.n_actions} — campaign legs "
                f"must share map dimensions")
        name = (defender_override or leg.get("defender")
                or scen["defender"] or "random")
        tier = leg.get("difficulty") or scen["difficulty"]
        defender = gdef.resolve(name, leg.get("script_lines"),
                                difficulty=tier)
        out(f"leg {i + 1}/{len(legs)}: {leg['scenario']} vs {name} "
            f"(tier {tier})")
        states_before = len(agent.table)
        report = run_session(agent, env, defender, eps=eps,
                             train_lr=0.1 if learn else None,
                             difficulty=tier)
        leg_reports.append({
            "leg": i + 1, "scenario": leg["scenario"], "defender": name,
            "difficulty": tier, "result": report["result"],
            "turns": report["turns"],
            "states_before": states_before,
            "debrief": report,
        })
        out(f"  -> {report['result']} in {report['turns']} turns")
    wins = sum(1 for r in leg_reports if r["result"] == "attacker_win")
    return agent, {
        "legs": leg_reports,
        "wins": wins,
        "result": "campaign_clear" if wins == len(legs) else "campaign_held",
        "final_states": len(agent.table),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="escalation_arc",
                   help="campaign name from kb/campaigns.yaml")
    p.add_argument("--defender", default=None,
                   choices=["human", "script", *DEF_NAMES],
                   help="override every leg's defender policy")
    p.add_argument("--script", type=Path, default=None,
                   help="defender script file for 'script' legs")
    p.add_argument("--player-id", default=None)
    p.add_argument("--policy", type=Path, default=None)
    p.add_argument("--learn", action="store_true",
                   help="fine-tune mid-game and save the campaign table")
    p.add_argument("--eps", type=float, default=0.05)
    p.add_argument("--debrief", type=Path, default=None,
                   help="write the campaign summary JSON here")
    args = p.parse_args()

    legs = load_campaign(args.name)
    script_lines = (args.script.read_text().splitlines()
                    if args.script else None)
    for leg in legs:
        leg.setdefault("script_lines", script_lines)

    policy = args.policy or (
        policy_path(f"campaign_{args.name}", args.player_id)
        if args.player_id else None)
    agent, summary = run_campaign(legs, defender_override=args.defender,
                                  policy=policy, learn=args.learn,
                                  eps=args.eps)

    print(f"\ncampaign: {summary['result']} — "
          f"{summary['wins']}/{len(legs)} legs won")
    if args.debrief:
        args.debrief.write_text(json.dumps(summary, indent=2))
        print(f"debrief -> {args.debrief}")
    if args.learn and policy is not None:
        agent.save(policy)
        print(f"adapted policy -> {policy} ({summary['final_states']} states)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

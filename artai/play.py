"""Human-vs-AI entrypoint.

  python -m artai.play --defender human --scenario 9 --policy runs/x/policy.pkl

The AI loads its checkpoint, plays one game, optionally fine-tunes on your
defender moves (--learn), and saves the adapted table back — so the
adversary you face next session is the one that studied you.
"""

import argparse
import json
import sys
from pathlib import Path

from artai.agent.qlearn import QLearner
from artai.env.catalog import load_scenario
from artai.env.defender import DEF_NAMES
from artai.env.redteam_env import RedTeamEnv
from artai.game import defenders as gdef
from artai.game.session import run_session
from artai.profiles import policy_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--defender", default=None,
                   choices=["human", "script", *DEF_NAMES],
                   help="defender policy; defaults to the scenario's own")
    p.add_argument("--difficulty", type=int, choices=(1, 2, 3), default=None,
                   help="defender tier; defaults to the scenario's own")
    p.add_argument("--scenario", default="training_ground",
                   help="scenario name, a map seed, or 'none'")
    p.add_argument("--policy", type=Path, default=None,
                   help="explicit checkpoint path; overrides --player-id")
    p.add_argument("--player-id", default=None,
                   help="use policies/<scenario>/<id>.pkl — the adversary "
                        "remembers you across sessions")
    p.add_argument("--script", type=Path, default=None,
                   help="defender script file, one 'action node' per line")
    p.add_argument("--eps", type=float, default=0.05)
    p.add_argument("--learn", action="store_true",
                   help="fine-tune the policy on this defender mid-game")
    p.add_argument("--debrief", type=Path, default=None,
                   help="write the post-game report JSON here")
    args = p.parse_args()

    scen = load_scenario(None if args.scenario == "none" else args.scenario)
    env = RedTeamEnv(scenario_seed=scen["seed"],
                     n_workstations=scen["n_workstations"],
                     n_servers=scen["n_servers"])
    policy = args.policy or (
        policy_path(args.scenario, args.player_id)
        if args.player_id else Path("policy.pkl"))
    agent = QLearner(env.action_space.n, seed=0)
    if policy.exists():
        # checkpoints are trusted artifacts — pickle load, don't point at
        # files you didn't produce
        agent.load(policy)
        print(f"loaded {policy} ({len(agent.table)} states)")
    else:
        print(f"no checkpoint at {policy} — playing untrained policy")

    script_lines = (args.script.read_text().splitlines()
                    if args.script else None)
    defender_name = args.defender or scen.get("defender") or "random"
    difficulty = args.difficulty or scen["difficulty"]
    defender = gdef.resolve(defender_name, script_lines,
                            difficulty=difficulty)

    report = run_session(agent, env, defender, eps=args.eps,
                         train_lr=0.1 if args.learn else None,
                         difficulty=difficulty)

    print(f"\nresult: {report['result']} in {report['turns']} turns "
          f"(defender {defender_name}, tier {difficulty})")
    print(f"attempted: {report['techniques_attempted']}")
    print(f"detected : {report['techniques_detected']}")
    print(f"unseen   : {report['techniques_undetected']}")
    if args.debrief:
        args.debrief.write_text(json.dumps(report, indent=2))
        print(f"debrief -> {args.debrief}")
    if args.learn:
        agent.save(policy)
        print(f"adapted policy -> {policy} ({len(agent.table)} states)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

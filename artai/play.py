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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--defender", default="random",
                   choices=["human", "script", *DEF_NAMES])
    p.add_argument("--scenario", default="training_ground",
                   help="scenario name, a map seed, or 'none'")
    p.add_argument("--policy", type=Path, default=Path("policy.pkl"))
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
    agent = QLearner(env.action_space.n, seed=0)
    if args.policy.exists():
        # checkpoints are trusted artifacts — pickle load, don't point at
        # files you didn't produce
        agent.load(args.policy)
        print(f"loaded {args.policy} ({len(agent.table)} states)")
    else:
        print(f"no checkpoint at {args.policy} — playing untrained policy")

    script_lines = (args.script.read_text().splitlines()
                    if args.script else None)
    defender = gdef.resolve(args.defender, script_lines)

    report = run_session(agent, env, defender, eps=args.eps,
                         train_lr=0.1 if args.learn else None)

    print(f"\nresult: {report['result']} in {report['turns']} turns")
    print(f"attempted: {report['techniques_attempted']}")
    print(f"detected : {report['techniques_detected']}")
    print(f"unseen   : {report['techniques_undetected']}")
    if args.debrief:
        args.debrief.write_text(json.dumps(report, indent=2))
        print(f"debrief -> {args.debrief}")
    if args.learn:
        agent.save(args.policy)
        print(f"adapted policy -> {args.policy} ({len(agent.table)} states)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

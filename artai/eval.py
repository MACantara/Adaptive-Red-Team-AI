"""Paired-seed policy evaluation — the project's measurement substrate.

Two policies play the same episode seeds on the same env: identical maps,
identical detection/defender rolls. Differences in outcome are policy
quality, not luck. Win rates lie under variance; mean reward diff is the
signal.

Usage: python -m artai.eval --policy runs/x/policy.pkl \
         --defender patch_on_alert --scenario training_ground
"""

import argparse
import sys
from pathlib import Path

import numpy as np

from artai.agent.baselines import RandomAgent
from artai.agent.qlearn import QLearner
from artai.env.catalog import load_scenario
from artai.env.defender import DEF_NAMES, make as make_defender
from artai.env.redteam_env import RedTeamEnv


def paired_eval(make_a, make_b, env, reps=40, seed0=1000):
    """Run both policies on identical episode seeds.

    make_x(rep) -> policy callable (key, mask) -> action. Factories give
    fresh per-rep policies so stochastic baselines can reseed while the
    environment's rng (and hence its luck) stays paired.
    Returns {"reward_a", "reward_b", "reward_diff", "wins_a", "wins_b"}.
    """
    rewards = {"a": [], "b": []}
    wins = {"a": 0, "b": 0}
    for rep in range(reps):
        for tag, mk in (("a", make_a), ("b", make_b)):
            policy = mk(rep)
            _, info = env.reset(seed=seed0 + rep)
            key = env.state_key()
            total = 0.0
            while True:
                _, r, term, trunc, info = env.step(policy(key, info["action_mask"]))
                key = env.state_key()
                total += r
                if term or trunc:
                    wins[tag] += env.won
                    break
            rewards[tag].append(total)
    ra, rb = np.asarray(rewards["a"]), np.asarray(rewards["b"])
    return {
        "reward_a": float(ra.mean()), "reward_b": float(rb.mean()),
        "reward_diff": float((ra - rb).mean()),
        "wins_a": wins["a"], "wins_b": wins["b"],
    }


def qlearn_policy(agent: QLearner):
    """Deterministic greedy policy; same across reps."""
    return lambda rep: (lambda key, mask: agent.act(key, mask, 0.0))


def random_baseline():
    """RandomAgent reseeded per rep — its own luck varies, the env's is paired."""
    return lambda rep: (lambda key, mask: RandomAgent(seed=rep).act(None, mask))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", type=Path, required=True)
    p.add_argument("--defender", default="patch_on_alert", choices=DEF_NAMES)
    p.add_argument("--scenario", default="training_ground",
                   help="scenario name, a map seed, or 'none'")
    p.add_argument("--reps", type=int, default=40)
    p.add_argument("--baseline", choices=("random",), default="random")
    args = p.parse_args()

    if not args.policy.exists():
        print(f"error: no checkpoint at {args.policy}", file=sys.stderr)
        return 1
    scen = load_scenario(None if args.scenario == "none" else args.scenario)
    env = RedTeamEnv(scenario_seed=scen["seed"],
                     n_workstations=scen["n_workstations"],
                     n_servers=scen["n_servers"],
                     defender=make_defender(args.defender))
    agent = QLearner(env.action_space.n, seed=0)
    agent.load(args.policy)

    res = paired_eval(qlearn_policy(agent), random_baseline(), env,
                      reps=args.reps)
    print(f"policy   : {args.policy}")
    print(f"defender : {args.defender}   scenario: {args.scenario}")
    print(f"wins     : {res['wins_a']}/{args.reps} vs baseline "
          f"{res['wins_b']}/{args.reps}")
    print(f"reward   : {res['reward_a']:.2f} vs {res['reward_b']:.2f} "
          f"(diff {res['reward_diff']:+.2f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

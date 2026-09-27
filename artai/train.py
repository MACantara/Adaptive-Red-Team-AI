"""Training entrypoint: RL adversary vs a defender population.

Usage: python -m artai.train --episodes 2000 [--algo qlearn|dqn] [--resume x.pt]
Writes JSONL metrics to runs/<run>/metrics.jsonl and a policy checkpoint to
runs/<run>/policy.{pkl,pt}. --resume continues from a checkpoint.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from artai.agent.dqn import DQNAgent
from artai.agent.qlearn import QLearner
from artai.env.catalog import load_scenario
from artai.env.defender import make as make_defender
from artai.env.redteam_env import RedTeamEnv

DEFENDER_POPULATION = ("noop", "random", "patch_on_alert", "scan_and_reimage")


def train(episodes: int, run_dir: Path, algo: str = "qlearn",
          resume: Path | None = None, seed: int = 0,
          scenario: str | int | None = None,
          eps_start=1.0, eps_end=0.05, eps_decay_episodes=0.6,
          quiet: bool = False):
    run_dir.mkdir(parents=True, exist_ok=True)
    scen = load_scenario(scenario)
    env = RedTeamEnv(scenario_seed=scen["seed"],
                     n_workstations=scen["n_workstations"],
                     n_servers=scen["n_servers"])

    if algo == "qlearn":
        agent = QLearner(env.action_space.n, seed=seed)
        ckpt_path = resume or run_dir / "policy.pkl"
    else:
        agent = DQNAgent(env.observation_space.shape[0],
                         env.action_space.n, seed=seed)
        ckpt_path = resume or run_dir / "policy.pt"
    if resume is not None and not resume.exists():
        print(f"warning: --resume {resume} not found; training fresh")
    if resume and resume.exists():
        agent.load(resume)

    rng = np.random.default_rng(seed)
    eps_decay = eps_decay_episodes * episodes
    recent: list[float] = []
    t0 = time.time()
    with open(run_dir / "metrics.jsonl", "a", encoding="utf-8") as log:
        for ep in range(episodes):
            if ep < episodes * 0.15:  # warmup: learn mechanics before opponents
                defender = make_defender("noop")
            else:
                defender = make_defender(
                    DEFENDER_POPULATION[rng.integers(len(DEFENDER_POPULATION))]
                )
            env.defender = defender
            obs, info = env.reset(seed=seed + ep)  # reproducible runs
            eps = max(eps_end, eps_start - (eps_start - eps_end) * ep / max(eps_decay, 1))
            key = env.state_key()

            total, won = 0.0, False
            while True:
                a = agent.act(key if algo == "qlearn" else obs,
                              info["action_mask"], eps)
                prev_obs = obs
                obs, r, term, trunc, info = env.step(a)
                done = term or trunc
                if algo == "qlearn":
                    key2 = env.state_key()
                    agent.update(key, a, r, key2, done, info["action_mask"])
                    key = key2
                else:
                    agent.remember(prev_obs, a, r, obs, done,
                                   info["action_mask"])
                    agent.train_step()
                total += r
                won = won or (term and r > 10)
                if done:
                    break

            recent.append(float(won))
            if len(recent) > 100:
                recent.pop(0)
            win_rate = sum(recent) / len(recent)
            rec = {"ep": ep, "defender": defender.name, "reward": round(total, 2),
                   "won": won, "win_rate_100": round(win_rate, 3),
                   "eps": round(eps, 3)}
            if algo == "dqn":
                rec["loss"] = round(agent.last_loss, 4)
            log.write(json.dumps(rec) + "\n")
            if not quiet and ep % 100 == 0:
                extra = f"states={len(agent.table)}" if algo == "qlearn" else f"loss={agent.last_loss:.3f}"
                print(f"ep {ep:5d} win_rate_100={win_rate:.2f} eps={eps:.2f} "
                      f"{extra} [{time.time()-t0:.0f}s]")
            if ep % 200 == 0:
                agent.save(ckpt_path)

    agent.save(ckpt_path)
    return {"win_rate_100": recent and (sum(recent) / len(recent)),
            "ckpt": str(ckpt_path)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=2000)
    p.add_argument("--algo", choices=("qlearn", "dqn"), default="qlearn")
    p.add_argument("--run", default=str(int(time.time())))
    p.add_argument("--resume", type=Path, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--scenario", default=None,
                   help="scenario name from kb/scenarios.yaml, a map seed, "
                        "or omit for a fresh map each episode")
    args = p.parse_args()
    out = train(args.episodes, Path("runs") / args.run, algo=args.algo,
                resume=args.resume, seed=args.seed, scenario=args.scenario)
    print(json.dumps(out))


if __name__ == "__main__":
    main()

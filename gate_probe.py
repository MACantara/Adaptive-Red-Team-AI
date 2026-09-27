"""E2 gate run: tabular vs DQN on held-out map seeds. Disposable probe."""
from pathlib import Path

from artai.agent.dqn import DQNAgent
from artai.agent.qlearn import QLearner
from artai.env.defender import make
from artai.env.redteam_env import RedTeamEnv
from artai.eval import dqn_policy, eval_transfer, qlearn_policy, random_baseline
from artai.train import train

SEEDS = list(range(200, 220))
EPOCHS = 500

env0 = RedTeamEnv(scenario_seed=SEEDS[0])
n_actions = env0.action_space.n
obs_dim = env0.observation_space.shape[0]
defender = lambda: make("patch_on_alert")  # noqa: E731 — fresh per eval

print("=== tabular: training on unpinned maps ===", flush=True)
train(EPOCHS, Path("runs/gate_tab"), algo="qlearn", scenario=None, quiet=True)
tab = QLearner(n_actions, seed=0)
tab.load(Path("runs/gate_tab/policy.pkl"))

print("=== dqn: same budget, same unpinned regime ===", flush=True)
train(EPOCHS, Path("runs/gate_dqn"), algo="dqn", scenario=None, quiet=True)
dqn = DQNAgent(obs_dim, n_actions, seed=0)
dqn.load(Path("runs/gate_dqn/policy.pt"))

base = eval_transfer(random_baseline(), SEEDS, defender=defender())
tres = eval_transfer(qlearn_policy(tab), SEEDS, defender=defender())
dres = eval_transfer(dqn_policy(dqn), SEEDS, defender=defender())

print(f"\nbaseline : {base['reward']:+.2f} reward, {base['wins']}/{len(SEEDS)} wins")
print(f"tabular  : {tres['reward']:+.2f} reward, {tres['wins']}/{len(SEEDS)} wins "
      f"(diff {tres['reward'] - base['reward']:+.2f})")
print(f"dqn      : {dres['reward']:+.2f} reward, {dres['wins']}/{len(SEEDS)} wins "
      f"(diff {dres['reward'] - base['reward']:+.2f})")
margin = dres["reward"] - tres["reward"]
print(f"\nGATE: dqn - tabular = {margin:+.2f} "
      f"({'PASS' if margin >= 1.0 else 'FAIL — keep tabular'})")

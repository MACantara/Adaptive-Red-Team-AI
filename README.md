# Adaptive Red Team AI

A Python training platform where an RL adversary attacks a simulated
enterprise network and adapts to whoever defends it. Attack actions are
MITRE ATT&CK techniques; the defense can be a scripted policy ladder or a
human playing blue team over a CLI.

This is a simulation sandbox — no real offensive tooling, no network I/O.

## Quickstart

```bash
pip install -e ".[dev]"

# train the adversary on a pinned scenario vs the defender population
python -m artai.train --episodes 2000 --scenario training_ground --run my_run

# play blue team against the trained policy (it learns your moves)
python -m artai.play --defender human --scenario training_ground \
    --policy runs/my_run/policy.pkl --learn --debrief debrief.json

# or watch it play a scripted defender
python -m artai.play --defender patch_on_alert --scenario contested_dmz
```

`--learn` fine-tunes the Q-table on *your* defender behavior mid-game and
saves it back — the adversary you face next session is the one that
studied you.

## How it works

- **Environment** (`artai/env/`): Gymnasium env over a seeded network
  graph. Hosts carry vulns, credentials, privileges, and alert state.
  Actions are ATT&CK techniques loaded from `kb/techniques.yaml`,
  exposed through a validity mask — invalid actions are never sampled.
- **Agent** (`artai/agent/`): tabular Q-learning over `env.state_key()`
  with a replay buffer. Defender actions feed the state — the policy is
  opponent-conditioned. A masked DQN (`agent/dqn.py`) is the scale-up
  path for unseen topologies.
- **Defenders** (`artai/env/defender.py`): noop / random / patch-on-alert
  / scan-and-reimage ladder. Defenders see alert noise, never ground
  truth. Tier-0 (internet edge) can't be patched or isolated — the edge
  is evictable, not hardenable.
- **Game** (`artai/game/`, `artai/play.py`): turn-based — attacker acts a
  burst of steps, defender gets one action per turn from the alert view.
  Post-game debrief reports ATT&CK techniques attempted/detected/missed.

## Scenario packs

`artai/kb/scenarios.yaml` names map seeds and sizes:

| name | seed | size |
|------|------|------|
| `training_ground` | 9 | 3 ws + 3 srv |
| `contested_dmz` | 3 | 3 ws + 3 srv |
| `sprawl` | 11 | 4 ws + 4 srv |

`--scenario` accepts a name, an int seed, or `none` (fresh map each game).

## Design bets (the honest ones)

- **Fixed-scenario training is the product mode.** A Q-table can't
  generalize across randomized topologies; on a pinned map it converges
  hard. Cross-map generalization is what the DQN exists for.
- **Adaptation is tested, not claimed.** `test_agent.py` proves the
  trained table outscores random play on identical luck, and that after
  a defender swap the re-trained table beats its own frozen snapshot.
- **Noise has no pattern.** The agent learns vs patterned defenders;
  against `random` defender there is nothing to adapt to — by design.

## Tests

```bash
python -m pytest tests/          # full suite
python -m ruff check artai/ tests/
```

## Roadmap

PPO/DQN training for unseen maps, a defender-action histogram as a
richer opponent feature, self-play blue agent, scenario difficulty
ratings in the debrief.

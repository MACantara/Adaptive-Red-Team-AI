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

# or on a curriculum that escalates defender tiers as it wins
python -m artai.train --episodes 2000 --scenario training_ground --curriculum

# play blue team against the trained policy (it learns your moves)
python -m artai.play --defender human --scenario training_ground --policy runs/my_run/policy.pkl --learn --debrief debrief.json

# watch it live in the TUI — map on the left, event feed on the right
python -m artai.play --defender patch_on_alert --scenario contested_dmz --policy runs/my_run/policy.pkl --ui textual

# per-player profiles: the adversary remembers *you*
python -m artai.train --episodes 500 --scenario training_ground --player-id alice
python -m artai.play --defender human --player-id alice --learn --ui textual

# campaign: one agent across escalating legs, table carries forward
python -m artai.campaign --name escalation_arc --player-id alice --learn

# score a checkpoint vs random play on paired luck
python -m artai.eval --policy runs/my_run/policy.pkl --scenario training_ground

# scrub a saved debrief: n/→ next step, a autoplay
python -m artai.play --replay debrief.json
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
  with a replay buffer. The key carries a defender-behavior signature
  (dominant action, activity level) and a classified archetype — the
  table learns *policies of opponents*. A double-DQN (`agent/dqn.py`)
  is the proven generalization path for unseen topologies.
- **Defenders** (`artai/env/defender.py`): noop / random / patch-on-alert
  / scan-and-reimage ladder with difficulty tiers (act probability).
  Defenders see alert noise, never ground truth. Tier-0 (internet edge)
  can't be patched or isolated — the edge is evictable, not hardenable.
- **Game** (`artai/game/`, `artai/play.py`): turn-based — attacker acts a
  burst of steps, defender gets one action per turn from the alert view.
  Debrief reports ATT&CK techniques, a kill-chain timeline, and defender
  efficiency (useful vs wasted actions). `artai/campaign.py` strings
  scenarios into escalating legs with table carry-over.

## Scenario packs

`artai/kb/scenarios.yaml` names map seeds, sizes, and default opposition:

| name | seed | size | defender | tier |
|------|------|------|----------|------|
| `training_ground` | 9 | 3 ws + 3 srv | patch_on_alert | 1 |
| `contested_dmz` | 3 | 3 ws + 3 srv | patch_on_alert | 2 |
| `sprawl` | 11 | 4 ws + 4 srv | scan_and_reimage | 3 |

`--scenario` accepts a name, an int seed, or `none` (fresh map each game).
`--defender`/`--difficulty` override the pack defaults.

## Design bets (the honest ones)

- **Fixed-scenario training is the product mode for tabular.** On a
  pinned map the Q-table converges hard. On unseen maps it scores *worse
  than random* — that's the measured DQN gate (+11.1 mean reward on
  60 held-out seeds, `gate_probe.py` reproduces it).
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

See `ROADMAP.md` — self-play blue agent, an LLM strategy layer over the
proven executor, and export to established frameworks.

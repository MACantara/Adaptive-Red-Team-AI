# Roadmap

Where this project can go next, ordered by how much each step deepens the
core claim — *an adversary that learns from your moves* — rather than by
how impressive it sounds.

Current state: a Gymnasium env over seeded ATT&CK-mapped networks, a
tabular Q-learner that provably adapts to patterned defenders on pinned
scenarios, a turn-based human-defender CLI, and contract tests that make
"learns" falsifiable.

## Tier 1 — deepen the learning (small diffs, real payoff)

### Opponent-model features in the state key
`state_key()` already carries `last_def_action` and alert counts. Add a
bucketed histogram of the defender's recent actions (`def_hist` exists in
the observation but not the key). Unlocks: the table can learn *policies
of opponents*, not just "something reacted last turn".
Size: small. Files: `redteam_env.py::state_key`, contract tests.

### Per-defender persistent profiles
`--player-id` on `play.py` namespaces the checkpoint file, so Alice and
Bob each face an adversary trained on *them*. Almost free — the
serialization path already exists; it's a filename convention plus a
lookup table of profiles. Unlocks the literal product promise.
Size: small. Files: `play.py`, `train.py`.

### Defender-conditioned state segmentation
Split Q-rows by defender archetype detected *in-episode* (a Bayesian
guess from `def_hist`: "this opponent patches → probably patch_on_alert").
The attacker then exploits the *identified* defender's weakness
mid-game — the actual adaptive behavior, not just post-hoc re-training.
Size: medium. Files: `redteam_env.py` (key), `qlearn.py` (optionally
shadow-tables per archetype).

### Difficulty-tiered defender ladder
Concrete knobs on `ScanAndReimageDefender`/`PatchOnAlertDefender`
(probability, reaction delay, view noise) exposed as named tiers in
`scenarios.yaml`. Gives the debrief a difficulty rating and players a
progression path.
Size: small. Files: `defender.py`, `scenarios.yaml`, `catalog.py`.

## Tier 2 — generalization (the real research problem)

### Masked DQN that earns its place — PASSED
Gate verdict (500-episode spike, double-DQN only): on 60 held-out map
seeds vs patch_on_alert, DQN beat the shared-table baseline by
**+11.1 mean reward** (+9.6 over random baseline; tabular transfer was
*worse than random* at -1.5 — exact-match keys can't fire on new maps).
Win rates were ~equal; the margin is efficiency — faster, quieter wins.
DQN stays as the generalization path. Tabular remains the default for
pinned-scenario play, where it converges harder.
Size: large. Files: `dqn.py` (double-DQN landed), `eval.py`,
`gate_probe.py` (the runnable evidence).

### Curriculum training
Ordered population: NoOp → patch_on_alert → scan_and_reimage → mixed.
`train.py` already has a warmup phase; formalize it as a schedule with
gates (promote when win_rate crosses a threshold). Standard trick for
the defended-game spike in difficulty.
Size: small-medium. Files: `train.py`.

### Reward shaping iteration
Contained/burned/detection penalties exist; tune them against the
paired-eval harness before touching model complexity. Cheap experiment,
often the highest-leverage knob in sparse-goal RL.
Size: small. Files: `redteam_env.py` constants.

## Tier 3 — the game worth playing

### Richer debrief
Kill-chain timeline (turns per ATT&CK tactic phase), defender efficiency
metrics (alerts-touched vs noise-chased), and a per-run difficulty
rating. The JSON export already exists — this is purely a better report
over the same `log` data.
Size: small. Files: `session.py::debrief`.

### Campaign mode
A sequence of scenarios with carry-over: the adversary keeps its table
across maps (transfer test for the player, not just the AI), defenders
escalate tier each map. Small orchestration over existing pieces.
Size: medium. Files: `play.py`, new `campaign.py`.

### Human-quality UX
The CLI is functional but terse: map rendering of node states (owned/
alerted/isolated), defender action hints, and a replay viewer for past
sessions (the `log` already records everything needed). Textual or a
small web UI — pick Textual if staying in-terminal, it was the original
choice for a reason.
Size: medium-large. Files: `game/`, new render layer.

## Tier 4 — research bets (expensive, uncertain, honest)

### Self-play blue agent
Train a defender policy in the same gym (defender actions are already a
proper space). Red and blue co-train → an arms race. This is where the
"adaptive" story gets genuinely interesting — but it doubles the RL
problem and needs the DQN to work first. Gate: Tier 2's DQN milestone.

### LLM strategist over the RL executor
An LLM proposes a campaign plan ("prioritize credential access this
phase") that re-weights the Q-table's action preferences — CAGE-4-style
hierarchical control. Cheap to prototype (prompt + reranking layer),
expensive to make *better* than the table alone. Prototype only after
the executor is provably strong on its own.

### Export to real frameworks
Caldera ability-file export or CybORG scenario export of generated
maps+playbooks — positions this as a *generator* for established
frameworks rather than a competitor. Check license/attribution
expectations first.

## Explicitly not doing (revisit only with new information)

- **MITRE Caldera / CyberBattleSim as the env.** Evaluated and rejected:
  the glue tax exceeds the reuse value at this granularity, and their
  action spaces don't map cleanly to per-technique ATT&CK mechanics.
- **networkx.** Adjacency dicts are sufficient at ≤16 nodes; don't add a
  dependency for aesthetics.
- **Real offensive tooling.** This is a simulation gym. Any step that
  touches a real network changes the project category entirely — out of
  scope permanently.
- **A web dashboard before the game is good.** Render polish on a weak
  adversary is wasted work; Tier 1-2 first.

## Definition of done per milestone

Every roadmap item ships with: a paired-eval or contract test proving
the claim (not vibes), a focused test file, an atomic commit ≤50 chars,
and a README line. The project's culture is *evidence over marketing* —
keep it that way.

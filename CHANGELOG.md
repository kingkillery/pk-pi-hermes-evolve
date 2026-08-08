# Changelog

## 0.3.0 - 2026-08-08

### Evaluation-architecture correctness fixes

Hard gates are now fail-closed with no recovery path, and every promotion decision rests on paired, executor-grounded measurements:

- **never promote a gate-failing candidate**: the fallback path that promoted the best fully-evaluated iteration even when `accepted === false` (clearing its drift/test results in the process) is removed; when no iteration passes every hard gate the engine retains the baseline as a "no safe improvement" outcome, reuses the baseline's own holdout measurement (improvement is exactly 0 instead of re-measured noise), and records `promotion.blockedReason` in `manifest.json`
- **paired, candidate-aware tiered gate**: `cohortJudgeFunc` now receives the artifact text under evaluation, and the cohort tier compares the candidate against a baseline judged once per run on the SAME cohort — previously the candidate's cohort score was compared against the sealed holdout baseline aggregate (an unpaired cross-split comparison that also leaked the holdout into search); `coherenceCheck` receives the candidate text explicitly
- **fail closed on evaluator uncertainty**: `TieredGateResult` gains a tri-state `status` (`pass`/`fail`/`unknown`) — judge/callback outages report `cohort_judge_error`/`coherence_error` as `unknown` instead of masquerading as measured regressions; a drift-judge failure blocks the candidate instead of fabricating a passing 0.2 score; real-executor calls retry once and any remaining failure marks the evaluation (`ArtifactEvaluation.executorFailureCount`) so it can neither accept a candidate, enter the Pareto pool, nor justify a PR
- **real execution traces feed reflection**: the train-subset reflection minibatch is now executor-grounded for every pool entry (baseline, ancestor seed, and candidates), so mutation prompts see genuine pi transcripts under `hasRealExecution` while validation/holdout stay hidden from the proposer; baseline holdout/validation executor traces are also retained for reporting
- **span-based secret redaction**: `scanForSecrets` returns exact character spans for EVERY match (global, all rules) and findings no longer embed a preview of the matched secret; new `redactSecrets` replaces merged spans back-to-front, fixing the old preview-string `replace` that usually left the full secret in place and only handled the first match
- **PR automation in a disposable worktree**: the winning candidate is committed on a new branch inside a temporary `git worktree` (removed in `finally`) and the PR is opened via the `gh` CLI — previously the code ran `git pr create` (not a git subcommand, so no PR was ever created) and switched branches/overwrote the target file in the caller's active checkout
- **promotion gating**: `createPR` requires a strictly-accepted candidate, a positive holdout improvement, and clean executor measurements on both sides of the holdout comparison; the decision and reason are persisted under `manifest.json#promotion`

### Improvement-impact fixes for the evolution loop

Five changes targeting measurement validity and compounding, so that reported "improvement" reflects real gains rather than evaluation artifacts:

- ground baseline validation/holdout scoring (and the winner's holdout confirmation) in the real `pi` executor, matching the regime candidates are already measured under; previously the baseline was judge-predicted while candidates were executor-grounded, biasing every improvement delta
- blind the judge to the artifact's prose whenever a real executor observation exists — the judge now scores what the agent actually did, not how persuasive the artifact text reads
- stop leaking the validation split into reflection: accepted pool entries now carry their train-minibatch traces/evaluation for reflection-prompt assembly, so mutations are no longer steered by the same instances used for candidate selection (the selection metric itself still uses the full validation pass)
- surface real executor stdout excerpts (up to 3 failing traces, 1200 chars each, flagged via the new `ExecutionTrace.hasRealExecution`) in the mutation reflection prompt, giving the mutator observed behavior instead of only judge summaries
- difficulty-stratified `splitExamples` with size-first allocation for n≥5, so validation/holdout splits get representative difficulty mixes instead of positional slices; n≤4 behavior is unchanged
- seed the Pareto pool with the best cross-run ancestor: new `resolveAncestorBody` in `src/lineage.ts` hash-verifies the ancestor run's persisted `best-candidate.md` against `lineage.jsonl` before the engine constraint-checks it, evaluates it on validation, and adds it as an `ancestor` pool entry — successive runs now compound instead of restarting from the raw artifact

### GEPA-Pareto review fixes

Addresses CodeRabbit review findings on the GEPA-Pareto PR:

- fix merge candidates being minibatch-filtered and score-deltaed against an unrelated sampled `parent` instead of their actual merge inputs `a`/`b`; the comparison baseline is now the stronger of the two merge parents
- fail closed instead of silently skipping when the tiered gate itself throws (as opposed to a tier resolving `passed: false`, which `runTieredGate` already handles internally) — an exception now rejects the iteration rather than letting the candidate through unchecked
- fix the no-fully-evaluated-candidate fallback promoting a draft under its parent's (or a partial minibatch) score; it now only promotes from iterations that completed a real full-validation pass on themselves, and retains the baseline (with baseline's own genuine evaluation) rather than misattributing a score when nothing did
- `CandidateRecord`/`IterationRecord` gain `parentCandidates?: string[]` (replacing the synthetic `"a+b"` string lineage for merges) alongside the existing single-parent `IterationRecord.parentCandidate` for backward compatibility
- `tests/parity.test.ts` now parses the README parity table's status/evidence columns per row instead of a whole-section substring check, and rejects duplicate capability rows

### GEPA-Pareto upgrade to the reflective loop

Aligns the iterative loop in `src/engine.ts` with GEPA ("Reflective Prompt Evolution Can Outperform Reinforcement Learning", arXiv:2507.19457) — the current state-of-the-art reflective/evolutionary prompt optimizer, which this package's `optimizerUsed` label already claimed lineage from without implementing the algorithm's core mechanisms.

- fix a correctness bug where every mutation was generated from the pristine original artifact body regardless of `parentName`, so accepted gains from prior iterations were discarded rather than compounded; `generateOneCandidateDraft` now takes `parentBody` and mutates the actual selected parent
- replace single-lineage greedy hill-climbing (`accepted = scoreDelta > priorComposite`, chaining from the single last-accepted candidate) with a Pareto-frontier candidate pool: `computeParetoFrontier` tracks per-validation-instance winners and prunes dominated candidates; `selectParetoParent` samples the next mutation parent proportional to frontier appearance, which avoids collapsing to a local optimum once the first easy gains are exhausted
- add a cheap train-set minibatch pre-filter (`evaluateArtifact` on 1-2 examples) before paying for a full validation pass with the real `pi` executor; only a clear regression vs. the parent's minibatch score is rejected, matching GEPA's "35x fewer rollouts" efficiency lever
- move the tiered regression gate (typecheck → cohort → coherence) to run immediately after constraint validation, before the minibatch filter and full judge pass, since it is a cheap, quality-orthogonal safety signal that should short-circuit expensive rollouts rather than run after them
- add a bounded system-aware merge (`generateMergeCandidateDraft`, GEPA Appendix F "crossover"): every third iteration, if the pool has ≥2 distinct mutation lineages on the frontier, attempt synthesizing one candidate from two frontier candidates' complementary per-instance strengths; capped at 2 attempts per run
- final `bestCandidate` selection now considers every fully-validated pool member (not just the accepted chain), matching GEPA's "return candidate with best aggregate performance on validation set" stopping rule
- extend `types.ts` additively: `CandidateRecord.parentCandidate/selectionMethod/minibatchScore`, `IterationRecord.selectionMethod/paretoFrontierSize/minibatchFiltered`, `EvolutionRunResult.paretoFrontier/mergeAttempts/minibatchFilteredCount`
- `optimizerUsed` renamed from `"gepa-iterative"` to `"gepa-pareto"`; `report.md` gains an "Optimization strategy" section and the candidates table gains Parent/Method columns
- README parity table and `tests/parity.test.ts` gain rows for the Pareto-frontier pool and system-aware merge

### TypeScript-native Hermes Phase 1 parity

The TypeScript engine is now the source-of-truth implementation of the Hermes Phase 1 workflow. Python/DSPy is an optional acceleration mode rather than the primary backend.

- replace single-shot candidate fan-out with an iterative reflective loop in `src/engine.ts`; each iteration builds a `ReflectionPrompt` from prior-round failure traces and judge feedback, persists an `IterationRecord` to `iterations/<n>.json`, and decides accept or reject based on score delta plus constraint and test outcomes
- add `src/pi-executor.ts` (`executeCandidateInPi`) which installs the candidate into an ephemeral `.pi/skills/<slot>` and spawns `pi -p --no-session` to capture real stdout for the judge instead of predicting agent behavior; logs to `executor/<iter>/<exampleIndex>/{stdout.log,stderr.log,meta.json}`
- add `src/tiered-gate.ts` (`runTieredGate`) for a typecheck → cohort → coherence regression gate with distinct reason codes (`typecheck_failed`, `cohort_regression`, `coherence_failed`); early-terminates on first failure
- add `src/constraints-structure.ts` (`checkSkillStructure`, `buildSkillStructureReport`) enforcing `name:` and `description:` in the first 500 chars for skill artifacts; extends the `ConstraintName` union with `skill_structure`
- add `src/lineage.ts` (`appendLineageEntry`, `loadLineage`, `loadBestAncestor`) writing `.pi/hermes-self-evolution/lineage.jsonl` so consecutive runs can pick the Pareto-best ancestor; matching uses content-hash when `artifactContent` is supplied
- extend `src/types.ts` with `IterationRecord`, `ReflectionPrompt`, `ExecutionObservation`, `TieredGateResult`, `LineageEntry`, `SkillStructureReport`, `BackendMode`
- regenerate the README parity table to cover all 18 capabilities, mark every Phase 1 row complete, mark Python/DSPy and benchmark scripts as optional acceleration
- add `tests/api-snapshot.test.ts` to freeze public exports at type level; `tests/e2e-golden.test.ts` to document the expected run-dir shape; `tests/parity.test.ts` with `npm run test:parity` to assert all 18 parity rows remain in the README
- add `docs/ownership-map.md` documenting the 5-lane disjoint-ownership pattern used to build the above

### Engine hooks promoted to EvolutionOptions

- add `seed?: number` to `EvolutionOptions`; `splitExamples` now uses a deterministic mulberry32 RNG when supplied, falling back to unseeded `Math.random` otherwise
- add `cohortExamples?: EvalExample[]`, `cohortJudgeFunc?`, and `coherenceCheck?` to `EvolutionOptions`; threaded through to the `runTieredGate` call in the iterative loop, enabling real `cohort_regression` and `coherence_failed` reason codes instead of the previous `skipped_no_cohort` / `skipped_no_check` defaults
- engine now writes a top-level `gate.json` (containing the best candidate's `TieredGateResult[]`) per run directory, alongside the existing `iterations/<n>.json#gateResults`
- refactor `scripts/smoke-test.ts` to consume the new hooks; remove the global `Math.random` monkey-patch and synthetic `gate.json` writes that were previously documented as Known Limitations in `tests/smoke-test-report.md`
- remove the two `// SOFT-SPOT(coherence-default)` and `// SOFT-SPOT(cohort-default)` comments at the `runTieredGate` call site; the corresponding entries in `tests/smoke-test-report.md` Known Limitations are removed

### Hermes parity gap closure

- **diff rendering**: every evolution run now writes a `diff.patch` file (original body → best candidate body) and embeds a `## Diff` section in `report.md`
- **apply/approve workflow**: new `/evolve apply [runDir]` command copies the best candidate to the target file after showing the diff and requiring explicit `"yes"` confirmation; uses the last session run when no `runDir` is provided
- **artifact-type rubric presets**: the judge now receives type-specific scoring guidance (`skill` / `prompt` / `instructions`) so correctness, procedure-following, and conciseness are weighted appropriately for each artifact kind; parity added in both TypeScript and Python backends

### Earlier Unreleased work

- add `scripts/ralph_otel.py`, a traced Ralph loop for Hermes-parity gap closure work in this repo
- add `scripts/tasks/hermes_parity_task.json` as the default parity task spec
- upgrade the Ralph judge with deterministic repo-deliverable checks for parity targets like execution traces, validation splits, and golden datasets
- include OpenTelemetry Python dependencies and repo scripts for the Ralph loop workflow
- add `scripts/sokoban_benchmark.py` plus bundled benchmark assets under `benchmarks/sokoban/`
- add a scaffolded baseline-vs-improvement 5-attempt benchmark workflow with attempt preparation, CSV recording, and summary analysis

## 0.2.1 - 2026-04-12

- fix Python backend syntax so CI and `python:check` pass cleanly
- republish hybrid extension package with the corrected Python backend

## 0.2.0 - 2026-04-12

- add optional Python DSPy/GEPA hybrid backend under `python_backend/`
- add automatic backend selection (`auto` / `python` / `typescript`)
- add GitHub Actions CI and npm release workflows
- update npm package metadata and README for public distribution

## 0.1.0 - 2026-04-12

- initial pi-native Hermes-inspired self-evolution extension
- `/evolve` command and `self_evolve_artifact` tool
- TypeScript-only reflective evaluation loop with report generation

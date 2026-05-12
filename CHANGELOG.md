# Changelog

## Unreleased

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

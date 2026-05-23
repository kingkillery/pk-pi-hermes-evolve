# Engine Hooks Follow-Up — Orchestration Overview

## Purpose

The runtime smoke-test pipeline closed Phase 1 parity but deferred two engine-API hooks. The smoke driver currently monkey-patches `Math.random` globally (because `EvolutionOptions` has no `seed?` field) and synthesizes `gate.json` for forced-failure runs (because the engine calls `runTieredGate` at `engine.ts:499` without `cohortExamples` / `cohortJudgeFunc` / `coherenceCheck`). This dispatch promotes both to first-class `EvolutionOptions` fields, threads them into the engine, makes the engine write a per-run `gate.json`, refactors the smoke driver off its workarounds, and confirms `npm run test:gates` stays green.

Parallel dispatch is justified: types pre-phase + engine wiring + smoke driver refactor + docs + acceptance gate are five naturally disjoint lanes once types land first.

## Letter-group dispatch table

| Letter | Lane | Archetype | Effort | Depends on | File |
|---|---|---|---|---|---|
| A | types-extension | `[pre-phase]` | TINY | none | `.prd/engine-hooks-A-types.md` |
| B | engine-bundle | `[parallel-builder]` | MEDIUM | A | `.prd/engine-hooks-B-engine-bundle.md` |
| C | smoke-driver-refactor | `[parallel-builder]` | SMALL | A | `.prd/engine-hooks-C-smoke-driver.md` |
| D | docs-and-changelog | `[parallel-builder]` | TINY | A | `.prd/engine-hooks-D-docs.md` |
| E | acceptance-and-parity-gate | `[acceptance-gate]` | SMALL | B, C, D | `.prd/engine-hooks-E-acceptance.md` |

## 5-row dispatch table (operational form)

| Lane | Gap letters | Owned files | Effort | Model | Subagent type | Isolation | Depends on | Verify commands |
|---|---|---|---|---|---|---|---|---|
| types-extension | A | `src/types.ts` | TINY | sonnet | executor | worktree | none | `npm run typecheck` |
| engine-bundle | B | `src/engine.ts` | MEDIUM | opus | executor | worktree | A | `npm run typecheck`, `npm run smoke` |
| smoke-driver-refactor | C | `scripts/smoke-test.ts` | SMALL | sonnet | executor | worktree | A | `npm run typecheck`, `npm run smoke` |
| docs-and-changelog | D | `docs/configuration.md`, `docs/output-layout.md`, `CHANGELOG.md` | TINY | sonnet | executor | worktree | A | `npm run typecheck` |
| acceptance-and-parity-gate | E | `tests/smoke-test-report.md` | SMALL | sonnet | executor | worktree | B, C, D | `npm run test:gates`, grep checks |

Model routing override: even though A and D are TINY, sonnet is appropriate — they are mechanical edits, not reasoning-heavy. Lane B is MEDIUM (multi-site engine surgery + new per-run output file) and warrants opus.

## File-ownership matrix

| File | Lane A | Lane B | Lane C | Lane D | Lane E |
|---|---|---|---|---|---|
| `src/types.ts` | own | – | – | – | – |
| `src/engine.ts` | – | own (includes SOFT-SPOT comment removal at :499) | – | – | – |
| `scripts/smoke-test.ts` | – | – | own | – | – |
| `docs/configuration.md` | – | – | – | own | – |
| `docs/output-layout.md` | – | – | – | own | – |
| `CHANGELOG.md` | – | – | – | own | – |
| `tests/smoke-test-report.md` | – | – | – | – | own |

Empty intersection across parallel lanes B, C, D. Lane E does not edit any source file — only the evidence report. The two `// SOFT-SPOT(coherence-default)` and `// SOFT-SPOT(cohort-default)` comments at `engine.ts:499` are removed by Lane B as part of its wiring change (since the call site itself is being rewritten with real callbacks). Lane E confirms via grep that they are gone.

## Execution sequence

1. **Phase 0**: Lane A solo in worktree. Block, commit, fast-forward into `main`. ~5 minutes.
2. **Phase 1**: Lanes B, C, D dispatched in parallel with `run_in_background: true`. Lanes C and D may use `@ts-expect-error` bridges if they reference `EvolutionOptions` fields before they land on main (unlikely — A lands first). Wait for all 3 completion notifications. ~10-15 minutes.
3. **Phase 1.5**: Sequential `git merge --no-ff <branch>` of each lane. No conflicts expected (disjoint ownership). Run `npm run typecheck` to catch any field-name drift between Lane A and Lane B/C/D.
4. **Phase 2**: Lane E solo. Confirm SOFT-SPOT comments removed, update Known Limitations in `tests/smoke-test-report.md`, run full `npm run test:gates`, commit acceptance.

## Acceptance criteria checklist

Every claim links to a concrete file, test name, or grep target.

- [ ] `npm run typecheck` exits 0 on `main` after Lane E merges
- [ ] `npm run python:check` exits 0 on `main` after Lane E merges
- [ ] `npm run test:parity` exits 0 (18/18 rows)
- [ ] `npm run test:smoke` exits 0 (all 4 verifiers pass)
- [ ] `npm run test:gates` exits 0 (composite of the above)
- [ ] `grep -n "SOFT-SPOT(coherence-default)\|SOFT-SPOT(cohort-default)" src/engine.ts` returns no matches
- [ ] `grep -n "Math.random = " scripts/smoke-test.ts` returns no matches (no monkey-patch)
- [ ] `grep -n "writeFileSync.*gate.json\|writeFile.*gate.json" scripts/smoke-test.ts` returns no matches (no synthetic gate.json writes)
- [ ] `EvolutionOptions` in `src/types.ts` declares all four new optional fields: `seed`, `cohortExamples`, `cohortJudgeFunc`, `coherenceCheck`
- [ ] Each fresh `npm run smoke` invocation produces a top-level `gate.json` in every run dir (default and forced modes)
- [ ] For a fixed `seed`, two consecutive runs produce byte-identical `dataset.json` splits (determinism gate)
- [ ] `tests/smoke-test-report.md` Known Limitations section no longer lists `coherence-default` or `cohort-default`
- [ ] `docs/configuration.md` Tool Parameters table lists all four new fields with type signatures
- [ ] `docs/output-layout.md` Per-run directory format includes `gate.json` as a required file
- [ ] `CHANGELOG.md` Unreleased section has an entry describing the new engine hooks and `gate.json` output

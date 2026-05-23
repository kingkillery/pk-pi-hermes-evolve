# Lane D — Docs and Changelog Update [parallel-builder]

## 1. Mission + read-first

You are a parallel-builder sub-agent. You update three documentation files to reflect the new engine-hook surface that Lane A added to `EvolutionOptions` and Lane B wired into the engine. Nothing you write executes; your job is field-name and shape parity between the docs and `src/types.ts`.

**Read first** (each in full):
- `.prd/engine-hooks-orchestration.md` — pipeline context
- `src/types.ts` — the four new optional fields on `EvolutionOptions` (already on main from Lane A): `seed?: number`, `cohortExamples?: EvalExample[]`, `cohortJudgeFunc?: (examples) => Promise<{ composite: number }>`, `coherenceCheck?: () => Promise<{ passed: boolean; detail: string }>`
- `docs/configuration.md` — current Tool Parameters table you will extend
- `docs/output-layout.md` — current per-run directory format you will extend with `gate.json`
- `CHANGELOG.md` — Unreleased section you will add an entry to
- `src/tiered-gate.ts` — callback shape reference (do not edit; just confirm signatures match `cohortJudgeFunc` / `coherenceCheck`)

## 2. Owned files

You may ONLY edit:
- `docs/configuration.md` (existing)
- `docs/output-layout.md` (existing)
- `CHANGELOG.md` (existing)

You may NOT edit `src/types.ts`, `src/engine.ts`, `tests/smoke-test-report.md`, or any other file. If you find a discrepancy between docs and types that requires a type change, flag it for Lane E — do not patch types yourself.

## 3. Gap (verbatim from the table)

> D — Docs and Changelog Update [parallel-builder]: Add the 4 new optional fields to the tool parameters table in `docs/configuration.md`; add `gate.json` to the required-files list in `docs/output-layout.md`; add an Unreleased CHANGELOG entry describing the new engine hooks and per-run `gate.json` output. (0% complete) [TINY] depends on: A | files: `docs/configuration.md`, `docs/output-layout.md`, `CHANGELOG.md`

## 4. What to build

### `docs/configuration.md` — Tool Parameters table

Find the existing Tool Parameters table (the table headed `| Parameter | Type | Default | Description |`). Add four new rows after the last existing row (after `persistGolden`), in this order and exact format:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `seed` | number | undefined | Deterministic RNG seed for `splitExamples` and other RNG consumers. When unset, falls back to unseeded `Math.random`. |
| `cohortExamples` | `EvalExample[]` | undefined | Examples used by the tiered gate's cohort-regression tier. Required to activate `cohort_regression` reason code. |
| `cohortJudgeFunc` | `(examples) => Promise<{composite: number}>` | undefined | Judge callback for the cohort tier. Required when `cohortExamples` is supplied. |
| `coherenceCheck` | `() => Promise<{passed: boolean; detail: string}>` | undefined | Coherence check callback for the tiered gate's coherence tier. When unset, the tier returns `skipped_no_check`. |

Also add a brief paragraph after the table (before the existing "Effort and cost guidance" subsection) describing what activating these hooks does. Keep it under 60 words. Reference `src/tiered-gate.ts` for the gate semantics.

### `docs/output-layout.md` — `gate.json` in per-run directory format

Find the per-run directory format section (the ASCII tree showing `original.md`, `best-candidate.md`, `report.md`, `manifest.json`, `dataset.json`, `candidates/`, `iterations/`, `executor/`, `traces/`). Add `gate.json` as a new entry in the tree, placed alphabetically between `executor/` and `iterations/`:

```
<timestamp>-<artifact-slug>/
├── original.md
├── best-candidate.md
├── report.md
├── manifest.json
├── dataset.json
├── gate.json                       ← top-level TieredGateResult[] for the best candidate
├── candidates/
├── iterations/
├── executor/
└── traces/
```

Then add a new subsection after the existing per-iteration `gateResults` description (or wherever the per-iteration tier results are documented) titled `### gate.json (top-level)` with this content:

```markdown
### gate.json (top-level)

Top-level array of `TieredGateResult` objects representing the gate outcomes for the best (selected winner) candidate. Mirrors the contents of the corresponding `iterations/<n>.json#gateResults` for the iteration that produced the winner.

Shape: `TieredGateResult[]` — see [src/types.ts](../src/types.ts) for the field definitions (`tier`, `passed`, `reasonCode`, `detail`, `durationMs`).

Always written, even if empty (`[]`). Empty array indicates no gate ran or the winner had no `gateResults` recorded.
```

### `CHANGELOG.md` — Unreleased entry

Find the `## Unreleased` section. Add a new sub-heading inside it (after any existing sub-headings) titled `### Engine hooks promoted to EvolutionOptions` with this content:

```markdown
### Engine hooks promoted to EvolutionOptions

- add `seed?: number` to `EvolutionOptions`; `splitExamples` now uses a deterministic mulberry32 RNG when supplied, falling back to unseeded `Math.random` otherwise
- add `cohortExamples?: EvalExample[]`, `cohortJudgeFunc?`, and `coherenceCheck?` to `EvolutionOptions`; threaded through to the `runTieredGate` call in the iterative loop, enabling real `cohort_regression` and `coherence_failed` reason codes instead of the previous `skipped_no_cohort` / `skipped_no_check` defaults
- engine now writes a top-level `gate.json` (containing the best candidate's `TieredGateResult[]`) per run directory, alongside the existing `iterations/<n>.json#gateResults`
- refactor `scripts/smoke-test.ts` to consume the new hooks; remove the global `Math.random` monkey-patch and synthetic `gate.json` writes that were previously documented as Known Limitations in `tests/smoke-test-report.md`
- remove the two `// SOFT-SPOT(coherence-default)` and `// SOFT-SPOT(cohort-default)` comments at the `runTieredGate` call site; the corresponding entries in `tests/smoke-test-report.md` Known Limitations are removed
```

## 5. Hard constraints

1. No new npm dependencies.
2. `npm run typecheck` must pass at the end (no TS files touched, so this is a sanity check that you haven't accidentally broken anything via a bad include or similar).
3. No edits outside the three owned files. Verify via `git diff --name-only`.
4. Field names and types in `docs/configuration.md` must match `src/types.ts` exactly. If `src/types.ts` says `cohortExamples?: EvalExample[]`, the docs must use `EvalExample[]`, not `EvalExample list` or `array<EvalExample>`.
5. The CHANGELOG entry must use lowercase imperative-style bullets matching the existing style in that file.
6. No emoji. Match the existing terse professional tone of the docs.

## 6. Verification

```bash
npm run typecheck
git diff --name-only
grep -nE "\bseed\?\b|\bcohortExamples\?\b|\bcohortJudgeFunc\?\b|\bcoherenceCheck\?\b" docs/configuration.md
grep -n "gate.json" docs/output-layout.md
grep -A2 "Engine hooks promoted" CHANGELOG.md
```

Expected:
- typecheck exit 0
- `git diff --name-only` lists only the three owned files
- All four field names found in `docs/configuration.md`
- `gate.json` referenced in `docs/output-layout.md`
- The new CHANGELOG sub-heading is present

## 7. Commit message

`docs: document new EvolutionOptions hooks + per-run gate.json output (Gap D)`

## 8. Final report

```
### Lane D final report
- Worktree path / branch:
- Files modified: docs/configuration.md, docs/output-layout.md, CHANGELOG.md
- New field rows in docs/configuration.md: [list]
- New section in docs/output-layout.md: [section heading]
- New CHANGELOG sub-heading: [heading]
- Lines added / removed:
- Verification:
  - npm run typecheck exit: ___
  - grep for new fields in configuration.md: ___
  - grep for gate.json in output-layout.md: ___
  - grep for new CHANGELOG heading: ___
  - git diff --name-only: ___
- Flags / blockers:
```

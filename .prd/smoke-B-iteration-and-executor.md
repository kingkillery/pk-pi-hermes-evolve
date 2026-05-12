# Lane B — Iteration + Executor Verifiers [parallel-verifier]

## Agent prompt (paste verbatim into `Agent({prompt})`)

You are one of **three parallel verifier sub-agents** (B, C, D) for the Hermes Phase 1 runtime smoke-test pipeline at `C:\dev\Desktop-Projects\pi-hermes-self-evolution`. Lane A has already produced shared smoke artifacts (two run dirs under `.pi/hermes-self-evolution/runs/`). You will be dispatched concurrently with Lanes C and D. Lane E waits on all three of you.

**Read `.prd/smoke-test-orchestration.md` first** for context, then `.prd/gap-analysis.md`, then `.prd/current-state.md`.

## Owned files (only these)
- `tests/smoke-iterations.test.ts` (new)
- `tests/smoke-executor.test.ts` (new)
- `tests/smoke-findings-B.md` (new — findings stub for Lane E)

You may **NOT** edit any file under `src/`, `scripts/`, `tests/fixtures/`, or any other test file. You may **NOT** invoke `runEvolution`; you only read Lane A's emitted run-dir artifacts.

## What you must produce

### 1. `tests/smoke-iterations.test.ts`

A typecheck-clean Node ESM TypeScript file that exports a runnable function `runIterationVerifier(runDirs: string[])` and asserts the following against each run dir under `.pi/hermes-self-evolution/runs/`:

- `iterations/` subdirectory exists.
- ≥2 files matching `iterations/<n>.json` exist (the iterative loop must have actually iterated; if only 1, the silent-fallback soft spot in `src/engine.ts` was triggered — record this).
- Each `iterations/<n>.json` parses as `IterationRecord` shape (from `src/types.ts`): has `iteration`, `mutationRationale`, `reflectionPrompt`, `candidate`, `evaluation`, `traces`, `scoreDelta`, `accepted` fields.
- For iteration index ≥2: `reflectionPrompt.priorTraces` must be a non-empty array AND `reflectionPrompt.priorJudgeFeedback` must contain at least one string. This proves the GEPA-like reflection actually used prior-round signals (not just regenerated independently).
- Record observed behavior of the "silent acceptance fallback" — count how many iterations have `accepted: true` vs. `accepted: false` but were promoted to `bestCandidate` anyway. Read the run's `manifest.json` to find the best candidate's source iteration.

Include a `main` block so the file is runnable via `node --experimental-strip-types tests/smoke-iterations.test.ts <run-dir-1> <run-dir-2>`.

### 2. `tests/smoke-executor.test.ts`

A typecheck-clean Node ESM TypeScript file that exports `runExecutorVerifier(runDirs: string[])` and asserts:

- `executor/` subdirectory exists.
- ≥1 path matching `executor/<iter>/<exampleIndex>/stdout.log` exists.
- At least one `stdout.log` has non-empty content (>0 bytes). If all are empty, this is evidence the smoke mock is bypassing the pi-executor entirely — record as a finding.
- Each `executor/<iter>/<exampleIndex>/meta.json` (if present per Lane A's mock approach) parses as `ExecutionObservation` shape: `stdout`, `stderr`, `exitCode`, `durationMs`.
- Verify executor logs were used in judging: cross-reference the iteration's `traces[].rawOutput` field — at least some `rawOutput` entries should match content from the corresponding `executor/<iter>/<ex>/stdout.log`. If they never match, the pi-executor is wired but its output isn't reaching the judge (a real soft spot to record).

Include a runnable `main` block.

### 3. `tests/smoke-findings-B.md`

A short markdown fragment (≤80 lines) Lane E will consume and delete. Format:

```markdown
# Lane B findings

## Iteration verifier observations
- run 1: N iterations recorded, M strictly accepted, best candidate from iter <i> (accepted=<bool>)
- run 2: ...
- silent-fallback fired: yes/no (cite which run)
- reflection-prompt evidence from iter 2: <"non-empty priorTraces" | "empty">

## Executor verifier observations
- run 1: <N> executor stdout.log files, <M> non-empty, <K> matched against traces[].rawOutput
- run 2: ...
- pi-executor wiring evidence: <"working" | "logs written but not consumed" | "logs not written">

## Soft spots observed
1. iteration-acceptance silent fallback: <observed/not-observed/inconclusive>
2. executor-log → judge wiring: <observed/not-observed/inconclusive>

## Recommended remediations (for Lane E)
- [bullet list, ≤5 items]
```

## Constraints

1. **No engine edits.** If a verifier needs an API the engine doesn't expose, document the gap in `smoke-findings-B.md` for Lane E.
2. **Read-only against run dirs.** Never write into `.pi/hermes-self-evolution/runs/`.
3. **No new npm dependencies.** Use Node built-ins (`node:fs/promises`, `node:path`, `node:assert/strict`).
4. **Type imports only from `../src/types.js`.** Do not import runtime engine code.
5. **Style**: normal formatting. No comments except non-obvious WHY.
6. **Deterministic**: assertions must produce identical results given identical run dirs.

## Verification (run before declaring done)
```bash
npm run typecheck
node --experimental-strip-types tests/smoke-iterations.test.ts \
  .pi/hermes-self-evolution/runs/<SMOKE_RUN_1> .pi/hermes-self-evolution/runs/<SMOKE_RUN_2>
node --experimental-strip-types tests/smoke-executor.test.ts \
  .pi/hermes-self-evolution/runs/<SMOKE_RUN_1> .pi/hermes-self-evolution/runs/<SMOKE_RUN_2>
git diff --name-only
```

Expected:
- `npm run typecheck`: exit 0
- Both verifiers: exit 0 (or 1 if they detect a real failure — that's also valid output for Lane E)
- `git diff --name-only` lists only your three owned files

Commit message: `test(smoke): iteration and executor verifiers (smoke B)`

## Final report
```
### Lane B final report
- Worktree path / branch:
- Files created: tests/smoke-iterations.test.ts, tests/smoke-executor.test.ts, tests/smoke-findings-B.md
- Iteration verifier outcome: [pass / fail-with-evidence / inconclusive]
- Executor verifier outcome: [pass / fail-with-evidence / inconclusive]
- Soft spots observed: [list with run-dir refs]
- Verification: typecheck exit ___; iter verifier exit ___; exec verifier exit ___
- Flags/blockers: ___
```

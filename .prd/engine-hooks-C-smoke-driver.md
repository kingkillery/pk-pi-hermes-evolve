# Lane C — Smoke Driver Refactor [parallel-builder]

## 1. Mission + read-first

You are a parallel-builder sub-agent. You refactor `scripts/smoke-test.ts` to use the new real engine hooks instead of its current workarounds: replace the global `Math.random = mulberry32(...)` monkey-patch with `seed: 0xC0FFEE` passed to `runEvolution`; replace the synthetic `gate.json` writes for `MOCK_MODE=force-*` modes with mocked `coherenceCheck` / `cohortExamples` / `cohortJudgeFunc` callbacks.

**Read first** (each in full):
- `.prd/engine-hooks-orchestration.md` — pipeline context
- `tests/smoke-test-report.md` — Known Limitations section explaining what your changes close
- `scripts/smoke-test.ts` — current driver with the workarounds you are removing
- `src/tiered-gate.ts` — callback signatures you must match in your mocks

The new `EvolutionOptions` fields you consume are already on `main` from Lane A: `seed`, `cohortExamples`, `cohortJudgeFunc`, `coherenceCheck`.

## 2. Owned files

You may ONLY edit:
- `scripts/smoke-test.ts` (existing)

You may NOT edit `scripts/smoke-shim/mock-pi.cjs`, `scripts/smoke-shim/pi`, `scripts/smoke-shim/pi.cmd`, `scripts/smoke-shim/ts-resolver.mjs`, or `tests/fixtures/smoke-skill/*`. The mock-pi dispatcher and PATH-shim stay as-is; only the orchestrator at `scripts/smoke-test.ts` changes.

## 3. Gap (verbatim from the table)

> C — Smoke Driver Refactor [parallel-builder]: Remove the global `Math.random` mulberry32 monkey-patch and pass `seed: 0xC0FFEE` via the real option; remove synthetic `gate.json` writes for `MOCK_MODE=force-*` and instead supply mocked `coherenceCheck` / `cohortExamples` / `cohortJudgeFunc` callbacks; keep `MOCK_MODE` failure shapes intact. (0% complete) [SMALL] depends on: A | files: `scripts/smoke-test.ts`

## 4. What to build

### Change 1 — Remove `Math.random` monkey-patch

Find the section in `scripts/smoke-test.ts` near orchestrator entry where `Math.random` is reassigned (e.g., `Math.random = mulberry32(0xC0FFEE)` or similar). Delete that block. Pass `seed: 0xC0FFEE` as part of the `EvolutionOptions` object handed to `runEvolution` for each smoke run.

```typescript
const options: EvolutionOptions = {
  targetPath: "tests/fixtures/smoke-skill/SKILL.md",
  // ... existing fields ...
  seed: 0xC0FFEE,
};
```

### Change 2 — Replace synthetic gate.json writes with real callbacks

For each `MOCK_MODE=force-*` branch, instead of post-run synthesizing a `gate.json` with the forced reason code, pass callbacks at run-config time that cause the engine's `runTieredGate` to emit that reason code naturally.

- **`MOCK_MODE=force-cohort-fail`**: supply
  ```typescript
  options.cohortExamples = [/* one or two dummy EvalExamples */];
  options.cohortJudgeFunc = async () => ({ composite: 0.1 });  // far below baseline
  ```
  This produces `reasonCode: "cohort_regression"`.

- **`MOCK_MODE=force-coherence-fail`**: supply
  ```typescript
  options.coherenceCheck = async () => ({ passed: false, detail: "smoke: forced coherence failure" });
  ```
  This produces `reasonCode: "coherence_failed"`.

- **`MOCK_MODE=force-typecheck-fail`**: the typecheck tier runs real `tsc --noEmit` and is not directly mockable via callbacks. Keep the existing approach (smoke driver may still need to influence this tier by other means — for example, a temp `tsconfig.smoke-broken.json` referenced via a future `tsConfigPath` option, OR the existing synthesis if no clean alternative exists). **If the only way to force-fail the typecheck tier requires further engine changes, leave a clear note in your final report and Lane E will decide**: either ship without `force-typecheck-fail` coverage and document it as a limitation, or accept a small additional synthesis for that one mode only. Do not bloat the change to wire a typecheck-fail hook in this lane.

Default mode (no `MOCK_MODE` set): no callbacks supplied. `runTieredGate` produces `skipped_no_cohort` and `skipped_no_check` as before — that is correct behavior under default smoke and matches what the verifiers expect.

### Change 3 — Delete the synthetic `gate.json` write code path

Find and remove every block in `scripts/smoke-test.ts` that calls `writeFileSync` / `writeFile` against a path ending in `gate.json`. The engine (Lane B) now writes `gate.json` itself for every run dir.

### Style and structure

- Match the existing `scripts/smoke-test.ts` style — no JSDoc, normal formatting.
- Keep `SMOKE_RUN_1` / `SMOKE_RUN_2` stdout markers intact so Lane E's verification still works.
- Imports: `EvolutionOptions` from `../src/types.js` (use the existing import style).

## 5. Hard constraints

1. No new npm dependencies.
2. `npm run typecheck` must pass at the end.
3. No edits outside `scripts/smoke-test.ts`. Verify via `git diff --name-only`.
4. No `Math.random` reassignment anywhere in the file (verify via grep).
5. No `writeFile` / `writeFileSync` targeting `gate.json` anywhere in the file (verify via grep).
6. `MOCK_MODE=force-cohort-fail` and `MOCK_MODE=force-coherence-fail` produce real engine output containing the corresponding `reasonCode`. `MOCK_MODE=force-typecheck-fail` either does the same OR is documented as deferred in your final report.
7. Determinism must be preserved (same seed → same dataset splits across runs).

## 6. Verification

```bash
npm run typecheck
npm run smoke
MOCK_MODE=force-cohort-fail npm run smoke
MOCK_MODE=force-coherence-fail npm run smoke
grep -n "Math.random\s*=" scripts/smoke-test.ts
grep -nE "writeFile.*gate\.json|writeFileSync.*gate\.json" scripts/smoke-test.ts
git diff --name-only
```

Expected:
- typecheck exit 0
- Three smoke invocations exit 0
- Both `grep` checks return no matches
- `git diff --name-only` lists only `scripts/smoke-test.ts`

For the cohort-fail and coherence-fail runs, optionally also verify the run-dir `gate.json` contains the expected reason code (engine-emitted):
```bash
cat .pi/hermes-self-evolution/runs/<latest-cohort-fail>/gate.json | grep -o "cohort_regression"
cat .pi/hermes-self-evolution/runs/<latest-coherence-fail>/gate.json | grep -o "coherence_failed"
```

## 7. Commit message

`refactor(smoke): use real engine hooks (seed + callbacks) instead of monkey-patch and synthetic gate.json (Gap C)`

## 8. Final report

```
### Lane C final report
- Worktree path / branch:
- Files modified: scripts/smoke-test.ts
- Changes applied:
  - Math.random monkey-patch removed: [yes/no]
  - seed: 0xC0FFEE wired via EvolutionOptions: [yes/no]
  - Synthetic gate.json writes removed: [yes/no]
  - MOCK_MODE=force-cohort-fail wiring: [yes/no, brief description]
  - MOCK_MODE=force-coherence-fail wiring: [yes/no, brief description]
  - MOCK_MODE=force-typecheck-fail: [wired via real path / left as deferred / partial]
- Lines added / removed:
- Verification:
  - npm run typecheck exit: ___
  - npm run smoke exit: ___
  - force-cohort-fail run exit + reason code observed: ___
  - force-coherence-fail run exit + reason code observed: ___
  - force-typecheck-fail run exit + reason code observed: ___
  - grep Math.random result: ___
  - grep gate.json write result: ___
  - git diff --name-only: ___
- Flags / blockers:
```

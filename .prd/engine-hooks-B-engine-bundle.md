# Lane B — Engine Wiring Bundle [parallel-builder]

## 1. Mission + read-first

You are a parallel-builder sub-agent. You thread three engine changes through `src/engine.ts` as one consistent bundle: (1) seeded RNG in `splitExamples`, (2) real tiered-gate callbacks at the `runTieredGate` call site, (3) per-run top-level `gate.json` output. You also remove the two `// SOFT-SPOT(...)` comments at the `runTieredGate` call site since they describe limitations you are now fixing.

**Read first** (each in full):
- `.prd/engine-hooks-orchestration.md` — pipeline context, dispatch table
- `tests/smoke-test-report.md` — Soft-Spot Dispositions table; the two items you are closing
- `src/tiered-gate.ts` — current `runTieredGate` signature you will pass real callbacks to
- `src/engine.ts` lines 290-305 (`splitExamples` with unseeded `Math.random`)
- `src/engine.ts` line ~499 (current `runTieredGate` call with the SOFT-SPOT comments)
- `src/engine.ts` lines ~521-562 (per-run output writes; you will add `gate.json` here)
- `docs/output-layout.md` — the `gate.json` format already documented

The new `EvolutionOptions` fields you consume are already on `main` from Lane A: `seed?`, `cohortExamples?`, `cohortJudgeFunc?`, `coherenceCheck?`.

## 2. Owned files

You may ONLY edit:
- `src/engine.ts` (existing)

You may NOT edit any other file, including `src/types.ts` (Lane A owns that), `src/tiered-gate.ts` (do not change its signature), or `scripts/smoke-test.ts` (Lane C owns that). If you find `src/tiered-gate.ts` needs a callback-shape adjustment, flag it for Lane E — do not patch.

## 3. Gap (verbatim from the table)

> B — Engine Wiring Bundle [parallel-builder]: Replace unseeded `Math.random` in `splitExamples` with seeded mulberry32 RNG; thread `cohortExamples`, `cohortJudgeFunc`, `coherenceCheck`, `baselineScore`, and `maxRegressionPct` into the `runTieredGate` call at engine.ts:499; write a top-level `gate.json` with the best candidate's `TieredGateResult[]` per run dir alongside existing `iterations/<n>.json#gateResults`. (0% complete) [MEDIUM] depends on: A | files: `src/engine.ts`

## 4. What to build

### Change 1 — Seeded RNG in `splitExamples`

The current `splitExamples` (around `src/engine.ts:290-298`) uses a Fisher-Yates shuffle with unseeded `Math.random`. Replace with a deterministic mulberry32 RNG when `options.seed` is provided; fall back to `Math.random` when unset.

Implementation sketch:
```typescript
function mulberry32(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6D2B79F5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function splitExamples(examples: EvalExample[], seed?: number): { ... } {
  const rng = seed !== undefined ? mulberry32(seed) : Math.random;
  const s = [...examples];
  for (let i = s.length - 1; i > 0; i -= 1) {
    const j = Math.floor(rng() * (i + 1));
    [s[i], s[j]] = [s[j]!, s[i]!];
  }
  // ... rest unchanged
}
```

Thread `options.seed` from `runTypeScriptEvolution` into the `splitExamples` call. **Important**: the unseeded path must remain functionally identical to today's behavior so existing callers see no change.

### Change 2 — Real callbacks at the `runTieredGate` call site

Locate the current call near `engine.ts:499`. It looks roughly like:

```typescript
// SOFT-SPOT(coherence-default): ...
// SOFT-SPOT(cohort-default): ...
try { gateResults = await runTieredGate({ cwd: options.cwd, candidateText: fullText, signal: options.signal }); } catch { /* gate unavailable; skip */ }
```

Replace with a call that threads the new options through. Pass `cohortExamples`, `cohortJudgeFunc` (as `judgeFunc`), and `coherenceCheck` directly. Compute `baselineScore` from `baselineHoldout.aggregate.composite` (already in scope at this point in the run; if not, hoist it). Pass `maxRegressionPct` only if you choose to expose a default override (otherwise omit and let `runTieredGate` use its 0.02 default).

```typescript
try {
  gateResults = await runTieredGate({
    cwd: options.cwd,
    candidateText: fullText,
    signal: options.signal,
    cohortExamples: options.cohortExamples,
    judgeFunc: options.cohortJudgeFunc,
    coherenceCheck: options.coherenceCheck,
    baselineScore: baselineHoldout.aggregate.composite,
  });
} catch { /* gate unavailable; skip */ }
```

**Remove the two `// SOFT-SPOT(coherence-default)` and `// SOFT-SPOT(cohort-default)` comments above this call site.** Their justification no longer applies — the hooks are now real.

### Change 3 — Write top-level `gate.json` per run

Locate the per-run output writes near `engine.ts:521-562` (where `report.md`, `manifest.json`, `dataset.json`, `traces/all-traces.json`, etc. are written). Add a write for `gate.json` containing the best candidate's `TieredGateResult[]`:

```typescript
// Top-level gate.json with best candidate's tier results
const bestGateResults = bestCandidate.gateResults ?? [];
await safeWriteFile(path.join(runDir, "gate.json"), JSON.stringify(bestGateResults, null, 2));
```

This must run before the function returns, alongside the existing writes. The shape is already documented in `docs/output-layout.md`. If `bestCandidate.gateResults` is undefined (no candidate produced gate results), write an empty array `[]` — never omit the file.

### Threading

Make sure `runEvolution(options)` forwards `options.seed`, `options.cohortExamples`, `options.cohortJudgeFunc`, `options.coherenceCheck` into `runTypeScriptEvolution` and that `runTypeScriptEvolution` makes them available at the three sites above. If the existing `options` object pass-through already covers this (likely), no extra plumbing is needed — only verify it.

## 5. Hard constraints

1. No new npm dependencies.
2. `npm run typecheck` must pass at the end.
3. No edits outside `src/engine.ts`. Verify via `git diff --name-only`.
4. No breaking changes to existing exports or behavior when the new options are unset. The unseeded path of `splitExamples` must produce the same statistical behavior as before. The `runTieredGate` call without callbacks must still produce `skipped_no_cohort` / `skipped_no_check` codes (just via the `runTieredGate` skip logic, not via the old call shape).
5. Match the existing dense single-line `;` idiom in `engine.ts` where you edit; use normal formatting if you add a new helper function (e.g., `mulberry32`).
6. Do NOT modify any other call sites or refactor unrelated logic.
7. Remove both `// SOFT-SPOT(coherence-default)` and `// SOFT-SPOT(cohort-default)` comments at the `runTieredGate` call site as part of this change.

## 6. Verification

```bash
npm run typecheck
npm run smoke
grep -n "SOFT-SPOT(coherence-default)\|SOFT-SPOT(cohort-default)" src/engine.ts
ls .pi/hermes-self-evolution/runs/*/gate.json | tail -3
git diff --name-only
```

Expected:
- typecheck exit 0
- `npm run smoke` exit 0; produces two run dirs (the smoke driver from Lane A's smoke-test pipeline still works in its current form)
- `grep` returns no matches (both SOFT-SPOT comments removed)
- `gate.json` exists in each new run dir
- `git diff --name-only` lists only `src/engine.ts`

Note: with the smoke driver unrefactored (Lane C may not have landed yet), `cohortExamples` and `coherenceCheck` will still be undefined and tier results will be skipped. That is fine — your job is the engine plumbing. Lane C wires the actual smoke-driver callbacks.

## 7. Commit message

`feat(engine): thread seed + tiered-gate callbacks + per-run gate.json; remove SOFT-SPOT comments (Gap B)`

## 8. Final report

```
### Lane B final report
- Worktree path / branch:
- Files modified: src/engine.ts
- Changes applied:
  - splitExamples: [seeded mulberry32 wired]
  - runTieredGate call: [callbacks threaded, baselineScore passed]
  - gate.json write: [path]
  - SOFT-SPOT comments removed: [yes/no]
- @ts-expect-error suppressors added: none
- Lines added / removed:
- Verification:
  - npm run typecheck exit: ___
  - npm run smoke exit: ___
  - grep SOFT-SPOT result: ___
  - gate.json present in run dirs: ___
  - git diff --name-only: ___
- Flags / blockers:
```

# Lane A — Types Extension [pre-phase]

## 1. Mission + read-first

You are the pre-phase sub-agent for the engine-hooks follow-up at `C:\dev\Desktop-Projects\pi-hermes-self-evolution`. You extend `EvolutionOptions` with four optional fields that downstream lanes (B engine, C smoke driver, D docs) consume.

**Read first** (each in full):
- `.prd/engine-hooks-orchestration.md` — pipeline context, dispatch table, acceptance criteria
- `tests/smoke-test-report.md` — Known Limitations section explaining what these hooks unblock
- `src/types.ts` — current `EvolutionOptions` interface (lines 148-163)
- `src/tiered-gate.ts` — current `TieredGateOptions` interface; your new fields mirror its callback shapes

## 2. Owned files

You may ONLY edit:
- `src/types.ts` (existing)

You may NOT edit any other file. If you find a downstream module needs a type adjustment you cannot make additively, flag it in your final report — do not patch.

## 3. Gap (verbatim from the table)

> A — Types Surface Extension [pre-phase]: Add `seed?: number`, `cohortExamples?: EvalExample[]`, `cohortJudgeFunc`, and `coherenceCheck` as optional fields on `EvolutionOptions`; ensure imports for `EvalExample` and tiered-gate callback signatures resolve. (0% complete) [TINY] depends on: none | files: `src/types.ts`

## 4. What to build

Extend the existing `EvolutionOptions` interface in `src/types.ts` with these four optional fields (additive only — do not remove or rename any existing field):

```typescript
export interface EvolutionOptions {
  // ... existing fields unchanged ...

  /** Deterministic seed for splitExamples and any future RNG consumers. When unset, falls back to unseeded Math.random. */
  seed?: number;

  /** Cohort of EvalExamples used by the tiered gate's cohort-regression tier. */
  cohortExamples?: EvalExample[];

  /** Judge callback invoked by the tiered gate to score the cohort. Required when cohortExamples is supplied. */
  cohortJudgeFunc?: (examples: EvalExample[]) => Promise<{ composite: number }>;

  /** Coherence check callback invoked by the tiered gate's coherence tier. */
  coherenceCheck?: () => Promise<{ passed: boolean; detail: string }>;
}
```

Notes:
- `EvalExample` is already exported from `src/types.ts` (lines 24-30) — no new import needed.
- The callback signatures must match `TieredGateOptions.judgeFunc` and `TieredGateOptions.coherenceCheck` in `src/tiered-gate.ts` so Lane B can pass them through directly.
- Place the four new fields at the **end** of `EvolutionOptions`, after `persistGolden`, in the order listed above.
- No JSDoc beyond the one-line `/** ... */` comments shown.

## 5. Hard constraints

1. No new npm dependencies.
2. `npm run typecheck` must pass at the end.
3. No edits outside `src/types.ts`. Verify via `git diff --name-only`.
4. No breaking changes to existing exports. Strictly additive.
5. Match the existing style in `src/types.ts` — minimal comments, no JSDoc beyond a single-line slash-star comment per new field.

## 6. Verification

```bash
npm run typecheck
git diff --name-only
```

Expected:
- typecheck exit 0
- `git diff --name-only` lists only `src/types.ts`

## 7. Commit message

`feat(types): add seed / cohortExamples / cohortJudgeFunc / coherenceCheck to EvolutionOptions (Gap A)`

## 8. Final report

Fill in and return at the end of your response:

```
### Lane A final report
- Worktree path / branch:
- Files modified: src/types.ts
- Public fields added: seed, cohortExamples, cohortJudgeFunc, coherenceCheck
- Lines added / removed:
- Verification:
  - npm run typecheck exit: ___
  - git diff --name-only: ___
- Flags / blockers:
```

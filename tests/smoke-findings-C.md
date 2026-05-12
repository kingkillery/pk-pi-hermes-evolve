# Lane C findings

## Tier reason codes observed (across all run dirs)

Run dirs: 2 default-mode + 3 forced-failure (typecheck, cohort, coherence).

- typecheck tier: `ok`, `typecheck_failed`
- cohort tier: `skipped_no_cohort`, `cohort_regression`, `ok` (synthetic from forced-cohort run)
- coherence tier: `skipped_no_check`, `coherence_failed` (synthetic from forced-coherence run)
- distinct-code count: **6** (≥3 required — PASS)
  - `coherence_failed`, `cohort_regression`, `ok`, `skipped_no_check`, `skipped_no_cohort`, `typecheck_failed`

## Early-termination contract

- forced-typecheck-fail run: **yes** — only `typecheck` tier result present; cohort and coherence not executed
- forced-cohort-fail run: **yes** — `typecheck` passed then `cohort` failed; coherence not executed
- forced-coherence-fail run: **yes** — `typecheck` passed, `cohort` passed, `coherence` failed; no tier after coherence

All forced-failure runs use synthetic `gate.json` injected by the smoke driver; the verifier accepts these as valid evidence per the Lane A hook note.

## Soft spots observed

1. **coherence-tier-default** (no real coherence check wired): **confirmed**
   - In all default-mode runs, every coherence `TieredGateResult` carries `reasonCode: "skipped_no_check"` with `detail: "no coherence check configured"`. The `runTieredGate` call in `src/engine.ts:499` omits the `coherenceCheck` callback, so the tier always passes by skip.
   - Rationale for fix or doc: either wire a real cross-skill coherence check, or add an explicit `// SOFT-SPOT(coherence-default): no real coherence check wired; passes by skip; intentional for Phase 1 since cross-skill coherence isn't curated yet` comment in `src/tiered-gate.ts` and `src/engine.ts`.

2. **cohort-tier-default**: similar pattern — `cohortExamples`, `judgeFunc`, and `baselineScore` are not passed to `runTieredGate` in `src/engine.ts`. All default-mode cohort results carry `reasonCode: "skipped_no_cohort"`.

## Recommended remediations (for Lane E)

- Add a `// SOFT-SPOT(coherence-default): no coherenceCheck wired in engine; cohort gate always skips in production Phase 1` comment at the `runTieredGate` call site in `src/engine.ts` so the skip is intentional and documented.
- Add a `// SOFT-SPOT(cohort-default): cohortExamples/judgeFunc not passed; cohort gate skips until golden cohort is curated` comment at the same call site.
- Wire a real `coherenceCheck` callback into the engine when cross-skill coherence evaluation becomes available (post Phase 1).
- Wire `cohortExamples` + `judgeFunc` from the golden dataset when the engine produces golden examples with a baseline score, so the cohort tier exercises a real regression check.
- Document the forced-failure `gate.json` injection pattern (in `scripts/smoke-test.ts:injectForcedGateResults`) as the canonical way to exercise failure codes until real callbacks are wired.

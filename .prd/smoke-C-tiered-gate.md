# Lane C — Tiered Gate Verifier [parallel-verifier]

## Agent prompt (paste verbatim into `Agent({prompt})`)

You are one of **three parallel verifier sub-agents** (B, C, D) for the Hermes Phase 1 runtime smoke-test pipeline at `C:\dev\Desktop-Projects\pi-hermes-self-evolution`. Lane A has produced shared smoke artifacts (≥2 run dirs plus forced-failure variants). You will be dispatched concurrently with Lanes B and D. Lane E waits on all three of you.

**Read `.prd/smoke-test-orchestration.md` first** for context, then `.prd/gap-analysis.md`, then `.prd/current-state.md`.

## Owned files (only these)
- `tests/smoke-tiered-gate.test.ts` (new)
- `tests/smoke-findings-C.md` (new — findings stub for Lane E)

You may **NOT** edit any file under `src/`, `scripts/`, `tests/fixtures/`, or any other test file. You may **NOT** invoke `runEvolution`; you only read Lane A's emitted run-dir artifacts.

## What you must produce

### 1. `tests/smoke-tiered-gate.test.ts`

A typecheck-clean Node ESM TypeScript file that exports `runTieredGateVerifier(runDirs: string[])` and asserts the following across **all** run dirs Lane A produced (the two normal runs plus the forced-failure-mode runs):

- For each candidate accepted into the run, `gateResults?` (on the `CandidateRecord`) — or per-iteration `iterations/<n>.json` field — contains an array of `TieredGateResult` entries.
- Across the full set of run dirs, **at least 3 distinct `reasonCode` values** are observed. The set must include codes from all three tiers (e.g., `typecheck_failed`, `cohort_regression`, `coherence_failed`) when forced-failure modes ran. Acceptable success codes (`ok`, `skipped_no_cohort`, `skipped_no_check`) count toward the set as long as the three failure codes also appear.
- Each `TieredGateResult` matches the type shape from `src/types.ts:TieredGateResult` (`tier`, `passed`, `reasonCode`, `detail`, `durationMs` all present, types correct).
- For the coherence tier specifically: confirm whether Lane A's smoke runs invoked the `coherenceCheck` callback or fell through to `skipped_no_check`. If always `skipped_no_check`, record this as the **coherence-tier-default soft spot** — no real coherence check is wired in production code.
- Optional but valuable: confirm that when a tier fails, **subsequent tiers are not executed** (the `TieredGateResult[]` ends at the failing tier). This validates the early-termination contract in `src/tiered-gate.ts`.

Include a runnable `main` block:
```bash
node --experimental-strip-types tests/smoke-tiered-gate.test.ts \
  .pi/hermes-self-evolution/runs/<RUN_1> \
  .pi/hermes-self-evolution/runs/<RUN_2> \
  [<FORCED_FAIL_RUN_1> ...]
```

### 2. `tests/smoke-findings-C.md`

```markdown
# Lane C findings

## Tier reason codes observed (across all run dirs)
- typecheck tier: <list of codes>
- cohort tier: <list of codes>
- coherence tier: <list of codes>
- distinct-code count: <N>  (≥3 required)

## Early-termination contract
- forced-typecheck-fail run: <subsequent tiers skipped? yes/no>
- forced-cohort-fail run: <subsequent tiers skipped? yes/no>
- forced-coherence-fail run: <subsequent tiers skipped? yes/no>

## Soft spots observed
1. coherence-tier-default (no real coherence check wired): <confirmed/not-confirmed>
   - rationale for fix or doc:
2. (other observations)

## Recommended remediations (for Lane E)
- [bullet list, ≤5 items, e.g.:
  - wire a real cross-skill coherence check, OR
  - add an explicit `// SOFT-SPOT(coherence-default): no real check yet; passes by skip; intentional for Phase 1 since cross-skill cohort isn't curated` comment in `src/tiered-gate.ts`
  ]
```

## Constraints

1. **No engine or tiered-gate code edits.** Document remediations in `smoke-findings-C.md` for Lane E.
2. **Read-only against run dirs.**
3. **No new npm dependencies.**
4. **Type imports only from `../src/types.js`.**
5. **Style**: normal formatting. No comments except non-obvious WHY.

## Verification (run before declaring done)
```bash
npm run typecheck
node --experimental-strip-types tests/smoke-tiered-gate.test.ts <run-dirs...>
git diff --name-only
```

Expected:
- `npm run typecheck`: exit 0
- Verifier: exit 0 (or 1 if a real failure is found — document in findings)
- `git diff --name-only` lists only your two owned files

Commit message: `test(smoke): tiered gate verifier (smoke C)`

## Final report
```
### Lane C final report
- Worktree path / branch:
- Files created: tests/smoke-tiered-gate.test.ts, tests/smoke-findings-C.md
- Distinct reason codes observed: [list] (count: ___)
- Early-termination contract held: [yes/no/partial]
- coherence-tier-default soft spot: [confirmed/not-confirmed/inconclusive]
- Verification: typecheck exit ___; verifier exit ___
- Flags/blockers: ___
```

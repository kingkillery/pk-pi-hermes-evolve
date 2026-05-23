# Lane E — Acceptance and Parity Gate [acceptance-gate]

## 1. Mission + read-first

You are the acceptance-gate sub-agent. Lanes A, B, C, D have all merged into main. Your job is to update the smoke-test evidence document to reflect closed limitations, run the full verification gate, and confirm every item in the acceptance checklist below resolves to a concrete observation.

**Read first** (each in full):
- `.prd/engine-hooks-orchestration.md` — pipeline context and the full acceptance checklist
- `tests/smoke-test-report.md` — Known Limitations section you will update
- `src/engine.ts` lines around the `runTieredGate` call (formerly engine.ts:499) — confirm SOFT-SPOT comments are gone (Lane B should have removed them)
- `src/types.ts` — confirm `EvolutionOptions` has the four new optional fields (from Lane A)
- `scripts/smoke-test.ts` — confirm `Math.random` reassignment and synthetic `gate.json` writes are gone (from Lane C)

## 2. Owned files

You may ONLY edit:
- `tests/smoke-test-report.md` (existing)

You may NOT edit any source file, doc file, or other test file. If Lane B forgot to remove a SOFT-SPOT comment, or Lane C left a `Math.random` reassignment, or Lane D missed a field in the docs, **document the gap in your final report and stop** — do not patch other lanes' files yourself. The orchestrator decides whether to dispatch a remediation lane or fix in main.

## 3. Gap (verbatim from the table)

> E — Acceptance and Parity Gate [acceptance-gate]: Remove both `// SOFT-SPOT(coherence-default)` and `// SOFT-SPOT(cohort-default)` comments at engine.ts:499; clear the two corresponding entries from the Known Limitations section of `tests/smoke-test-report.md`; run `npm run test:gates` (typecheck + python:check + test:parity + test:smoke) and confirm all four smoke verifiers pass against real engine output with no edits; verify top-level `gate.json` matches best-candidate `iterations/<n>.json#gateResults`. (0% complete) [SMALL] depends on: B, C, D | files: `src/engine.ts`, `tests/smoke-test-report.md`, `tests/smoke-iterations.test.ts`, `tests/smoke-executor.test.ts`, `tests/smoke-tiered-gate.test.ts`, `tests/smoke-lineage.test.ts`

Note: the gap's `files:` list is the full **read** set. Your actual **owned** (writable) set is only `tests/smoke-test-report.md`. The verifier test files are read-only to you; Lane B already handled the SOFT-SPOT comment removal in `src/engine.ts`.

## 4. What to build

### Step 1 — Pre-flight grep checks

Confirm Lanes B, C, D did their work. Run:
```bash
grep -n "SOFT-SPOT(coherence-default)\|SOFT-SPOT(cohort-default)" src/engine.ts
grep -nE "Math\.random\s*=" scripts/smoke-test.ts
grep -nE "writeFile.*gate\.json|writeFileSync.*gate\.json" scripts/smoke-test.ts
grep -nE "\bseed\b" src/types.ts
grep -n "gate.json" docs/output-layout.md
grep -n "Engine hooks promoted" CHANGELOG.md
```

Expected:
- First three return no matches (Lane B + Lane C cleanup confirmed)
- Last three return matches (Lane A + Lane D additions confirmed)

If any expectation fails, stop and report the missing/extra piece.

### Step 2 — Run the full gate

```bash
npm run test:gates
```

This runs typecheck, python:check, test:parity, and test:smoke. All four must exit 0. The smoke driver from Lane C should now exercise the real engine hooks from Lane B with no monkey-patching or synthesis.

### Step 3 — Cross-verify gate.json parity

Pick the most recent default-mode run dir produced by `npm run test:smoke`. Compare its top-level `gate.json` to the gateResults inside the iteration that the manifest names as the best candidate:

```bash
LATEST=$(ls -dt .pi/hermes-self-evolution/runs/* | head -1)
BEST_ITER=$(jq -r '.bestCandidate.acceptanceMode // ""; .bestCandidate.name' "$LATEST/manifest.json")
# extract the iteration number from the best candidate (e.g., "iter-2-foo" -> 2)
# compare gate.json to iterations/<n>.json#gateResults
```

The contents should match (same `TieredGateResult[]`). Document the comparison result in the report.

### Step 4 — Update `tests/smoke-test-report.md`

Find the Known Limitations section. Remove the two bullet items describing:
- The smoke driver's global `Math.random` monkey-patch
- The smoke driver's synthetic `gate.json` writes for forced-failure modes

Replace them with a single short note in a new "Closed in follow-up" subsection at the end of Known Limitations:

```markdown
### Closed in follow-up

The two engine-hook gaps deferred in the original smoke-test run have been closed:

- `seed?: number` on `EvolutionOptions` is now threaded into `splitExamples` via a local mulberry32 RNG. The smoke driver passes `seed: 0xC0FFEE` directly; the global `Math.random` monkey-patch is removed.
- `cohortExamples` / `cohortJudgeFunc` / `coherenceCheck` are now first-class `EvolutionOptions` fields, threaded into the `runTieredGate` call. The engine writes a top-level `gate.json` per run dir. The smoke driver invokes real callbacks for `MOCK_MODE=force-cohort-fail` and `MOCK_MODE=force-coherence-fail`; no synthetic `gate.json` writes remain.

See CHANGELOG `Unreleased` § "Engine hooks promoted to EvolutionOptions" for the full set of changes.
```

If `MOCK_MODE=force-typecheck-fail` is still using a workaround (per Lane C's final report), add a one-line bullet to the regular Known Limitations explaining that one mode specifically.

Also update the Summary section (~120 words at top of the report) to mention that the two engine hooks have been promoted.

### Step 5 — Final commit

Single commit. Message:

`chore(smoke): close engine-hooks limitations in smoke-test report (Gap E)`

## 5. Hard constraints

1. No new npm dependencies.
2. `npm run test:gates` must exit 0 at the end. If it doesn't, you cannot commit — stop and report.
3. No edits outside `tests/smoke-test-report.md`. Verify via `git diff --name-only`.
4. The Known Limitations section must no longer contain the two deferred items by name.
5. The Closed-in-follow-up subsection must reference concrete file paths and the CHANGELOG entry.
6. Do not invent new fixes for issues other lanes were supposed to handle. If a lane left work undone, report and stop.

## 6. Verification

```bash
grep -n "SOFT-SPOT(coherence-default)\|SOFT-SPOT(cohort-default)" src/engine.ts          # expect no matches
grep -nE "Math\.random\s*=" scripts/smoke-test.ts                                        # expect no matches
grep -nE "writeFile.*gate\.json|writeFileSync.*gate\.json" scripts/smoke-test.ts         # expect no matches
grep -A1 "Closed in follow-up" tests/smoke-test-report.md                                # expect new subsection
npm run test:gates                                                                       # expect exit 0
git diff --name-only                                                                     # expect only tests/smoke-test-report.md
```

## 7. Commit message

`chore(smoke): close engine-hooks limitations in smoke-test report (Gap E)`

## 8. Final report

```
### Lane E final report
- Worktree path / branch:
- Files modified: tests/smoke-test-report.md
- Pre-flight checks:
  - SOFT-SPOT comments removed from src/engine.ts: [yes/no]
  - Math.random reassignment removed from scripts/smoke-test.ts: [yes/no]
  - Synthetic gate.json writes removed from scripts/smoke-test.ts: [yes/no]
  - seed field present in src/types.ts: [yes/no]
  - gate.json referenced in docs/output-layout.md: [yes/no]
  - CHANGELOG entry present: [yes/no]
- gate.json parity check (top-level vs best-iteration gateResults): [match/mismatch + brief notes]
- Verification:
  - npm run test:gates exit: ___
  - typecheck exit: ___
  - python:check exit: ___
  - test:parity exit: ___ (rowsChecked: ___)
  - test:smoke exit: ___ (verifier failure count: ___)
  - git diff --name-only: ___
- Flags / blockers:
```

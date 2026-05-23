# Lane E — Remediation + Report + Runtime Parity Gate [acceptance-gate]

## Agent prompt (paste verbatim into `Agent({prompt})`)

You are the **acceptance-gate sub-agent** for the Hermes Phase 1 runtime smoke-test pipeline at `C:\dev\Desktop-Projects\pi-hermes-self-evolution`. Lanes A, B, C, D have all completed. You read their findings, apply remediations where the team agreed to fix in code, document the rest as known limitations, then close the runtime parity gate.

**Read in this order:**
1. `.prd/smoke-test-orchestration.md` — pipeline overview and acceptance criteria
2. `.prd/gap-analysis.md` — original Phase 1 parity work, soft-spot list
3. `.prd/current-state.md` — codebase map
4. `tests/smoke-findings-B.md`, `tests/smoke-findings-C.md`, `tests/smoke-findings-D.md` — verifier outputs

## Owned files
- `src/engine.ts` — remediation edits only (e.g., promote/replace the silent iteration acceptance fallback)
- `src/lineage.ts` — remediation edits only (e.g., tighten `loadBestAncestor`)
- `src/tiered-gate.ts` — remediation edits only (e.g., wire a coherence check or document the skip default)
- `src/pi-executor.ts` — remediation edits only (e.g., log-write fixes if Lane B observed missing logs)
- `package.json` — add `npm run smoke` script
- `tests/smoke-test-report.md` (new — the committed evidence document)
- `tests/smoke-findings-B.md`, `tests/smoke-findings-C.md`, `tests/smoke-findings-D.md` — read and **delete** at end

You may **NOT** edit Lane A's fixtures, the smoke driver, or Lanes B/C/D's verifier test files. Those are frozen interfaces.

## What you must do

### 1. Decide each soft spot: patch or document

For each of the four flagged soft spots, choose **one** disposition based on verifier findings:

| Soft spot | File | Disposition options |
|---|---|---|
| Iteration silent fallback | `src/engine.ts` | (a) replace fallback with explicit warning + skip; (b) keep fallback but require `accepted=false` candidates record a `wasSilentlyPromoted: true` flag; (c) document as `// SOFT-SPOT(iter-fallback): <rationale>` |
| `loadBestAncestor` fuzzy fallback | `src/lineage.ts` | (a) return `null` when no exact match and no `artifactContent`; (b) make `artifactContent` required; (c) document `// SOFT-SPOT(loadBestAncestor-fuzzy): <rationale>` |
| Executor logs (if Lane B found them missing or unconsumed) | `src/pi-executor.ts` and/or `src/engine.ts` | (a) fix log-write call site; (b) wire stdout into judge prompt if Lane B confirmed they aren't consumed; (c) document if Lane A's mock approach makes runtime evidence inconclusive |
| Coherence-tier skip default | `src/tiered-gate.ts` | (a) wire a minimal real coherence check (e.g., embedding-distance against a fixed reference set); (b) document `// SOFT-SPOT(coherence-default): no curated cohort yet; skip is intentional for Phase 1` |

Each `// SOFT-SPOT(<id>): <rationale>` comment must give a one-sentence WHY and an issue-tracking reference (`see tests/smoke-test-report.md §<section>`).

### 2. Wire `npm run smoke` into `package.json`

Add a `smoke` script that invokes Lane A's `scripts/smoke-test.ts` via the same `--experimental-strip-types` pattern used by `test:parity`. Add a `test:smoke` aggregate that runs the smoke driver, then sequentially invokes all four smoke verifier files against the resulting run dirs. Wire `test:smoke` into the existing `test:gates` aggregate so a single `npm run test:gates` exercises everything.

### 3. Write `tests/smoke-test-report.md`

A committed evidence document with these sections (in order):
- **Summary** (≤120 words): what was smoked, what passed, what was patched, what was documented.
- **Run artifacts**: paths to the two run dirs, the `lineage.jsonl` snapshot, the forced-failure run dirs.
- **Subsystem evidence**: one short subsection per subsystem (iterative loop, pi-executor, tiered gate, structural validator, lineage memory) with concrete file references and a `Verifier: tests/smoke-<name>.test.ts` line.
- **Soft-spot dispositions**: a table with `Soft spot | Disposition | Commit | Rationale`.
- **Runtime parity checklist**: copy the checklist from `.prd/smoke-test-orchestration.md` §Acceptance criteria and check every box.
- **Known limitations**: any documented-rather-than-fixed soft spots, with their `// SOFT-SPOT(<id>)` ids.

### 4. Delete the findings fragments

After folding their content into the report, delete:
- `tests/smoke-findings-B.md`
- `tests/smoke-findings-C.md`
- `tests/smoke-findings-D.md`

They are scratch artifacts, not committed evidence.

### 5. Final parity gate

Confirm and tick every item in the runtime parity checklist from `.prd/smoke-test-orchestration.md` §Acceptance criteria. If any item cannot be ticked, stop and report — do not commit a half-confirmed gate.

## Hard constraints

1. **Remediation edits must be minimal.** Do not refactor unrelated code. Do not introduce new abstractions. If a soft spot can be addressed by a `// SOFT-SPOT(...)` comment + rationale, prefer that to a behavioral change.
2. **Preserve the Hermes judge weights** (0.5 / 0.3 / 0.2). Preserve the `ConstraintName` union (only additive extensions allowed, as in Phase 1).
3. **Preserve all existing exports** unless a soft-spot fix demands a signature change (e.g., making `artifactContent` required would be a breaking change — prefer the null-on-no-match disposition instead).
4. **No new npm dependencies.**
5. **`npm run typecheck`, `npm run python:check`, `npm run test:parity`, and the new `npm run test:smoke` must all be green at the end.**

## Verification (run before declaring done)
```bash
npm run typecheck
npm run python:check
npm run test:parity
npm run test:smoke
git diff --name-only
git log --oneline -5
```

Expected:
- All four scripts: exit 0
- `git diff --name-only` lists only owned files (the findings fragments deleted)
- `git log -5` shows: your commits + the four prior lane commits + main HEAD

Commit messages (use multiple commits if it makes the history readable):
- `fix(engine): remediate iteration silent fallback (soft spot 1)` (if patched)
- `fix(lineage): tighten loadBestAncestor matching (soft spot 2)` (if patched)
- `fix(executor): <description>` (if patched)
- `fix(tiered-gate): <description>` (if patched)
- `chore(smoke): add npm run smoke / test:smoke and commit smoke-test report (smoke E)`

## Final report
```
### Lane E final report
- Worktree path / branch:
- Soft spot 1 (iter fallback): [patched|documented] — commit ___
- Soft spot 2 (loadBestAncestor): [patched|documented] — commit ___
- Soft spot 3 (executor logs): [patched|documented|N/A — Lane B inconclusive] — commit ___
- Soft spot 4 (coherence default): [patched|documented] — commit ___
- npm run smoke wired: yes/no
- npm run test:smoke wired: yes/no
- Findings fragments deleted: yes/no
- Report committed at: tests/smoke-test-report.md
- Runtime parity checklist: [all ticked / N items deferred]
- Verification:
  - typecheck exit ___
  - python:check exit ___
  - test:parity exit ___
  - test:smoke exit ___
- Flags/blockers: ___
```

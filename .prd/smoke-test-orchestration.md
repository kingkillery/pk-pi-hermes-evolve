# Smoke-Test Orchestration — Hermes Phase 1 Runtime Confirmation

## Purpose

The Phase 1 parity code on `main` (commits c417bfd → 5559ad4) typechecks and passes `npm run test:parity` (18/18 rows), but **has never executed against a real `SKILL.md`**. This pipeline produces runtime evidence that each new subsystem actually behaves as designed, then either remediates or explicitly documents the four flagged soft spots:

1. Iteration acceptance has a silent fallback that promotes the highest-validation candidate when zero iterations strictly accept (`src/engine.ts`)
2. `loadBestAncestor` uses a fuzzy runId-substring match when `artifactContent` isn't passed (`src/lineage.ts`)
3. Executor logs were specified but never observed (`src/pi-executor.ts` writes to `executor/<iter>/<ex>/`)
4. Tiered gate's coherence tier defaults to passed-with-skip-reason — no real coherence check is wired (`src/tiered-gate.ts`)

## Letter groups (5 lanes, ≤5 sub-agents)

| Letter | Lane | Archetype | Effort | Depends on | File |
|---|---|---|---|---|---|
| **A** | smoke fixture + mock driver | pre-phase | LARGE | none | `.prd/smoke-A-fixture-and-runs.md` |
| **B** | iteration + executor verifiers | parallel-verifier | MEDIUM | A | `.prd/smoke-B-iteration-and-executor.md` |
| **C** | tiered-gate verifier | parallel-verifier | SMALL | A | `.prd/smoke-C-tiered-gate.md` |
| **D** | lineage verifier + findings consolidation | parallel-verifier | SMALL | A | `.prd/smoke-D-lineage-and-findings.md` |
| **E** | remediation + report + runtime parity gate | acceptance-gate | MEDIUM | A, B, C, D | `.prd/smoke-E-remediation-and-gate.md` |

## Execution sequence

**Phase 0 — solo pre-phase (Lane A)**
- One sub-agent in a worktree.
- Produces the shared smoke artifacts: `scripts/smoke-test.ts`, `tests/fixtures/smoke-skill/SKILL.md`, `tests/fixtures/smoke-skill/mock-llm-responses.json`, plus two consecutive run directories under `.pi/hermes-self-evolution/runs/` (second linked to first via lineage parent ref).
- Blocks Phase 1 until merged.

**Phase 1 — 3-way parallel verifiers (Lanes B, C, D)**
- Three sub-agents in worktrees, branched off the merged main (post-A).
- Each consumes Lane A's run dirs **read-only**. None may call `runEvolution`.
- File ownership is disjoint by design — each verifier owns only its own test file.
- Each emits a findings stub (markdown fragment) consumed by Lane E.

**Phase 2 — solo acceptance gate (Lane E)**
- One sub-agent.
- Reads all three verifier findings, applies remediations to `src/engine.ts` and/or `src/lineage.ts` (or documents limitations inline with rationale), wires `npm run smoke` into `package.json`, commits `tests/smoke-test-report.md`, runs the final parity checklist.

## Dispatch rules (orchestrator contract)

1. **Strict serialization**: A → {B, C, D in parallel} → E. No overlap between phases.
2. **Worktree isolation**: all 5 lanes run in `isolation: "worktree"` to prevent cross-pollution. The orchestrator merges sequentially.
3. **Model routing**: Lane A is LARGE → opus. Lanes B is MEDIUM → opus. Lanes C, D are SMALL → sonnet. Lane E is MEDIUM → opus.
4. **Subagent type**: `executor` for all lanes (matches the pattern that produced commits c417bfd–5559ad4).
5. **Verification before merge**: every lane must run `npm run typecheck` and produce a clean `git diff --name-only` against its owned files. Lane A additionally runs `npm run smoke` end-to-end and confirms two run dirs were emitted.
6. **No live LLM, no live pi**: Lane A's mock driver must run on a clean machine with no `pi` binary on PATH and no LLM credentials. If A relies on a PATH-shim or spawn-mock, document the approach in its final report.
7. **Findings handoff**: Lanes B, C, D each commit a short `tests/smoke-findings-<letter>.md` file (≤80 lines) that Lane E consumes. These are deletable artifacts; Lane E folds their contents into the final `tests/smoke-test-report.md` and deletes the fragments before its commit.

## Disjoint file-ownership matrix

| File | Lane A | Lane B | Lane C | Lane D | Lane E |
|---|---|---|---|---|---|
| `scripts/smoke-test.ts` | own | – | – | – | – |
| `tests/fixtures/smoke-skill/**` | own | – | – | – | – |
| `tests/smoke-iterations.test.ts` | – | own | – | – | – |
| `tests/smoke-executor.test.ts` | – | own | – | – | – |
| `tests/smoke-tiered-gate.test.ts` | – | – | own | – | – |
| `tests/smoke-lineage.test.ts` | – | – | – | own | – |
| `tests/smoke-findings-B.md` | – | own (write) | – | – | own (delete) |
| `tests/smoke-findings-C.md` | – | – | own (write) | – | own (delete) |
| `tests/smoke-findings-D.md` | – | – | – | own (write) | own (delete) |
| `tests/smoke-test-report.md` | – | – | – | – | own |
| `src/engine.ts` | – | – | – | – | own (remediation only) |
| `src/lineage.ts` | – | – | – | – | own (remediation only) |
| `src/tiered-gate.ts` | – | – | – | – | own (remediation only) |
| `package.json` | – | – | – | – | own (add `smoke` script) |

The empty intersection across B, C, D is the parallelism enabler. Lane E owns all post-merge edits.

## Acceptance criteria (Lane E's runtime parity checklist)

A run is considered Hermes-Phase-1-runtime-confirmed when **all** of the following hold:

- [ ] `npm run smoke` exits 0 and produces ≥2 run dirs under `.pi/hermes-self-evolution/runs/`
- [ ] The two run dirs are linked via `lineage.jsonl` (`parentRunId` set on the second)
- [ ] Iteration shape verifier (B-iter) confirms ≥2 `iterations/<n>.json` files per run with non-empty `reflectionPrompt.priorTraces` from iteration 2 onward
- [ ] Executor verifier (B-exec) confirms ≥1 non-empty `executor/<iter>/<ex>/stdout.log` exists
- [ ] Tiered-gate verifier (C) observes ≥3 distinct `reasonCode` values across forced-failure mock modes (e.g., `typecheck_failed`, `cohort_regression`, `coherence_failed`)
- [ ] Lineage verifier (D) confirms `loadBestAncestor` returns a non-null entry on the second run when called with the same artifact content as the first
- [ ] Each of the four flagged soft spots has either a remediation commit OR a documented-limitation comment in code with `// SOFT-SPOT(<id>): <rationale>` format
- [ ] `tests/smoke-test-report.md` references the concrete run dir paths and links every checklist item to a verifier test name

## Out of scope for this pipeline

- Live LLM-judged smoke runs (deferred; requires credentials and budget)
- Performance benchmarking (deferred to a separate pipeline)
- Hermes Phase 2 (tool-description evolution) — distinct work, not gated on this

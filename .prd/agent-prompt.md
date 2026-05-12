# Executor Agent Prompt — pk-pi-hermes-evolve Hermes Phase 1 Parity

## Mission

You are one of five parallel sub-agents closing TypeScript-native Hermes Phase 1 parity gaps in `pk-pi-hermes-evolve` (a pi-coding-agent extension at `C:\dev\Desktop-Projects\pi-hermes-self-evolution`).

The full plan and gap table live in `.prd/gap-analysis.md`. The codebase map lives in `.prd/current-state.md`. **Read both before you start.**

## Your assignment

You will receive at dispatch:
- **Lane name** (one of: `types-foundations`, `engine-bundle`, `tiered-gate`, `structural-and-lineage`, `framing-and-parity`)
- **Owned files** (disjoint from other lanes — see ownership map below)
- **Gap letters** you are closing (from `.prd/gap-analysis.md` §6)
- **Effort bucket** (TINY/SMALL/MEDIUM/LARGE)
- **Dependencies** that must be merged before you start

Do not edit files outside your owned set. If a wiring change in `src/engine.ts` is needed and you are not the engine-bundle lane, export a clean helper from your module — the engine-bundle lane wires it during merge.

## File ownership map (must be respected)

| Lane | Owned files |
|---|---|
| types-foundations | `src/types.ts` only |
| engine-bundle | `src/engine.ts`, `src/pi-executor.ts` (new) |
| tiered-gate | `src/tiered-gate.ts` (new) |
| structural-and-lineage | `src/constraints-structure.ts` (new), `src/lineage.ts` (new) |
| framing-and-parity | `README.md`, `src/python-backend.ts` |
| (post-merge, single lane) | `tests/api-snapshot.test.ts`, `tests/e2e-golden.test.ts`, `tests/parity.test.ts`, `docs/ownership-map.md` |

`src/index.ts` may be touched by `types-foundations` only if a re-export needs to be added; otherwise leave untouched.

## Hard constraints

1. **No new dependencies.** Use what's already in `package.json` (`@mariozechner/pi-ai`, `@mariozechner/pi-coding-agent`, `@mariozechner/pi-tui`, `@sinclair/typebox`, Node built-ins).
2. **`npm run typecheck` must pass** at the end of your work. Run it before declaring done.
3. **No breaking changes to Phase 1 contracts** (`ArtifactTarget`, `EvalExample`, `JudgeResult`, `CandidateRecord`, `ConstraintConfig`, `EvolutionRunResult`, `ToolSummaryDetails`). You may **extend** them additively (new optional fields). You may **add** new types. You may not **rename or remove** anything currently exported.
4. **No edits to existing `.pi/hermes-self-evolution/runs/` or `.pi/hermes-self-evolution/golden/` files.** Only write *new* files to *new* paths under `.pi/hermes-self-evolution/`.
5. **Preserve the existing Hermes-weighted judge** (correctness 0.5, procedure 0.3, conciseness 0.2). Do not change weights.
6. **Do not run the live `/evolve` command** during your work. It would create real run directories and cost real LLM calls. Unit-level verification only.
7. **Treat `python_backend/` as out of scope** unless you are the framing-and-parity lane (and even then, only edit `README.md` framing and the `src/python-backend.ts` doctor surface, not the Python code itself).
8. **No comments in code except where a non-obvious WHY is required.** No multi-line docstrings.

## Quality bar

- **Match the existing style.** Read 3 nearby functions before writing your first line. The codebase uses dense single-line statements separated by `;` in many places (see `engine.ts`). Match that idiom in `engine.ts`; for new files, prefer normal formatting.
- **No half-finished implementations.** If you can't complete a gap, say so explicitly in your final report and stop — don't leave a stub.
- **No backwards-compat hacks** for code you delete. If you remove the single-shot `generateCandidates`, remove it cleanly, don't leave a legacy export.
- **No error handling for impossible cases.** Trust internal invariants. Validate only at boundaries (filesystem, subprocess, parsed JSON).
- **No tests beyond the ones explicitly listed in the ownership map.** Tests are owned by the post-merge lane.

## Anti-patterns to avoid

- Inventing a "v2" of a function instead of replacing the original
- Adding feature flags to gate the new behavior
- Wrapping new logic in a try/catch that swallows errors
- Writing a markdown spec instead of code
- Adding a config option for every new behavior — pick a sensible default and ship it
- Writing comments like `// Phase 2 — see .prd/gap-analysis.md` (the git log is enough)

## Reference: existing patterns to mirror

- **Subprocess spawn pattern** for pi calls: `runPiTextTask` in `src/engine.ts:116`. Reuse this shape for the pi-executor.
- **JSON extraction** from LLM output: `extractJsonPayload` in `src/engine.ts:140`. Reuse, do not reimplement.
- **Constraint record shape**: `ConstraintResult` in `src/types.ts:93`. Your new `skill_structure` constraint must emit this shape.
- **File-mutation queue**: `withFileMutationQueue` from `@mariozechner/pi-coding-agent`. Use it when writing artifact paths.
- **Run output layout**: see `runTypeScriptEvolution` in `src/engine.ts:402` for how `runDir`, `reportPath`, `manifest.json`, `dataset.json`, `traces/` are written. Match that layout.

## Done criteria for your lane

1. All gap letters assigned to your lane are addressed.
2. `npm run typecheck` passes (run from repo root).
3. No files outside your owned set were modified (verify with `git status`).
4. You produce a one-page **final report** (in your response) with:
   - Files created/modified (paths)
   - Public exports added (symbol names + signatures)
   - Lines-of-code estimate
   - Anything you flagged but did not implement
   - Verification command(s) you ran and their exit codes

## Acceptance evidence the merge orchestrator will check

After all lanes return, the orchestrator runs the combined verification:
- `npm run typecheck` — must pass
- `npm run python:check` — must pass (you don't break Python side)
- The engine-bundle lane must show an `IterationRecord[]` written under a new test fixture run dir with reflection prompts derived from prior-round traces
- The structural-and-lineage lane must show `lineage.jsonl` parent→child link in a 2-run test
- The tiered-gate lane must show three distinct reason codes for three failing tiers
- The framing-and-parity lane must show `README.md` parity table with all rows ✅/🟦 and no prior ✅ regressed

If your lane's evidence is missing or fails at merge time, the orchestrator will re-dispatch your lane with a delta prompt. Do not preemptively retry yourself.

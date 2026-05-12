# Contributing

This document covers how the codebase is organized, the conventions to follow when changing it, the parallel-PRD dispatch pattern used for non-trivial work, and the verification gates every change must clear.

## Code organization

| Path | Purpose |
|---|---|
| `src/` | TypeScript engine, executor, gate, lineage, constraints, types, pi extension entrypoint |
| `tests/` | Snapshot, parity, and shape tests. No runtime test framework — tests are TypeScript modules that compile clean and can be invoked via `node --experimental-strip-types`. |
| `scripts/` | Auxiliary runners: traced Ralph loop, Sokoban benchmark scaffold, smoke-test driver |
| `python_backend/` | Optional DSPy/GEPA acceleration sidecar |
| `benchmarks/` | Bundled benchmark assets (Sokoban pack) |
| `docs/` | Architecture, configuration, output-layout, ownership-map, this file |
| `.prd/` | Planning artifacts for in-flight or recent work. Committed; not part of the npm package. |

See [docs/architecture.md](docs/architecture.md) for a per-module description.

## Style conventions

### TypeScript

- ESM imports throughout. Every relative import ends in `.js` even when the source file is `.ts`. This is the standard ESM TypeScript pattern; do not break it.
- `src/engine.ts` uses a dense single-line-with-`;` idiom in several places (see lines 130, 199, 226). When editing existing code in `engine.ts`, match the surrounding idiom. When creating a new file, prefer normal formatting.
- Imports from `./types.js` are type-only where possible. Use `import type` to avoid runtime imports of pure type modules.
- The `ConstraintName` union and the `EvolutionOptions.backend` union are intentionally additive. New values are appended; existing values are not renamed or removed.

### Comments and documentation

Default to writing no comments. Comments are reserved for non-obvious WHY: hidden constraints, subtle invariants, workarounds for specific issues. Do not comment on WHAT the code does — well-named identifiers and types convey that.

The exception is `// SOFT-SPOT(<id>): <rationale>` comments used to document accepted limitations identified during smoke-testing or review. The id appears in the corresponding evidence document (e.g., `tests/smoke-test-report.md`).

### Markdown

- Sentence-case headings.
- Minimal emoji usage. The only emoji currently in committed docs is the parity-table status indicator (✅ / 🟦). Do not introduce others.
- Tables for parameter lists, options, file-ownership matrices, and parity rows.
- Code fences for shell commands and file content excerpts.
- Reference files by their repo-relative path with backticks.

## Verification gates

Every change must pass:

```bash
npm run typecheck       # tsc --noEmit
npm run python:check    # compile Python sidecar + scripts
npm run test:parity     # 18-row README parity check
```

The aggregate target is:

```bash
npm run test:gates      # runs all three of the above
```

For changes touching the engine or new subsystems, also run any relevant verifier tests under `tests/smoke-*.test.ts` once the smoke-test pipeline lands.

CI should invoke `npm run test:gates` on every PR.

## Backwards compatibility

The public API surface is intentionally narrow. Breaking changes to any of the following require a major version bump and a CHANGELOG entry calling them out explicitly:

- The `EvolutionOptions` shape (`src/types.ts`)
- The `EvolutionRunResult` shape
- The `ToolSummaryDetails` shape (this is the tool's machine-readable return value; downstream automation depends on it)
- The exported function signatures of `runEvolution`, `resolveArtifactTarget`, `loadGoldenDataset`, `scanForSecrets`
- The run directory layout documented in [docs/output-layout.md](docs/output-layout.md)
- The constraint-pipeline pass / fail decision rules

Additive extensions (new optional fields, new union members, new exported helpers) are not breaking changes.

## Parallel-PRD dispatch pattern

Non-trivial work in this repo (the Phase 1 parity rollout, the runtime smoke-test pipeline) is decomposed into ≤5 parallel lanes under disjoint file ownership, dispatched as sub-agents in git worktrees, and merged sequentially under an acceptance gate. The pattern is captured as the `qs-parallelprd` skill (folder name: `parallel-pipeline-dispatch`). See [docs/ownership-map.md](docs/ownership-map.md) for the version of the pattern that landed Phase 1 parity.

The pattern is appropriate when:

- The task spans 4+ files or 3+ subsystems
- A natural pre-phase establishes shared contracts (types, fixtures, run artifacts)
- A natural acceptance phase integrates and verifies
- File ownership can be carved disjoint across parallel lanes

The pattern is not appropriate for single-file edits, sequential refactors that share one orchestrator file, or exploratory work where the deliverable shape is not yet known.

### Lane archetypes

| Archetype | Mandate | May touch |
|---|---|---|
| `[pre-phase]` | Establish shared contracts. Solo. | One or two foundational files |
| `[parallel-builder]` | Build a new module that other lanes will import | Its own new file(s) |
| `[parallel-verifier]` | Verify outputs from a previous lane. Read-only. | Its own test file plus a findings fragment |
| `[remediation]` | Apply fixes flagged by verifiers | Specific src/ files named in upstream findings |
| `[acceptance-gate]` | Integrate, verify, commit final evidence | The report file, post-merge cleanup, package wiring |

### Authoring a lane prompt

Every lane `.md` file in `.prd/` follows the 8-section template documented in the skill's `SKILL.md`. Skipping sections produces lanes that are not actually self-contained and forces sub-agents to re-derive context.

### Authoring an orchestration overview

Every dispatch needs one `.prd/<feature>-orchestration.md` file containing the dispatch table, file-ownership matrix, execution sequence, and acceptance criteria checklist. The matrix is critical — confirm an empty intersection across parallel lanes before dispatching.

## Soft spots

When you find an issue that you choose not to fix immediately, document it inline with:

```typescript
// SOFT-SPOT(<short-id>): <one-sentence WHY> see <evidence-doc-path>
```

The id must appear in a corresponding evidence document (currently `tests/smoke-test-report.md` once that pipeline lands). Soft spots are not bug reports; they are intentional accepted limitations with a paper trail.

Current soft spots flagged for smoke-test remediation:

- `iter-fallback` — iteration acceptance promotes the highest-validation candidate even when zero iterations strictly accept; `src/engine.ts`
- `loadBestAncestor-fuzzy` — lineage ancestor lookup falls back to runId substring match when `artifactContent` is not supplied; `src/lineage.ts`
- `executor-logs` — pi-executor log writes specified but not yet observed end-to-end in production; `src/pi-executor.ts`
- `coherence-default` — tiered-gate coherence tier passes by skip-reason when no `coherenceCheck` callback is wired; `src/tiered-gate.ts`

## Commit conventions

Commits use Conventional Commits:

- `feat(<scope>):` new capability
- `fix(<scope>):` bug fix
- `docs(<scope>):` documentation only
- `test(<scope>):` test additions or changes
- `chore(<scope>):` tooling, dependencies, build
- `refactor(<scope>):` non-functional restructuring
- `merge(<scope>):` worktree integration (parallel-PRD lanes)

Scopes used so far: `engine`, `executor`, `lineage`, `tiered-gate`, `types`, `smoke`, `parity`, `framing`, `merge`.

Subject line ≤ 72 chars. Body wrapped at ≤ 80 chars when present.

## Pull request guidelines

- One concern per PR. Lanes from a parallel-PRD dispatch are typically squash-merged into a single PR with the orchestration overview as the description.
- The PR description should include: summary of changes, links to relevant `.prd/` files if a dispatch ran, the `npm run test:gates` evidence, and any soft-spot dispositions.
- A reviewer should be able to verify the PR by running `npm run test:gates` locally.

## Release process

1. Update `CHANGELOG.md` with the new version entry. Move items from `Unreleased` to the new section.
2. Update `package.json` version.
3. Verify `npm run test:gates` passes.
4. Commit: `chore(release): vX.Y.Z`.
5. Tag: `git tag vX.Y.Z`.
6. Push and publish via the existing GitHub Actions workflow.

The version policy is semver. Phase 2 parity work warrants a `0.3.0` minor bump because all additions are additive but the framing change (TypeScript as primary, Python as acceleration) is a meaningful product-level shift.

# Architecture

This document describes the runtime architecture of the TypeScript engine. It is intended for contributors and integrators. For installation and usage, see the [README](../README.md). For the full configuration surface, see [docs/configuration.md](configuration.md). For the run-directory format, see [docs/output-layout.md](output-layout.md).

## Overview

The package is a pi-coding-agent extension that ships as a single TypeScript engine plus an optional Python sidecar. The TypeScript engine implements the Hermes Phase 1 workflow end-to-end. Python/DSPy is available as an opt-in acceleration mode.

A single evolution run takes one local markdown artifact (a `SKILL.md`, prompt template, `AGENTS.md`, or `SYSTEM.md`), produces a held-out evaluation dataset, runs a reflective iterative search for a better version, gates each candidate against constraints and regression tests, and writes a reviewable report under `.pi/hermes-self-evolution/runs/<timestamp>-<artifact>/`. The original file is never overwritten.

## Module map

| Module | Responsibility | Key exports |
|---|---|---|
| `src/index.ts` | pi extension entrypoint; registers `/evolve` command and `self_evolve_artifact` tool | default export `hermesSelfEvolutionExtension(pi)` |
| `src/engine.ts` | orchestrator: artifact resolution, dataset generation, iterative loop, judge, constraint pipeline, drift, PR automation, lineage wiring | `runEvolution`, `resolveArtifactTarget`, `loadGoldenDataset`, `scanForSecrets`, `buildToolSummary`, `toToolSummaryDetails` |
| `src/pi-executor.ts` | spawns `pi -p --no-session` against a temp skill slot, captures stdout/stderr/exit/duration | `executeCandidateInPi` |
| `src/tiered-gate.ts` | sequential typecheck → cohort → coherence gate with distinct reason codes | `runTieredGate`, `TieredGateOptions` |
| `src/constraints-structure.ts` | SKILL.md frontmatter / name / description validator | `checkSkillStructure`, `buildSkillStructureReport` |
| `src/lineage.ts` | parent→child run linkage in `lineage.jsonl`; Pareto-best ancestor lookup | `appendLineageEntry`, `loadLineage`, `loadBestAncestor` |
| `src/session-history.ts` | mines pi session JSONL for evaluation snippets | `mineSessionSnippets`, `buildKeywordSet` |
| `src/python-backend.ts` | detects and invokes the optional Python DSPy/GEPA sidecar | `detectPythonBackend`, `runPythonBackend` |
| `src/types.ts` | all shared types and discriminated unions | 32 exported types |

## End-to-end run pipeline

The full lifecycle of one `runEvolution(options)` call:

```
                  ┌──────────────────────────────────────────────┐
                  │  resolveArtifactTarget(targetPath, cwd)      │
                  │  - read file                                 │
                  │  - split frontmatter / body                  │
                  │  - extract placeholders, top heading         │
                  │  - compute originalBytes, maxBytes (×1.2)    │
                  └──────────────┬───────────────────────────────┘
                                 │
                  ┌──────────────▼───────────────────────────────┐
                  │  Lineage lookup                              │
                  │  - loadBestAncestor(cwd, target.path,        │
                  │    target.fullText) returns parent or null   │
                  └──────────────┬───────────────────────────────┘
                                 │
                  ┌──────────────▼───────────────────────────────┐
                  │  Dataset                                     │
                  │  - loadGoldenDataset if goldenTaskId set     │
                  │  - else generateDataset (one LLM call)       │
                  │  - splitExamples 50/20/30 train/val/holdout  │
                  │  - stripSecretsFromExamples                  │
                  │  - saveGoldenDataset if persistGolden        │
                  └──────────────┬───────────────────────────────┘
                                 │
                  ┌──────────────▼───────────────────────────────┐
                  │  Baseline evaluation                         │
                  │  - evaluateArtifact on train (traces)        │
                  │  - evaluateArtifact on validation            │
                  │  - evaluateArtifact on holdout               │
                  └──────────────┬───────────────────────────────┘
                                 │
                  ┌──────────────▼───────────────────────────────┐
                  │  Iterative reflective loop (Phase 2)         │
                  │  for iter in 1..candidateCount:              │
                  │    1. buildReflectionPrompt(priorTraces,     │
                  │       priorFeedback, weaknesses)             │
                  │    2. generateOneCandidateDraft via LLM      │
                  │    3. validateConstraints (7 checks)         │
                  │    4. evaluateArtifact on validation         │
                  │       (with useRealExecutor=true ⇒ pi spawn) │
                  │    5. computeSemanticDrift                   │
                  │    6. runTestCommand if testCommand set      │
                  │    7. runTieredGate (typecheck/cohort/       │
                  │       coherence)                             │
                  │    8. compute scoreDelta vs prior            │
                  │    9. accept iff constraintsPass &&          │
                  │       scoreDelta > 0 && testPassed != false  │
                  │   10. persist iterations/<n>.json            │
                  └──────────────┬───────────────────────────────┘
                                 │
                  ┌──────────────▼───────────────────────────────┐
                  │  Best-candidate selection                    │
                  │  - sort accepted candidates by validation    │
                  │    composite                                 │
                  │  - confirm winner on holdout                 │
                  │  - record holdoutEvaluation, executor logs   │
                  └──────────────┬───────────────────────────────┘
                                 │
                  ┌──────────────▼───────────────────────────────┐
                  │  Lineage write                               │
                  │  - appendLineageEntry with hashes, score,    │
                  │    parent reference                          │
                  └──────────────┬───────────────────────────────┘
                                 │
                  ┌──────────────▼───────────────────────────────┐
                  │  Optional PR automation                      │
                  │  - if createPR && improvement > 0:           │
                  │    git branch + commit + gh pr create        │
                  └──────────────┬───────────────────────────────┘
                                 │
                  ┌──────────────▼───────────────────────────────┐
                  │  Output                                      │
                  │  - report.md, manifest.json, dataset.json    │
                  │  - candidates/<name>.{md,json}               │
                  │  - traces/all-traces.json, failure-traces    │
                  │  - iterations/<n>.json (each iteration)      │
                  │  - executor/<iter>/<ex>/{stdout,stderr,meta} │
                  └──────────────────────────────────────────────┘
```

See [docs/output-layout.md](output-layout.md) for the exact run-directory contents.

## Iterative reflective loop

The single most important architectural change in Phase 2. The pre-Phase-2 engine asked the LLM for `N` candidate revisions in a single call, then evaluated each. This was PromptBreeder-shaped, not GEPA-shaped: there was no round-to-round feedback signal.

The Phase 2 loop replaces this with sequential iterations. Each iteration receives the prior iteration's failure traces and judge feedback strings as a `ReflectionPrompt` and produces exactly one candidate. The mutation prompt explicitly asks the LLM to address specific weaknesses observed in the previous round.

```
Iteration N:
  ReflectionPrompt {
    priorTraces:        ExecutionTrace[]   ← from iteration N-1
    priorJudgeFeedback: string[]           ← from iteration N-1's judge
    objective:          string             ← user-supplied
    weaknessSummary:    string             ← derived from prior eval
  }
  ↓
  generateOneCandidateDraft(prompt) → CandidateDraft
  ↓
  validate / evaluate / drift / test / gate
  ↓
  IterationRecord persisted to iterations/<N>.json
```

The first iteration's "prior traces" are the baseline traces. Subsequent iterations build on the most recent iteration.

Acceptance criterion (in `runTypeScriptEvolution`):

```
accepted = constraintsPass
        && scoreDelta > 0
        && (testPassed === undefined ? true : testPassed)
```

If zero iterations strictly accept, a fallback promotes the iteration with the highest validation composite to keep the run completing. This fallback is flagged as `SOFT-SPOT(iter-fallback)` and is on the runtime-smoke-test agenda.

## Pi-native executor

Phase 1's judge predicted how an agent following the artifact would respond. Phase 2 actually runs an agent and judges the real output.

`executeCandidateInPi(options)`:

1. Creates a temp directory under `.pi/hermes-self-evolution/.exec-tmp/<uuid>/skills/<artifactName>/`.
2. Writes the candidate's full text (frontmatter + body) as `SKILL.md` in that slot.
3. Spawns `pi -p --no-session --no-extensions --no-themes` with the task input piped to stdin and the temp skills directory included on PATH or via env.
4. Captures stdout, stderr, exit code, and duration into an `ExecutionObservation`.
5. Cleans up the temp slot.

The captured stdout becomes the response the judge scores. Per-example logs land under `executor/<iter>/<exampleIndex>/{stdout.log,stderr.log,meta.json}`.

`useRealExecutor` defaults to `false` for baseline and holdout-confirmation calls (cheap LLM-only judging) and `true` inside the iterative loop (real-execution-based judging). This keeps overall cost predictable.

## Tiered regression gate

`runTieredGate(options)` runs three sequential tiers. Each tier produces a `TieredGateResult` with `tier`, `passed`, `reasonCode`, `detail`, and `durationMs`. On first failure the array is returned with `passed: false` on the failing tier and subsequent tiers skipped.

| Tier | Reason codes | Default behavior |
|---|---|---|
| `typecheck` | `ok`, `typecheck_failed` | runs `npx tsc --noEmit -p <config>` |
| `cohort` | `ok`, `skipped_no_cohort`, `cohort_regression` | requires `cohortExamples`, `baselineScore`, and `judgeFunc` |
| `coherence` | `ok`, `skipped_no_check`, `coherence_failed` | requires a `coherenceCheck` callback; defaults to skip |

The coherence tier is intentionally open: there is no curated cross-skill cohort in the current package. Callers may pass a `coherenceCheck` callback (embedding distance against a reference set, semantic-drift average across a curated cohort, etc.). Without one, the tier passes with `reasonCode: "skipped_no_check"`.

## Constraint pipeline

`validateConstraints(target, candidateBody, candidateFullText, config)` runs the following checks in order. The first failure does not abort; all checks run and their results aggregate.

| Constraint | Failure condition | Severity |
|---|---|---|
| `non_empty` | candidate body is empty after trim | hard reject |
| `size_limit` | size exceeds `maxSizeBytes` (default: `originalBytes × 1.2 + 400`) | hard reject |
| `growth_limit` | growth ratio exceeds `maxGrowthRatio` (default 0.20) | hard reject |
| `placeholder_preservation` | any original `{{placeholder}}` missing | hard reject |
| `top_heading_preservation` | top-level markdown heading removed when one existed | warning |
| `frontmatter_preservation` | YAML frontmatter modified when one existed | hard reject |
| `semantic_drift` | LLM-judged drift score exceeds `maxDriftScore` (default 0.40) | hard reject |
| `skill_structure` | `name:` and `description:` not both in first 500 chars (skills only) | hard reject |

A candidate that fails any hard-reject constraint is discarded from the iteration. Soft warnings are recorded on the candidate but do not block promotion.

## Cross-run lineage

Each completed run appends one `LineageEntry` to `.pi/hermes-self-evolution/lineage.jsonl`:

```
{
  runId:             "2026-05-11_20-15-33-skill-foo",
  parentRunId:       "2026-05-11_19-02-11-skill-foo",   // optional
  artifactHash:      "<sha256(artifactText).slice(0,16)>",
  parentArtifactHash:"<...>",                            // optional
  score:             0.742,
  mutationRationale: "<best-candidate's rationale>",
  createdAt:         "<ISO timestamp>"
}
```

At run start, `loadBestAncestor(cwd, artifactPath, artifactContent?)` is consulted. When `artifactContent` is provided, the lookup uses an exact content-hash match. Without `artifactContent`, the lookup falls back to a runId-substring match against the artifact path's basename. Both modes return the highest-score matching entry or `null`.

The lineage memory enables future runs to seed against a known-good ancestor and to skip mutations that previously regressed.

## Backend selection

```
runEvolution(options.backend = "auto")
   │
   ├── detectPythonBackend()
   │      ├── locates python (PI_HERMES_EVOLVE_PYTHON || python3 || python)
   │      └── runs `--doctor` to check for DSPy / GEPA / MIPROv2
   │
   ├── if Python + DSPy available: runPythonBackend(...)
   │      └── spawn python with piped JSON request
   │
   └── else: runTypeScriptEvolution(...)
```

| Option | Resolution |
|---|---|
| `backend: "auto"` | Python if DSPy installed, else TypeScript |
| `backend: "typescript"` | TypeScript engine only |
| `backend: "python"` | Python sidecar; throws if unavailable |

The TypeScript engine is feature-complete for Phase 1 on its own. The Python sidecar is an optional acceleration layer.

## Determinism and reproducibility

The engine uses non-deterministic randomness in `splitExamples` (a Fisher-Yates shuffle using `Math.random`) and `crypto.randomUUID` for trace IDs. For deterministic smoke tests and golden-dataset reproduction, callers must either (a) supply a `goldenTaskId` that was persisted on an earlier run (the persisted splits are reused verbatim), or (b) seed `Math.random` before invoking `runEvolution`. There is no built-in seed parameter currently.

## Error handling and graceful degradation

The engine treats internal modules optimistically and external surfaces defensively:

- LLM call failures (`runPiTextTask`) raise. The caller may retry at the iteration level; the run does not auto-recover mid-iteration.
- Drift detection failures default to `{score: 0.2, feedback: "Drift detection failed."}` so a single LLM failure does not block a run.
- Tiered-gate calls are wrapped in `try/catch` so a missing module does not crash the engine. This is intentional: the gate is auxiliary, not load-bearing.
- PR automation failures restore the original file and return `undefined` so the user does not lose state.
- Secret patterns matching dataset content trigger redaction, not run abort.

## Extension points

The architecture is designed to make Phase 2 / 3 / 4 of Hermes additive:

| Future capability | Extension point |
|---|---|
| Tool-description evolution (Hermes Phase 2) | extend `ArtifactType` union; add tool-description detection in `detectArtifactType`; reuse the rest of the pipeline |
| System-prompt-segment evolution (Phase 3) | same as above with a `system_prompt_segment` artifact type |
| Code-organism evolution (Phase 4) | new artifact type plus a non-text constraint pipeline; the iterative loop and lineage are reusable |
| Curated cross-skill coherence cohort | implement a `coherenceCheck` callback in `src/tiered-gate.ts` consumers; no engine changes needed |
| Deterministic seeded runs | accept a `seed` option in `EvolutionOptions`; thread it into `splitExamples` |

# Current State Map — pk-pi-hermes-evolve

A pi-coding-agent TypeScript extension at `C:\dev\Desktop-Projects\pi-hermes-self-evolution`. Read this so you don't have to re-explore the tree.

## Top-level layout

```
.
├── .pi/                         # Pi project state
├── .github/                     # CI metadata
├── .omc/                        # OMC orchestration state (ignore)
├── benchmarks/sokoban/          # Sokoban benchmark assets
├── python_backend/              # Optional DSPy/GEPA sidecar (out of scope unless framing lane)
│   ├── pyproject.toml
│   ├── run_backend.py
│   └── pk_pi_hermes_evolve/backend.py
├── scripts/
│   ├── ralph_otel.py            # OTel-traced Ralph loop
│   ├── sokoban_benchmark.py     # Sokoban runner
│   └── tasks/                   # Task JSON for the ralph loop
├── src/                         # TypeScript engine
│   ├── engine.ts                # Main orchestrator (596 lines)
│   ├── index.ts                 # Pi extension entrypoint
│   ├── types.ts                 # Shared types
│   ├── python-backend.ts        # Python sidecar detection/runner
│   └── session-history.ts       # Pi session JSONL miner
├── tsconfig.json
├── package.json
├── README.md
└── CHANGELOG.md
```

## src/engine.ts — the orchestrator (596 lines)

Public exports the engine surfaces (used elsewhere):
- `runEvolution(options) → Promise<EvolutionSummaryDetails>` — entrypoint called from `src/index.ts`
- `runTypeScriptEvolution(options) → Promise<EvolutionRunResult>` — the all-TS path
- `resolveArtifactTarget(inputPath, cwd) → Promise<ArtifactTarget>`
- `loadGoldenDataset(cwd, goldenTaskId)` / `saveGoldenDataset(...)`
- `scanForSecrets(text) → SecretScanResult`
- `buildToolSummary(result)`, `toToolSummaryDetails(result)`
- Re-exports: `EvolutionSummaryDetails`, `ToolSummaryDetails`

Internal pipeline of `runTypeScriptEvolution` (line 402):
1. `resolveArtifactTarget` — read file, split frontmatter, detect type, extract placeholders, compute `originalBytes` and `maxBytes`
2. `buildConstraintConfig` — defaults: maxSize=originalBytes×1.2+400, maxGrowthRatio=0.2, testTimeoutMs=60000, checkSemanticDrift=true, maxDriftScore=0.4
3. **Dataset** — `loadGoldenDataset` if `goldenTaskId`, else `generateDataset` (one `runPiTextTask` call, returns `EvalExample[]`); `splitExamples` (50/20/30 train/val/holdout); `saveGoldenDataset` if `persistGolden`
4. **Baseline** — `evaluateArtifact` three times (train, holdout, validation), each example judged via one `runPiTextTask` call returning JSON `{responsePreview, correctness, procedureFollowing, conciseness, feedback, confidence}`. Composite = `0.5×correctness + 0.3×procedure + 0.2×conciseness`. Traces recorded per example.
5. **Candidates** — `generateCandidates` makes ONE `runPiTextTask` call asking for N variants in a JSON array. **This is the single-shot fan-out.** Failure traces are included in the prompt but only once.
6. **Per-candidate**: validate constraints → evaluate on validation → compute drift → run test command if provided → record
7. **Best candidate** — sort by validation composite, re-evaluate winner on holdout (`bestHoldoutEvaluation`)
8. **PR** — if `createPR && improvement > 0`, `createGitBranchWithCandidate` writes the candidate to disk, commits to a new branch `evolve/<slug>-<ts>`, optionally `gh pr create`, then restores the original file
9. **Output** — writes `original.md`, `best-candidate.md`, `dataset.json`, `manifest.json`, `report.md`, `candidates/*.md`+`.json`, `traces/all-traces.json`, `traces/failure-traces.json` under `.pi/hermes-self-evolution/runs/<ts>-<artifact>/`

Key idiom: dense single-line statements with `;` separators (see engine.ts:130, 199, 226). Match this style if editing engine.ts.

## src/types.ts — frozen contract surface

Existing public types (must remain backwards-compatible — additive extension only):
- `ArtifactType = "skill" | "prompt" | "instructions"`
- `EvalSource = "synthetic" | "session" | "mixed"`
- `Difficulty = "easy" | "medium" | "hard"`
- `ArtifactTarget` — `{path, name, type, fullText, body, frontmatter?, originalBytes, placeholders, topHeading?}`
- `SessionSnippet`, `EvalExample`, `JudgeResult`, `ExampleEvaluation`, `AggregateScore`, `ArtifactEvaluation`
- `CandidateDraft` — `{name, rationale, candidateBody}`
- `ExecutionTrace` — `{traceId, artifactText, taskInput, expectedBehavior, rawOutput, responsePreview, scores, feedback, isFailure, timestamp}`
- `ConstraintName = "non_empty" | "size_limit" | "growth_limit" | "placeholder_preservation" | "top_heading_preservation" | "frontmatter_preservation" | "semantic_drift"` — **extend this union** with `"skill_structure"`
- `ConstraintResult`, `ConstraintConfig`, `PRAutomationResult`, `SecretScanResult`, `GoldenDatasetManifest`
- `CandidateRecord` — `{...CandidateDraft, candidateFullText, evaluation, holdoutEvaluation?, executionTraces, constraints, warnings, semanticDriftScore?, testPassed?}`
- `EvolutionOptions`, `EvolutionPaths`, `GoldenDataset`, `EvolutionRunResult`
- `EvolutionSummaryDetails = ToolSummaryDetails`
- `ToolSummaryDetails` — the flat summary shape returned to the pi tool

New types to add (Phase 2):
- `IterationRecord` — one entry per GEPA-shaped iteration: `{iteration, parentCandidate?, mutationRationale, reflectionPrompt, candidate: CandidateDraft, evaluation: ArtifactEvaluation, traces: ExecutionTrace[], scoreDelta, accepted: boolean}`
- `ReflectionPrompt` — `{priorTraces: ExecutionTrace[], priorJudgeFeedback: string[], objective: string, weaknessSummary: string}`
- `ExecutionObservation` — `{stdout, stderr, exitCode, durationMs, capturedFiles?: Record<string,string>}`
- `TieredGateResult` — `{tier: "typecheck"|"cohort"|"coherence", passed: boolean, reasonCode: string, detail: string, durationMs}`
- `LineageEntry` — `{runId, parentRunId?, artifactHash, parentArtifactHash?, score, mutationRationale, createdAt}`
- `SkillStructureReport` — `{hasFrontmatter: boolean, hasName: boolean, hasDescription: boolean, nameInFirst500: boolean, descriptionInFirst500: boolean, errors: string[]}`
- `BackendMode = "typescript" | "python-accelerate"` (replace the current `"auto" | "typescript" | "python"` discriminator in a backwards-compat way: keep the option type but document `"auto"` and `"python"` as legacy aliases that resolve to `"python-accelerate"`)

## src/index.ts — pi extension entrypoint

- Registers a markdown renderer for `hermes-self-evolution` messages
- Registers `/evolve` command (interactive picker, `last`, explicit path)
- Registers `self_evolve_artifact` tool with TypeBox params: `targetPath`, optional `objective`, `evalSource`, `backend`, `candidateCount`, `maxExamples`, `sessionQuery`, `model`, `goldenTaskId`, `testCommand`, `testTimeout`, `createPR`, `persistGolden`
- Both paths call `runEvolution` from engine.ts and append a `LAST_RUN_ENTRY` to the session

Do not modify unless adding new tool params for tiered gate, executor mode, etc. — and even then, keep them optional with safe defaults.

## src/python-backend.ts (109 lines, dense)

- `detectPythonBackend()` — finds python via `PI_HERMES_EVOLVE_PYTHON`, `python3`, `python`; runs `--doctor` to check DSPy availability
- `runPythonBackend(python, options)` — spawns python and pipes JSON request

Framing lane will edit this to change "primary" vs "accelerate" semantics.

## src/session-history.ts

- `mineSessionSnippets({cwd, targetName, objective, artifactBody, sessionQuery, maxSnippets})` reads `.pi/sessions/*.jsonl` and returns `SessionSnippet[]`.

Do not modify.

## .pi/hermes-self-evolution/ run output layout

Per run:
```
.pi/hermes-self-evolution/runs/<ts>-<artifact-slug>/
├── original.md
├── best-candidate.md
├── report.md
├── manifest.json
├── dataset.json
├── candidates/
│   ├── <name>.md
│   └── <name>.json
└── traces/
    ├── all-traces.json
    └── failure-traces.json   (only if failures > 0)
```

Phase 2 will add per run:
```
├── iterations/
│   ├── 1.json                  # IterationRecord
│   ├── 2.json
│   └── ...
├── executor/
│   ├── 1/stdout.log            # captured pi -p stdout per example
│   ├── 1/stderr.log
│   └── ...
├── gate.json                   # TieredGateResult[]
└── lineage.json                # this run's LineageEntry (also appended to global lineage.jsonl)
```

Global lineage memory:
```
.pi/hermes-self-evolution/lineage.jsonl    # one LineageEntry per line, all runs
```

## package.json scripts

- `npm run typecheck` — `tsc --noEmit -p tsconfig.json` (must pass after every lane)
- `npm run python:check` — `python -m py_compile` over backend + scripts (must pass)
- `npm run ralph:parity` — runs the Ralph loop (don't run during your work)
- `npm run benchmark:sokoban:help` — prints CLI help only

## Git state

- Current branch: `main`
- Last commits: 3213440 (update), cb48524 (parity gaps closed), a94a5d6 (0.2.1 + python repair), 210151b (hybrid dspy backend + release flow)
- Working tree: clean as of session start

## What the engine **doesn't** do today (gaps to close)

1. No iterative loop — one LLM call produces all candidates
2. No real execution — judge predicts how an agent would respond
3. No tiered gate — single `testCommand` only
4. No skill-structure constraint
5. No lineage.jsonl
6. README + types frame Python as "real", TS as "proxy"

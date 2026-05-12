# Run Output Layout

Every evolution run writes a self-contained directory under `.pi/hermes-self-evolution/runs/<timestamp>-<artifact-slug>/`. The format below is the contract verifiers in `tests/` assert against.

For architectural context see [docs/architecture.md](architecture.md). For the parameters that influence what gets written, see [docs/configuration.md](configuration.md).

## Top-level project layout

```
.pi/hermes-self-evolution/
├── runs/
│   ├── 2026-05-11_20-15-33-skill-foo/      ← one directory per run
│   └── 2026-05-11_19-02-11-skill-foo/
├── golden/                                  ← persisted golden datasets
│   └── my-skill-v1/
│       ├── train.jsonl
│       ├── validation.jsonl
│       ├── holdout.jsonl
│       └── manifest.json
├── lineage.jsonl                            ← append-only run-to-run linkage
└── .exec-tmp/                               ← transient executor scratch (auto-cleaned)
```

## Per-run directory format

A complete run directory:

```
<timestamp>-<artifact-slug>/
├── original.md                    ← byte-exact copy of the input artifact
├── best-candidate.md              ← the chosen winner's full text
├── report.md                      ← human-readable summary
├── manifest.json                  ← machine-readable summary
├── dataset.json                   ← train/val/holdout splits + session snippets used
├── candidates/
│   ├── <name>.md                  ← one per evaluated candidate
│   └── <name>.json                ← scores, rationale, constraints
├── iterations/
│   ├── 1.json                     ← one per iteration (Phase 2)
│   ├── 2.json
│   └── ...
├── executor/                      ← only when useRealExecutor=true (default in iter loop)
│   ├── 1/
│   │   ├── 0/
│   │   │   ├── stdout.log
│   │   │   ├── stderr.log
│   │   │   └── meta.json
│   │   ├── 1/...
│   │   └── ...
│   └── 2/...
└── traces/
    ├── all-traces.json            ← every ExecutionTrace from baseline + candidates
    └── failure-traces.json        ← subset where composite < 0.5 (omitted if empty)
```

## File specifications

### `original.md`

A byte-exact copy of the input artifact at the moment the run started. Preserved so reviewers can diff against `best-candidate.md`.

### `best-candidate.md`

The full text (frontmatter + body) of the candidate selected as the winner. This file is what `createPR: true` would commit to a feature branch. The original input file on disk is never overwritten.

### `report.md`

A human-readable summary with the following sections:

- Run metadata: target, type, objective, source, model, run dir
- Split sizes and selection vs confirmation scores
- Baseline weaknesses (top 3 lowest-scoring tasks with their rubric and feedback)
- Execution traces summary (total + failure count + top 5 failure summaries)
- Candidates table: name, validation, holdout, correctness, procedure, conciseness, constraints, drift
- Best candidate: name, rationale, constraint results, drift
- Selected winner confirmation on holdout split
- Holdout weaknesses
- PR info (when `createPR` was used)
- Files index pointing at the rest of the run dir

### `manifest.json`

The machine-readable counterpart to `report.md`. Schema:

```jsonc
{
  "targetPath":          "/abs/path/to/SKILL.md",
  "objective":           "string",
  "evalSource":          "synthetic" | "session" | "mixed",
  "modelLabel":          "anthropic/claude-sonnet-4-6",
  "selectionSplit":      "validation",
  "confirmationSplit":   "holdout",
  "maxBytes":            12345,
  "splits": {
    "train":      5,
    "validation": 2,
    "holdout":    3
  },
  "goldenTaskId":        "my-skill-v1" | null,
  "usedPersistedGolden": true | false,
  "baselineValidation":  { "composite": 0.62, "correctness": 0.7, ... },
  "baselineHoldout":     { ... },
  "bestCandidate": {
    "name":                "candidate-iter-2",
    "rationale":           "...",
    "validationScore":     { ... },
    "holdoutScore":        { ... },
    "warnings":            [...],
    "constraints":         [ConstraintResult, ...],
    "semanticDriftScore":  0.18,
    "testPassed":          true | false | null
  },
  "candidates":          [ /* same shape, all accepted candidates */ ],
  "traces":              { "baselineCount": 8 },
  "prBranch":            "evolve/skill-foo-2026-05-11_20-15-33" | null,
  "createdAt":           "2026-05-11T20:15:33.000Z"
}
```

### `dataset.json`

```jsonc
{
  "train":      [ EvalExample, ... ],
  "validation": [ EvalExample, ... ],
  "holdout":    [ EvalExample, ... ],
  "golden":     { "id": "...", "description": "...", "exampleCount": 5 } | null,
  "sessionSnippets": [ SessionSnippet, ... ]
}
```

`EvalExample` shape (from `src/types.ts`):

```typescript
{
  taskInput:        string,    // ≤ 1800 chars
  expectedBehavior: string,    // ≤ 1800 chars (rubric)
  difficulty:       "easy" | "medium" | "hard",
  category:         string,
  source:           "synthetic" | "session"
}
```

### `candidates/<name>.md` and `<name>.json`

One pair per accepted candidate. The `.md` is the candidate's full text (frontmatter + body, suitable for direct use as an artifact replacement). The `.json` is its full evaluation record:

```jsonc
{
  "rationale":          "string",
  "warnings":           [...],
  "evaluation":         ArtifactEvaluation,
  "holdoutEvaluation":  ArtifactEvaluation | null,
  "constraints":        [ConstraintResult, ...],
  "semanticDriftScore": 0.18,
  "testPassed":         true | false | null
}
```

### `iterations/<n>.json`

One per iteration of the reflective loop, regardless of whether the iteration's candidate was ultimately accepted. Shape:

```jsonc
{
  "iteration":         1,
  "parentCandidate":   "candidate-iter-0" | null,
  "mutationRationale": "string",
  "reflectionPrompt": {
    "priorTraces":        [ExecutionTrace, ...],
    "priorJudgeFeedback": [string, ...],
    "objective":          "string",
    "weaknessSummary":    "string"
  },
  "candidate":         CandidateDraft,
  "evaluation":        ArtifactEvaluation,
  "traces":            [ExecutionTrace, ...],
  "scoreDelta":        0.034,
  "accepted":          true | false,
  // additional fields persisted alongside the IterationRecord:
  "candidateFullText": "string",
  "constraints":       [ConstraintResult, ...],
  "warnings":          [...],
  "semanticDriftScore": 0.18,
  "testPassed":         true | false | null,
  "gateResults":        [TieredGateResult, ...] | undefined
}
```

The first iteration's `parentCandidate` is `null` and its `reflectionPrompt.priorTraces` are the baseline traces. Subsequent iterations reference the most recent accepted candidate.

### `executor/<iter>/<exampleIndex>/`

Written by `executeCandidateInPi` when `useRealExecutor` is true (default inside the iterative loop, false in baseline and holdout-confirmation passes).

| File | Contents |
|---|---|
| `stdout.log` | Raw stdout from the spawned `pi -p` process |
| `stderr.log` | Raw stderr from the same |
| `meta.json` | `ExecutionObservation`: `{stdout, stderr, exitCode, durationMs}` (`stdout` and `stderr` here are truncated previews; the full logs are the `.log` files) |

`exampleIndex` is the 0-based index into the validation split for that iteration.

### `traces/all-traces.json`

Every `ExecutionTrace` produced during the run, tagged with its phase:

```jsonc
[
  {
    "phase":            "baseline",
    "traceId":          "abc123",
    "artifactText":     "<truncated to 2000 chars>",
    "taskInput":        "...",
    "expectedBehavior": "...",
    "rawOutput":        "<truncated to 2000 chars>",
    "responsePreview":  "<truncated to 500 chars>",
    "scores": {
      "correctness":        0.7,
      "procedureFollowing": 0.8,
      "conciseness":        0.6,
      "composite":          0.71
    },
    "feedback":  "string",
    "isFailure": false,
    "timestamp": "2026-05-11T20:15:33.000Z"
  },
  // ...
  { "phase": "candidate/candidate-iter-1", ... },
  // ...
]
```

### `traces/failure-traces.json`

Subset of `all-traces.json` where `composite < 0.5`. The file is omitted entirely when there are no failures.

## Global lineage log

`.pi/hermes-self-evolution/lineage.jsonl` is append-only JSONL. One line per completed run:

```jsonc
{
  "runId":              "2026-05-11_20-15-33-skill-foo",
  "parentRunId":        "2026-05-11_19-02-11-skill-foo" | undefined,
  "artifactHash":       "abc123def456...",
  "parentArtifactHash": "..." | undefined,
  "score":              0.742,
  "mutationRationale":  "string",
  "createdAt":          "2026-05-11T20:15:33.000Z"
}
```

The hashes are SHA-256 of the full artifact text, sliced to the first 16 hex chars. `score` is the best candidate's holdout composite.

## Golden dataset format

When `goldenTaskId` is set and `persistGolden` is not disabled, the validation split is preserved at:

```
.pi/hermes-self-evolution/golden/<goldenTaskId>/
├── train.jsonl          ← one EvalExample per line
├── validation.jsonl
├── holdout.jsonl
└── manifest.json
```

`manifest.json` shape:

```jsonc
{
  "id":              "my-skill-v1",
  "artifactPath":    "/abs/path/to/SKILL.md",
  "artifactName":    "skill-foo",
  "exampleCount":    10,
  "trainCount":      5,
  "validationCount": 2,
  "holdoutCount":    3,
  "createdAt":       "2026-05-11T18:00:00.000Z",
  "lastUsedAt":      "2026-05-11T20:15:33.000Z"
}
```

`lastUsedAt` is bumped every time the golden is loaded.

## Transient files

`.pi/hermes-self-evolution/.exec-tmp/<uuid>/` is created by `executeCandidateInPi` to host the candidate's `SKILL.md` while pi is spawned. The directory is removed after the executor returns. If you see persistent `.exec-tmp/` content, a previous run was interrupted; it is safe to delete the orphans.

## Diffing two runs

Useful one-liners:

```bash
diff -u .pi/hermes-self-evolution/runs/<run1>/best-candidate.md \
        .pi/hermes-self-evolution/runs/<run2>/best-candidate.md

jq '.bestCandidate.holdoutScore.composite' \
   .pi/hermes-self-evolution/runs/<run>/manifest.json

jq -s '.[].score' .pi/hermes-self-evolution/lineage.jsonl
```

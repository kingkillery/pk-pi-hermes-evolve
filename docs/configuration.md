# Configuration Reference

Complete reference for the `self_evolve_artifact` tool parameters, the `/evolve` command flags, environment variables, and the constraint configuration surface. For an end-to-end architectural overview, see [docs/architecture.md](architecture.md).

## Tool parameters (`self_evolve_artifact`)

These are the inputs accepted by the pi tool exposed in `src/index.ts`. They are also the fields of `EvolutionOptions` consumed by `runEvolution`.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `targetPath` | string | required | Path to a local markdown or instruction file to evolve. Relative paths resolve against the pi `cwd`. Leading `@` is stripped. |
| `objective` | string | "Improve trigger clarity, execution quality, and practical usefulness while preserving the artifact's intent." | What to improve. Free-text natural language. |
| `evalSource` | `"synthetic" \| "session" \| "mixed"` | `"mixed"` | Dataset composition. `synthetic` generates new examples from scratch. `session` mines pi session JSONL history. `mixed` uses both. |
| `backend` | `"auto" \| "typescript" \| "python"` | `"auto"` | Which engine runs the evolution. See [Backend selection](#backend-selection) below. |
| `candidateCount` | number | 3 (clamped 1-5) | Number of iterations in the reflective loop. Higher values increase cost roughly linearly. |
| `maxExamples` | number | 8 (clamped 4-12) | Total evaluation examples generated before splitting 50/20/30. |
| `sessionQuery` | string | undefined | Optional hint phrase for `mineSessionSnippets` to bias relevance scoring. |
| `model` | string | session-default | Model override in `provider/id` form (e.g., `anthropic/claude-sonnet-4-6`). |
| `goldenTaskId` | string | undefined | Reproducible task identifier. When set, the validation split is tagged as a golden dataset and reused on subsequent runs with the same id. |
| `testCommand` | string | undefined | Shell command run as a per-candidate validation gate (e.g., `"npm test"`). Non-zero exit rejects the candidate. |
| `testTimeout` | number (seconds) | 60 | Timeout for `testCommand` execution. |
| `createPR` | boolean | false | When true and the run produces an improvement, creates a `evolve/<slug>-<ts>` branch with the best candidate, commits it, and attempts `gh pr create`. Restores the original file on the source branch. |
| `persistGolden` | boolean | true when `goldenTaskId` is set | Persist the split as a golden dataset under `.pi/hermes-self-evolution/golden/<id>/`. |
| `seed` | number | undefined | Deterministic RNG seed for `splitExamples` and other RNG consumers. When unset, falls back to unseeded `Math.random`. |
| `cohortExamples` | `EvalExample[]` | undefined | Examples used by the tiered gate's cohort-regression tier. Required to activate `cohort_regression` reason code. |
| `cohortJudgeFunc` | `(examples: EvalExample[]) => Promise<{composite: number}>` | undefined | Judge callback for the cohort tier. Required when `cohortExamples` is supplied. |
| `coherenceCheck` | `() => Promise<{passed: boolean; detail: string}>` | undefined | Coherence check callback for the tiered gate's coherence tier. When unset, the tier returns `skipped_no_check`. |

When `cohortExamples` and `cohortJudgeFunc` are supplied together, the tiered gate runs a real cohort-regression check and can return a `cohort_regression` reason code. Supplying `coherenceCheck` enables the coherence tier and replaces the default `skipped_no_check` outcome. Supplying `seed` makes `splitExamples` deterministic. See `src/tiered-gate.ts` for the full gate semantics and tier ordering.

### Effort and cost guidance

| Setting | Cheap | Default | Thorough |
|---|---|---|---|
| `maxExamples` | 4 | 8 | 12 |
| `candidateCount` | 1 | 3 | 5 |
| `testCommand` | unset | `"npm run typecheck"` | full test suite |
| `useRealExecutor` (internal) | n/a | true (iterative) / false (baseline+holdout) | true everywhere |

A default run is roughly: 1 dataset call + 3 baseline-set judges × 8 examples + 3 iterations × (1 candidate gen + 1 drift + 8 judge calls) + 1 holdout-confirm × 8 examples = roughly 60-70 LLM calls. Scale linearly with `maxExamples` and `candidateCount`.

## `/evolve` command flags

The interactive command accepts the same arguments as the tool, with these shorthand forms:

| Invocation | Behavior |
|---|---|
| `/evolve` | Interactive picker; lists discovered artifacts and prompts for objective + eval source |
| `/evolve <path>` | Skip the picker; prompt only for objective + eval source |
| `/evolve last` | Print the report path of the most recent evolution recorded in the current pi session |
| `/evolve help` | Print usage |

The interactive picker discovers candidates under the project tree by matching:
- `SKILL.md`, `AGENTS.md`, `SYSTEM.md`, `APPEND_SYSTEM.md`
- Any `.md` under `.pi/prompts/` or `.agents/prompts/`

Directories `.git`, `node_modules`, `dist`, `build`, `coverage`, and `.pi/hermes-self-evolution/` are skipped during discovery.

## Environment variables

| Variable | Purpose |
|---|---|
| `PI_HERMES_EVOLVE_PYTHON` | Override Python interpreter path for the optional sidecar. Falls back to `python3`, then `python`. |
| `PI_SKIP_VERSION_CHECK` | Set to `1` by the engine when spawning sub-`pi` processes to suppress the version check. |
| Standard pi-coding-agent env vars | Inherited unchanged when spawning sub-`pi` processes (model selection, API keys, etc.). |

## Constraint configuration (`ConstraintConfig`)

The constraint pipeline runs against every candidate. Defaults are constructed by `buildConstraintConfig(target, maxBytes, overrides)` in `src/engine.ts`. Overrides may be supplied via tool parameters where exposed.

| Field | Type | Default | Source of override |
|---|---|---|---|
| `maxSizeBytes` | number | `max(originalBytes + 400, ceil(originalBytes × 1.2))` | computed from artifact |
| `maxGrowthRatio` | number | 0.20 | not currently exposed via tool params |
| `testCommand` | string \| undefined | undefined | tool param `testCommand` |
| `testTimeoutMs` | number | 60000 | tool param `testTimeout × 1000` |
| `checkSemanticDrift` | boolean | true | not currently exposed |
| `maxDriftScore` | number | 0.40 | not currently exposed |

The constraint surface is intentionally minimal at the tool layer. Power users who want to tune drift thresholds or growth ratios should call `runEvolution` directly rather than through the tool.

## Backend selection

`detectPythonBackend()` is invoked when `backend` is `auto` or `python`. It locates a Python interpreter via:

1. `process.env.PI_HERMES_EVOLVE_PYTHON` if set
2. `python3` on PATH
3. `python` on PATH

It then runs the sidecar's `--doctor` mode and inspects the JSON response for `dspy: true`. Behavior matrix:

| `backend` | DSPy installed | Outcome |
|---|---|---|
| `auto` | yes | Python sidecar |
| `auto` | no | TypeScript engine (silent fallback) |
| `typescript` | (irrelevant) | TypeScript engine |
| `python` | yes | Python sidecar |
| `python` | no | Throws `"Python backend requested but unavailable."` |

The TypeScript engine is feature-complete for Phase 1 without Python.

## Output configuration

Run output goes under `.pi/hermes-self-evolution/runs/<timestamp>-<artifact-slug>/`. The slug is derived from the artifact's basename via `slugify` (lowercase, alphanumeric + dashes, max 48 chars). See [docs/output-layout.md](output-layout.md) for the full per-run directory format.

Golden datasets persist under `.pi/hermes-self-evolution/golden/<goldenTaskId>/` with `train.jsonl`, `validation.jsonl`, `holdout.jsonl`, and `manifest.json`. Reusing the same `goldenTaskId` loads these splits verbatim instead of regenerating.

The lineage append-only log lives at `.pi/hermes-self-evolution/lineage.jsonl` and accumulates across all runs in the project.

## Verification gates

Add `npm run typecheck`, `npm run python:check`, and `npm run test:parity` (alias `npm run test:gates`) to CI to catch regressions. The parity gate reads the README and asserts every claimed capability row is still present and that no removed-framing markers (e.g., `typescript-proxy`) regress back in.

```yaml
# Example CI snippet
- run: npm install
- run: npm run test:gates
```

## Common configuration recipes

### Cheap smoke run

```text
Use self_evolve_artifact on .pi/skills/my-skill/SKILL.md
with maxExamples 4 and candidateCount 1.
```

### Reproducible golden run

```text
Use self_evolve_artifact on .pi/skills/my-skill/SKILL.md
with goldenTaskId "my-skill-v1" and persistGolden true.
```

First run generates and persists the splits. Subsequent runs with the same id reuse them, making the dataset stable across model and prompt variations.

### Test-gated run that auto-creates a PR

```text
Use self_evolve_artifact on AGENTS.md with
testCommand "npm run typecheck && npm test"
and createPR true.
```

Only candidates passing the test command are eligible. If the best candidate improves the holdout composite, the branch and PR are created automatically.

### Python-only run

```text
Use self_evolve_artifact on .pi/skills/foo/SKILL.md with backend "python".
```

Throws if DSPy is not installed in the resolved Python environment.

# pk-pi-hermes-evolve

A local pi package inspired by [Nous Research's Hermes Agent Self-Evolution](https://github.com/NousResearch/hermes-agent-self-evolution).

This package adapts the *Hermes Phase 1 idea* to pi:

- pick a local instruction artifact (`SKILL.md`, prompt template, `AGENTS.md`, `SYSTEM.md`, etc.)
- generate a compact evaluation set from synthetic tasks, recent pi session history, or both
- run a reflective candidate-generation loop
- proxy-score baseline vs candidates with an LLM judge
- save a reviewable report and candidate files under `.pi/hermes-self-evolution/`
- **never overwrite the original target automatically**

It is a **pi-native extension** with a **TypeScript-native engine**:

- **TypeScript backend**: the source-of-truth implementation — iterative reflective loop, Hermes-weighted judge, tiered constraint pipeline, execution traces, golden datasets, PR automation
- **Python acceleration mode**: optional sidecar (`--accelerate`) — activates when Python + DSPy are installed; adds a DSPy/GEPA optimizer path on top of the TS engine's same guardrails

The core loop is modeled after Hermes' mutation → evaluation → guardrails → human review flow, but adapted to pi extension APIs and local pi session history.

## Documentation

| Document | Contents |
|---|---|
| [docs/architecture.md](docs/architecture.md) | Module map, end-to-end run pipeline, iterative loop, executor, tiered gate, constraint pipeline, lineage, backend selection |
| [docs/configuration.md](docs/configuration.md) | Every tool parameter, `/evolve` flag, env var, constraint config option, and common recipes |
| [docs/output-layout.md](docs/output-layout.md) | Run-directory format, manifest / dataset / iteration / executor / trace schemas, lineage and golden dataset layouts |
| [docs/ownership-map.md](docs/ownership-map.md) | The 5-lane disjoint-ownership pattern used to land Phase 1 parity |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Code organization, style, verification gates, parallel-PRD dispatch pattern, soft-spot policy, release process |
| [CHANGELOG.md](CHANGELOG.md) | Version history |

The full documentation index is at [docs/README.md](docs/README.md).

## Pi docs reviewed for this package

This package was designed against pi's extension/package docs and examples, especially:

- `README.md`
- `docs/extensions.md`
- `docs/packages.md`
- `docs/session.md`
- `docs/tui.md`
- examples:
  - `examples/extensions/subagent/`
  - `examples/extensions/plan-mode/`
  - `examples/extensions/todo.ts`
  - `examples/extensions/with-deps/`

Key pi takeaways applied here:

- ship as a **pi package** with a `pi.extensions` manifest
- keep extension logic in TypeScript loaded directly by pi
- use a **command** for human-driven runs and a **tool** for model-driven runs
- keep state in session entries with `appendEntry()` instead of hidden external mutation
- use `.pi/...` paths for project-local generated artifacts
- rely on session JSONL history as a local source for evolution context

## What it supports

### Target artifacts

Best fit:

- `.pi/skills/**/SKILL.md`
- `.pi/prompts/*.md`
- `.agents/skills/**/SKILL.md`
- `AGENTS.md`
- `.pi/SYSTEM.md`
- `.pi/APPEND_SYSTEM.md`

The engine is optimized for **text instructions**, not general code evolution.

### Commands

- `/evolve` → interactive artifact picker
- `/evolve path/to/file.md` → evolve a specific file
- `/evolve last` → show the last saved report path in the current session

### Tool

- `self_evolve_artifact`

Use it when you explicitly want the model to improve a local instruction artifact and save reviewable candidates.

### Backends

- `typescript` → TypeScript engine (default; always available)
- `--accelerate` / `auto` → TypeScript engine + Python DSPy acceleration when available
- `python` → require the Python acceleration sidecar (error if unavailable)

## Install

### Local path install

From pi:

```bash
pi install npm:pk-pi-hermes-evolve
```

Or project-local:

```bash
pi install -l npm:pk-pi-hermes-evolve
```

### Direct extension loading for testing

```bash
pi -e npm:pk-pi-hermes-evolve
```

## Python acceleration mode (optional)

The npm package includes an optional Python sidecar under `python_backend/`. The TypeScript engine is fully functional without it.

Install the sidecar if you want DSPy/GEPA acceleration:

```bash
cd python_backend
pip install -e .
```

The extension searches for Python in this order:

1. `PI_HERMES_EVOLVE_PYTHON`
2. `python3`
3. `python`

When DSPy is detected and `backend` is `auto`, the Python acceleration sidecar is activated.
Without it, the TypeScript engine runs the full evolution loop on its own.

## Usage

### Interactive command

```text
/evolve
/evolve .pi/skills/my-skill/SKILL.md
/evolve AGENTS.md
```

The command will ask for:

- evolution objective
- evaluation source:
  - `mixed`
  - `synthetic`
  - `session`

The tool also accepts an optional backend override:

```text
Use self_evolve_artifact on AGENTS.md with backend python.
```

### Tool-driven usage

Example prompt to pi:

```text
Use self_evolve_artifact on .pi/skills/review/SKILL.md to improve trigger clarity and output quality.
```

With a golden task ID for reproducible validation:

```text
Use self_evolve_artifact on .pi/skills/review/SKILL.md with goldenTaskId "review-skill-v1".
```

## Output layout

Every run writes to a timestamped directory:

```text
.pi/hermes-self-evolution/runs/<timestamp>-<artifact>/
├── original.md
├── best-candidate.md
├── report.md
├── manifest.json
├── dataset.json
└── candidates/
    ├── candidate-1.md
    ├── candidate-1.json
    └── ...
```

### Dataset splits

Generated examples are split into three sets:

- **Train** (~50%): used for candidate generation and weakness analysis
- **Validation** (~20%): used for intermediate scoring and golden dataset tagging
- **Holdout** (~30%): used only for final evaluation, never during candidate generation

### Golden datasets

When `goldenTaskId` is provided (tool parameter or Ralph loop `--golden-task-id`), the validation split is tagged as a golden dataset. This enables:

- reproducible cross-run evaluation with a known example set
- consistent benchmarking across different candidate strategies
- traceability from evolution runs back to the originating task

## Guardrails

Current guardrails mirror Hermes' spirit, but stay lightweight and local:

- original file is preserved
- candidates are written separately
- frontmatter is preserved when present
- existing `{{placeholders}}` must survive candidate generation
- candidates over the size budget are rejected
- human review is always required before applying changes

## Hermes Phase 1 parity

The TypeScript engine implements the Hermes Phase 1 workflow end-to-end. Status by capability:

| Capability | Status | Where |
|---|---|---|
| 3-source dataset (synthetic / session / mixed) | ✅ | `generateDataset` in `src/engine.ts` |
| Train / validation / holdout split | ✅ | `splitExamples` in `src/engine.ts` |
| Golden dataset persistence by task id | ✅ | `saveGoldenDataset` / `loadGoldenDataset` |
| Hermes-weighted judge (0.5 / 0.3 / 0.2) | ✅ | `evaluateArtifact` in `src/engine.ts` |
| 7-check constraint validator (non_empty, size, growth, placeholder, heading, frontmatter, drift, skill_structure) | ✅ | `validateConstraints` + `src/constraints-structure.ts` |
| Execution traces (all + failures) | ✅ | `buildTrace` in `src/engine.ts` |
| Secret scanner on datasets | ✅ | `scanForSecrets` in `src/engine.ts` |
| Optional test-command gate | ✅ | `runTestCommand` in `src/engine.ts` |
| Optional PR automation (branch + `gh pr create`) | ✅ | `createGitBranchWithCandidate` |
| **Iterative reflective loop** (GEPA-Pareto: frontier-based parent selection, minibatch pre-filter, bounded system-aware merge) | ✅ | iteration loop in `runTypeScriptEvolution`; `IterationRecord[]` in `iterations/` |
| **Pareto-frontier candidate pool** (illumination-style parent sampling instead of greedy hill-climbing; mutation edits the selected parent's own body) | ✅ | `computeParetoFrontier` / `selectParetoParent` in `src/engine.ts` |
| **System-aware merge** (bounded crossover of two frontier candidates' complementary strengths) | ✅ | `generateMergeCandidateDraft` in `src/engine.ts` |
| **Pi-native executor** (real stdout, not predicted) | ✅ | `src/pi-executor.ts` → `executeCandidateInPi` |
| **Tiered regression gate** (typecheck → cohort → coherence) | ✅ | `src/tiered-gate.ts` → `runTieredGate` |
| **SKILL.md structural validator** (name + description in first 500 chars) | ✅ | `src/constraints-structure.ts` |
| **Cross-run lineage memory** (`lineage.jsonl`, Pareto-best ancestor) | ✅ | `src/lineage.ts` |
| **TS as default, Python as `--accelerate`** | ✅ | reframed throughout this README and `src/python-backend.ts` |
| Python DSPy/GEPA acceleration sidecar | 🟦 optional | `python_backend/` |
| OTel-traced Ralph loop for parity work | 🟦 optional | `scripts/ralph_otel.py` |
| Sokoban benchmark scaffold | 🟦 optional | `scripts/sokoban_benchmark.py` |

What is still out of scope versus the full Nous vision:

- Hermes Phases 2–5 (tool-description / system-prompt / tool-implementation-code evolution, continuous auto-improvement loop)
- code-organism evolution via Darwinian Evolver
- no built-in pytest gate (use `testCommand` to wire one)

This package implements **Hermes Phase 1 in TypeScript** as a first-class pi extension.

## Development

Install dev dependencies and type-check:

```bash
cd /c/dev/Desktop-Projects/pi-hermes-self-evolution
npm install
npm run typecheck
npm run python:check
```

## Sokoban benchmark scaffold

The repo now includes a reusable benchmark scaffold built around the provided Sokoban benchmark pack and CSV schema:

- benchmark assets live under `benchmarks/sokoban/`
- runner entrypoint: `scripts/sokoban_benchmark.py`
- supported workflow:
  - initialize a run
  - prepare a baseline or improvement attempt
  - record a completed attempt into `results.csv`
  - analyze attempt-1 vs attempt-5 and baseline-vs-improvement attempt-5 deltas

Example:

```bash
python scripts/sokoban_benchmark.py init \
  --run-id demo-sokoban \
  --training-levels level-a level-b level-c level-d \
  --heldout-level level-e

python scripts/sokoban_benchmark.py prepare-attempt \
  --run-id demo-sokoban \
  --arm improvement \
  --attempt 1

python scripts/sokoban_benchmark.py record-attempt \
  --run-id demo-sokoban \
  --arm improvement \
  --attempt 1

python scripts/sokoban_benchmark.py analyze \
  --run-id demo-sokoban
```

`prepare-attempt` creates:

- `prompt.md`
- `result.json`
- `postmortem.md`
- `input-skill.md`
- `updated-skill.md`

`record-attempt` validates the result payload against the CSV schema and carries the updated skill forward for the improvement arm.

## Ralph loop for Hermes parity work

The repo now includes a traced Ralph loop for closing the highest-value Hermes parity gaps:

```bash
cd /c/dev/Desktop-Projects/pi-hermes-self-evolution
python scripts/ralph_otel.py \
  --task-file scripts/tasks/hermes_parity_task.json \
  --repo . \
  --model zai/glm-5.1 \
  --telemetry-export console
```

What it does:

- runs a retryable multi-step loop against this repo
- calls `pi` as the execution worker for each step
- runs validation commands after every step
- runs deterministic repo-deliverable checks in the judge (for example execution traces, validation split support, and golden-dataset support)
- records OpenTelemetry spans for `ralph.run/<task>`, `loop.step`, `model.infer`, `tool`, and `judge`
- writes JSON step artifacts under `.pi/hermes-self-evolution/ralph-runs/`

Use `--telemetry-export otlp-http --otlp-endpoint http://host:4318` to ship traces and metrics to an OTLP collector.

## Next useful upgrades

- add real execution-based evaluation via subagent runs
- add prompt-template / skill-specific rubric presets
- add diff rendering in the final report
- add apply/approve workflows behind explicit confirmation
- add automatic browser/game automation for benchmark runs instead of scaffold-only preparation
- add benchmark/test gates to the Python backend so GEPA mutations are filtered by real task outcomes
- add repeated multi-run aggregation across several held-out boards instead of single-run summaries

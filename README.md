# pk-pi-hermes-evolve

A local pi package inspired by [Nous Research's Hermes Agent Self-Evolution](https://github.com/NousResearch/hermes-agent-self-evolution).

This package adapts the *Hermes Phase 1 idea* to pi:

- pick a local instruction artifact (`SKILL.md`, prompt template, `AGENTS.md`, `SYSTEM.md`, etc.)
- generate a compact evaluation set from synthetic tasks, recent pi session history, or both
- run a reflective candidate-generation loop
- proxy-score baseline vs candidates with an LLM judge
- save a reviewable report and candidate files under `.pi/hermes-self-evolution/`
- **never overwrite the original target automatically**

It is a **pi-native extension** with a **hybrid backend model**:

- **TypeScript backend**: always available, uses pi subprocess calls as a local proxy-evolution loop
- **Python backend**: optional, uses a real DSPy/GEPA-style path when Python + DSPy are installed

The core loop is modeled after Hermes' mutation → evaluation → guardrails → human review flow, but adapted to pi extension APIs and local pi session history.

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

- `auto` → prefer Python DSPy backend when available, otherwise TypeScript fallback
- `python` → require the Python backend
- `typescript` → force the TypeScript-only path

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

## Python DSPy backend

The npm package includes an optional Python sidecar under `python_backend/`.

Install it manually if you want the hybrid DSPy/GEPA path:

```bash
cd python_backend
pip install -e .
```

The extension looks for Python in this order:

1. `PI_HERMES_EVOLVE_PYTHON`
2. `python3`
3. `python`

If DSPy is installed, `backend: auto` will use the Python backend.
Otherwise it falls back to TypeScript.

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

## Important limitations

This is still **not** a full Hermes reproduction.

What the Python upgrade adds:

- a real Python backend bundled with the npm package
- DSPy-based dataset generation, judging, and candidate synthesis
- a GEPA path when the installed DSPy build exposes `dspy.GEPA`
- automatic fallback to MIPROv2 or plain Chain-of-Thought if GEPA is unavailable

What the latest parity upgrade adds:

- a **three-way dataset split** (train / validation / holdout) in both TypeScript and Python backends
- **golden dataset support**: when a `goldenTaskId` is provided, the validation split is tagged as a golden set for reproducible cross-run evaluation
- **traced Ralph loop** (`scripts/ralph_otel.py`) with OpenTelemetry spans for `ralph.run/<task>`, `loop.step`, `model.infer`, `tool`, and `judge`
- **deterministic repo-deliverable checks** in the judge (execution traces, validation split, golden datasets)
- a **Sokoban benchmark scaffold** (`scripts/sokoban_benchmark.py`) for initializing repeatable 5-attempt baseline/improvement runs, preparing per-attempt artifacts, recording results, and summarizing held-out performance

What is still missing versus the full Nous vision:

- no automatic pytest / external benchmark gate yet
- no code-organism evolution
- still optimized mainly for prompt/instruction artifacts, not general source code

So treat this as a practical **hybrid phase-1 self-evolution package for pi**.

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

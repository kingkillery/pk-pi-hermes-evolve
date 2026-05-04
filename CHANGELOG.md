# Changelog

## Unreleased

- **diff rendering**: every evolution run now writes a `diff.patch` file (original body → best candidate body) and embeds a `## Diff` section in `report.md`
- **apply/approve workflow**: new `/evolve apply [runDir]` command copies the best candidate to the target file after showing the diff and requiring explicit `"yes"` confirmation; uses the last session run when no `runDir` is provided
- **artifact-type rubric presets**: the judge now receives type-specific scoring guidance (`skill` / `prompt` / `instructions`) so correctness, procedure-following, and conciseness are weighted appropriately for each artifact kind; parity added in both TypeScript and Python backends

- add `scripts/ralph_otel.py`, a traced Ralph loop for Hermes-parity gap closure work in this repo
- add `scripts/tasks/hermes_parity_task.json` as the default parity task spec
- upgrade the Ralph judge with deterministic repo-deliverable checks for parity targets like execution traces, validation splits, and golden datasets
- include OpenTelemetry Python dependencies and repo scripts for the Ralph loop workflow
- add `scripts/sokoban_benchmark.py` plus bundled benchmark assets under `benchmarks/sokoban/`
- add a scaffolded baseline-vs-improvement 5-attempt benchmark workflow with attempt preparation, CSV recording, and summary analysis

## 0.2.1 - 2026-04-12

- fix Python backend syntax so CI and `python:check` pass cleanly
- republish hybrid extension package with the corrected Python backend

## 0.2.0 - 2026-04-12

- add optional Python DSPy/GEPA hybrid backend under `python_backend/`
- add automatic backend selection (`auto` / `python` / `typescript`)
- add GitHub Actions CI and npm release workflows
- update npm package metadata and README for public distribution

## 0.1.0 - 2026-04-12

- initial pi-native Hermes-inspired self-evolution extension
- `/evolve` command and `self_evolve_artifact` tool
- TypeScript-only reflective evaluation loop with report generation

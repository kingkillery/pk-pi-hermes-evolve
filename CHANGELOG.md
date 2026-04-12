# Changelog

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

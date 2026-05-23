# Documentation Index

| Document | Audience | Contents |
|---|---|---|
| [architecture.md](architecture.md) | contributors, integrators | Module map, end-to-end run pipeline diagram, iterative loop, executor, tiered gate, constraint pipeline, lineage, backend selection, extension points |
| [configuration.md](configuration.md) | users, integrators | Every tool parameter, `/evolve` flag, env var, and constraint config option; effort/cost guidance; common recipes |
| [output-layout.md](output-layout.md) | reviewers, automation authors | Complete run-directory format; manifest, dataset, candidate, iteration, executor, and trace file schemas; lineage and golden dataset layouts |
| [ownership-map.md](ownership-map.md) | contributors running parallel-PRD dispatches | The 5-lane disjoint-ownership pattern that landed Hermes Phase 1 parity; commit references and serialization rules |
| [../CONTRIBUTING.md](../CONTRIBUTING.md) | contributors | Code organization, style conventions, verification gates, parallel-PRD pattern, soft-spot policy, commit conventions, release process |
| [../README.md](../README.md) | users | Install, basic usage, parity table, Python acceleration mode |
| [../CHANGELOG.md](../CHANGELOG.md) | all | Version history including the TypeScript-native Phase 1 parity work |

## Reading order

**New user**: README → configuration.md.

**Reviewing a run**: output-layout.md → architecture.md (iterative loop section).

**Contributor**: CONTRIBUTING.md → architecture.md → relevant src/ file.

**Running a parallel-PRD dispatch**: ownership-map.md → CONTRIBUTING.md (parallel-PRD section) → the `qs-parallelprd` skill (`~/.claude/skills/parallel-pipeline-dispatch/SKILL.md` and its `references/` examples).

## Planning artifacts

The `.prd/` directory at the repo root holds in-flight and recently completed dispatch plans. These are committed for traceability but are not part of the npm package. Current contents:

- `.prd/gap-analysis.md`, `.prd/agent-prompt.md`, `.prd/current-state.md` — Phase 1 parity dispatch (completed; landed commits c417bfd through 5559ad4)
- `.prd/smoke-test-orchestration.md` + `.prd/smoke-{A,B,C,D,E}-*.md` — runtime smoke-test dispatch (in progress; see lane status notifications)

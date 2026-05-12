# File ownership map — Hermes Phase 1 parity work

This map was used to dispatch five sub-agents in parallel for the Phase 2 work that closed the Hermes Phase 1 workflow gaps. It is preserved here as a reference for future multi-agent dispatches against this repository.

## Lanes and ownership

| Lane | Owned files | Gap letters | Effort | Model | Commit |
|---|---|---|---|---|---|
| `pre-phase-A` (solo, blocks all others) | `src/types.ts` | A | SMALL | sonnet | `c417bfd` |
| `engine-bundle` (B+C bundled) | `src/engine.ts`, `src/pi-executor.ts` (new) | B, C | LARGE | opus | `326bb32` |
| `tiered-gate` | `src/tiered-gate.ts` (new) | D | MEDIUM | opus | `44b982f` |
| `structural-and-lineage` | `src/constraints-structure.ts` (new), `src/lineage.ts` (new) | E, F | MEDIUM | opus | `8f20ecc` |
| `framing-and-parity` | `README.md`, `src/python-backend.ts` | G | SMALL | sonnet | `d289d9b` |
| post-merge (solo, this orchestrator) | `tests/api-snapshot.test.ts`, `tests/e2e-golden.test.ts`, `tests/parity.test.ts`, `docs/ownership-map.md`, `README.md` parity section | H, I, J, K, Z | MEDIUM | — | (this commit) |

## Serialization constraint

The `engine-bundle` lane is the only lane that may edit `src/engine.ts`. Gaps B (iterative reflective loop) and C (pi-native executor) are bundled in a single lane because both heavily mutate `src/engine.ts`. They cannot be split into two parallel agents without serializing access to the file.

Other lanes (`tiered-gate`, `structural-and-lineage`) build self-contained modules that the `engine-bundle` lane imports. The imports are added by the `engine-bundle` lane with `@ts-expect-error` suppressors during parallel execution; the merge orchestrator removes those suppressors after all sibling modules land on disk.

## Wiring pattern (for future Phase 2/3 dispatches)

1. Pre-phase: solo `src/types.ts` extension. Block all other lanes until merged.
2. Parallel build: launch N lanes where N − 1 own disjoint new modules and one (`engine-bundle`) edits the orchestrator. The orchestrator-editing lane adds stub imports under `@ts-expect-error`.
3. Merge: fast-forward or no-conflict merges in any order — files are disjoint.
4. Cleanup: orchestrator removes `@ts-expect-error` suppressors and reconciles any field-name drift between the orchestrator-editing lane and the module-owning lanes.
5. Post-merge: solo lane writes tests, regenerates docs, runs the final parity gate.

## Disjointness verification

Per-lane file-touch sets (from `git show --stat` on each lane commit):

- `pre-phase-A`: `src/types.ts`
- `engine-bundle`: `src/engine.ts`, `src/pi-executor.ts`
- `tiered-gate`: `src/tiered-gate.ts`
- `structural-and-lineage`: `src/constraints-structure.ts`, `src/lineage.ts`
- `framing-and-parity`: `README.md`

Intersection of any two sets: empty. The `engine-bundle` lane is the only lane that imports the others' new modules; those imports are added during the lane's own work, not during merge.

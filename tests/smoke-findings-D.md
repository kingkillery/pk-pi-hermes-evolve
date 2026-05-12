# Lane D findings

## Lineage parent→child link
- lineage.jsonl entries: 2
- run 1 runId / artifactHash: `2026-05-12_03-51-39-skill` / `0690fb5f65640ac7`
- run 2 runId / artifactHash: `2026-05-12_03-51-46-skill` / `0690fb5f65640ac7`
- run 2 parentRunId matches run 1 runId: yes
- run 2 parentArtifactHash matches run 1 artifactHash: **no** (see finding below)
- score delta (run 2 − run 1): 0.0000 (flat — both runs converged to the same candidate via fallback acceptance)

### parentArtifactHash semantic mismatch
Both lineage entries carry `parentArtifactHash=00645ba901764838`, which is the sha256[:16] of the
original SKILL.md input. This value is the hash of the artifact *before* mutation, not the hash
of the prior run's winning output. The spec requires `parentArtifactHash` to equal run-1's
`artifactHash`; the engine instead stores the pre-mutation source hash in all runs, so
cross-run hash chaining is not implemented.

## loadBestAncestor probes
- Probe 1 (content-hash, strict): returned `null` — fixture SKILL.md hash (`00645ba901764838`) equals
  `parentArtifactHash` in lineage but no entry carries it as `artifactHash`; strict path returns null
- Probe 2 (path-only, fuzzy): returned `2026-05-12_03-51-39-skill`; matched via **global-highest-score fallback** —
  `SKILL.md` (the basename slug) is not a substring of any runId (runIds use `<ts>-skill` format, not `<ts>-SKILL.md`);
  both entries tie on score so the first entry is returned
- Probe 3 (wrong path): returned `2026-05-12_03-51-39-skill` — same fallback; wrong paths are indistinguishable
  from unknown-but-real paths, confirming false-positive fallback behavior

## Soft spots observed
1. `loadBestAncestor` fuzzy match returns false-positive on unknown path: **confirmed**
   - rationale: `path.basename(artifactPath)` produces `SKILL.md`; runIds are formatted as
     `<ts>-<slugified-name>` (e.g., `2026-05-12_03-51-39-skill`) so the substring never matches.
     The all-entries fallback then returns the highest-score entry for *any* unknown path.
2. `parentArtifactHash` records the pre-mutation source hash, not the prior run's output hash.
   Cross-run artifact chaining (spec: run-N's `artifactHash` → run-(N+1)'s `parentArtifactHash`) is
   not implemented. Both fields equal `sha256[:16]` of the unchanged original input.
3. Both runs produced the same candidate (score flat at 0.812) because neither iteration exceeded
   the strict acceptance delta. The engine's fallback-acceptance path promoted the same iter-1
   candidate in both runs, making the lineage a degenerate chain (both `artifactHash` identical).

## Recommended remediations (for Lane E)
- When writing a lineage entry, set `parentArtifactHash` to the *prior run's* `artifactHash` (load
  the ancestor entry via `loadBestAncestor` before writing the new entry, then copy its `artifactHash`).
- In `loadBestAncestor`, when `artifactContent` is absent, return `null` instead of the
  global-highest-score fallback so unknown paths do not silently produce false-positive results.
  Document the intentional behaviour change with `// SOFT-SPOT(loadBestAncestor-fuzzy): ...`.
- Alternatively, require `artifactContent` as a non-optional parameter and remove the path-only
  fuzzy path entirely; callers that need it can read the file themselves.
- Slugify the artifact basename consistently in runId generation so `path.basename(artifactPath)`
  is a reliable runId substring (e.g., `SKILL.md` → `skill`); this would make the heuristic work
  as designed without requiring the content-hash path.
- Add a lineage integrity check: after both runs, assert `run2.parentArtifactHash === run1.artifactHash`
  to catch degenerate chains early (flagging the fallback-acceptance path as a risk).

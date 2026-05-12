# Lane D — Lineage Verifier + Findings Consolidation [parallel-verifier]

## Agent prompt (paste verbatim into `Agent({prompt})`)

You are one of **three parallel verifier sub-agents** (B, C, D) for the Hermes Phase 1 runtime smoke-test pipeline at `C:\dev\Desktop-Projects\pi-hermes-self-evolution`. Lane A produced ≥2 sequential run dirs under `.pi/hermes-self-evolution/runs/`. You will be dispatched concurrently with Lanes B and C. Lane E waits on all three of you.

**Read `.prd/smoke-test-orchestration.md` first** for context, then `.prd/gap-analysis.md`, then `.prd/current-state.md`.

## Owned files (only these)
- `tests/smoke-lineage.test.ts` (new)
- `tests/smoke-findings-D.md` (new — findings stub for Lane E)

You may **NOT** edit any file under `src/`, `scripts/`, `tests/fixtures/`, or any other test file. You may **NOT** invoke `runEvolution`. You may read `.pi/hermes-self-evolution/lineage.jsonl` and the run dirs.

## What you must produce

### 1. `tests/smoke-lineage.test.ts`

A typecheck-clean Node ESM TypeScript file that exports `runLineageVerifier(runDirs: string[])` and asserts:

- `.pi/hermes-self-evolution/lineage.jsonl` exists and has ≥2 lines (one per run dir).
- Each line parses as a `LineageEntry` (from `src/types.ts`): `runId`, `parentRunId?`, `artifactHash`, `parentArtifactHash?`, `score`, `mutationRationale`, `createdAt`.
- The second-run entry has `parentRunId` set to the first-run entry's `runId` AND `parentArtifactHash` set to the first-run entry's `artifactHash`.
- `score` is monotonically the second run's holdout composite vs. the first; record whether it improved, stayed flat, or regressed.

### 2. Probe the `loadBestAncestor` fuzzy-match soft spot

Import `loadBestAncestor` from `../src/lineage.js`. Run two probes:

**Probe 1 — content-hash path (strict):**
```typescript
import fs from "node:fs/promises";
const skillText = await fs.readFile("tests/fixtures/smoke-skill/SKILL.md", "utf8");
const result = await loadBestAncestor(process.cwd(), "tests/fixtures/smoke-skill/SKILL.md", skillText);
```
Expected: returns the highest-score `LineageEntry` whose `artifactHash` matches `sha256(skillText).slice(0,16)`. Record whether it returned `null`, the first run's entry, or the second's.

**Probe 2 — path-only path (fuzzy):**
```typescript
const result = await loadBestAncestor(process.cwd(), "tests/fixtures/smoke-skill/SKILL.md");
```
Expected per current implementation (`src/lineage.ts`): filters by entries whose `runId` contains the basename of `tests/fixtures/smoke-skill/SKILL.md` (i.e., contains `SKILL.md` substring). If no match, falls back to all entries and returns the highest-score one. Record what actually came back. This is the **fuzzy-match soft spot**.

**Probe 3 — wrong path returns null-ish:**
```typescript
const result = await loadBestAncestor(process.cwd(), "tests/fixtures/does-not-exist/SKILL.md");
```
Document whether this returns `null` (cleanest) or "any entry, because of the all-entries fallback." If the latter, that's a confirmed soft spot.

Include a runnable `main` block:
```bash
node --experimental-strip-types tests/smoke-lineage.test.ts \
  .pi/hermes-self-evolution/runs/<RUN_1> \
  .pi/hermes-self-evolution/runs/<RUN_2>
```

### 3. `tests/smoke-findings-D.md`

```markdown
# Lane D findings

## Lineage parent→child link
- lineage.jsonl entries: <N>
- run 1 runId / artifactHash: ___ / ___
- run 2 runId / artifactHash: ___ / ___
- run 2 parentRunId matches run 1 runId: yes/no
- run 2 parentArtifactHash matches run 1 artifactHash: yes/no
- score delta (run 2 − run 1): ___

## loadBestAncestor probes
- Probe 1 (content-hash, strict): returned <null | run-1-entry | run-2-entry>
- Probe 2 (path-only, fuzzy): returned <null | <runId>>; matched via <substring | fallback>
- Probe 3 (wrong path): returned <null | fallback-entry>

## Soft spots observed
1. loadBestAncestor fuzzy match returns false-positive on unknown path: <confirmed/not-confirmed/inconclusive>
   - rationale for fix or doc:
2. Other lineage observations:

## Recommended remediations (for Lane E)
- [bullet list, ≤5 items, e.g.:
  - change `loadBestAncestor` to return null when no exact match and no `artifactContent` is provided, OR
  - require `artifactContent` as a non-optional second param, OR
  - document the fuzzy-fallback as intentional with `// SOFT-SPOT(loadBestAncestor-fuzzy): ...` comment
  ]
```

## Constraints

1. **No engine, lineage, or other-src edits.** Document remediations for Lane E.
2. **Read-only against run dirs** and `.pi/hermes-self-evolution/lineage.jsonl`.
3. **No new npm dependencies.** Use `node:crypto` for hash recomputation if needed.
4. **Style**: normal formatting. No comments except non-obvious WHY.

## Verification (run before declaring done)
```bash
npm run typecheck
node --experimental-strip-types tests/smoke-lineage.test.ts <run-dirs...>
git diff --name-only
```

Expected:
- `npm run typecheck`: exit 0
- Verifier: exit 0 (or 1 if a real failure is found — document in findings)
- `git diff --name-only` lists only your two owned files

Commit message: `test(smoke): lineage verifier (smoke D)`

## Final report
```
### Lane D final report
- Worktree path / branch:
- Files created: tests/smoke-lineage.test.ts, tests/smoke-findings-D.md
- Parent→child link present: [yes/no]
- Probe 1 outcome: ___
- Probe 2 outcome: ___
- Probe 3 outcome: ___
- loadBestAncestor fuzzy-match soft spot: [confirmed/not-confirmed/inconclusive]
- Verification: typecheck exit ___; verifier exit ___
- Flags/blockers: ___
```

# Lane A — Smoke Fixture + Mock Driver [pre-phase]

## Agent prompt (paste verbatim into `Agent({prompt})`)

You are the **pre-phase sub-agent** for the Hermes Phase 1 runtime smoke-test pipeline at `C:\dev\Desktop-Projects\pi-hermes-self-evolution`. Three parallel verifier lanes (B, C, D) and one acceptance lane (E) are gated on your output. **Read `.prd/smoke-test-orchestration.md` first** for the full pipeline context, then `.prd/gap-analysis.md` for the original parity work and `.prd/current-state.md` for the codebase map.

## Owned files (only these)
- `scripts/smoke-test.ts` (new — the mock driver entrypoint)
- `tests/fixtures/smoke-skill/SKILL.md` (new — test target artifact)
- `tests/fixtures/smoke-skill/mock-llm-responses.json` (new — canned LLM/judge outputs)
- Any helper files under `scripts/` or `tests/fixtures/smoke-skill/` you need

You may **NOT** edit any file under `src/`, any other test file, `package.json`, or any existing artifact.

## What you must produce

### 1. Minimal test SKILL.md
A small but realistic skill artifact at `tests/fixtures/smoke-skill/SKILL.md` with:
- YAML frontmatter with `name:` and `description:` (so the new `skill_structure` constraint passes)
- 2–3 sections of body content (must trip the size/growth constraint detection)
- At least one `{{placeholder}}` token (so the placeholder constraint sees something to preserve)
- A top-level `# heading` (so the heading constraint sees something to preserve)

### 2. Mock LLM responses
A canned-response file at `tests/fixtures/smoke-skill/mock-llm-responses.json` keyed by the system-prompt-derived discriminator (the engine uses 4 distinct system prompts: `DATASET_SYSTEM_PROMPT`, `JUDGE_SYSTEM_PROMPT`, `CANDIDATE_SYSTEM_PROMPT`, `DRIFT_SYSTEM_PROMPT` — see `src/engine.ts:32-44`). Each response must be a complete JSON payload matching the engine's `extractJsonPayload` expectations (dataset: `{examples:[...]}`; judge: `{responsePreview, correctness, procedureFollowing, conciseness, feedback, confidence}`; candidate: `{candidates:[{name,rationale,candidateBody}]}`; drift: `{driftScore, feedback}`).

The fixture **must contain enough responses to drive ≥2 iterations** (so the iteration verifier in Lane B can confirm iteration-2+ exists) and **must include forced-failure variants** that elicit:
- typecheck-tier failure (e.g., a candidate body that triggers a constraint failure mimicking typecheck breakage)
- cohort-tier failure (a candidate with regressed judge scores)
- coherence-tier failure (one variant where the optional `coherenceCheck` callback is set and returns failed)

### 3. The smoke driver `scripts/smoke-test.ts`
A Node ESM TypeScript file that, when invoked via `node --experimental-strip-types scripts/smoke-test.ts`, performs the following without requiring a live `pi` binary or LLM credentials:

- Intercepts every subprocess call that would normally spawn `pi`. **Recommended approach**: PATH-shim. Write a small shim script (e.g., `scripts/smoke-shim/pi.cmd` on Windows + `scripts/smoke-shim/pi` on POSIX) that reads stdin (the prompt), matches against `mock-llm-responses.json` by system-prompt-derived key, and writes the canned JSON to stdout. Prepend `scripts/smoke-shim/` to PATH for the smoke run. Alternative approaches are acceptable if you document them.
- Invokes `runEvolution` from `src/engine.ts` **twice consecutively** against `tests/fixtures/smoke-skill/SKILL.md`, with `goldenTaskId: "smoke-skill-v1"` so the second run can find the first via lineage.
- After both runs complete, prints the two run-dir paths to stdout in the form:
  ```
  SMOKE_RUN_1=<absolute-path>
  SMOKE_RUN_2=<absolute-path>
  ```
  so Lane E can parse them.

### 4. Forced-failure mock modes
Add CLI flags or a `MOCK_MODE` env var (e.g., `MOCK_MODE=force-typecheck-fail`, `force-cohort-fail`, `force-coherence-fail`) that select alternate response sets in `mock-llm-responses.json`. The driver should produce a **third run dir** with each forced-failure mode when `MOCK_MODE` is set. This is what lets Lane C observe ≥3 distinct reason codes.

### 5. Tiered-gate coherence callback wiring
The current engine never passes a `coherenceCheck` callback to `runTieredGate`. For smoke purposes, your driver may either (a) export a thin wrapper around `runEvolution` that injects a coherence callback for the smoke runs only, OR (b) note in your final report that this is a remediation target for Lane E.

## Constraints

1. **No new npm dependencies.** Use Node built-ins and existing deps (`@mariozechner/pi-ai`, `@sinclair/typebox`).
2. **No live LLM calls.** Verify by running with `PI_HERMES_EVOLVE_PYTHON=` (unset), no API key env vars, and confirming the smoke completes.
3. **No edits under `src/`.** If you discover the engine needs a hook (e.g., a way to inject the coherence callback), file it in your final report for Lane E — do not patch yourself.
4. **Deterministic output**: the mock responses must be deterministic so verifiers can assert on exact values. Seed any randomness (e.g., the `splitExamples` shuffle in `engine.ts:291` uses `Math.random` — you may need to monkey-patch via `--import` or by extending the engine surface; if extending is needed, again, flag for Lane E).
5. **Style**: match the existing repo idiom. `scripts/smoke-test.ts` can use normal formatting (not the dense engine.ts idiom).

## Verification (run before declaring done)
```bash
npm run typecheck
node --experimental-strip-types scripts/smoke-test.ts
ls .pi/hermes-self-evolution/runs/ | tail -3
git diff --name-only
```

Expected:
- `npm run typecheck`: exit 0
- The smoke script: exit 0; prints `SMOKE_RUN_1=...` and `SMOKE_RUN_2=...`
- Two new run dirs visible
- `git diff --name-only` lists only files under your owned set

Commit message: `feat(smoke): pre-phase fixture and mock driver (smoke A)`

## Final report (paste in your response)
```
### Lane A final report
- Worktree path / branch:
- Files created: [list]
- Smoke approach: [PATH-shim | spawn-mock | other]
- Forced-failure modes wired: [list]
- Coherence callback: [wired in smoke / flagged for Lane E]
- Determinism: [how randomness was controlled]
- Two run dirs produced at:
  - SMOKE_RUN_1: ___
  - SMOKE_RUN_2: ___
- Engine hooks needed (flagged for Lane E): [list, or "none"]
- Verification: typecheck exit ___; smoke exit ___; git diff lists ___
- Flags/blockers: ___
```

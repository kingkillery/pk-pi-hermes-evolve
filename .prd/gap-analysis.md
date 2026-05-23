# Hermes Phase 1 Parity — Gap Analysis (TypeScript-native)

**Repo:** `pk-pi-hermes-evolve` (this directory)
**Reference workflow:** [NousResearch/hermes-agent-self-evolution](https://github.com/NousResearch/hermes-agent-self-evolution), Phase 1 only.
**Goal:** Make the TypeScript backend the source-of-truth Hermes-shaped workflow inside a pi-coding-agent extension. Keep Python/DSPy as optional `--accelerate`, not the "real" path.

---

## 1. What Hermes Phase 1 actually does (workflow, not artifacts)

Concrete sequence in `evolution/skills/evolve_skill.py`:

1. Load target `SKILL.md`
2. Build eval dataset (synthetic OR session-mined OR hand-curated golden), split into train/val/holdout
3. Baseline constraint check
4. Wrap skill body as a `dspy.Signature`
5. **Iterative GEPA optimizer** — reads judge feedback text per round, refines the same lineage, multiple rounds
6. Constraint gate on output (fails abort)
7. Holdout evaluation: baseline vs. evolved via `LLMJudge` weighted (correctness 0.5, procedure 0.3, conciseness 0.2)
8. Tiered safety gates: `pytest` → constraints → fitness ≥10% → fast bench (TBLite-20) → full regression (TBLite-100, ≤2% drop) → coherence (YC-Bench) → mandatory PR review

Three distinguishing properties:
- **Reflective trace-aware mutation** — judge returns structured `feedback` text; GEPA reads it
- **Multi-tier unified target framework** — same infra for skills, tool descriptions, system prompts, code (phases 2–4)
- **Hard safety gates + mandatory human PR** — anti-drift mechanism

## 2. What this repo has

Implemented in TypeScript (`src/engine.ts`, `src/index.ts`, `src/types.ts`):

| Component | Status | Reference |
|---|---|---|
| 3-source dataset (synthetic / session / mixed) | ✅ | `generateDataset` engine.ts:333 |
| Train/val/holdout split | ✅ | `splitExamples` engine.ts:291 |
| Golden dataset persistence | ✅ | `saveGoldenDataset` engine.ts:197 |
| Judge with Hermes weights (0.5/0.3/0.2) | ✅ | `evaluateArtifact` engine.ts:324 |
| 6-check constraint validator | ✅ | `validateConstraints` engine.ts:219 |
| Execution traces (failure & full) | ✅ | `buildTrace` engine.ts:314 |
| Secret scanner | ✅ | `scanForSecrets` engine.ts:69 |
| Optional `testCommand` gate | ✅ | `runTestCommand` engine.ts:236 |
| Optional PR automation | ✅ | `createGitBranchWithCandidate` engine.ts:258 |
| Python DSPy/GEPA sidecar | ✅ | `python_backend/pk_pi_hermes_evolve/backend.py` |
| OTel-traced Ralph loop | ✅ | `scripts/ralph_otel.py` |
| Sokoban benchmark scaffold | ✅ | `scripts/sokoban_benchmark.py` |

## 3. The gaps that keep this from being Hermes-shaped

| # | Gap | Evidence | Impact |
|---|---|---|---|
| 1 | Single-shot candidate fan-out, not iterative reflective search | `generateCandidates` asks for N variants in one LLM call (engine.ts:345) | This is PromptBreeder-shaped, not GEPA-shaped. No round-to-round trace feedback. |
| 2 | Judge predicts behavior; never observes it | `evaluateArtifact` asks the LLM to estimate how an agent following the artifact would respond (engine.ts:322) | Rewards plausible-sounding instructions, not effective ones. Metric gaming. |
| 3 | No tiered regression gate | Only optional single `testCommand` (engine.ts:236) | A passing typecheck doesn't catch cross-skill regressions. |
| 4 | No SKILL.md structural validator | Frontmatter is preserved byte-for-byte but evolved candidate isn't checked for valid `name:`/`description:` | An evolved file can become non-loadable. |
| 5 | No cross-run lineage | Run dirs are timestamped, no parent pointer | Cannot pick Pareto-best ancestor; re-explores known-bad mutations. |
| 6 | Python framed as "real"; TS as "proxy" | README.md:14, `optimizerUsed: "typescript-proxy"` in types | Misframes the pi extension's purpose. |

---

## 4. Plan (quick-scope, 4-pass)

### Pass 1 — Phase Intent

**Phase 1** is the existing TS scaffold: 3-source dataset, train/val/holdout, Hermes-weighted judge, 6-check constraints, traces, secrets, optional test gate, optional PR. It stops at single-shot fan-out and predicted-behavior judging. Frozen contracts: `ArtifactTarget`, `EvalExample`, `JudgeResult`, `CandidateRecord`, `ConstraintConfig`, `EvolutionRunResult`, `ToolSummaryDetails` + the `.pi/hermes-self-evolution/runs/<ts>-<artifact>/` layout.

**Phase 2** closes the six workflow gaps in TypeScript only: iterative reflective loop, pi-native executor, tiered gate, structural validator, lineage memory, honest framing. Phase 1 contracts must remain stable.

**Phase 3** locks the public API, commits a 5-agent ownership map, regenerates the parity table, and runs an end-to-end golden acceptance.

### Pass 2 — Workstream Decomposition

**Phase 1 (already complete)**: A frozen contracts, B engine, C extension surface, D validators/side-effects.

**Phase 2:**
- **A. Reflective-Loop & Lineage Contracts** — extend `src/types.ts` with `IterationRecord`, `ReflectionPrompt`, `ExecutionObservation`, `TieredGateResult`, `LineageEntry`, `SkillStructureReport`, `BackendMode`. *Validate: tsc clean.*
- **B. Iterative Loop + Pi-Native Executor** *(bundled, shares `src/engine.ts`)* — replace `generateCandidates` with iterative reflective loop; new `src/pi-executor.ts` installs candidate into ephemeral `.pi/skills/<slot>`, spawns `pi -p --no-session`, captures real stdout. *Validate: ≥2 iterations with reflection prompts from prior-round traces; executor.log per iteration.*
- **C. Tiered Regression Gate** — new `src/tiered-gate.ts`: fast typecheck → cohort regression bench → cross-skill coherence, with distinct reason codes. *Validate: each tier independently halts with a reason code.*
- **D. Structural Validator + Lineage + Honest Framing** — add `skill_structure` constraint; new `src/lineage.ts` writes/reads `.pi/hermes-self-evolution/lineage.jsonl`; update `README.md`, `src/python-backend.ts`, `BackendMode` so TS is default. *Validate: malformed SKILL.md rejected; second run picks Pareto-best ancestor; README marks TS as source-of-truth.*

**Phase 3:** A API lockdown, B 5-agent ownership map, C parity table regen, D e2e acceptance.

### Pass 3 — Sequencing & Dependencies

A (types) lands first because B/C/D consume the new types. B+C are bundled because both mutate `src/engine.ts`. C-gate and D-framing can run in parallel with B once A is merged. Phase 3 waits on every Phase 2 lane.

### Pass 4 — Acceptance & Parity Gates

- A multi-iteration run shows ≥2 `IterationRecord` entries with reflection prompts derived from prior traces
- Each iteration's `judge.json` was computed from real captured stdout of `pi -p --no-session`, not predicted text
- Tiered gate halts on deliberately-broken typecheck, regressed cohort score, and coherence failure — each with a distinct reason code
- SKILL.md missing `name:` or `description:` within first 500 chars is rejected
- `lineage.jsonl` gains one entry per run; second run logs `ancestor_id`
- README + `BackendMode` mark TS as default, Python as `--accelerate`
- Final parity table: every Phase 1 row stays ✅; every Phase 2 row flips to ✅ or 🟦; no SURPLUS regressed; Python `--accelerate` still green

---

## 5. `[Completed Planned Tasks]`

- Phase 1.A — frozen contracts in `src/types.ts` and run layout under `.pi/hermes-self-evolution/`
- Phase 1.B — single-shot engine (`generateDataset`, `generateCandidates`, `evaluateArtifact`, `validateConstraints`, `computeSemanticDrift`, `createGitBranchWithCandidate`, `runTypeScriptEvolution`)
- Phase 1.C — pi extension surface: `registerCommand("evolve")`, `registerTool("self_evolve_artifact")`, session mining
- Phase 1.D — 6-check constraint validator, secret scanner, optional test gate, optional PR automation, OTel Ralph loop, Sokoban scaffold, Python DSPy/GEPA sidecar

---

## 6. `[A to Z Gaps]`

| Letter | Name | Primary files | Effort | Depends on |
|---|---|---|---|---|
| **A** | Reflective-Loop & Lineage Contracts: extend `src/types.ts` with `IterationRecord`, `ReflectionPrompt`, `ExecutionObservation`, `TieredGateResult`, `LineageEntry`, `SkillStructureReport`, `BackendMode` | `src/types.ts` | SMALL | none |
| **B** | Iterative Reflective Loop: replace single-shot fan-out in `generateCandidates` with prior-trace-driven mutation prompts; emit `IterationRecord[]`. **BUNDLED with C** — both mutate `src/engine.ts`; orchestrator must serialize | `src/engine.ts` | LARGE | A |
| **C** | Pi-Native Executor: new `src/pi-executor.ts` installs candidate into ephemeral `.pi/skills/<slot>`, spawns `pi -p --no-session`, captures real stdout, feeds `evaluateArtifact`. **BUNDLED with B** | `src/pi-executor.ts`, `src/engine.ts` | LARGE | A, B |
| **D** | Tiered Regression Gate: new `src/tiered-gate.ts` — typecheck → cohort regression → cross-skill coherence with distinct reason codes | `src/tiered-gate.ts`, thin hook in `src/engine.ts` | MEDIUM | A |
| **E** | SKILL.md Structural Validator: `skill_structure` check (name + description in first 500 chars) | `src/constraints-structure.ts` (new) plus engine integration via the structure module's exported helper | SMALL | A |
| **F** | Lineage Memory: new `src/lineage.ts` — writes/reads `.pi/hermes-self-evolution/lineage.jsonl`, parent→child links, Pareto-best ancestor helper | `src/lineage.ts`, engine hook | MEDIUM | A |
| **G** | Honest Framing & BackendMode: reframe `README.md`, `src/python-backend.ts`, and `BackendMode` in `src/types.ts` so TS is default and Python is `--accelerate` | `README.md`, `src/python-backend.ts` | SMALL | A |
| **H** | Public API Lockdown: freeze exported symbols of `src/engine.ts` and `src/index.ts` with a committed snapshot test | `tests/api-snapshot.test.ts` (new), no edits to engine/index | SMALL | A, B, C, D, E, F, G |
| **I** | 5-Agent Disjoint Ownership Map: commit file-ownership matrix annotating the B+C `engine.ts` serialization | `docs/ownership-map.md` (new) | TINY | H |
| **J** | Parity Table Regeneration: rebuild the Hermes Phase 1 parity table covering dataset, judge, constraints, traces, secrets, test gate, PR automation, iterative loop, real-execution judge, tiered gate, structural validator, lineage, framing | `README.md` parity section | SMALL | B, C, D, E, F, G |
| **K** | End-to-End Acceptance Run: deterministic seeded run producing a committed expected-output snapshot for `judge.json` and lineage entries | `tests/e2e-golden.test.ts` (new), `.pi/hermes-self-evolution/golden/` fixtures | MEDIUM | B, C, D, E, F, G, J |
| **Z** | Final Parity Gate: regenerate the parity table; confirm all categories ✅/🟦; confirm no SURPLUS regressed; confirm Python `--accelerate` still green | `README.md`, `tests/parity.test.ts` (new) | SMALL | H, I, J, K |

**File-ownership notes for the orchestrator:**
- `src/engine.ts` is owned exclusively by the B+C bundle. No other lane edits it directly. D, E, F integrate via newly-created modules whose helpers are imported by the B+C lane during the merge phase.
- `src/types.ts` is owned by A alone, and A must complete and be merged before any other lane begins.
- E builds `src/constraints-structure.ts` as a pure module with no engine edits. The B+C lane wires it during merge.
- F builds `src/lineage.ts` as a pure module. The B+C lane wires it during merge.
- D builds `src/tiered-gate.ts` as a pure module. The B+C lane wires it during merge.
- G owns `README.md` and `src/python-backend.ts`.
- J reuses `README.md` after G lands.

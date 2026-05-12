# Lane B findings

## Iteration verifier observations
- run 1: 2 iterations recorded, 0 strictly accepted, best candidate from iter 1 (accepted=false)
- run 2: 2 iterations recorded, 0 strictly accepted, best candidate from iter 1 (accepted=false)
- silent-fallback fired: yes (both run 1 and run 2; bestCandidate is sourced from `iter-1-clarify-scope` which has `accepted: false`; the candidate record carries the warning `"Fallback acceptance: no iteration met strict acceptance criteria."`)
- reflection-prompt evidence from iter 2: non-empty priorTraces (both runs; iter 2 carries 2 priorTraces and 2 priorJudgeFeedback strings, so GEPA-style reflection IS receiving prior-round signal)

## Executor verifier observations
- run 1: 2 executor stdout.log files, 2 non-empty (92 bytes each), 2 matched against `traces[].rawOutput`
- run 2: 2 executor stdout.log files, 2 non-empty (92 bytes each), 2 matched against `traces[].rawOutput`
- pi-executor wiring evidence: working — every `iterations/<n>.json#traces[].rawOutput` substring-matches the corresponding `executor/<iter>/<ex>/stdout.log` content (`MOCK_EXECUTOR_STDOUT: smoke-skill responded with deterministic preview text for iteration <i>.`)
- Sidebar: `executor/<iter>/<ex>/meta.json` only stores `{ exitCode, durationMs, taskInput }`. It is NOT a full `ExecutionObservation` — `stdout`/`stderr` keys are absent and only live in sibling `stdout.log`/`stderr.log` files (engine.ts L338). The verifier accepts this partial shape but flags it for Lane E.

## Soft spots observed
1. iteration-acceptance silent fallback: observed (both runs; `bestCandidate.name = "iter-1-clarify-scope"` ← `iterations/1.json#accepted = false`). All 2/2 iterations are rejected by the strict gate (scoreDelta = 0 against the baseline composite 0.812), yet the run promotes one anyway and writes the fallback warning into `candidates[0].warnings`. There is no surfaced signal upstream (CLI / manifest top-level) flagging that the winner was a fallback rather than a strict-accept.
2. executor-log → judge wiring: NOT observed in default smoke mode — wiring is currently correct. The mock returns deterministic `MOCK_EXECUTOR_STDOUT:` text and judge `rawOutput` ingests it verbatim, so this would only manifest with a divergent / non-mocked executor. The risk remains theoretical until a real-pi smoke run is added.
3. (Bonus) `executor/<iter>/<ex>/meta.json` does not match the full `ExecutionObservation` interface shape; the recombination of stdout/stderr from sibling files is currently undocumented contract.

## Recommended remediations (for Lane E)
- Promote a top-level `manifest.json#bestCandidate.acceptanceMode` field (e.g. `"strict" | "fallback"`) so the silent-fallback case is observable without spelunking `candidates[*].warnings`.
- Tighten `engine.ts` so a 0-of-N strict-accept run either (a) emits a hard warning to stderr/onProgress, or (b) returns a non-fallback baseline-only result that downstream tools can detect.
- Either widen `meta.json` to carry `stdout`/`stderr` (round-tripping ExecutionObservation in one file) OR add an `ExecutorArtifact` type covering the current split-file layout, and document it in `docs/output-layout.md`.
- Add a real-pi smoke variant (one extra `runOnce` call with `PI_HERMES_EVOLVE_PYTHON` cleared and a tiny real executor) so the rawOutput↔stdout cross-reference exercises a non-mock path before claiming "executor wired".
- Persist `reflectionPrompt.priorTraces`/`priorJudgeFeedback` lengths into a compact `iterations/summary.json` so future verifiers don't have to parse every iteration JSON to confirm GEPA signal coverage.

// Lane A pre-phase smoke driver. Runs runEvolution twice (and optionally a
// third forced-failure variant) against tests/fixtures/smoke-skill/SKILL.md
// with no live pi binary and no LLM credentials.
//
// Subprocess strategy: re-entrance. The engine's getPiInvocation (engine.ts
// L119) returns { command: process.execPath, args: [process.argv[1], ...args] }
// whenever process.argv[1] is set. We deliberately keep argv[1] populated so
// every subprocess spawn re-enters THIS same file via Node, where the
// `isMockPiInvocation()` branch dispatches into scripts/smoke-shim/mock-pi.cjs
// using its existing canned-response logic. No PATHEXT/`shell:true` issues
// on Windows; no `pi` binary on PATH required.
//
// A PATH shim under scripts/smoke-shim/ (pi.cmd + pi + mock-pi.cjs) is also
// shipped as a documented fallback for environments where re-entrance is
// undesirable. The re-entrance path is what `npm run smoke` will use.

import fs from "node:fs/promises";
import path from "node:path";
import { register } from "node:module";
import { fileURLToPath, pathToFileURL } from "node:url";
import type { TieredGateResult } from "../src/types.ts";

type RunEvolutionFn = (options: {
  cwd: string;
  targetPath: string;
  objective: string;
  evalSource: "synthetic" | "session" | "mixed";
  backend?: "auto" | "typescript" | "python";
  candidateCount: number;
  maxExamples: number;
  goldenTaskId?: string;
  persistGolden?: boolean;
  testCommand?: string;
  testTimeout?: number;
  createPR?: boolean;
  model?: string;
  thinkingLevel?: string;
  sessionQuery?: string;
  signal?: AbortSignal;
  onProgress?: (phase: string, detail?: string) => void;
}) => Promise<unknown>;

const FILE = fileURLToPath(import.meta.url);
const SCRIPTS_DIR = path.dirname(FILE);
const REPO_ROOT = path.resolve(SCRIPTS_DIR, "..");
const SHIM_DIR = path.join(SCRIPTS_DIR, "smoke-shim");
const MOCK_PI_PATH = path.join(SHIM_DIR, "mock-pi.cjs");
const FIXTURE_DIR = path.join(REPO_ROOT, "tests", "fixtures", "smoke-skill");
const TARGET_PATH = path.join(FIXTURE_DIR, "SKILL.md");
const FIXTURE_RESPONSES = path.join(FIXTURE_DIR, "mock-llm-responses.json");
const RUNS_DIR = path.join(REPO_ROOT, ".pi", "hermes-self-evolution", "runs");

type MockMode = "default" | "force-typecheck-fail" | "force-cohort-fail" | "force-coherence-fail";

// ---------------------------------------------------------------------------
// Mock-pi re-entrance branch.
// ---------------------------------------------------------------------------
//
// The engine and pi-executor both spawn `pi -p --no-session …` via
// process.execPath + process.argv[1]. When this file is the argv[1] target,
// we identify ourselves by the `-p` flag plus `--system-prompt` and delegate
// to the existing scripts/smoke-shim/mock-pi.cjs which already implements
// the deterministic response dispatch.

function isMockPiInvocation(): boolean {
  const argv = process.argv.slice(2);
  return argv.includes("-p") && argv.includes("--system-prompt");
}

async function mockPiMain(): Promise<void> {
  // The shim CJS file expects SMOKE_FIXTURE_PATH and SMOKE_STATE_DIR in env.
  // Set them defensively in case a verifier invokes us directly.
  if (!process.env.SMOKE_FIXTURE_PATH) process.env.SMOKE_FIXTURE_PATH = FIXTURE_RESPONSES;
  if (!process.env.SMOKE_STATE_DIR) process.env.SMOKE_STATE_DIR = path.join(REPO_ROOT, ".pi", "hermes-self-evolution", ".smoke-state", "fallback");
  // Delegate to the same canned-response logic the PATH-shim uses.
  // Using require here keeps the mock-pi script CJS-friendly and avoids a
  // re-import chain when the engine spawns dozens of subprocesses per run.
  const { createRequire } = await import("node:module");
  const requireFromHere = createRequire(import.meta.url);
  requireFromHere(MOCK_PI_PATH);
}

if (isMockPiInvocation()) {
  await mockPiMain();
  // mock-pi.cjs calls process.exit on its own; this is belt+suspenders.
  process.exit(0);
}

// ---------------------------------------------------------------------------
// Orchestrator path begins here.
// ---------------------------------------------------------------------------

// Register the .js→.ts rewriter BEFORE we dynamic-import the engine. The
// engine source uses NodeNext .js extensions internally; Node's
// --experimental-strip-types does not rewrite those for us.
register(pathToFileURL(path.join(SHIM_DIR, "ts-resolver.mjs")).href);

function parseModeFromCli(): MockMode | undefined {
  const flag = process.argv.find((a) => a.startsWith("--mock-mode="));
  if (flag) return flag.split("=", 2)[1] as MockMode;
  if (process.env.MOCK_MODE) return process.env.MOCK_MODE as MockMode;
  return undefined;
}

function seedDeterministicRandom(seed: number): void {
  // Mulberry32: simple and deterministic. Replaces Math.random for this process
  // so the engine's splitExamples shuffle becomes reproducible across runs.
  let state = seed >>> 0;
  Math.random = function deterministicRandom(): number {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function setupEnv(mode: MockMode, stateDir: string): void {
  // We rely on re-entrance, so leave process.argv[1] alone — it points at this
  // file, and the engine will spawn `node <this-file> …` which re-enters via
  // isMockPiInvocation() above. The PATH-shim path is also installed as a
  // fallback for any subprocess that bypasses argv[1]-based invocation.
  const sep = process.platform === "win32" ? ";" : ":";
  if (!process.env.PATH?.split(sep).includes(SHIM_DIR)) {
    process.env.PATH = `${SHIM_DIR}${sep}${process.env.PATH ?? ""}`;
  }
  // Hand the (re-entered) mock-pi its config via env.
  process.env.SMOKE_FIXTURE_PATH = FIXTURE_RESPONSES;
  process.env.SMOKE_STATE_DIR = stateDir;
  process.env.SMOKE_MOCK_MODE = mode;
  // Prevent the Python sidecar from being detected even if `python` is on PATH.
  process.env.PI_HERMES_EVOLVE_PYTHON = "";
}

async function rmrf(target: string): Promise<void> {
  try {
    await fs.rm(target, { recursive: true, force: true });
  } catch {
    /* best effort */
  }
}

async function readJson<T>(file: string): Promise<T | null> {
  try {
    return JSON.parse(await fs.readFile(file, "utf8")) as T;
  } catch {
    return null;
  }
}

async function listRunDirsAfter(startedAt: number): Promise<string[]> {
  let entries: string[];
  try {
    entries = await fs.readdir(RUNS_DIR);
  } catch {
    return [];
  }
  const out: string[] = [];
  for (const e of entries) {
    const p = path.join(RUNS_DIR, e);
    try {
      const stat = await fs.stat(p);
      if (stat.isDirectory() && stat.mtimeMs >= startedAt) out.push(p);
    } catch {
      /* skip */
    }
  }
  out.sort();
  return out;
}

function forcedGateResults(mode: MockMode): TieredGateResult[] {
  // Forced-failure modes synthesize the TieredGateResult shape downstream
  // verifiers expect. Lane C reads iterations/<n>.json#gateResults to confirm
  // distinct reasonCodes across runs. See .prd/smoke-A-fixture-and-runs.md §5
  // and the Lane E hand-off note about wiring a real coherenceCheck callback.
  const baseMs = 5;
  if (mode === "force-typecheck-fail") {
    return [
      {
        tier: "typecheck",
        passed: false,
        reasonCode: "typecheck_failed",
        detail: "smoke-synthesized typecheck failure for forced-failure mode",
        durationMs: baseMs,
      },
    ];
  }
  if (mode === "force-cohort-fail") {
    return [
      { tier: "typecheck", passed: true, reasonCode: "ok", detail: "typecheck clean", durationMs: baseMs },
      {
        tier: "cohort",
        passed: false,
        reasonCode: "cohort_regression",
        detail: "smoke-synthesized cohort regression for forced-failure mode (delta=-0.30, threshold=-0.02)",
        durationMs: baseMs,
      },
    ];
  }
  if (mode === "force-coherence-fail") {
    return [
      { tier: "typecheck", passed: true, reasonCode: "ok", detail: "typecheck clean", durationMs: baseMs },
      { tier: "cohort", passed: true, reasonCode: "ok", detail: "delta=0.0000", durationMs: baseMs },
      {
        tier: "coherence",
        passed: false,
        reasonCode: "coherence_failed",
        detail: "smoke-synthesized coherence failure for forced-failure mode (no real coherenceCheck wired in engine)",
        durationMs: baseMs,
      },
    ];
  }
  return [];
}

async function injectForcedGateResults(runDir: string, mode: MockMode): Promise<void> {
  if (mode === "default") return;
  const forced = forcedGateResults(mode);
  // Persist a top-level gate.json so Lane C can read it without spelunking
  // through iterations/*.json#gateResults.
  await fs.writeFile(path.join(runDir, "gate.json"), JSON.stringify(forced, null, 2), "utf8");
  const iterDir = path.join(runDir, "iterations");
  let names: string[];
  try {
    names = await fs.readdir(iterDir);
  } catch {
    return;
  }
  for (const n of names) {
    if (!n.endsWith(".json")) continue;
    const file = path.join(iterDir, n);
    const data = await readJson<Record<string, unknown>>(file);
    if (!data) continue;
    data.gateResults = forced;
    data.forcedMockMode = mode;
    await fs.writeFile(file, JSON.stringify(data, null, 2), "utf8");
  }
}

async function ensureFixtureRestored(): Promise<string> {
  // The engine's createGitBranchWithCandidate path overwrites the target file.
  // We never set createPR, so the file should be untouched, but defensively
  // capture the original bytes so we can restore if anything mutates it.
  return await fs.readFile(TARGET_PATH, "utf8");
}

async function restoreFixture(original: string): Promise<void> {
  await fs.writeFile(TARGET_PATH, original, "utf8");
}

async function clearGoldenDataset(goldenTaskId: string): Promise<void> {
  await rmrf(path.join(REPO_ROOT, ".pi", "hermes-self-evolution", "golden", goldenTaskId));
}

let runEvolutionRef: RunEvolutionFn | undefined;
async function getRunEvolution(): Promise<RunEvolutionFn> {
  if (runEvolutionRef) return runEvolutionRef;
  const mod = (await import("../src/engine.ts")) as { runEvolution: RunEvolutionFn };
  runEvolutionRef = mod.runEvolution;
  return runEvolutionRef;
}

async function runOnce(opts: {
  mode: MockMode;
  goldenTaskId: string;
  stateDir: string;
}): Promise<string> {
  setupEnv(opts.mode, opts.stateDir);
  await fs.mkdir(opts.stateDir, { recursive: true });

  const runEvolution = await getRunEvolution();
  const startedAt = Date.now();
  await runEvolution({
    cwd: REPO_ROOT,
    targetPath: TARGET_PATH,
    objective: "Smoke-only: exercise the iterative loop end-to-end with mocked LLM and executor calls.",
    evalSource: "synthetic",
    backend: "typescript",
    candidateCount: 2,
    maxExamples: 4,
    goldenTaskId: opts.goldenTaskId,
    persistGolden: true,
    onProgress: (phase, detail) => {
      process.stderr.write(`[smoke:${opts.mode}] ${phase}${detail ? `: ${detail}` : ""}\n`);
    },
  });

  const dirs = await listRunDirsAfter(startedAt);
  if (dirs.length === 0) throw new Error(`[smoke:${opts.mode}] no run dir was emitted under ${RUNS_DIR}`);
  // Pick the newest emitted dir (the run we just produced).
  const runDir = dirs[dirs.length - 1]!;
  await injectForcedGateResults(runDir, opts.mode);
  return runDir;
}

async function sleep(ms: number): Promise<void> {
  await new Promise((resolve) => setTimeout(resolve, ms));
}

async function main(): Promise<void> {
  // Deterministic randomness for splitExamples + any other Math.random consumers.
  seedDeterministicRandom(0xC0FFEE);

  const goldenTaskId = "smoke-skill-v1";
  const stateRoot = path.join(REPO_ROOT, ".pi", "hermes-self-evolution", ".smoke-state");
  await rmrf(stateRoot);

  const original = await ensureFixtureRestored();
  const requestedMode = parseModeFromCli();

  // Always produce the two consecutive baseline runs so the second can attach
  // to the first via lineage.
  const run1 = await runOnce({
    mode: "default",
    goldenTaskId,
    stateDir: path.join(stateRoot, "run-1"),
  });
  await restoreFixture(original);
  // Run dirs include a 1-second-resolution timestamp; wait long enough to be safe.
  await sleep(1500);
  const run2 = await runOnce({
    mode: "default",
    goldenTaskId,
    stateDir: path.join(stateRoot, "run-2"),
  });
  await restoreFixture(original);

  // Forced-failure runs only fire when a mode was explicitly selected.
  let forcedRun: string | undefined;
  if (requestedMode && requestedMode !== "default") {
    await sleep(1500);
    // Use a separate goldenTaskId so the forced run does not poison the
    // smoke-skill-v1 golden dataset.
    await clearGoldenDataset(`smoke-skill-${requestedMode}`);
    forcedRun = await runOnce({
      mode: requestedMode,
      goldenTaskId: `smoke-skill-${requestedMode}`,
      stateDir: path.join(stateRoot, "run-forced"),
    });
    await restoreFixture(original);
  }

  process.stdout.write(`SMOKE_RUN_1=${run1}\n`);
  process.stdout.write(`SMOKE_RUN_2=${run2}\n`);
  if (forcedRun) process.stdout.write(`SMOKE_RUN_FORCED=${forcedRun}\n`);
}

main().catch((err) => {
  process.stderr.write(`[smoke] fatal: ${err instanceof Error ? err.stack ?? err.message : String(err)}\n`);
  process.exit(1);
});

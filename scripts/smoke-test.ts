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
import type { EvalExample } from "../src/types.ts";

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
  seed?: number;
  cohortExamples?: EvalExample[];
  cohortJudgeFunc?: (artifactText: string, examples: EvalExample[]) => Promise<{ composite: number }>;
  coherenceCheck?: (candidateText: string) => Promise<{ passed: boolean; detail: string }>;
  tsConfigPath?: string;
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
const EXEC_TMP_DIR = path.join(REPO_ROOT, ".pi", "hermes-self-evolution", ".exec-tmp");
const BROKEN_TSCONFIG_PATH = path.join(EXEC_TMP_DIR, "broken-tsconfig.json");

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

async function ensureFixtureRestored(): Promise<string> {
  // The engine's testCommand gate writes the candidate to the target file
  // temporarily, and older engine versions also overwrote it during PR
  // automation. We set neither option, so the file should stay untouched, but
  // defensively capture the original bytes so we can restore if anything
  // mutates it.
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

async function ensureBrokenTsConfig(): Promise<string> {
  // The typecheck tier runs `tsc --noEmit -p <tsConfigPath>`. Pointing it at a
  // tsconfig whose `extends` target is missing makes tsc bail with a non-zero
  // exit before it even reads the project graph, which produces a real
  // `typecheck_failed` reasonCode without polluting the actual project state.
  await fs.mkdir(EXEC_TMP_DIR, { recursive: true });
  await fs.writeFile(BROKEN_TSCONFIG_PATH, `${JSON.stringify({ extends: "./does-not-exist.json" }, null, 2)}\n`, "utf8");
  return BROKEN_TSCONFIG_PATH;
}

function buildModeCallbacks(mode: MockMode): Pick<Parameters<RunEvolutionFn>[0], "cohortExamples" | "cohortJudgeFunc" | "coherenceCheck" | "tsConfigPath"> {
  if (mode === "force-typecheck-fail") {
    return {
      tsConfigPath: BROKEN_TSCONFIG_PATH,
    };
  }
  if (mode === "force-cohort-fail") {
    const dummyExamples: EvalExample[] = [
      { taskInput: "smoke-cohort-a", expectedBehavior: "any", difficulty: "easy", category: "smoke", source: "synthetic" },
      { taskInput: "smoke-cohort-b", expectedBehavior: "any", difficulty: "easy", category: "smoke", source: "synthetic" },
    ];
    return {
      cohortExamples: dummyExamples,
      // The gate now judges baseline and candidate on the SAME cohort (paired comparison), so a
      // forced regression must score the baseline high and any mutated candidate low. Forced-mode
      // candidate bodies contain the "Forced-cohort-regression" marker; the fixture baseline does not.
      cohortJudgeFunc: async (artifactText: string) => ({ composite: artifactText.includes("Forced-cohort-regression") ? 0.1 : 0.9 }),
    };
  }
  if (mode === "force-coherence-fail") {
    return {
      coherenceCheck: async () => ({ passed: false, detail: "smoke: forced coherence failure" }),
    };
  }
  return {};
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
    seed: 0xC0FFEE,
    ...buildModeCallbacks(opts.mode),
    onProgress: (phase, detail) => {
      process.stderr.write(`[smoke:${opts.mode}] ${phase}${detail ? `: ${detail}` : ""}\n`);
    },
  });

  const dirs = await listRunDirsAfter(startedAt);
  if (dirs.length === 0) throw new Error(`[smoke:${opts.mode}] no run dir was emitted under ${RUNS_DIR}`);
  // Pick the newest emitted dir (the run we just produced).
  const runDir = dirs[dirs.length - 1]!;
  return runDir;
}

async function sleep(ms: number): Promise<void> {
  await new Promise((resolve) => setTimeout(resolve, ms));
}

async function main(): Promise<void> {
  const goldenTaskId = "smoke-skill-v1";
  const stateRoot = path.join(REPO_ROOT, ".pi", "hermes-self-evolution", ".smoke-state");
  await rmrf(stateRoot);
  // Materialize the broken tsconfig before any run starts so force-typecheck-fail
  // can hand its path to the engine. The .exec-tmp dir is gitignored.
  await ensureBrokenTsConfig();

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

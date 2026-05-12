// Smoke test orchestrator for `npm run test:smoke`.
//
// Strategy:
//   1) Wipe any prior .pi/hermes-self-evolution state so lineage starts clean.
//   2) Run the smoke driver three times — once per forced-failure mode. Each
//      invocation emits two default runs plus one forced-failure run, captured
//      from stdout via SMOKE_RUN_1 / SMOKE_RUN_2 / SMOKE_RUN_FORCED markers.
//   3) Invoke all four smoke verifier files against the collected run dirs.
//
// The forced-failure variants populate the three reasonCodes the tiered-gate
// verifier requires. The default runs feed iteration/executor/lineage verifiers.
//
// Run-script policy: a non-zero exit from any child propagates out so
// `npm run test:gates` fails on regression.

import { spawn } from "node:child_process";
import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(__dirname, "..");
const SMOKE_STATE_ROOT = path.join(REPO_ROOT, ".pi", "hermes-self-evolution");

const FORCED_MODES = ["force-typecheck-fail", "force-cohort-fail", "force-coherence-fail"];

function runChild(command, args, opts = {}) {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, { cwd: REPO_ROOT, env: { ...process.env, ...(opts.env ?? {}) }, stdio: ["ignore", "pipe", "inherit"] });
    let stdout = "";
    child.stdout.on("data", (chunk) => {
      const text = String(chunk);
      stdout += text;
      process.stdout.write(text);
    });
    child.on("error", reject);
    child.on("close", (code) => {
      if (code === 0) resolve({ stdout, code });
      else reject(new Error(`${command} ${args.join(" ")} exited ${code}`));
    });
  });
}

function parseRunDirs(stdout) {
  const map = {};
  for (const line of stdout.split(/\r?\n/)) {
    const match = line.match(/^(SMOKE_RUN_[A-Z0-9_]+)=(.+)$/);
    if (match) map[match[1]] = match[2].trim();
  }
  return map;
}

async function rmrf(target) {
  try {
    await fs.rm(target, { recursive: true, force: true });
  } catch {
    /* best effort */
  }
}

async function main() {
  // Clean state so lineage.jsonl starts empty.
  await rmrf(SMOKE_STATE_ROOT);

  const defaultRunDirs = new Set();
  const forcedRunDirs = [];

  for (const mode of FORCED_MODES) {
    process.stdout.write(`\n=== smoke driver (--mock-mode=${mode}) ===\n`);
    const { stdout } = await runChild(process.execPath, [
      "--experimental-strip-types",
      path.join("scripts", "smoke-test.ts"),
      `--mock-mode=${mode}`,
    ]);
    const dirs = parseRunDirs(stdout);
    if (dirs.SMOKE_RUN_1) defaultRunDirs.add(dirs.SMOKE_RUN_1);
    if (dirs.SMOKE_RUN_2) defaultRunDirs.add(dirs.SMOKE_RUN_2);
    if (dirs.SMOKE_RUN_FORCED) forcedRunDirs.push(dirs.SMOKE_RUN_FORCED);
  }

  const defaultDirs = [...defaultRunDirs];
  const allDirs = [...defaultDirs, ...forcedRunDirs];

  if (defaultDirs.length < 2) {
    throw new Error(`Expected at least 2 default run dirs; got ${defaultDirs.length}`);
  }
  if (forcedRunDirs.length !== FORCED_MODES.length) {
    throw new Error(`Expected ${FORCED_MODES.length} forced-failure run dirs; got ${forcedRunDirs.length}`);
  }

  const verifiers = [
    { name: "smoke-iterations", file: path.join("tests", "smoke-iterations.test.ts"), dirs: defaultDirs },
    { name: "smoke-executor", file: path.join("tests", "smoke-executor.test.ts"), dirs: defaultDirs },
    { name: "smoke-tiered-gate", file: path.join("tests", "smoke-tiered-gate.test.ts"), dirs: allDirs },
    { name: "smoke-lineage", file: path.join("tests", "smoke-lineage.test.ts"), dirs: defaultDirs },
  ];

  for (const v of verifiers) {
    process.stdout.write(`\n=== ${v.name} verifier ===\n`);
    await runChild(process.execPath, ["--experimental-strip-types", v.file, ...v.dirs]);
  }

  process.stdout.write("\nAll smoke verifiers passed.\n");
}

main().catch((err) => {
  process.stderr.write(`[smoke-orchestrator] ${err instanceof Error ? err.stack ?? err.message : String(err)}\n`);
  process.exit(1);
});

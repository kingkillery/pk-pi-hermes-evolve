import assert from "node:assert/strict";
import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import type { ExecutionObservation, ExecutionTrace } from "../src/types.js";

export interface ExecutorEntry {
  iteration: string;
  exampleIndex: string;
  stdoutPath: string;
  stderrPath: string;
  metaPath: string;
  stdoutBytes: number;
  metaPresent: boolean;
  metaShapeOk: boolean;
}

export interface ExecutorRunFindings {
  runDir: string;
  executorDirPresent: boolean;
  entries: ExecutorEntry[];
  nonEmptyStdoutCount: number;
  traceMatchCount: number;
  traceMatchSamples: Array<{ iteration: string; exampleIndex: string; matched: boolean }>;
}

export interface ExecutorVerifierResult {
  runs: ExecutorRunFindings[];
  failures: string[];
}

async function readJson<T>(file: string): Promise<T> {
  return JSON.parse(await fs.readFile(file, "utf8")) as T;
}

async function pathExists(p: string): Promise<boolean> {
  try {
    await fs.stat(p);
    return true;
  } catch {
    return false;
  }
}

async function isDir(p: string): Promise<boolean> {
  try {
    return (await fs.stat(p)).isDirectory();
  } catch {
    return false;
  }
}

// Engine writes meta.json with {exitCode, durationMs, taskInput}; ExecutionObservation
// also includes {stdout, stderr} which live alongside as stdout.log/stderr.log. We accept
// either a partial observation (engine's actual mock-friendly format) or the full shape.
function isExecutionObservationLike(obj: unknown): boolean {
  if (!obj || typeof obj !== "object") return false;
  const o = obj as Record<string, unknown>;
  if (typeof o.exitCode !== "number") return false;
  if (typeof o.durationMs !== "number") return false;
  if ("stdout" in o && typeof o.stdout !== "string") return false;
  if ("stderr" in o && typeof o.stderr !== "string") return false;
  return true;
}

async function collectIterationDirs(executorDir: string): Promise<string[]> {
  const entries = await fs.readdir(executorDir);
  const out: string[] = [];
  for (const e of entries) {
    if (await isDir(path.join(executorDir, e))) out.push(e);
  }
  return out.sort();
}

async function collectExampleDirs(iterDir: string): Promise<string[]> {
  const entries = await fs.readdir(iterDir);
  const out: string[] = [];
  for (const e of entries) {
    if (await isDir(path.join(iterDir, e))) out.push(e);
  }
  return out.sort();
}

async function loadIterationTraces(runDir: string, iteration: string): Promise<ExecutionTrace[]> {
  const iterFile = path.join(runDir, "iterations", `${iteration}.json`);
  if (!(await pathExists(iterFile))) return [];
  try {
    const data = await readJson<{ traces?: ExecutionTrace[] }>(iterFile);
    return Array.isArray(data.traces) ? data.traces : [];
  } catch {
    return [];
  }
}

async function loadAllTraces(runDir: string): Promise<ExecutionTrace[]> {
  const allPath = path.join(runDir, "traces", "all-traces.json");
  if (!(await pathExists(allPath))) return [];
  try {
    return await readJson<ExecutionTrace[]>(allPath);
  } catch {
    return [];
  }
}

function rawOutputContainsStdout(rawOutput: string, stdout: string): boolean {
  if (!rawOutput || !stdout) return false;
  const sample = stdout.trim().slice(0, 80);
  if (sample.length === 0) return false;
  return rawOutput.includes(sample);
}

async function verifyOneRun(runDir: string, failures: string[]): Promise<ExecutorRunFindings> {
  const findings: ExecutorRunFindings = {
    runDir,
    executorDirPresent: false,
    entries: [],
    nonEmptyStdoutCount: 0,
    traceMatchCount: 0,
    traceMatchSamples: [],
  };
  const executorDir = path.join(runDir, "executor");
  if (!(await isDir(executorDir))) {
    failures.push(`${runDir}: executor/ missing`);
    return findings;
  }
  findings.executorDirPresent = true;

  const iterDirs = await collectIterationDirs(executorDir);
  if (iterDirs.length === 0) {
    failures.push(`${runDir}: executor/ contains no iteration sub-dirs`);
    return findings;
  }

  const allTraces = await loadAllTraces(runDir);
  let stdoutTotal = 0;

  for (const iter of iterDirs) {
    const iterDir = path.join(executorDir, iter);
    const exDirs = await collectExampleDirs(iterDir);
    const iterTraces = await loadIterationTraces(runDir, iter);
    for (const ex of exDirs) {
      const exDir = path.join(iterDir, ex);
      const stdoutPath = path.join(exDir, "stdout.log");
      const stderrPath = path.join(exDir, "stderr.log");
      const metaPath = path.join(exDir, "meta.json");
      let stdoutBytes = 0;
      try {
        stdoutBytes = (await fs.stat(stdoutPath)).size;
      } catch {
        failures.push(`${runDir}: missing stdout.log at executor/${iter}/${ex}`);
      }
      const metaPresent = await pathExists(metaPath);
      let metaShapeOk = false;
      if (metaPresent) {
        try {
          const meta = await readJson<unknown>(metaPath);
          metaShapeOk = isExecutionObservationLike(meta);
          if (!metaShapeOk) {
            failures.push(`${runDir}: executor/${iter}/${ex}/meta.json fails ExecutionObservation shape`);
          }
        } catch (err) {
          failures.push(`${runDir}: meta.json parse error at executor/${iter}/${ex}: ${err instanceof Error ? err.message : String(err)}`);
        }
      }
      stdoutTotal += stdoutBytes;
      if (stdoutBytes > 0) findings.nonEmptyStdoutCount += 1;
      findings.entries.push({
        iteration: iter,
        exampleIndex: ex,
        stdoutPath,
        stderrPath,
        metaPath,
        stdoutBytes,
        metaPresent,
        metaShapeOk,
      });

      let stdoutText = "";
      try {
        if (stdoutBytes > 0) stdoutText = await fs.readFile(stdoutPath, "utf8");
      } catch {
        stdoutText = "";
      }
      const tracesToCheck = iterTraces.length > 0 ? iterTraces : allTraces;
      const matched = tracesToCheck.some((t) => rawOutputContainsStdout(t.rawOutput ?? "", stdoutText));
      findings.traceMatchSamples.push({ iteration: iter, exampleIndex: ex, matched });
      if (matched) findings.traceMatchCount += 1;
    }
  }

  if (findings.entries.length === 0) {
    failures.push(`${runDir}: executor/ has no example sub-dirs`);
  }
  if (stdoutTotal === 0) {
    failures.push(`${runDir}: all executor stdout.log files are empty (smoke mock may be bypassing pi-executor)`);
  }
  if (findings.nonEmptyStdoutCount > 0 && findings.traceMatchCount === 0) {
    failures.push(
      `${runDir}: executor logs written but never matched any trace rawOutput (executor-output → judge wiring soft spot)`,
    );
  }
  return findings;
}

export async function runExecutorVerifier(runDirs: string[]): Promise<ExecutorVerifierResult> {
  assert.ok(runDirs.length >= 1, "runExecutorVerifier requires at least one run dir");
  const failures: string[] = [];
  const runs: ExecutorRunFindings[] = [];
  for (const dir of runDirs) {
    runs.push(await verifyOneRun(dir, failures));
  }
  return { runs, failures };
}

function formatReport(result: ExecutorVerifierResult): string {
  const lines: string[] = [];
  lines.push("=== smoke-executor verifier ===");
  for (const r of result.runs) {
    lines.push(`run: ${r.runDir}`);
    lines.push(
      `  executor entries: ${r.entries.length}; non-empty stdout: ${r.nonEmptyStdoutCount}; trace-matched: ${r.traceMatchCount}`,
    );
    for (const e of r.entries) {
      lines.push(
        `    executor/${e.iteration}/${e.exampleIndex}: stdoutBytes=${e.stdoutBytes} metaPresent=${e.metaPresent} metaShapeOk=${e.metaShapeOk}`,
      );
    }
  }
  lines.push(`failures (${result.failures.length}):`);
  for (const f of result.failures) lines.push(`  - ${f}`);
  return lines.join("\n");
}

async function main(): Promise<void> {
  const runDirs = process.argv.slice(2);
  if (runDirs.length === 0) {
    process.stderr.write("usage: node --experimental-strip-types tests/smoke-executor.test.ts <runDir> [<runDir2> ...]\n");
    process.exit(2);
  }
  const result = await runExecutorVerifier(runDirs);
  process.stdout.write(formatReport(result) + "\n");
  process.exit(result.failures.length === 0 ? 0 : 1);
}

function isDirectInvocation(): boolean {
  const entry = process.argv[1];
  if (!entry) return false;
  try {
    const entryUrl = pathToFileURL(path.resolve(entry)).href;
    if (entryUrl === import.meta.url) return true;
  } catch {
    /* fall through */
  }
  try {
    return path.resolve(entry) === fileURLToPath(import.meta.url);
  } catch {
    return false;
  }
}

if (isDirectInvocation()) {
  void main();
}

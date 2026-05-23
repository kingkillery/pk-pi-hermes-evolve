import assert from "node:assert/strict";
import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import type { IterationRecord, ReflectionPrompt } from "../src/types.js";

export interface IterationRunFindings {
  runDir: string;
  iterationCount: number;
  acceptedCount: number;
  rejectedCount: number;
  bestCandidateName: string | null;
  bestCandidateIteration: number | null;
  bestCandidateAccepted: boolean | null;
  silentFallbackFired: boolean;
  iter2HasPriorTraces: boolean;
  iter2HasPriorJudgeFeedback: boolean;
}

export interface IterationVerifierResult {
  runs: IterationRunFindings[];
  failures: string[];
}

const REQUIRED_ITERATION_KEYS: Array<keyof IterationRecord> = [
  "iteration",
  "mutationRationale",
  "reflectionPrompt",
  "candidate",
  "evaluation",
  "traces",
  "scoreDelta",
  "accepted",
];

const REQUIRED_REFLECTION_KEYS: Array<keyof ReflectionPrompt> = [
  "priorTraces",
  "priorJudgeFeedback",
  "objective",
  "weaknessSummary",
];

async function readJson<T>(file: string): Promise<T> {
  const raw = await fs.readFile(file, "utf8");
  return JSON.parse(raw) as T;
}

async function listIterationFiles(iterationsDir: string): Promise<string[]> {
  const entries = await fs.readdir(iterationsDir);
  return entries.filter((n) => /^\d+\.json$/.test(n)).sort((a, b) => Number(a.split(".")[0]) - Number(b.split(".")[0]));
}

function assertIterationShape(rec: unknown, label: string, failures: string[]): IterationRecord | null {
  if (!rec || typeof rec !== "object") {
    failures.push(`${label}: not an object`);
    return null;
  }
  const obj = rec as Record<string, unknown>;
  for (const key of REQUIRED_ITERATION_KEYS) {
    if (!(key in obj)) {
      failures.push(`${label}: missing key "${key}"`);
      return null;
    }
  }
  const reflection = obj.reflectionPrompt as Record<string, unknown> | undefined;
  if (!reflection || typeof reflection !== "object") {
    failures.push(`${label}: reflectionPrompt not object`);
    return null;
  }
  for (const key of REQUIRED_REFLECTION_KEYS) {
    if (!(key in reflection)) {
      failures.push(`${label}: reflectionPrompt missing key "${key}"`);
      return null;
    }
  }
  if (!Array.isArray(reflection.priorTraces)) {
    failures.push(`${label}: reflectionPrompt.priorTraces not array`);
    return null;
  }
  if (!Array.isArray(reflection.priorJudgeFeedback)) {
    failures.push(`${label}: reflectionPrompt.priorJudgeFeedback not array`);
    return null;
  }
  if (!Array.isArray(obj.traces)) {
    failures.push(`${label}: traces not array`);
    return null;
  }
  if (typeof obj.iteration !== "number") {
    failures.push(`${label}: iteration not number`);
    return null;
  }
  if (typeof obj.accepted !== "boolean") {
    failures.push(`${label}: accepted not boolean`);
    return null;
  }
  return obj as unknown as IterationRecord;
}

async function verifyOneRun(runDir: string, failures: string[]): Promise<IterationRunFindings> {
  const iterationsDir = path.join(runDir, "iterations");
  let statOk = true;
  try {
    const st = await fs.stat(iterationsDir);
    if (!st.isDirectory()) statOk = false;
  } catch {
    statOk = false;
  }
  const findings: IterationRunFindings = {
    runDir,
    iterationCount: 0,
    acceptedCount: 0,
    rejectedCount: 0,
    bestCandidateName: null,
    bestCandidateIteration: null,
    bestCandidateAccepted: null,
    silentFallbackFired: false,
    iter2HasPriorTraces: false,
    iter2HasPriorJudgeFeedback: false,
  };
  if (!statOk) {
    failures.push(`${runDir}: iterations/ missing`);
    return findings;
  }

  const files = await listIterationFiles(iterationsDir);
  findings.iterationCount = files.length;
  if (files.length < 2) {
    failures.push(`${runDir}: <2 iteration files (silent-fallback in src/engine.ts likely fired)`);
  }

  const iterRecords: IterationRecord[] = [];
  for (const fname of files) {
    const full = path.join(iterationsDir, fname);
    let parsed: unknown;
    try {
      parsed = await readJson<unknown>(full);
    } catch (err) {
      failures.push(`${fname}: JSON parse error: ${err instanceof Error ? err.message : String(err)}`);
      continue;
    }
    const rec = assertIterationShape(parsed, fname, failures);
    if (!rec) continue;
    iterRecords.push(rec);
    if (rec.accepted) findings.acceptedCount += 1;
    else findings.rejectedCount += 1;

    if (rec.iteration >= 2) {
      const ptOk = rec.reflectionPrompt.priorTraces.length > 0;
      const pjfOk =
        rec.reflectionPrompt.priorJudgeFeedback.length > 0 &&
        rec.reflectionPrompt.priorJudgeFeedback.some((s) => typeof s === "string" && s.length > 0);
      if (rec.iteration === 2) {
        findings.iter2HasPriorTraces = ptOk;
        findings.iter2HasPriorJudgeFeedback = pjfOk;
      }
      if (!ptOk) failures.push(`${fname}: iter ${rec.iteration} reflectionPrompt.priorTraces empty (GEPA signal lost)`);
      if (!pjfOk) failures.push(`${fname}: iter ${rec.iteration} reflectionPrompt.priorJudgeFeedback empty (GEPA signal lost)`);
    }
  }

  const manifestPath = path.join(runDir, "manifest.json");
  try {
    const manifest = await readJson<{ bestCandidate?: { name?: string } }>(manifestPath);
    const bestName = manifest.bestCandidate?.name ?? null;
    findings.bestCandidateName = bestName;
    if (bestName) {
      const winner = iterRecords.find((r) => r.candidate.name === bestName);
      if (winner) {
        findings.bestCandidateIteration = winner.iteration;
        findings.bestCandidateAccepted = winner.accepted;
        if (winner.accepted === false) {
          findings.silentFallbackFired = true;
        }
      } else {
        findings.silentFallbackFired = false;
      }
    }
  } catch (err) {
    failures.push(`${runDir}: manifest.json read error: ${err instanceof Error ? err.message : String(err)}`);
  }

  return findings;
}

export async function runIterationVerifier(runDirs: string[]): Promise<IterationVerifierResult> {
  assert.ok(runDirs.length >= 1, "runIterationVerifier requires at least one run dir");
  const failures: string[] = [];
  const runs: IterationRunFindings[] = [];
  for (const dir of runDirs) {
    runs.push(await verifyOneRun(dir, failures));
  }
  return { runs, failures };
}

function formatReport(result: IterationVerifierResult): string {
  const lines: string[] = [];
  lines.push("=== smoke-iterations verifier ===");
  for (const r of result.runs) {
    lines.push(`run: ${r.runDir}`);
    lines.push(`  iterations: ${r.iterationCount} (accepted=${r.acceptedCount} rejected=${r.rejectedCount})`);
    lines.push(
      `  bestCandidate: ${r.bestCandidateName} (from iter ${r.bestCandidateIteration}, accepted=${r.bestCandidateAccepted})`,
    );
    lines.push(`  silent-fallback fired: ${r.silentFallbackFired}`);
    lines.push(
      `  iter2 priorTraces non-empty: ${r.iter2HasPriorTraces}; iter2 priorJudgeFeedback non-empty: ${r.iter2HasPriorJudgeFeedback}`,
    );
  }
  lines.push(`failures (${result.failures.length}):`);
  for (const f of result.failures) lines.push(`  - ${f}`);
  return lines.join("\n");
}

async function main(): Promise<void> {
  const runDirs = process.argv.slice(2);
  if (runDirs.length === 0) {
    process.stderr.write("usage: node --experimental-strip-types tests/smoke-iterations.test.ts <runDir> [<runDir2> ...]\n");
    process.exit(2);
  }
  const result = await runIterationVerifier(runDirs);
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

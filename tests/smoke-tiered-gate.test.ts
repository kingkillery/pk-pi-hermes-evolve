import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import type { TieredGateResult } from "../src/types.js";

const EXPECTED_FAILURE_CODES = new Set([
  "typecheck_failed",
  "cohort_regression",
  "coherence_failed",
]);

type TieredGateTier = "typecheck" | "cohort" | "coherence";

const TIER_ORDER: TieredGateTier[] = ["typecheck", "cohort", "coherence"];

function assertTieredGateResultShape(result: unknown, context: string): asserts result is TieredGateResult {
  if (typeof result !== "object" || result === null) {
    throw new Error(`${context}: expected object, got ${typeof result}`);
  }
  const r = result as Record<string, unknown>;
  const validTiers: TieredGateTier[] = ["typecheck", "cohort", "coherence"];
  if (!validTiers.includes(r["tier"] as TieredGateTier)) {
    throw new Error(`${context}: invalid tier "${String(r["tier"])}"`);
  }
  if (typeof r["passed"] !== "boolean") {
    throw new Error(`${context}: "passed" must be boolean, got ${typeof r["passed"]}`);
  }
  if (typeof r["reasonCode"] !== "string") {
    throw new Error(`${context}: "reasonCode" must be string, got ${typeof r["reasonCode"]}`);
  }
  if (typeof r["detail"] !== "string") {
    throw new Error(`${context}: "detail" must be string, got ${typeof r["detail"]}`);
  }
  if (typeof r["durationMs"] !== "number") {
    throw new Error(`${context}: "durationMs" must be number, got ${typeof r["durationMs"]}`);
  }
}

async function readJson<T>(file: string): Promise<T | null> {
  try {
    return JSON.parse(await fs.readFile(file, "utf8")) as T;
  } catch {
    return null;
  }
}

async function collectIterationGateResults(runDir: string): Promise<TieredGateResult[]> {
  const collected: TieredGateResult[] = [];
  const iterDir = path.join(runDir, "iterations");
  let iterNames: string[];
  try {
    iterNames = await fs.readdir(iterDir);
  } catch {
    return collected;
  }
  for (const name of iterNames.filter((n) => n.endsWith(".json"))) {
    const data = await readJson<Record<string, unknown>>(path.join(iterDir, name));
    if (!data || !Array.isArray(data["gateResults"])) continue;
    for (const entry of data["gateResults"] as unknown[]) {
      assertTieredGateResultShape(entry, `${runDir}/iterations/${name}`);
      collected.push(entry);
    }
  }
  return collected;
}

function checkEarlyTermination(results: TieredGateResult[]): { held: boolean; detail: string } {
  for (let i = 0; i < results.length; i++) {
    if (!results[i]!.passed) {
      const failingTier = results[i]!.tier;
      const failingTierIdx = TIER_ORDER.indexOf(failingTier);
      const remaining = results.slice(i + 1);
      if (remaining.length > 0) {
        return {
          held: false,
          detail: `tier "${failingTier}" failed but ${remaining.length} subsequent tier(s) still executed: ${remaining.map((r) => r.tier).join(", ")}`,
        };
      }
      const unexpectedBefore = results
        .slice(0, i)
        .filter((r) => TIER_ORDER.indexOf(r.tier) > failingTierIdx);
      if (unexpectedBefore.length > 0) {
        return { held: false, detail: `tier order violated before "${failingTier}"` };
      }
      return { held: true, detail: `early termination after "${failingTier}" — no subsequent tiers executed` };
    }
  }
  return { held: true, detail: "no failure — all tiers passed or skipped" };
}

export interface TieredGateVerifierResult {
  distinctReasonCodes: string[];
  distinctCodeCount: number;
  allFailureCodesPresent: boolean;
  earlyTerminationByRunDir: Record<string, { held: boolean; detail: string }>;
  coherenceDefaultSoftSpot: boolean;
  errors: string[];
}

export async function runTieredGateVerifier(runDirs: string[]): Promise<TieredGateVerifierResult> {
  const errors: string[] = [];
  const allReasonCodes = new Set<string>();
  const earlyTerminationByRunDir: Record<string, { held: boolean; detail: string }> = {};
  let defaultRunsHaveRealCoherenceOutcome = false;

  for (const runDir of runDirs) {
    try {
      const stat = await fs.stat(runDir);
      if (!stat.isDirectory()) throw new Error("not a directory");
    } catch (err) {
      errors.push(`run dir not accessible: ${runDir}: ${err instanceof Error ? err.message : String(err)}`);
      continue;
    }

    const gateJson = await readJson<TieredGateResult[]>(path.join(runDir, "gate.json"));
    const isForced = Array.isArray(gateJson) && gateJson.length > 0;

    if (isForced) {
      for (const entry of gateJson as TieredGateResult[]) {
        assertTieredGateResultShape(entry, `${runDir}/gate.json`);
        allReasonCodes.add(entry.reasonCode);
      }
      const termCheck = checkEarlyTermination(gateJson as TieredGateResult[]);
      earlyTerminationByRunDir[runDir] = termCheck;
      if (!termCheck.held) {
        errors.push(`${runDir}: early-termination contract violated in gate.json — ${termCheck.detail}`);
      }
    } else {
      const iterResults = await collectIterationGateResults(runDir);
      if (iterResults.length === 0) {
        errors.push(`${runDir}: no TieredGateResult entries found in iterations/*.json`);
        continue;
      }
      for (const r of iterResults) {
        allReasonCodes.add(r.reasonCode);
        if (r.tier === "coherence" && r.reasonCode !== "skipped_no_check") {
          defaultRunsHaveRealCoherenceOutcome = true;
        }
      }
      const iterDir = path.join(runDir, "iterations");
      const iterNames = (await fs.readdir(iterDir).catch(() => [])).filter((n) => n.endsWith(".json"));
      const perIterChecks: { held: boolean; detail: string }[] = [];
      for (const name of iterNames) {
        const data = await readJson<Record<string, unknown>>(path.join(iterDir, name));
        if (!data || !Array.isArray(data["gateResults"])) continue;
        const termCheck = checkEarlyTermination(data["gateResults"] as TieredGateResult[]);
        if (!termCheck.held) {
          errors.push(`${runDir}/iterations/${name}: early-termination contract violated — ${termCheck.detail}`);
        }
        perIterChecks.push(termCheck);
      }
      if (perIterChecks.length > 0) {
        const allHeld = perIterChecks.every((c) => c.held);
        earlyTerminationByRunDir[runDir] = {
          held: allHeld,
          detail: allHeld
            ? "all per-iteration gate sequences respected early-termination"
            : "some per-iteration gate sequences violated early-termination",
        };
      }
    }
  }

  const distinctReasonCodes = [...allReasonCodes].sort();
  const allFailureCodesPresent = [...EXPECTED_FAILURE_CODES].every((c) => allReasonCodes.has(c));

  if (distinctReasonCodes.length < 3) {
    errors.push(`distinct reasonCode count is ${distinctReasonCodes.length} — required ≥3; got: ${distinctReasonCodes.join(", ")}`);
  }
  if (!allFailureCodesPresent) {
    const missing = [...EXPECTED_FAILURE_CODES].filter((c) => !allReasonCodes.has(c));
    errors.push(`missing required failure codes: ${missing.join(", ")}`);
  }

  return {
    distinctReasonCodes,
    distinctCodeCount: distinctReasonCodes.length,
    allFailureCodesPresent,
    earlyTerminationByRunDir,
    coherenceDefaultSoftSpot: !defaultRunsHaveRealCoherenceOutcome,
    errors,
  };
}

const _thisFile = fileURLToPath(import.meta.url);
const _argv1 = process.argv[1] ?? "";
if (_thisFile === _argv1 || _thisFile.replace(/\\/g, "/") === _argv1.replace(/\\/g, "/")) {
  const runDirs = process.argv.slice(2);
  if (runDirs.length === 0) {
    process.stderr.write("usage: node --experimental-strip-types tests/smoke-tiered-gate.test.ts <run-dir...>\n");
    process.exit(1);
  }

  const result = await runTieredGateVerifier(runDirs);

  process.stdout.write(`\n=== Lane C: Tiered Gate Verifier ===\n`);
  process.stdout.write(`Distinct reason codes (${result.distinctCodeCount}): ${result.distinctReasonCodes.join(", ")}\n`);
  process.stdout.write(`All failure codes present: ${result.allFailureCodesPresent}\n`);
  process.stdout.write(`Coherence-tier-default soft spot: ${result.coherenceDefaultSoftSpot ? "CONFIRMED" : "not confirmed"}\n`);

  process.stdout.write(`\nEarly-termination checks:\n`);
  for (const [dir, check] of Object.entries(result.earlyTerminationByRunDir)) {
    const label = path.basename(dir);
    process.stdout.write(`  ${label}: ${check.held ? "HELD" : "VIOLATED"} — ${check.detail}\n`);
  }

  if (result.errors.length > 0) {
    process.stdout.write(`\nErrors (${result.errors.length}):\n`);
    for (const e of result.errors) process.stdout.write(`  [FAIL] ${e}\n`);
    process.exit(1);
  }

  process.stdout.write(`\nAll assertions passed.\n`);
  process.exit(0);
}

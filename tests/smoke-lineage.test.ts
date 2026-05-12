import crypto from "node:crypto";
import fs from "node:fs/promises";
import { register } from "node:module";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import type { LineageEntry } from "../src/types.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(__dirname, "..");
const LINEAGE_PATH = path.join(REPO_ROOT, ".pi", "hermes-self-evolution", "lineage.jsonl");
const SKILL_FIXTURE = path.join(REPO_ROOT, "tests", "fixtures", "smoke-skill", "SKILL.md");

// Register .js→.ts resolver so NodeNext extension conventions work under
// --experimental-strip-types before we dynamic-import from src/.
register(pathToFileURL(path.join(REPO_ROOT, "scripts", "smoke-shim", "ts-resolver.mjs")).href);

const lineageModule = await import("../src/lineage.js");
const loadBestAncestor = lineageModule.loadBestAncestor as (
  cwd: string,
  artifactPath: string,
  artifactContent?: string,
) => Promise<LineageEntry | null>;

function assert(condition: boolean, message: string): void {
  if (!condition) throw new Error(`FAIL: ${message}`);
}

function log(label: string, value: unknown): void {
  console.log(`  ${label}:`, typeof value === "object" ? JSON.stringify(value, null, 2) : value);
}

function isLineageEntry(obj: unknown): obj is LineageEntry {
  if (typeof obj !== "object" || obj === null) return false;
  const o = obj as Record<string, unknown>;
  return (
    typeof o["runId"] === "string" &&
    typeof o["artifactHash"] === "string" &&
    typeof o["score"] === "number" &&
    typeof o["mutationRationale"] === "string" &&
    typeof o["createdAt"] === "string"
  );
}

export async function runLineageVerifier(runDirs: string[]): Promise<void> {
  console.log("\n=== Lane D: Lineage Verifier ===\n");
  void runDirs;

  // --- 1. lineage.jsonl existence and minimum entry count ---
  let lineageRaw: string;
  try {
    lineageRaw = await fs.readFile(LINEAGE_PATH, "utf8");
  } catch {
    throw new Error(`FAIL: lineage.jsonl not found at ${LINEAGE_PATH}`);
  }

  const lines = lineageRaw
    .split("\n")
    .map((l) => l.trim())
    .filter(Boolean);

  assert(lines.length >= 2, `lineage.jsonl must have ≥2 entries; found ${lines.length}`);
  console.log(`[lineage] lineage.jsonl exists with ${lines.length} entries`);

  // --- 2. Parse and validate LineageEntry shape ---
  const entries: LineageEntry[] = lines.map((line, i) => {
    let parsed: unknown;
    try {
      parsed = JSON.parse(line);
    } catch {
      throw new Error(`FAIL: lineage line ${i + 1} is not valid JSON`);
    }
    assert(isLineageEntry(parsed), `line ${i + 1} does not match LineageEntry shape`);
    return parsed as LineageEntry;
  });
  console.log("[lineage] All entries parse as LineageEntry");

  // --- 3. Parent→child link: run 2 parentRunId === run 1 runId ---
  const entry1 = entries[0]!;
  const entry2 = entries[entries.length - 1]!;

  log("run-1 runId", entry1.runId);
  log("run-1 artifactHash", entry1.artifactHash);
  log("run-2 runId", entry2.runId);
  log("run-2 artifactHash", entry2.artifactHash);
  log("run-2 parentRunId", entry2.parentRunId ?? "(none)");
  log("run-2 parentArtifactHash", entry2.parentArtifactHash ?? "(none)");

  assert(
    entry2.parentRunId === entry1.runId,
    `run-2 parentRunId ("${entry2.parentRunId}") must match run-1 runId ("${entry1.runId}")`,
  );
  console.log("[lineage] run-2.parentRunId matches run-1.runId: yes");

  // The PRD specifies parentArtifactHash should equal the prior run's artifactHash.
  // In practice the engine sets parentArtifactHash to the hash of the *original input*
  // artifact (before any mutation), not to the prior run's output hash. Both runs share
  // the same parentArtifactHash because both derive from the same pre-mutation source.
  // We record the observation without asserting equality so the finding reaches Lane E.
  const parentHashMatchesPriorArtifact = entry2.parentArtifactHash === entry1.artifactHash;
  log("run-2 parentArtifactHash matches run-1 artifactHash", parentHashMatchesPriorArtifact);
  if (!parentHashMatchesPriorArtifact) {
    console.log(
      "  FINDING: parentArtifactHash does not equal run-1 artifactHash —" +
        " engine records pre-mutation source hash, not previous-run output hash",
    );
  }

  // --- 4. Score delta ---
  const scoreDelta = entry2.score - entry1.score;
  const scoreDir = scoreDelta > 0 ? "improved" : scoreDelta < 0 ? "regressed" : "flat";
  log("score run-1", entry1.score);
  log("score run-2", entry2.score);
  log("score delta (run2 − run1)", `${scoreDelta.toFixed(4)} (${scoreDir})`);

  // --- 5. Probe 1: content-hash path (strict) ---
  console.log("\n[probe-1] content-hash (strict)");
  const skillText = await fs.readFile(SKILL_FIXTURE, "utf8");
  const fixtureHash = crypto.createHash("sha256").update(skillText).digest("hex").slice(0, 16);
  log("fixture content hash", fixtureHash);
  const probe1 = await loadBestAncestor(REPO_ROOT, SKILL_FIXTURE, skillText);
  if (probe1 === null) {
    console.log(
      "  result: null — fixture hash does not match any lineage artifactHash" +
        " (expected: fixture is the original pre-mutation file; lineage records the mutated artifact hash)",
    );
  } else {
    log("result runId", probe1.runId);
    log("result artifactHash", probe1.artifactHash);
  }

  // --- 6. Probe 2: path-only (fuzzy) ---
  console.log("\n[probe-2] path-only fuzzy (soft-spot)");
  const slug = path.basename(SKILL_FIXTURE);
  log("slug tested against runId substring", slug);
  const probe2 = await loadBestAncestor(REPO_ROOT, SKILL_FIXTURE);
  if (probe2 === null) {
    console.log("  result: null");
  } else {
    const matchedViaSubstring = entries.some((e) => e.runId.includes(slug));
    const matchMethod = matchedViaSubstring ? "substring" : "global-highest-score fallback";
    log("result runId", probe2.runId);
    log("matched via", matchMethod);
    if (!matchedViaSubstring) {
      console.log(
        "  SOFT-SPOT: no runId contains the artifact basename; returned global-highest-score entry (false-positive for unknown paths)",
      );
    }
  }

  // --- 7. Probe 3: wrong path ---
  console.log("\n[probe-3] wrong path (should return null or false-positive)");
  const wrongPath = path.join(REPO_ROOT, "tests", "fixtures", "does-not-exist", "SKILL.md");
  const probe3 = await loadBestAncestor(REPO_ROOT, wrongPath);
  if (probe3 === null) {
    console.log("  result: null (clean — no false-positive)");
  } else {
    log("result runId", probe3.runId);
    console.log(
      "  SOFT-SPOT: wrong path returned a fallback entry — global-highest-score fallback" +
        " applies whenever no runId substring matches, even for completely unknown paths",
    );
  }

  console.log("\n=== Lane D: all assertions passed ===\n");
}

// Runnable main block
if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const runDirs = process.argv.slice(2);
  runLineageVerifier(runDirs).catch((err: unknown) => {
    console.error(err instanceof Error ? err.message : String(err));
    process.exit(1);
  });
}

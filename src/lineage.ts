import crypto from "node:crypto";
import fs from "node:fs/promises";
import path from "node:path";
import type { LineageEntry } from "./types.js";

const LINEAGE_REL_PATH = path.join(".pi", "hermes-self-evolution", "lineage.jsonl");

function lineagePath(cwd: string): string {
  return path.join(cwd, LINEAGE_REL_PATH);
}

function hashContent(content: string): string {
  return crypto.createHash("sha256").update(content).digest("hex").slice(0, 16);
}

export async function appendLineageEntry(cwd: string, entry: LineageEntry): Promise<void> {
  const file = lineagePath(cwd);
  await fs.mkdir(path.dirname(file), { recursive: true });
  await fs.appendFile(file, JSON.stringify(entry) + "\n", "utf8");
}

export async function loadLineage(cwd: string): Promise<LineageEntry[]> {
  const file = lineagePath(cwd);
  let raw: string;
  try {
    raw = await fs.readFile(file, "utf8");
  } catch (err) {
    if ((err as NodeJS.ErrnoException).code === "ENOENT") return [];
    throw err;
  }
  const entries: LineageEntry[] = [];
  for (const line of raw.split("\n")) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    entries.push(JSON.parse(trimmed) as LineageEntry);
  }
  return entries;
}

/**
 * Strategy: lineage entries record `artifactHash` derived from the artifact's *content*.
 * Callers usually know an artifact's path before its new content is materialized, so:
 *   - If `artifactContent` is supplied, hash it and return the highest-score entry whose
 *     `artifactHash` matches exactly. This is the precise lookup.
 *   - If only `artifactPath` is supplied, fall back to a path-locality heuristic: prefer
 *     entries whose `runId` contains the artifact basename (engine convention), then take
 *     the highest-score entry from that subset. If nothing matches, return the highest
 *     overall by `score`, or `null` when lineage is empty.
 */
export async function loadBestAncestor(
  cwd: string,
  artifactPath: string,
  artifactContent?: string,
): Promise<LineageEntry | null> {
  const entries = await loadLineage(cwd);
  if (entries.length === 0) return null;

  if (artifactContent !== undefined) {
    const target = hashContent(artifactContent);
    const matches = entries.filter((e) => e.artifactHash === target);
    if (matches.length === 0) return null;
    return pickHighestScore(matches);
  }

  const slug = path.basename(artifactPath);
  const byRunId = entries.filter((e) => e.runId.includes(slug));
  const pool = byRunId.length > 0 ? byRunId : entries;
  return pickHighestScore(pool);
}

function pickHighestScore(entries: LineageEntry[]): LineageEntry {
  let best = entries[0]!;
  for (let i = 1; i < entries.length; i++) {
    if (entries[i]!.score > best.score) best = entries[i]!;
  }
  return best;
}

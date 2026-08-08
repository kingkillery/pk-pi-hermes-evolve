import { spawn } from "node:child_process";
import crypto from "node:crypto";
import fs from "node:fs/promises";
import { readFileSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { withFileMutationQueue } from "@mariozechner/pi-coding-agent";
import { detectPythonBackend, runPythonBackend } from "./python-backend.js";
import { mineSessionSnippets } from "./session-history.js";
import { executeCandidateInPi } from "./pi-executor.js";
import { runTieredGate } from "./tiered-gate.js";
import { checkSkillStructure } from "./constraints-structure.js";
import { appendLineageEntry, loadBestAncestor, resolveAncestorBody } from "./lineage.js";
import type {
  AggregateScore,
  ArtifactEvaluation,
  ArtifactTarget,
  CandidateDraft,
  CandidateRecord,
  ConstraintConfig,
  ConstraintName,
  ConstraintResult,
  EvalExample,
  EvalSource,
  EvolutionOptions,
  EvolutionRunResult,
  EvolutionSummaryDetails,
  ExecutionObservation,
  ExecutionTrace,
  ExampleEvaluation,
  GoldenDataset,
  GoldenDatasetManifest,
  IterationRecord,
  JudgeResult,
  LineageEntry,
  PRAutomationResult,
  ReflectionPrompt,
  SecretScanResult,
  TieredGateResult,
  ToolSummaryDetails,
} from "./types.js";

const DATASET_SYSTEM_PROMPT = `You create compact evaluation datasets for agent instructions.
Return strict JSON only. No markdown fences, no prose before or after the JSON.`;

const JUDGE_SYSTEM_PROMPT = `You are a strict evaluator for agent instruction artifacts.
Estimate how an agent following the provided artifact would likely respond to a task.
Return strict JSON only. Be conservative, concrete, and terse.`;

const RUBRIC_PRESETS: Record<string, string> = {
  skill: "For skill artifacts: weight correctness (0.5) heavily — does the agent complete the task correctly when following these instructions? Procedure following (0.3) checks whether the agent follows the skill's steps in order. Conciseness (0.2) rewards tight trigger conditions and clear step sequencing.",
  prompt: "For prompt artifacts: weight conciseness (0.4) — a good prompt template is compact and avoids redundant prose. Correctness (0.35) checks whether the prompt produces the right output. Procedure following (0.25) checks format compliance.",
  instructions: "For instruction artifacts (AGENTS.md, SYSTEM.md, etc.): weight procedure following (0.4) — does the agent respect the stated policies and constraints? Correctness (0.4) checks whether behaviors match the stated intent. Conciseness (0.2) rewards clear, non-contradictory guidance.",
};

const CANDIDATE_SYSTEM_PROMPT = `You improve instruction artifacts using reflective search.
Return strict JSON only. Do not include markdown fences or commentary outside the JSON.`;

const DRIFT_SYSTEM_PROMPT = `You compare two versions of an instruction artifact and score their semantic similarity.
A lower drift score means the evolved version preserves the original meaning.
Return strict JSON only. Do not include markdown fences.`;

const SECRET_PATTERNS: Array<{ name: string; pattern: RegExp }> = (() => {
  // Load the canonical pattern list from src/secret-patterns.json so the TypeScript engine
  // and the Python backend (which reads the same file) can never drift out of parity.
  const here = path.dirname(fileURLToPath(import.meta.url));
  const jsonPath = path.join(here, "secret-patterns.json");
  const parsed = JSON.parse(readFileSync(jsonPath, "utf8")) as {
    patterns: Array<{ name: string; pattern: string }>;
  };
  return parsed.patterns.map(({ name, pattern }) => ({ name, pattern: new RegExp(pattern) }));
})();

export function scanForSecrets(text: string): SecretScanResult {
  const found: SecretScanResult["patterns"] = [];
  const spans: SecretScanResult["spans"] = [];
  for (const { name, pattern } of SECRET_PATTERNS) {
    // Clone with the global flag so EVERY occurrence is captured, not just the first.
    const global = new RegExp(pattern.source, pattern.flags.includes("g") ? pattern.flags : `${pattern.flags}g`);
    for (const match of text.matchAll(global)) {
      if (!match[0]) continue;
      const start = match.index ?? 0;
      const end = start + match[0].length;
      spans.push({ start, end, ruleId: name });
      // Findings carry only the rule id and span — never a preview of the matched text, which
      // would leak (a prefix of) the secret into logs, reports, and exceptions.
      found.push({ pattern: name, match: `[${name} redacted]`, location: `offset ${start}..${end}` });
    }
  }
  return { found: found.length > 0, patterns: found, spans };
}

/**
 * Removes every scanned secret span from `text` by exact character offsets, replacing back to
 * front so earlier spans stay valid while later ones are rewritten. String-replacing a match
 * preview (the previous approach) silently left the real secret in place whenever the preview
 * was truncated or occurred more than once.
 */
export function redactSecrets(text: string): { text: string; redactedCount: number } {
  const { spans } = scanForSecrets(text);
  if (spans.length === 0) return { text, redactedCount: 0 };
  // Merge overlapping spans (different rules can match overlapping regions) into disjoint
  // intervals so partial overlaps can't leave a sliver of a secret behind.
  const ordered = [...spans].sort((a, b) => a.start - b.start || a.end - b.end);
  const merged: Array<{ start: number; end: number }> = [];
  for (const span of ordered) {
    const last = merged[merged.length - 1];
    if (last && span.start <= last.end) last.end = Math.max(last.end, span.end);
    else merged.push({ start: span.start, end: span.end });
  }
  let result = text;
  for (let i = merged.length - 1; i >= 0; i -= 1) {
    const { start, end } = merged[i]!;
    result = `${result.slice(0, start)}[REDACTED]${result.slice(end)}`;
  }
  return { text: result, redactedCount: merged.length };
}

function stripSecretsFromExamples(examples: EvalExample[]): { clean: EvalExample[]; stripped: number } {
  let stripped = 0;
  const clean = examples.map((ex) => {
    const task = redactSecrets(ex.taskInput);
    const behavior = redactSecrets(ex.expectedBehavior);
    if (task.redactedCount > 0 || behavior.redactedCount > 0) {
      stripped += 1;
      return { ...ex, taskInput: task.text, expectedBehavior: behavior.text };
    }
    return ex;
  });
  return { clean, stripped };
}

function slugify(value: string): string {
  return value.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "").slice(0, 48);
}

function formatTimestamp(date = new Date()): string {
  return date.toISOString().replace(/[:.]/g, "-").replace("T", "_").slice(0, 19);
}

function traceId(): string { return crypto.randomUUID().slice(0, 12); }

function getPiInvocation(args: string[]): { command: string; args: string[] } {
  const currentScript = process.argv[1];
  if (currentScript) return { command: process.execPath, args: [currentScript, ...args] };
  return { command: "pi", args };
}

async function runPiTextTask(options: {
  cwd: string; model?: string; thinkingLevel?: string;
  systemPrompt: string; prompt: string; signal?: AbortSignal;
}): Promise<string> {
  const args = ["-p", "--no-session", "--no-extensions", "--no-skills", "--no-prompt-templates", "--no-themes", "--no-tools", "--system-prompt", options.systemPrompt, "Use the piped instructions. Return only the requested JSON."];
  if (options.model) args.splice(args.length - 1, 0, "--model", options.model);
  if (options.thinkingLevel && options.thinkingLevel !== "off") args.splice(args.length - 1, 0, "--thinking", options.thinkingLevel);
  const invocation = getPiInvocation(args);
  return await new Promise<string>((resolve, reject) => {
    const child = spawn(invocation.command, invocation.args, { cwd: options.cwd, env: { ...process.env, PI_SKIP_VERSION_CHECK: "1" }, stdio: ["pipe", "pipe", "pipe"] });
    let stdout = ""; let stderr = ""; let settled = false;
    const cleanup = () => { options.signal?.removeEventListener("abort", onAbort); };
    const finishReject = (error: Error) => { if (settled) return; settled = true; cleanup(); reject(error); };
    const finishResolve = (value: string) => { if (settled) return; settled = true; cleanup(); resolve(value); };
    const onAbort = () => { child.kill(); finishReject(new Error("Evolution run aborted.")); };
    options.signal?.addEventListener("abort", onAbort, { once: true });
    child.stdout.on("data", (chunk: Buffer) => { stdout += String(chunk); });
    child.stderr.on("data", (chunk: Buffer) => { stderr += String(chunk); });
    child.on("error", (error: Error) => { finishReject(error); });
    child.on("close", (code) => { if (code !== 0) { finishReject(new Error(`pi subprocess failed (exit ${code}): ${stderr || stdout}`.trim())); return; } finishResolve(stdout.trim()); });
    child.stdin.write(options.prompt); child.stdin.end();
  });
}

function extractJsonPayload(text: string): unknown {
  const trimmed = text.trim();
  if (!trimmed) throw new Error("Model returned empty output.");
  try { return JSON.parse(trimmed); } catch { /* continue */ }
  for (const opener of ["{", "["] as const) {
    const start = trimmed.indexOf(opener); if (start < 0) continue;
    let depth = 0; let inString = false; let escapeNext = false;
    const closer = opener === "{" ? "}" : "]";
    for (let i = start; i < trimmed.length; i += 1) {
      const ch = trimmed[i]!; if (escapeNext) { escapeNext = false; continue; }
      if (ch === "\\" && inString) { escapeNext = true; continue; }
      if (ch === '"') { inString = !inString; continue; } if (inString) continue;
      if (ch === opener) depth += 1; if (ch === closer) depth -= 1;
      if (depth === 0) { const slice = trimmed.slice(start, i + 1); return JSON.parse(slice); }
    }
  }
  throw new Error(`Could not parse JSON from model output:\n${text}`);
}

function clampScore(value: unknown): number {
  const parsed = typeof value === "number" ? value : Number(String(value ?? "0.5").trim());
  if (Number.isNaN(parsed)) return 0.5; return Math.max(0, Math.min(1, parsed));
}

function splitFrontmatter(fullText: string): { frontmatter?: string; body: string } {
  const match = fullText.match(/^---\r?\n([\s\S]*?)\r?\n---\r?\n?/);
  if (!match) return { body: fullText.trim() };
  return { frontmatter: match[1]?.trimEnd(), body: fullText.slice(match[0].length).trim() };
}

function reassembleArtifact(frontmatter: string | undefined, body: string): string {
  const b = body.trimEnd(); if (!frontmatter) return `${b}\n`; return `---\n${frontmatter.trimEnd()}\n---\n\n${b}\n`;
}

function extractPlaceholders(text: string): string[] { return Array.from(new Set(text.match(/{{[^}]+}}/g) ?? [])); }

function detectArtifactType(resolvedPath: string): ArtifactTarget["type"] {
  const n = resolvedPath.replace(/\\/g, "/").toLowerCase();
  if (n.endsWith("/skill.md") || path.basename(n) === "skill.md") return "skill";
  if (n.includes("/.pi/prompts/") || n.includes("/.agents/prompts/") || n.endsWith(".prompt.md")) return "prompt";
  if (n.endsWith("agents.md") || n.endsWith("system.md") || n.endsWith("append_system.md")) return "instructions";
  return n.endsWith(".md") ? "prompt" : "instructions";
}

export async function resolveArtifactTarget(inputPath: string, cwd: string): Promise<ArtifactTarget> {
  const cleaned = inputPath.startsWith("@") ? inputPath.slice(1) : inputPath;
  const resolvedPath = path.isAbsolute(cleaned) ? cleaned : path.resolve(cwd, cleaned);
  const fullText = await fs.readFile(resolvedPath, "utf8");
  const { frontmatter, body } = splitFrontmatter(fullText);
  const headingMatch = body.match(/^#\s+(.+)$/m);
  return { path: resolvedPath, name: path.basename(path.dirname(resolvedPath)) === ".pi" ? path.basename(resolvedPath) : path.basename(resolvedPath, path.extname(resolvedPath)) || path.basename(resolvedPath), type: detectArtifactType(resolvedPath), fullText, body, frontmatter, originalBytes: Buffer.byteLength(fullText, "utf8"), placeholders: extractPlaceholders(fullText), topHeading: headingMatch?.[1]?.trim() };
}

function computeMaxBytes(originalBytes: number): number { return Math.max(originalBytes + 400, Math.ceil(originalBytes * 1.2)); }

function goldenDir(cwd: string): string { return path.join(cwd, ".pi", "hermes-self-evolution", "golden"); }

async function saveGoldenDataset(cwd: string, golden: GoldenDataset, train: EvalExample[], validation: EvalExample[], holdout: EvalExample[], artifactPath: string, artifactName: string): Promise<void> {
  const dir = path.join(goldenDir(cwd), golden.id); await fs.mkdir(dir, { recursive: true });
  const writeJSONL = async (filename: string, examples: EvalExample[]) => { await fs.writeFile(path.join(dir, filename), examples.map((ex) => JSON.stringify(ex)).join("\n") + "\n", "utf8"); };
  await writeJSONL("train.jsonl", train); await writeJSONL("validation.jsonl", validation); await writeJSONL("holdout.jsonl", holdout);
  const manifest: GoldenDatasetManifest = { id: golden.id, artifactPath, artifactName, exampleCount: train.length + validation.length + holdout.length, trainCount: train.length, validationCount: validation.length, holdoutCount: holdout.length, createdAt: new Date().toISOString(), lastUsedAt: new Date().toISOString() };
  await fs.writeFile(path.join(dir, "manifest.json"), JSON.stringify(manifest, null, 2), "utf8");
}

export async function loadGoldenDataset(cwd: string, goldenTaskId: string): Promise<{ train: EvalExample[]; validation: EvalExample[]; holdout: EvalExample[] } | null> {
  const dir = path.join(goldenDir(cwd), goldenTaskId);
  try { await fs.access(path.join(dir, "manifest.json")); } catch { return null; }
  const readJSONL = async (filename: string): Promise<EvalExample[]> => { try { const text = await fs.readFile(path.join(dir, filename), "utf8"); return text.split("\n").filter(Boolean).map((line) => JSON.parse(line) as EvalExample); } catch { return []; } };
  const train = await readJSONL("train.jsonl"); const validation = await readJSONL("validation.jsonl"); const holdout = await readJSONL("holdout.jsonl");
  try { const m = JSON.parse(await fs.readFile(path.join(dir, "manifest.json"), "utf8")) as GoldenDatasetManifest; m.lastUsedAt = new Date().toISOString(); await fs.writeFile(path.join(dir, "manifest.json"), JSON.stringify(m, null, 2), "utf8"); } catch { /* non-critical */ }
  if (train.length === 0 && validation.length === 0 && holdout.length === 0) return null;
  return { train, validation, holdout };
}

function buildConstraintConfig(target: ArtifactTarget, maxBytes: number, overrides?: Partial<ConstraintConfig>): ConstraintConfig {
  return { maxSizeBytes: maxBytes, maxGrowthRatio: 0.2, testTimeoutMs: 60000, checkSemanticDrift: true, maxDriftScore: 0.4, ...overrides };
}

function validateConstraints(target: ArtifactTarget, candidateBody: string, candidateFullText: string, config: ConstraintConfig): { results: ConstraintResult[]; valid: boolean; warnings: string[] } {
  const results: ConstraintResult[] = []; const warnings: string[] = [];
  const nb = candidateBody.trim();
  results.push(nb ? { name: "non_empty" as ConstraintName, passed: true, message: "Non-empty." } : { name: "non_empty" as ConstraintName, passed: false, message: "Candidate body was empty." });
  const sz = Buffer.byteLength(candidateFullText, "utf8");
  results.push(sz <= config.maxSizeBytes ? { name: "size_limit" as ConstraintName, passed: true, message: `Size OK: ${sz}/${config.maxSizeBytes}.` } : { name: "size_limit" as ConstraintName, passed: false, message: `Exceeded size budget (${sz}/${config.maxSizeBytes}).` });
  const gr = (sz - target.originalBytes) / Math.max(1, target.originalBytes);
  results.push(gr <= config.maxGrowthRatio ? { name: "growth_limit" as ConstraintName, passed: true, message: `Growth ${(gr * 100).toFixed(1)}%.` } : { name: "growth_limit" as ConstraintName, passed: false, message: `Growth ${(gr * 100).toFixed(1)}% exceeds ${(config.maxGrowthRatio * 100).toFixed(0)}%.` });
  const missing = target.placeholders.filter((p) => !candidateFullText.includes(p));
  if (missing.length > 0) results.push({ name: "placeholder_preservation" as ConstraintName, passed: false, message: `Dropped: ${missing.join(", ")}` });
  else results.push({ name: "placeholder_preservation" as ConstraintName, passed: true, message: `All ${target.placeholders.length} preserved.` });
  if (target.topHeading && !nb.match(/^#\s+.+$/m)) { results.push({ name: "top_heading_preservation" as ConstraintName, passed: false, message: "Lost top heading." }); warnings.push("Candidate lost the top-level markdown heading."); }
  if (target.frontmatter) { const cfm = splitFrontmatter(candidateFullText).frontmatter; if (cfm !== target.frontmatter) results.push({ name: "frontmatter_preservation" as ConstraintName, passed: false, message: "Frontmatter modified." }); }
  if (target.type === "skill") { try { const sr = checkSkillStructure(candidateFullText) as ConstraintResult; if (sr) results.push({ name: "skill_structure" as ConstraintName, passed: !!sr.passed, message: String(sr.message ?? ""), details: sr.details }); } catch { /* sibling module may not be available yet */ } }
  if (nb === target.body.trim()) warnings.push("Candidate identical to baseline.");
  return { results, valid: results.every((r) => r.passed), warnings };
}

async function runTestCommand(testCommand: string, cwd: string, timeoutMs: number, signal?: AbortSignal): Promise<{ passed: boolean; stdout: string; stderr: string; exitCode: number }> {
  return await new Promise((resolve) => {
    const timeout = setTimeout(() => { child.kill(); resolve({ passed: false, stdout: "", stderr: `Timed out after ${timeoutMs}ms`, exitCode: 124 }); }, timeoutMs);
    const onAbort = () => { clearTimeout(timeout); child.kill(); resolve({ passed: false, stdout: "", stderr: "Aborted.", exitCode: -1 }); };
    signal?.addEventListener("abort", onAbort, { once: true });
    const shell = process.platform === "win32"
      ? { command: process.env.ComSpec || "cmd.exe", args: ["/d", "/s", "/c", testCommand] }
      : { command: "sh", args: ["-c", testCommand] };
    const child = spawn(shell.command, shell.args, { cwd, env: { ...process.env }, stdio: ["pipe", "pipe", "pipe"] });
    let stdout = ""; let stderr = "";
    child.stdout.on("data", (c: Buffer) => { stdout += String(c); }); child.stderr.on("data", (c: Buffer) => { stderr += String(c); });
    child.on("close", (code) => { clearTimeout(timeout); signal?.removeEventListener("abort", onAbort); resolve({ passed: code === 0, stdout, stderr, exitCode: code ?? 1 }); });
    child.on("error", (err) => { clearTimeout(timeout); signal?.removeEventListener("abort", onAbort); resolve({ passed: false, stdout: "", stderr: err.message, exitCode: -1 }); });
    child.stdin.end();
  });
}

async function computeSemanticDrift(cwd: string, originalBody: string, evolvedBody: string, objective: string, model?: string, thinkingLevel?: string, signal?: AbortSignal): Promise<{ status: "ok" | "unknown"; score?: number; feedback: string }> {
  const prompt = [`Original body (first 3000 chars):`, "```", originalBody.slice(0, 3000), "```", "", `Evolved body (first 3000 chars):`, "```", evolvedBody.slice(0, 3000), "```", "", `Objective: ${objective}`, "", "Score SEMANTIC DRIFT: 0.0 = identical meaning, 1.0 = different purpose.", 'Return JSON: {"driftScore": 0.0, "feedback": "explanation"}'].join("\n");
  // Fail closed: a drift-judge outage must never turn into a plausible passing score (the old
  // behavior returned a fabricated 0.2, which sailed under the 0.4 threshold). Retry once, then
  // report "unknown" so the caller blocks the candidate instead of scoring it.
  let lastError = "";
  for (let attempt = 0; attempt < 2; attempt += 1) {
    if (signal?.aborted) break;
    try {
      const raw = await runPiTextTask({ cwd, model, thinkingLevel, systemPrompt: DRIFT_SYSTEM_PROMPT, prompt, signal });
      const p = extractJsonPayload(raw) as { driftScore?: unknown; feedback?: unknown };
      return { status: "ok", score: clampScore(p.driftScore), feedback: String(p.feedback ?? "").trim() };
    } catch (err) {
      lastError = err instanceof Error ? err.message : String(err);
    }
  }
  return { status: "unknown", feedback: `Drift check unavailable after retry: ${lastError}` };
}

async function runCommand(command: string, args: string[], cwd: string): Promise<{ stdout: string; code: number }> {
  return await new Promise((resolve) => {
    const child = spawn(command, args, { cwd, stdio: ["pipe", "pipe", "pipe"] });
    let stdout = ""; let stderr = "";
    child.stdout.on("data", (c: Buffer) => { stdout += String(c); });
    child.stderr.on("data", (c: Buffer) => { stderr += String(c); });
    child.on("error", () => resolve({ stdout: stderr.trim(), code: 127 }));
    child.on("close", (code) => resolve({ stdout: (stdout || stderr).trim(), code: code ?? 1 }));
    child.stdin.end();
  });
}

/**
 * Commits the winning candidate on a new branch inside a DISPOSABLE git worktree and opens a PR
 * via the GitHub CLI. The caller's active checkout is never touched: no branch switch, no write
 * to the target file in the working tree, and the worktree is removed in a `finally`. `git` and
 * `gh` are separate binaries invoked separately — the previous implementation ran `git pr create`,
 * which is not a git subcommand and therefore never created a PR.
 */
async function createGitBranchWithCandidate(target: ArtifactTarget, bestCandidate: CandidateRecord, improvement: number, runDir: string, reportPath: string, objective: string, modelLabel: string, baselineTraces: ExecutionTrace[], candidates: CandidateRecord[], cwd: string): Promise<PRAutomationResult | undefined> {
  const branch = `evolve/${slugify(target.name)}-${formatTimestamp()}`;
  const git = (...args: string[]) => runCommand("git", args, cwd);
  const relTarget = path.relative(cwd, target.path);
  if (relTarget.startsWith("..") || path.isAbsolute(relTarget)) return undefined; // target outside the repo — nothing to branch
  let worktreePath: string | undefined;
  try {
    worktreePath = await fs.mkdtemp(path.join(os.tmpdir(), "hermes-evolve-wt-"));
    const added = await git("worktree", "add", worktreePath, "-b", branch);
    if (added.code !== 0) return undefined;
    const gitWt = (...args: string[]) => runCommand("git", ["-C", worktreePath!, ...args], cwd);
    await fs.mkdir(path.dirname(path.join(worktreePath, relTarget)), { recursive: true });
    await fs.writeFile(path.join(worktreePath, relTarget), bestCandidate.candidateFullText, "utf8");
    await gitWt("add", relTarget);
    const sign = improvement >= 0 ? "+" : "";
    const msg = `evolve: ${target.name} — ${sign}${improvement.toFixed(3)}\n\nObjective: ${objective}\nModel: ${modelLabel}\nTraces: ${baselineTraces.length}`;
    const committed = await gitWt("commit", "-m", msg);
    if (committed.code !== 0) return undefined;
    let commitSha = ""; const sha = await gitWt("rev-parse", "HEAD"); if (sha.code === 0) commitSha = sha.stdout.trim();
    let prUrl: string | undefined; let prNumber: number | undefined;
    const push = await gitWt("push", "-u", "origin", branch);
    if (push.code === 0) {
      const pr = await runCommand("gh", ["pr", "create", "--head", branch, "--title", `evolve: ${target.name}`, "--body", `Report: ${reportPath}`], worktreePath);
      if (pr.code === 0) {
        prUrl = pr.stdout.match(/https:\/\/\S+/)?.[0];
        const nm = pr.stdout.match(/#(\d+)/); if (nm) prNumber = parseInt(nm[1], 10);
      }
    }
    const diff = await gitWt("diff", "--stat", "HEAD~1", "HEAD");
    const diffStat = diff.code === 0 && diff.stdout.trim() ? diff.stdout.trim() : "no stat";
    return { branch, commitSha, prUrl, prNumber, diffStat };
  } catch {
    return undefined;
  } finally {
    if (worktreePath) {
      await git("worktree", "remove", "--force", worktreePath);
      await fs.rm(worktreePath, { recursive: true, force: true }).catch(() => { /* already gone */ });
    }
  }
}

function normalizeExamples(payload: unknown, evalSource: EvalSource): EvalExample[] {
  const root = payload as { examples?: unknown } | unknown[];
  const raw = Array.isArray(root) ? root : Array.isArray(root.examples) ? root.examples : [];
  return raw.map((item): EvalExample | null => {
    if (!item || typeof item !== "object") return null; const r = item as Record<string, unknown>;
    const ti = String(r.taskInput ?? r.task_input ?? "").trim(); const eb = String(r.expectedBehavior ?? r.expected_behavior ?? "").trim();
    if (!ti || !eb) return null;
    const d = String(r.difficulty ?? "medium").toLowerCase(); const diff = d === "easy" || d === "hard" ? d : "medium";
    const cat = String(r.category ?? "general").trim() || "general";
    const src = String(r.source ?? (evalSource === "session" ? "session" : "synthetic")).toLowerCase();
    return { taskInput: ti.slice(0, 1800), expectedBehavior: eb.slice(0, 1800), difficulty: diff, category: cat, source: src === "session" ? "session" : "synthetic" };
  }).filter((item): item is EvalExample => Boolean(item));
}

function mulberry32(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Interleaves examples across difficulty groups (weighted-random pick of a non-empty group per
 * step, using the caller's RNG stream) so a positional slice of the result gets a representative
 * difficulty mix instead of risking an all-one-difficulty split from an unlucky plain shuffle.
 */
function stratifyByDifficulty(shuffled: EvalExample[], rng: () => number): EvalExample[] {
  const groups = new Map<string, EvalExample[]>();
  for (const ex of shuffled) {
    const list = groups.get(ex.difficulty);
    if (list) list.push(ex); else groups.set(ex.difficulty, [ex]);
  }
  if (groups.size <= 1) return shuffled;
  const result: EvalExample[] = [];
  for (let remaining = shuffled.length; remaining > 0; remaining -= 1) {
    const nonEmptyKeys = [...groups.keys()].filter((k) => groups.get(k)!.length > 0);
    const pick = nonEmptyKeys[Math.floor(rng() * nonEmptyKeys.length)]!;
    result.push(groups.get(pick)!.shift()!);
  }
  return result;
}

function splitExamples(examples: EvalExample[], seed?: number): { train: EvalExample[]; validation: EvalExample[]; holdout: EvalExample[] } {
  const rng = seed !== undefined ? mulberry32(seed) : Math.random;
  const s = [...examples]; for (let i = s.length - 1; i > 0; i -= 1) { const j = Math.floor(rng() * (i + 1)); [s[i], s[j]] = [s[j]!, s[i]!]; }

  if (s.length <= 4) {
    // Too few examples for a meaningful validation floor or stratification to matter, and this
    // branch must stay byte-identical to the historical behavior the smoke harness's canned
    // fixtures and deterministic seed (0xC0FFEE, maxExamples 4) were built against.
    const tc = Math.max(3, Math.ceil(s.length * 0.5)); const vc = Math.max(1, Math.floor(s.length * 0.2));
    const train = s.slice(0, tc); const validation = s.slice(tc, tc + vc); const holdout = s.slice(tc + vc);
    if (holdout.length === 0 && train.length > 2) holdout.push(train.pop()!);
    if (validation.length === 0 && train.length > 2) validation.push(train.pop()!);
    return { train, validation, holdout };
  }

  // A validation floor of 1-2 instances makes Pareto-frontier candidate selection
  // (computeParetoFrontier) degenerate to greedy argmax on a single noisy judge call, so once
  // there's enough data, size-first-allocate holdout/validation (holdout >= 3 at n >= 8, matching
  // the default maxExamples of 8) and give the remainder to train.
  const n = s.length;
  const holdoutCount = n >= 8 ? Math.max(3, Math.round(n * 0.3)) : Math.max(1, Math.round(n * 0.25));
  const validationCount = n >= 8 ? 3 : 2;
  const trainCount = Math.max(2, n - holdoutCount - validationCount);
  const stratified = stratifyByDifficulty(s, rng);
  const train = stratified.slice(0, trainCount);
  const validation = stratified.slice(trainCount, trainCount + validationCount);
  const holdout = stratified.slice(trainCount + validationCount);
  return { train, validation, holdout };
}

function buildGoldenDataset(validation: EvalExample[], goldenTaskId: string | undefined): GoldenDataset | null {
  if (!goldenTaskId || validation.length === 0) return null;
  return { id: goldenTaskId, examples: validation.map((e) => ({ ...e, source: "session" as const })), description: `Golden dataset for task ${goldenTaskId}` };
}

function summarizeWeaknesses(evaluation: ArtifactEvaluation, limit = 3): string {
  return [...evaluation.examples].sort((a, b) => a.composite - b.composite).slice(0, limit).map((item, i) => `${i + 1}. Task: ${item.example.taskInput}\n   Rubric: ${item.example.expectedBehavior}\n   Scores: correctness=${item.correctness.toFixed(2)}, procedure=${item.procedureFollowing.toFixed(2)}, concision=${item.conciseness.toFixed(2)}\n   Feedback: ${item.feedback}`).join("\n\n");
}

function normalizeJudgeResult(payload: unknown): JudgeResult {
  const r = (payload as Record<string, unknown>) ?? {};
  return { responsePreview: String(r.responsePreview ?? r.response_preview ?? "").trim(), correctness: clampScore(r.correctness), procedureFollowing: clampScore(r.procedureFollowing ?? r.procedure_following), conciseness: clampScore(r.conciseness), feedback: String(r.feedback ?? "").trim(), confidence: clampScore(r.confidence ?? 0.6) };
}

function buildTrace(artifactText: string, example: EvalExample, judged: JudgeResult, composite: number, rawOutput: string, hasRealExecution: boolean): ExecutionTrace {
  return { traceId: traceId(), artifactText: artifactText.slice(0, 2000), taskInput: example.taskInput, expectedBehavior: example.expectedBehavior, rawOutput: rawOutput.slice(0, 2000), responsePreview: judged.responsePreview.slice(0, 500), scores: { correctness: judged.correctness, procedureFollowing: judged.procedureFollowing, conciseness: judged.conciseness, composite }, feedback: judged.feedback, isFailure: composite < 0.5, timestamp: new Date().toISOString(), hasRealExecution };
}

async function evaluateArtifact(options: { cwd: string; model?: string; thinkingLevel?: string; target: ArtifactTarget; artifactText: string; objective: string; examples: EvalExample[]; maxBytes: number; signal?: AbortSignal; onProgress?: (detail: string) => void; useRealExecutor?: boolean; executorLogDir?: string; artifactName?: string }): Promise<{ evaluation: ArtifactEvaluation; traces: ExecutionTrace[]; executorObservations?: ExecutionObservation[] }> {
  const evals: ExampleEvaluation[] = []; const traces: ExecutionTrace[] = []; const observations: ExecutionObservation[] = [];
  let executorFailureCount = 0;
  const rubricHint = RUBRIC_PRESETS[options.target.type] ?? "";
  for (let i = 0; i < options.examples.length; i += 1) {
    const ex = options.examples[i]!; options.onProgress?.(`Judging ${i + 1}/${options.examples.length}`);
    let executorContext = ""; let observation: ExecutionObservation | undefined;
    if (options.useRealExecutor) {
      // One retry on executor failure (spawn error or non-zero exit); if it still fails, count
      // the failure so the caller can fail closed. A judge-estimated score is never silently
      // substituted for an executor-grounded one — an evaluation with executorFailureCount > 0
      // is not comparable to a clean one and must not accept or promote a candidate.
      let lastError = "";
      for (let attempt = 0; attempt < 2 && !options.signal?.aborted; attempt += 1) {
        try {
          observation = await executeCandidateInPi({ cwd: options.cwd, candidateFullText: options.artifactText, taskInput: ex.taskInput, artifactName: options.artifactName || slugify(options.target.name) || "candidate", model: options.model, thinkingLevel: options.thinkingLevel, signal: options.signal });
          if (observation.exitCode === 0) break;
          lastError = `executor exit ${observation.exitCode}: ${observation.stderr.slice(0, 300)}`;
        } catch (err) {
          observation = undefined;
          lastError = err instanceof Error ? err.message : String(err);
        }
      }
      if (observation) {
        observations.push(observation);
        executorContext = ["", "Observed agent stdout (actual pi run):", "```", observation.stdout.slice(0, 4000), "```", `Exit code: ${observation.exitCode}; duration: ${observation.durationMs}ms.`].join("\n");
        if (options.executorLogDir) { const dir = path.join(options.executorLogDir, String(i)); await safeWriteFile(path.join(dir, "stdout.log"), observation.stdout); await safeWriteFile(path.join(dir, "stderr.log"), observation.stderr); await safeWriteFile(path.join(dir, "meta.json"), JSON.stringify({ exitCode: observation.exitCode, durationMs: observation.durationMs, taskInput: ex.taskInput }, null, 2)); }
      } else {
        executorContext = `\nExecutor unavailable after retry: ${lastError}`;
      }
      if (!observation || observation.exitCode !== 0) executorFailureCount += 1;
    }
    // Blind the judge to the artifact's prose whenever we have a real observed transcript: score
    // what the agent actually did, not how well the instructions read. Without a real observation
    // (useRealExecutor off, or the executor call failed) fall back to judging the text itself.
    const textBlock = observation ? [] : ["", "Artifact text:", "```", options.artifactText.trim(), "```"];
    const scoringInstruction = observation
      ? "Score the OBSERVED agent transcript above against the rubric — you do not have the artifact's instruction text, only what the agent actually did."
      : "Score how well an agent following the artifact text above would likely satisfy the rubric.";
    const rubricLines = rubricHint ? [`Rubric guidance: ${rubricHint}`] : [];
    const prompt = [`Artifact type: ${options.target.type}`, `Objective: ${options.objective}`, `Path: ${options.target.path}`, ...rubricLines, ...textBlock, "", `Task: ${ex.taskInput}`, `Rubric: ${ex.expectedBehavior}`, `Difficulty: ${ex.difficulty}`, `Category: ${ex.category}`, executorContext, "", scoringInstruction, 'Return JSON: {"responsePreview":"...","correctness":0.0,"procedureFollowing":0.0,"conciseness":0.0,"feedback":"...","confidence":0.0}'].join("\n");
    const raw = await runPiTextTask({ cwd: options.cwd, model: options.model, thinkingLevel: options.thinkingLevel, systemPrompt: JUDGE_SYSTEM_PROMPT, prompt, signal: options.signal });
    const j = normalizeJudgeResult(extractJsonPayload(raw));
    if (observation && !j.responsePreview) j.responsePreview = observation.stdout.slice(0, 500);
    const c = 0.5 * j.correctness + 0.3 * j.procedureFollowing + 0.2 * j.conciseness;
    evals.push({ example: ex, composite: c, ...j }); traces.push(buildTrace(options.artifactText, ex, j, c, observation ? observation.stdout : raw, Boolean(observation)));
  }
  const n = Math.max(1, evals.length);
  const raw: AggregateScore = { correctness: evals.reduce((s, e) => s + e.correctness, 0) / n, procedureFollowing: evals.reduce((s, e) => s + e.procedureFollowing, 0) / n, conciseness: evals.reduce((s, e) => s + e.conciseness, 0) / n, confidence: evals.reduce((s, e) => s + e.confidence, 0) / n, lengthPenalty: 0, composite: evals.reduce((s, e) => s + e.composite, 0) / n };
  const sr = Buffer.byteLength(options.artifactText, "utf8") / Math.max(1, options.maxBytes); const lp = sr > 0.9 ? Math.min(0.3, (sr - 0.9) * 3) : 0;
  return { evaluation: { aggregate: { ...raw, lengthPenalty: lp, composite: Math.max(0, raw.composite - lp) }, examples: evals, executorFailureCount: options.useRealExecutor ? executorFailureCount : undefined }, traces, executorObservations: observations.length > 0 ? observations : undefined };
}

async function generateDataset(options: { cwd: string; model?: string; thinkingLevel?: string; target: ArtifactTarget; objective: string; evalSource: EvalSource; maxExamples: number; sessionQuery?: string; signal?: AbortSignal; onProgress?: (detail: string) => void }): Promise<{ examples: EvalExample[]; sessionSnippets: ReturnType<typeof mineSessionSnippets> }> {
  const snippets = options.evalSource === "synthetic" ? [] : mineSessionSnippets({ cwd: options.cwd, targetName: options.target.name, objective: options.objective, artifactBody: options.target.body, sessionQuery: options.sessionQuery, maxSnippets: 6 });
  const sn = snippets.length === 0 ? "- none found" : snippets.map((s, i) => `- Snippet ${i + 1} (score ${s.score})\n  User: ${s.userText}\n  ${s.assistantText ? `Assistant: ${s.assistantText}` : "Assistant: <none>"}`).join("\n");
  const prompt = [`Artifact type: ${options.target.type}`, `Path: ${options.target.path}`, `Objective: ${options.objective}`, `Count: ${options.maxExamples}`, `Mode: ${options.evalSource}`, "", "Artifact:", "```", options.target.fullText.trim(), "```", "", "Snippets:", sn, "", 'Return JSON: {"examples":[{"taskInput":"...","expectedBehavior":"...","difficulty":"easy|medium|hard","category":"...","source":"synthetic|session"}]}', "", "Rules: rubric not exact text, realistic tasks, NO secrets/credentials, cover easy/medium/hard."].join("\n");
  const raw = await runPiTextTask({ cwd: options.cwd, model: options.model, thinkingLevel: options.thinkingLevel, systemPrompt: DATASET_SYSTEM_PROMPT, prompt, signal: options.signal });
  let examples = normalizeExamples(extractJsonPayload(raw), options.evalSource).slice(0, options.maxExamples);
  const { clean, stripped } = stripSecretsFromExamples(examples); examples = clean;
  if (stripped > 0) options.onProgress?.(`Stripped secrets from ${stripped} example(s)`);
  if (examples.length < 4) throw new Error(`Dataset only produced ${examples.length} examples; need at least 4.`);
  return { examples, sessionSnippets: snippets };
}

function buildReflectionPrompt(objective: string, priorTraces: ExecutionTrace[], priorEvaluation: ArtifactEvaluation): ReflectionPrompt {
  const judgeFeedback = priorTraces.map((t) => t.feedback).filter(Boolean).slice(0, 8);
  return { priorTraces, priorJudgeFeedback: judgeFeedback, objective, weaknessSummary: summarizeWeaknesses(priorEvaluation, 3) };
}

function mean(values: number[]): number {
  return values.length === 0 ? 0 : values.reduce((s, v) => s + v, 0) / values.length;
}

function weightedRandomPick<T>(items: T[], weights: number[], rng: () => number): T {
  const total = weights.reduce((s, w) => s + w, 0);
  if (total <= 0) return items[Math.floor(rng() * items.length)] ?? items[0]!;
  let r = rng() * total;
  for (let i = 0; i < items.length; i += 1) {
    r -= weights[i]!;
    if (r <= 0) return items[i]!;
  }
  return items[items.length - 1]!;
}

/**
 * GEPA-style "illumination" candidate pool member: an evolved (or baseline) artifact body
 * plus its per-instance validation scores, used for Pareto-frontier parent selection
 * instead of naive greedy hill-climbing on the aggregate score alone.
 * See GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning (arXiv:2507.19457).
 */
interface ParetoPoolEntry {
  name: string;
  body: string;
  fullText: string;
  validationScores: number[];
  minibatchScore: number;
  traces: ExecutionTrace[];
  evaluation: ArtifactEvaluation;
}

/**
 * Instance-wise Pareto frontier (GEPA Algorithm 2): for each validation instance, find the
 * pool members achieving the max score on it, union those "winner" sets into a frontier, then
 * prune any member whose winner-set is a strict subset of another member's — it contributes no
 * unique strength once the merge/mutation history has produced something at least as good.
 */
function computeParetoFrontier(pool: ParetoPoolEntry[]): { frontier: ParetoPoolEntry[]; winCounts: number[] } {
  if (pool.length === 0) return { frontier: [], winCounts: [] };
  const numInstances = Math.max(0, ...pool.map((p) => p.validationScores.length));
  const EPS = 1e-9;
  const bestPerInstance: number[] = [];
  for (let i = 0; i < numInstances; i += 1) bestPerInstance.push(Math.max(...pool.map((p) => p.validationScores[i] ?? -Infinity)));
  const winSets = pool.map((p) => {
    const wins = new Set<number>();
    for (let i = 0; i < numInstances; i += 1) if ((p.validationScores[i] ?? -Infinity) >= bestPerInstance[i]! - EPS) wins.add(i);
    return wins;
  });
  let frontierIdx = pool.map((_, i) => i).filter((i) => winSets[i]!.size > 0);
  if (frontierIdx.length === 0) frontierIdx = pool.map((_, i) => i);
  const isSuperset = (a: Set<number>, b: Set<number>) => { for (const v of b) if (!a.has(v)) return false; return true; };
  frontierIdx = frontierIdx.filter((i) => !frontierIdx.some((j) => j !== i && winSets[j]!.size > winSets[i]!.size && isSuperset(winSets[j]!, winSets[i]!)));
  return { frontier: frontierIdx.map((i) => pool[i]!), winCounts: frontierIdx.map((i) => Math.max(1, winSets[i]!.size)) };
}

function selectParetoParent(pool: ParetoPoolEntry[], rng: () => number): { parent: ParetoPoolEntry; frontierSize: number } {
  const { frontier, winCounts } = computeParetoFrontier(pool);
  if (frontier.length === 0) return { parent: pool[0]!, frontierSize: 0 };
  return { parent: weightedRandomPick(frontier, winCounts, rng), frontierSize: frontier.length };
}

function parseCandidateDraftPayload(raw: string, fallbackName: string): { name: string; rationale: string; candidateBody: string } {
  const payload = extractJsonPayload(raw) as Record<string, unknown> | unknown[];
  let r: Record<string, unknown> = {};
  if (Array.isArray(payload)) r = (payload[0] as Record<string, unknown>) ?? {};
  else { const p = payload as Record<string, unknown>; if (Array.isArray(p.candidates) && p.candidates.length > 0) r = (p.candidates[0] as Record<string, unknown>) ?? {}; else r = p; }
  let cb = String(r.candidateBody ?? r.candidate_body ?? "").trim();
  if (!cb) throw new Error(`${fallbackName}: model returned no candidateBody.`);
  if (cb.startsWith("---")) cb = splitFrontmatter(cb).body;
  return { name: slugify(String(r.name ?? fallbackName)) || fallbackName, rationale: String(r.rationale ?? "").trim() || `${fallbackName} mutation.`, candidateBody: cb };
}

async function generateOneCandidateDraft(options: { cwd: string; model?: string; thinkingLevel?: string; target: ArtifactTarget; objective: string; trainExamples: EvalExample[]; reflection: ReflectionPrompt; maxBytes: number; iteration: number; parentName?: string; parentBody: string; signal?: AbortSignal }): Promise<CandidateDraft> {
  const failures = options.reflection.priorTraces.filter((t) => t.isFailure);
  // Surface real executor stdout (not judge JSON — the `hasRealExecution` gate matters: showing
  // the judge's own speculative output as "what the agent did" would mislead the mutation rather
  // than inform it) for the worst few failures, so the mutator can target the actual observed
  // defect (wrong step order, missing action) instead of paraphrasing a one-line judge opinion.
  let excerptsShown = 0;
  const traceLines = failures.slice(0, 5).map((t, i) => {
    const base = `  ${i + 1}. Task: ${t.taskInput}\n     Scores: correctness=${t.scores.correctness.toFixed(2)}, procedure=${t.scores.procedureFollowing.toFixed(2)}, composite=${t.scores.composite.toFixed(2)}\n     Feedback: ${t.feedback}`;
    if (t.hasRealExecution && excerptsShown < 3) {
      excerptsShown += 1;
      return `${base}\n     Observed agent stdout (truncated): ${t.rawOutput.slice(0, 1200).replace(/\n/g, "\n       ")}`;
    }
    return base;
  });
  const traceSection = failures.length > 0 ? ["", "Observed failure traces (from the parent candidate below):", ...traceLines].join("\n") : "";
  const feedbackSection = options.reflection.priorJudgeFeedback.length > 0 ? ["", "Prior judge feedback to address:", ...options.reflection.priorJudgeFeedback.slice(0, 6).map((f, i) => `  ${i + 1}. ${f}`)].join("\n") : "";
  const parentSection = options.parentName ? `\nParent candidate: ${options.parentName} (this is the Pareto-frontier candidate selected to mutate; edit its BODY below, do not regress its gains).` : "\nParent candidate: <baseline>";
  const prompt = [`Iteration: ${options.iteration}`, `Artifact type: ${options.target.type}`, `Path: ${options.target.path}`, `Objective: ${options.objective}`, `Max bytes: ${options.maxBytes}`, `Placeholders: ${options.target.placeholders.length > 0 ? options.target.placeholders.join(", ") : "none"}`, `Top heading: ${options.target.topHeading ?? "none"}`, parentSection, "", "Parent BODY (mutate this, not the original):", "```", options.parentBody.trim(), "```", "", "Training tasks:", options.trainExamples.map((e, i) => `${i + 1}. ${e.taskInput}\n   Rubric: ${e.expectedBehavior}`).join("\n\n"), "", "Weaknesses (parent):", options.reflection.weaknessSummary, traceSection, feedbackSection, "", 'Return JSON: {"name":"short-kebab","rationale":"paragraph explaining mutation","candidateBody":"full revised body"}', "", "Rules: produce ONE revision of the parent BODY that targets its failure traces and judge feedback, preserve placeholders, keep markdown valid, never mention evaluation/scores."].join("\n");
  const raw = await runPiTextTask({ cwd: options.cwd, model: options.model, thinkingLevel: options.thinkingLevel, systemPrompt: CANDIDATE_SYSTEM_PROMPT, prompt, signal: options.signal });
  return parseCandidateDraftPayload(raw, `iter-${options.iteration}`);
}

/**
 * System-aware merge (GEPA "crossover", Appendix F): pick two Pareto-frontier candidates from
 * distinct mutation lineages that each win on different validation instances, and ask the model
 * to synthesize a single body combining their complementary strengths. Bounded by MAX_MERGE_ATTEMPTS
 * in the caller so this stays a rare, targeted operation rather than a per-iteration cost.
 */
async function generateMergeCandidateDraft(options: { cwd: string; model?: string; thinkingLevel?: string; target: ArtifactTarget; objective: string; a: ParetoPoolEntry; b: ParetoPoolEntry; maxBytes: number; iteration: number; signal?: AbortSignal }): Promise<CandidateDraft> {
  const prompt = [
    `Iteration: ${options.iteration} (system-aware merge)`,
    `Artifact type: ${options.target.type}`, `Path: ${options.target.path}`, `Objective: ${options.objective}`, `Max bytes: ${options.maxBytes}`,
    `Placeholders: ${options.target.placeholders.length > 0 ? options.target.placeholders.join(", ") : "none"}`,
    "",
    `Candidate A ("${options.a.name}") — strongest per-instance judge feedback:`,
    summarizeWeaknesses(options.a.evaluation, 2),
    "Candidate A BODY:", "```", options.a.body.trim(), "```",
    "",
    `Candidate B ("${options.b.name}") — strongest per-instance judge feedback:`,
    summarizeWeaknesses(options.b.evaluation, 2),
    "Candidate B BODY:", "```", options.b.body.trim(), "```",
    "",
    "Both candidates are on the Pareto frontier: each wins on validation instances the other does not.",
    'Return JSON: {"name":"short-kebab","rationale":"paragraph explaining which sections came from A vs B","candidateBody":"full merged body"}',
    "",
    "Rules: synthesize ONE body that keeps A's strengths where A wins and B's strengths where B wins, resolve overlaps in favor of whichever is more concrete, preserve placeholders, keep markdown valid, never mention evaluation/scores.",
  ].join("\n");
  const raw = await runPiTextTask({ cwd: options.cwd, model: options.model, thinkingLevel: options.thinkingLevel, systemPrompt: CANDIDATE_SYSTEM_PROMPT, prompt, signal: options.signal });
  return parseCandidateDraftPayload(raw, `merge-${options.iteration}`);
}

async function safeWriteFile(filePath: string, content: string): Promise<void> { await fs.mkdir(path.dirname(filePath), { recursive: true }); await withFileMutationQueue(filePath, async () => { await fs.writeFile(filePath, content, "utf8"); }); }

function computeUnifiedDiff(label: string, original: string, candidate: string): string {
  const origLines = original.split("\n");
  const candLines = candidate.split("\n");

  // Myers LCS-based diff — build edit script with context
  const m = origLines.length;
  const n = candLines.length;

  // Compute LCS length table
  const dp: number[][] = Array.from({ length: m + 1 }, () => new Array<number>(n + 1).fill(0));
  for (let i = m - 1; i >= 0; i -= 1) {
    for (let j = n - 1; j >= 0; j -= 1) {
      if (origLines[i] === candLines[j]) {
        dp[i]![j] = (dp[i + 1]?.[j + 1] ?? 0) + 1;
      } else {
        dp[i]![j] = Math.max(dp[i + 1]?.[j] ?? 0, dp[i]?.[j + 1] ?? 0);
      }
    }
  }

  // Build edit ops: "=" keep, "-" delete, "+" insert
  const ops: Array<{ op: "=" | "-" | "+"; line: string }> = [];
  let i = 0; let j = 0;
  while (i < m || j < n) {
    if (i < m && j < n && origLines[i] === candLines[j]) {
      ops.push({ op: "=", line: origLines[i]! });
      i += 1; j += 1;
    } else if (j < n && (i >= m || (dp[i]?.[j + 1] ?? 0) >= (dp[i + 1]?.[j] ?? 0))) {
      ops.push({ op: "+", line: candLines[j]! });
      j += 1;
    } else {
      ops.push({ op: "-", line: origLines[i]! });
      i += 1;
    }
  }

  // Collect hunks with 3 lines of context
  const CONTEXT = 3;
  const hunks: string[][] = [];
  let hunk: string[] | null = null;
  let trailingContext = 0;
  let origPos = 1; let candPos = 1;
  let hunkOrigStart = 1; let hunkCandStart = 1;
  let hunkOrigCount = 0; let hunkCandCount = 0;

  const flushHunk = () => {
    if (hunk && hunk.length > 0) {
      hunk.unshift(`@@ -${hunkOrigStart},${hunkOrigCount} +${hunkCandStart},${hunkCandCount} @@`);
      hunks.push(hunk);
    }
    hunk = null; trailingContext = 0;
  };

  const pendingContext: string[] = [];

  for (const { op, line } of ops) {
    if (op === "=") {
      if (hunk) {
        hunk.push(` ${line}`);
        hunkOrigCount += 1; hunkCandCount += 1;
        trailingContext += 1;
        if (trailingContext >= CONTEXT * 2) flushHunk();
      } else {
        pendingContext.push(` ${line}`);
        if (pendingContext.length > CONTEXT) pendingContext.shift();
      }
      origPos += 1; candPos += 1;
    } else {
      if (!hunk) {
        hunk = [...pendingContext];
        const ctxCount = pendingContext.length;
        hunkOrigStart = origPos - ctxCount;
        hunkCandStart = candPos - ctxCount;
        hunkOrigCount = ctxCount; hunkCandCount = ctxCount;
        pendingContext.length = 0;
      }
      trailingContext = 0;
      if (op === "-") { hunk.push(`-${line}`); hunkOrigCount += 1; origPos += 1; }
      else { hunk.push(`+${line}`); hunkCandCount += 1; candPos += 1; }
    }
  }
  flushHunk();

  if (hunks.length === 0) return `--- ${label} (original)\n+++ ${label} (best-candidate)\n(no textual changes)\n`;
  const header = `--- ${label} (original)\n+++ ${label} (best-candidate)\n`;
  return header + hunks.map((h) => h.join("\n")).join("\n") + "\n";
}

function buildReportMarkdown(result: EvolutionRunResult): string {
  const baselineValidation = result.baselineValidation.aggregate.composite;
  const bestValidation = result.bestCandidate.evaluation.aggregate.composite;
  const baselineHoldout = result.baselineHoldout.aggregate.composite;
  const bestHoldout = result.bestCandidate.holdoutEvaluation?.aggregate.composite ?? bestValidation;
  const totalTraces = result.baselineTraces.length + result.candidates.reduce((s, c) => s + c.executionTraces.length, 0);
  const failures = [...result.baselineTraces, ...result.candidates.flatMap((c) => c.executionTraces)].filter((t) => t.isFailure);
  const diffText = computeUnifiedDiff(result.target.name, result.target.body, result.bestCandidate.candidateBody);
  const diffLines = diffText.split("\n").length;
  const diffPreview = diffText.length > 4000
    ? `${diffText.slice(0, 4000).trimEnd()}\n… (truncated — full diff in diff.patch)`
    : diffText.trimEnd();
  return [
    "# Hermes-style Self-Evolution Report", "",
    `- **Target:** ${result.target.path}`, `- **Type:** ${result.target.type}`, `- **Objective:** ${result.objective}`,
    `- **Source:** ${result.evalSource}`, `- **Model:** ${result.modelLabel}`, `- **Run dir:** ${result.paths.runDir}`,
    `- **Selection split:** ${result.selectionSplit}`, `- **Confirmation split:** ${result.confirmationSplit}`,
    `- **Baseline validation:** ${baselineValidation.toFixed(3)}`, `- **Best validation:** ${bestValidation.toFixed(3)}`,
    `- **Baseline holdout:** ${baselineHoldout.toFixed(3)}`, `- **Confirmed holdout:** ${bestHoldout.toFixed(3)}`, `- **Improvement:** ${result.improvement >= 0 ? "+" : ""}${result.improvement.toFixed(3)}`,
    `- **Traces:** ${totalTraces} captured, ${failures.length} failures`, "",
    "## Guardrails", "- Original preserved, never auto-overwritten.", "- Frontmatter preserved verbatim.", "- Placeholders required to survive.", "- Size budget enforced.", "- Growth limited to 20%.", "- Semantic drift checked (threshold 0.4, fail-closed when unmeasurable).", "- Secret scanning on datasets (span-based redaction).", "- Hard gates have no recovery path: a gate-failing candidate is never promoted; the fallback outcome is the baseline itself.", "",
    "## Optimization strategy (GEPA-Pareto)",
    "- Parent selection: Pareto-frontier sampling over per-instance validation scores (arXiv:2507.19457), not a single greedy best-so-far chain.",
    "- Mutation edits the selected parent's own body, so accepted gains compound instead of being re-derived from the original each iteration.",
    `- Minibatch pre-filter: ${result.minibatchFilteredCount ?? 0} draft(s) rejected on a cheap train-set subset before a full validation + real-executor pass.`,
    `- System-aware merge attempts: ${result.mergeAttempts ?? 0}.`,
    `- Final Pareto frontier: ${result.paretoFrontier && result.paretoFrontier.length > 0 ? result.paretoFrontier.join(", ") : "n/a"}.`,
    "",
    "## Baseline weaknesses", summarizeWeaknesses(result.baselineTrain, 3), "",
    "## Execution traces", `- Total: ${totalTraces}`, `- Failures: ${failures.length}`, ...failures.slice(0, 5).map((t, i) => `${i + 1}. [${t.traceId}] composite=${t.scores.composite.toFixed(2)} — ${t.feedback.slice(0, 120)}`), "",
    "## Candidates", "| Name | Parent | Method | Validation | Holdout | Correctness | Procedure | Conciseness | Constraints | Drift |", "|---|---|---|---:|---:|---:|---:|---:|---|---|",
    ...result.candidates.map((c) => `| ${c.name} | ${c.parentCandidates && c.parentCandidates.length > 0 ? c.parentCandidates.join(" + ") : "—"} | ${c.selectionMethod ?? "mutation"} | ${c.evaluation.aggregate.composite.toFixed(3)} | ${c.holdoutEvaluation?.aggregate.composite.toFixed(3) ?? "—"} | ${c.evaluation.aggregate.correctness.toFixed(3)} | ${c.evaluation.aggregate.procedureFollowing.toFixed(3)} | ${c.evaluation.aggregate.conciseness.toFixed(3)} | ${c.constraints.every((x) => x.passed) ? "✅" : "❌"} | ${c.semanticDriftScore?.toFixed(2) ?? "—"} |`),
    "",
    "## Best candidate", `- **Name:** ${result.bestCandidate.name}`, `- **Rationale:** ${result.bestCandidate.rationale}`,
    ...result.bestCandidate.constraints.map((c) => `- ${c.passed ? "✅" : "❌"} **${c.name}**: ${c.message}`),
    `- **Drift:** ${result.bestCandidate.semanticDriftScore?.toFixed(3) ?? "not checked"}`,
    "",
    "## Diff (original → best candidate)",
    `_${diffLines} lines — see \`diff.patch\` for the full patch._`,
    "",
    "```diff",
    diffPreview,
    "```",
    "",
    "## Selected winner confirmation",
    `- **Winner chosen on:** ${result.selectionSplit}`,
    `- **Validation score:** ${bestValidation.toFixed(3)}`,
    `- **Holdout confirmation:** ${bestHoldout.toFixed(3)}`,
    "",
    "### Holdout weaknesses",
    summarizeWeaknesses(result.bestCandidate.holdoutEvaluation ?? result.bestCandidate.evaluation, 3),
    result.prResult ? `\n## PR\n- **Branch:** ${result.prResult.branch}\n- **Commit:** ${result.prResult.commitSha.slice(0, 12)}\n- **URL:** ${result.prResult.prUrl ?? "not created"}` : "",
    "", "## Files", `- Original: ${result.paths.originalPath}`, `- Best: ${result.paths.bestCandidatePath}`, `- Diff: ${result.paths.runDir}/diff.patch`, `- Dataset: ${result.paths.datasetPath}`, `- Manifest: ${result.paths.manifestPath}`, `- Traces: ${result.paths.runDir}/traces/`, `- Report: ${result.paths.reportPath}`,
  ].join("\n");
}

async function runTypeScriptEvolution(options: {
  cwd: string; targetPath: string; objective: string; evalSource: EvalSource; model?: string; thinkingLevel?: string;
  candidateCount: number; maxExamples: number; sessionQuery?: string; goldenTaskId?: string;
  testCommand?: string; testTimeout?: number; createPR?: boolean; persistGolden?: boolean;
  seed?: number; cohortExamples?: EvalExample[];
  cohortJudgeFunc?: (artifactText: string, examples: EvalExample[]) => Promise<{ composite: number }>;
  coherenceCheck?: (candidateText: string) => Promise<{ passed: boolean; detail: string }>;
  tsConfigPath?: string;
  signal?: AbortSignal; onProgress?: (phase: string, detail?: string) => void;
}): Promise<EvolutionRunResult> {
  const target = await resolveArtifactTarget(options.targetPath, options.cwd);
  const modelLabel = options.model ?? "current-session-model";
  const runDir = path.join(options.cwd, ".pi", "hermes-self-evolution", "runs", `${formatTimestamp()}-${slugify(target.name || "artifact")}`);
  const maxBytes = computeMaxBytes(target.originalBytes);
  const constraintConfig = buildConstraintConfig(target, maxBytes, { testCommand: options.testCommand, testTimeoutMs: (options.testTimeout ?? 60) * 1000 });

  // Golden dataset
  let usedPersistedGolden = false; let train: EvalExample[] = []; let validation: EvalExample[] = []; let holdout: EvalExample[] = [];
  let sessionSnippets: ReturnType<typeof mineSessionSnippets> = [];
  if (options.goldenTaskId && options.persistGolden !== false) {
    const loaded = await loadGoldenDataset(options.cwd, options.goldenTaskId);
    if (loaded && loaded.train.length > 0) { train = loaded.train; validation = loaded.validation; holdout = loaded.holdout; usedPersistedGolden = true; options.onProgress?.("dataset", `Loaded golden "${options.goldenTaskId}"`); }
  }
  if (!usedPersistedGolden) {
    options.onProgress?.("dataset", "Generating evaluation set");
    const ds = await generateDataset({ cwd: options.cwd, model: options.model, thinkingLevel: options.thinkingLevel, target, objective: options.objective, evalSource: options.evalSource, maxExamples: Math.max(4, options.maxExamples), sessionQuery: options.sessionQuery, signal: options.signal, onProgress: (d) => options.onProgress?.("dataset", d) });
    const splits = splitExamples(ds.examples, options.seed); train = splits.train; validation = splits.validation; holdout = splits.holdout; sessionSnippets = ds.sessionSnippets;
  }
  const golden = buildGoldenDataset(validation, options.goldenTaskId);
  if (golden && options.persistGolden !== false && !usedPersistedGolden) await saveGoldenDataset(options.cwd, golden, train, validation, holdout, target.path, target.name);

  // Baseline evaluation with traces. Holdout and validation are executor-grounded (not just
  // train) because their scores feed the Pareto pool's baseline entry and the headline
  // improvement number — those need to be measured under the same regime as candidates
  // (which are always executor-grounded on validation, see the iteration loop below) or
  // "improvement" is comparing judged prose against judged real behavior. Train stays
  // judge-only: it only informs the baseline weakness summary/reflection, never a candidate
  // comparison, so the regime doesn't need to match.
  const baselineArtifactName = slugify(target.name) || "baseline";
  options.onProgress?.("baseline", "Train"); const { evaluation: baselineTrain, traces: btt } = await evaluateArtifact({ cwd: options.cwd, model: options.model, thinkingLevel: options.thinkingLevel, target, artifactText: target.fullText, objective: options.objective, examples: train, maxBytes, signal: options.signal, onProgress: (d) => options.onProgress?.("baseline", d) });
  options.onProgress?.("baseline", "Holdout"); const { evaluation: baselineHoldout, traces: bht } = await evaluateArtifact({ cwd: options.cwd, model: options.model, thinkingLevel: options.thinkingLevel, target, artifactText: target.fullText, objective: options.objective, examples: holdout, maxBytes, signal: options.signal, useRealExecutor: true, executorLogDir: path.join(runDir, "executor", "baseline-holdout"), artifactName: baselineArtifactName });
  options.onProgress?.("baseline", "Validation"); const { evaluation: baselineValidation, traces: bvt } = await evaluateArtifact({ cwd: options.cwd, model: options.model, thinkingLevel: options.thinkingLevel, target, artifactText: target.fullText, objective: options.objective, examples: validation, maxBytes, signal: options.signal, useRealExecutor: true, executorLogDir: path.join(runDir, "executor", "baseline-validation"), artifactName: baselineArtifactName });
  // Keep the executor-grounded baseline traces too (not just the judge-only train ones): they
  // are the real-behavior record for the retained-baseline outcome and for trace reporting.
  // They feed reporting only — reflection pulls from pool-entry minibatch traces, so the
  // validation/holdout splits still never reach the proposer.
  const baselineTraces = [...btt, ...bht, ...bvt];

  // Lineage: try to find best ancestor for this artifact path, and — if its winning body is
  // still hash-verifiable and would still pass the CURRENT baseline's constraints — resolve it
  // for seeding the Pareto pool below. Without this, `loadBestAncestor`'s result was write-only
  // (a progress log plus lineage parentRunId/parentArtifactHash chaining): run N+1 always started
  // its search from the on-disk baseline, re-deriving whatever run N already found instead of
  // building on it. Gated on hash verification (a human edit or pruned run invalidates it) and
  // re-validated against the current target's constraints (frontmatter/placeholders/size may
  // have changed on disk since that run).
  let ancestor: Awaited<ReturnType<typeof loadBestAncestor>> | undefined;
  let ancestorSeed: { body: string; fullText: string; entry: LineageEntry } | undefined;
  try {
    ancestor = await loadBestAncestor(options.cwd, target.path);
    if (ancestor) {
      options.onProgress?.("lineage", `ancestor_id=${ancestor.runId} score=${ancestor.score.toFixed(3)}`);
      const ancestorFullText = await resolveAncestorBody(options.cwd, ancestor);
      if (ancestorFullText && ancestorFullText !== target.fullText) {
        const { body: ancestorBody } = splitFrontmatter(ancestorFullText);
        const ancestorConstraints = validateConstraints(target, ancestorBody, ancestorFullText, constraintConfig);
        if (ancestorConstraints.valid) {
          ancestorSeed = { body: ancestorBody, fullText: ancestorFullText, entry: ancestor };
          options.onProgress?.("lineage", "ancestor body hash-verified and constraint-valid; seeding pool");
        } else {
          options.onProgress?.("lineage", `ancestor body failed current constraints; not seeding (${ancestorConstraints.results.filter((r) => !r.passed).map((r) => r.name).join(", ")})`);
        }
      }
    }
  } catch { /* sibling module unavailable */ }

  // Iterative reflective loop, GEPA-Pareto style (arXiv:2507.19457): maintain a candidate pool
  // and select each iteration's mutation parent via Pareto-frontier sampling instead of a single
  // greedy best-so-far chain (which collapses to a local optimum once the first easy gains are
  // exhausted). New drafts mutate the SELECTED PARENT's own body (not the pristine original) so
  // gains actually compound. A cheap train-set minibatch pre-filters drafts before paying for a
  // full validation pass with the real pi executor, and a bounded system-aware merge periodically
  // combines two frontier candidates' complementary strengths into one.
  const iterationCount = Math.min(5, Math.max(1, options.candidateCount || 3));
  options.onProgress?.("iterations", `Running ${iterationCount} reflective iteration(s)`);
  const iterations: IterationRecord[] = []; const candidates: CandidateRecord[] = [];
  const paretoRng = options.seed !== undefined ? mulberry32(options.seed ^ 0x9e3779b9) : Math.random;
  const minibatch = train.slice(0, Math.max(1, Math.min(2, train.length)));
  // The reflection minibatch (a train subset) is EXECUTOR-GROUNDED for every pool entry,
  // baseline included: pool-entry traces are what reflection prompts show the mutator, so they
  // must be real pi execution transcripts, not judge speculation — and the minibatch pre-filter
  // compares parent vs. child minibatch scores, so both sides must be measured under the same
  // regime. Train examples are used, keeping validation/holdout hidden from the proposer.
  options.onProgress?.("baseline", "Reflection minibatch");
  const { evaluation: baselineMinibatch, traces: baselineMinibatchTraces } = await evaluateArtifact({ cwd: options.cwd, model: options.model, thinkingLevel: options.thinkingLevel, target, artifactText: target.fullText, objective: options.objective, examples: minibatch, maxBytes, signal: options.signal, useRealExecutor: true, executorLogDir: path.join(runDir, "executor", "baseline-minibatch"), artifactName: baselineArtifactName });
  const pool: ParetoPoolEntry[] = [{
    name: "baseline",
    body: target.body,
    fullText: target.fullText,
    validationScores: baselineValidation.examples.map((e) => e.composite),
    minibatchScore: baselineMinibatch.aggregate.composite,
    traces: baselineMinibatchTraces,
    evaluation: baselineMinibatch,
  }];
  if (ancestorSeed) {
    // Same evaluation regime as every other pool entry: executor-grounded validation score for
    // Pareto bookkeeping, executor-grounded minibatch score/traces (matching the mutation-parent
    // filter reference and reflection source) for everything else.
    options.onProgress?.("lineage", "Evaluating ancestor seed on validation");
    const ancestorArtifactName = slugify(`ancestor-${ancestorSeed.entry.runId}`) || "ancestor";
    const { evaluation: ancestorValidation } = await evaluateArtifact({ cwd: options.cwd, model: options.model, thinkingLevel: options.thinkingLevel, target, artifactText: ancestorSeed.fullText, objective: options.objective, examples: validation, maxBytes, signal: options.signal, useRealExecutor: true, executorLogDir: path.join(runDir, "executor", "ancestor-seed"), artifactName: ancestorArtifactName });
    const { evaluation: ancestorMinibatch, traces: ancestorMinibatchTraces } = await evaluateArtifact({ cwd: options.cwd, model: options.model, thinkingLevel: options.thinkingLevel, target, artifactText: ancestorSeed.fullText, objective: options.objective, examples: minibatch, maxBytes, signal: options.signal, useRealExecutor: true, executorLogDir: path.join(runDir, "executor", "ancestor-minibatch"), artifactName: ancestorArtifactName });
    pool.push({
      name: "ancestor",
      body: ancestorSeed.body,
      fullText: ancestorSeed.fullText,
      validationScores: ancestorValidation.examples.map((e) => e.composite),
      minibatchScore: ancestorMinibatch.aggregate.composite,
      traces: ancestorMinibatchTraces,
      evaluation: ancestorMinibatch,
    });
  }
  // Same-cohort paired baseline for the tiered gate's cohort tier, computed ONCE per run by
  // judging the baseline artifact on the exact cohort candidates will be judged on. The previous
  // code compared the candidate's cohort score against the sealed holdout baseline aggregate —
  // a different example set (an invalid unpaired comparison) that also leaked the holdout
  // baseline into the search loop. If baseline judging fails even after a retry, the score stays
  // undefined and the cohort tier fails closed for every candidate.
  let cohortBaselineScore: number | undefined;
  if (options.cohortExamples && options.cohortExamples.length > 0 && options.cohortJudgeFunc) {
    for (let attempt = 0; attempt < 2 && cohortBaselineScore === undefined; attempt += 1) {
      try { cohortBaselineScore = (await options.cohortJudgeFunc(target.fullText, options.cohortExamples)).composite; }
      catch (err) { options.onProgress?.("gate", `cohort baseline judge attempt ${attempt + 1} failed: ${err instanceof Error ? err.message : String(err)}`); }
    }
    if (cohortBaselineScore === undefined) options.onProgress?.("gate", "cohort baseline unavailable; cohort tier will fail closed");
  }

  const MERGE_EVERY = 3; const MAX_MERGE_ATTEMPTS = 2; let mergeAttempts = 0; let minibatchFilteredCount = 0;
  for (let iter = 1; iter <= iterationCount; iter += 1) {
    const { parent, frontierSize } = selectParetoParent(pool, paretoRng);
    const parentReflection = buildReflectionPrompt(options.objective, parent.traces, parent.evaluation);
    let draft: CandidateDraft | undefined; let selectionMethod: "mutation" | "merge" = "mutation";
    let parentNames: string[] = [parent.name]; let recordedReflection: ReflectionPrompt = parentReflection; let reportedFrontierSize = frontierSize;
    // Comparison baseline for the minibatch filter and scoreDelta: the sampled mutation `parent`
    // for a mutation draft, or the stronger of the two merge inputs for a merge draft (a merge
    // must be compared against what it was actually built from, not an unrelated sampled parent).
    let comparisonMinibatchScore = parent.minibatchScore;
    let comparisonValidationScore = mean(parent.validationScores);

    // Merge eligibility uses distinct FRONTIER MEMBERS (not a persistent lineage id): mutation
    // children are pool entries in their own right, each a candidate "lineage tip" with its own
    // per-instance strengths, so any two distinct frontier members are valid merge parents.
    if (iter > 1 && iter % MERGE_EVERY === 0 && mergeAttempts < MAX_MERGE_ATTEMPTS && pool.length >= 3) {
      const { frontier } = computeParetoFrontier(pool);
      if (frontier.length >= 2) {
        const [a, b] = [frontier[0]!, frontier[frontier.length - 1]!];
        options.onProgress?.("iteration", `${iter}/${iterationCount} merge ${a.name} + ${b.name}`);
        try {
          draft = await generateMergeCandidateDraft({ cwd: options.cwd, model: options.model, thinkingLevel: options.thinkingLevel, target, objective: options.objective, a, b, maxBytes, iteration: iter, signal: options.signal });
          selectionMethod = "merge"; mergeAttempts += 1; parentNames = [a.name, b.name]; reportedFrontierSize = frontier.length;
          comparisonMinibatchScore = Math.max(a.minibatchScore, b.minibatchScore);
          comparisonValidationScore = Math.max(mean(a.validationScores), mean(b.validationScores));
          recordedReflection = { priorTraces: [], priorJudgeFeedback: [], objective: options.objective, weaknessSummary: `System-aware merge of Pareto-frontier candidates "${a.name}" and "${b.name}".` };
        } catch (err) { options.onProgress?.("iteration", `${iter} merge skipped: ${err instanceof Error ? err.message : String(err)}`); }
      }
    }

    if (!draft) {
      options.onProgress?.("iteration", `${iter}/${iterationCount} reflect+mutate (parent=${parent.name}, frontier=${frontierSize})`);
      try { draft = await generateOneCandidateDraft({ cwd: options.cwd, model: options.model, thinkingLevel: options.thinkingLevel, target, objective: options.objective, trainExamples: train, reflection: parentReflection, maxBytes, iteration: iter, parentName: parent.name, parentBody: parent.body, signal: options.signal }); }
      catch (err) { options.onProgress?.("iteration", `${iter} skipped: ${err instanceof Error ? err.message : String(err)}`); continue; }
    }

    const fullText = reassembleArtifact(target.frontmatter, draft.candidateBody);
    const cr = validateConstraints(target, draft.candidateBody, fullText, constraintConfig);
    if (!cr.valid) {
      const iterRecord: IterationRecord = { iteration: iter, parentCandidate: parentNames[0], parentCandidates: parentNames, mutationRationale: draft.rationale, reflectionPrompt: recordedReflection, candidate: { name: draft.name, rationale: draft.rationale, candidateBody: draft.candidateBody }, evaluation: parent.evaluation, traces: [], scoreDelta: 0, accepted: false, selectionMethod, paretoFrontierSize: reportedFrontierSize };
      iterations.push(iterRecord);
      await safeWriteFile(path.join(runDir, "iterations", `${iter}.json`), JSON.stringify({ ...iterRecord, candidateFullText: fullText, constraints: cr.results, warnings: cr.warnings }, null, 2));
      options.onProgress?.("iteration", `${iter} rejected: constraints failed`);
      continue;
    }

    // Tiered regression gate (typecheck → cohort → coherence) runs first: it's cheap (no judge
    // LLM calls beyond an optional injected callback) and orthogonal to draft quality, so a
    // failing tier should short-circuit before paying for judge/executor rollouts at all. An
    // exception from the gate itself (as opposed to a tier resolving passed:false, which
    // runTieredGate already handles internally) is treated as a rejection, not a silent skip —
    // letting a candidate through because the safety check errored would defeat the gate.
    let gateResults: TieredGateResult[] | undefined;
    let gateError: string | undefined;
    try {
      gateResults = await runTieredGate({
        cwd: options.cwd,
        candidateText: fullText,
        signal: options.signal,
        cohortExamples: options.cohortExamples,
        judgeFunc: options.cohortJudgeFunc,
        coherenceCheck: options.coherenceCheck,
        tsConfigPath: options.tsConfigPath,
        baselineScore: cohortBaselineScore,
      });
    } catch (err) { gateError = err instanceof Error ? err.message : String(err); }
    if (gateError !== undefined) {
      const iterRecord: IterationRecord = { iteration: iter, parentCandidate: parentNames[0], parentCandidates: parentNames, mutationRationale: draft.rationale, reflectionPrompt: recordedReflection, candidate: { name: draft.name, rationale: draft.rationale, candidateBody: draft.candidateBody }, evaluation: parent.evaluation, traces: [], scoreDelta: 0, accepted: false, selectionMethod, paretoFrontierSize: reportedFrontierSize };
      iterations.push(iterRecord);
      await safeWriteFile(path.join(runDir, "iterations", `${iter}.json`), JSON.stringify({ ...iterRecord, candidateFullText: fullText, constraints: cr.results, warnings: [...cr.warnings, `Tiered gate execution error: ${gateError}`] }, null, 2));
      options.onProgress?.("iteration", `${iter} rejected: tiered gate execution error: ${gateError}`);
      continue;
    }
    if (gateResults && gateResults.some((r) => !r.passed)) {
      const failedTier = gateResults.find((r) => !r.passed)!;
      const iterRecord: IterationRecord = { iteration: iter, parentCandidate: parentNames[0], parentCandidates: parentNames, mutationRationale: draft.rationale, reflectionPrompt: recordedReflection, candidate: { name: draft.name, rationale: draft.rationale, candidateBody: draft.candidateBody }, evaluation: parent.evaluation, traces: [], scoreDelta: 0, accepted: false, selectionMethod, paretoFrontierSize: reportedFrontierSize, gateResults };
      iterations.push(iterRecord);
      await safeWriteFile(path.join(runDir, "iterations", `${iter}.json`), JSON.stringify({ ...iterRecord, candidateFullText: fullText, constraints: cr.results, warnings: cr.warnings, gateResults }, null, 2));
      options.onProgress?.("iteration", `${iter} rejected: tiered gate failed at "${failedTier.tier}" (${failedTier.reasonCode})`);
      continue;
    }

    // Minibatch pre-filter: an executor-grounded 1-2 example train subset before paying for a
    // full validation pass. Only reject a CLEAR regression vs. the comparison baseline on the
    // same subset — ties proceed, since a 1-2 example score is noisy and a tie carries no real
    // signal to reject on. This is GEPA's main lever for the "35x fewer rollouts" result. The
    // comparison baseline is the sampled parent for a mutation, or the stronger of the two merge
    // inputs for a merge (see comparisonMinibatchScore above). Real execution here also gives the
    // pool entry genuine transcripts for descendant reflection prompts.
    options.onProgress?.("iteration", `${iter} minibatch pre-filter`);
    const { evaluation: minibatchEval, traces: minibatchTraces } = await evaluateArtifact({ cwd: options.cwd, model: options.model, thinkingLevel: options.thinkingLevel, target, artifactText: fullText, objective: options.objective, examples: minibatch, maxBytes, signal: options.signal, useRealExecutor: true, executorLogDir: path.join(runDir, "executor", `minibatch-${iter}`), artifactName: draft.name });
    if ((minibatchEval.executorFailureCount ?? 0) > 0) {
      // Fail closed: an executor failure means this minibatch score is not comparable to the
      // parent's executor-grounded one — do not filter on it, and do not proceed to validation
      // under a mixed-regime measurement.
      const iterRecord: IterationRecord = { iteration: iter, parentCandidate: parentNames[0], parentCandidates: parentNames, mutationRationale: draft.rationale, reflectionPrompt: recordedReflection, candidate: { name: draft.name, rationale: draft.rationale, candidateBody: draft.candidateBody }, evaluation: minibatchEval, traces: minibatchTraces, scoreDelta: 0, accepted: false, selectionMethod, paretoFrontierSize: reportedFrontierSize, minibatchFiltered: true, gateResults };
      iterations.push(iterRecord);
      await safeWriteFile(path.join(runDir, "iterations", `${iter}.json`), JSON.stringify({ ...iterRecord, candidateFullText: fullText, constraints: cr.results, warnings: [...cr.warnings, `Executor failed on ${minibatchEval.executorFailureCount} minibatch example(s); rejected fail-closed.`], gateResults }, null, 2));
      options.onProgress?.("iteration", `${iter} rejected: executor failed on ${minibatchEval.executorFailureCount} minibatch example(s) (fail-closed)`);
      continue;
    }
    const minibatchScore = minibatchEval.aggregate.composite;
    if (minibatchScore < comparisonMinibatchScore - 1e-9) {
      minibatchFilteredCount += 1;
      const iterRecord: IterationRecord = { iteration: iter, parentCandidate: parentNames[0], parentCandidates: parentNames, mutationRationale: draft.rationale, reflectionPrompt: recordedReflection, candidate: { name: draft.name, rationale: draft.rationale, candidateBody: draft.candidateBody }, evaluation: minibatchEval, traces: minibatchTraces, scoreDelta: minibatchScore - comparisonMinibatchScore, accepted: false, selectionMethod, paretoFrontierSize: reportedFrontierSize, minibatchFiltered: true, gateResults };
      iterations.push(iterRecord);
      await safeWriteFile(path.join(runDir, "iterations", `${iter}.json`), JSON.stringify({ ...iterRecord, candidateFullText: fullText, constraints: cr.results, warnings: cr.warnings, gateResults }, null, 2));
      options.onProgress?.("iteration", `${iter} minibatch-filtered: ${minibatchScore.toFixed(3)} < comparison baseline ${comparisonMinibatchScore.toFixed(3)}`);
      continue;
    }

    options.onProgress?.("iteration", `${iter} judge on validation`);
    const execLogDir = path.join(runDir, "executor", String(iter));
    const { evaluation, traces: cTraces } = await evaluateArtifact({ cwd: options.cwd, model: options.model, thinkingLevel: options.thinkingLevel, target, artifactText: fullText, objective: options.objective, examples: validation, maxBytes, signal: options.signal, useRealExecutor: true, executorLogDir: execLogDir, artifactName: draft.name });
    // Drift. An unavailable drift judge blocks the candidate (fail-closed) — it is never mapped
    // to a passing score.
    let driftScore: number | undefined;
    if (constraintConfig.checkSemanticDrift) {
      options.onProgress?.("drift", draft.name);
      const d = await computeSemanticDrift(options.cwd, target.body, draft.candidateBody, options.objective, options.model, options.thinkingLevel, options.signal);
      if (d.status === "ok" && d.score !== undefined) {
        driftScore = d.score;
        cr.results.push({ name: "semantic_drift" as ConstraintName, passed: d.score <= constraintConfig.maxDriftScore, message: `Drift: ${d.score.toFixed(3)} (max ${constraintConfig.maxDriftScore}). ${d.feedback}` });
        if (d.score > constraintConfig.maxDriftScore) cr.warnings.push(`Semantic drift too high: ${d.score.toFixed(3)}`);
      } else {
        cr.results.push({ name: "semantic_drift" as ConstraintName, passed: false, message: `Drift unmeasurable (fail-closed): ${d.feedback}` });
        cr.warnings.push("Semantic drift could not be measured; candidate blocked fail-closed.");
      }
    }
    // Test gate
    let testPassed: boolean | undefined;
    if (options.testCommand) {
      options.onProgress?.("test", draft.name);
      await safeWriteFile(target.path, fullText); const tr = await runTestCommand(options.testCommand, options.cwd, constraintConfig.testTimeoutMs, options.signal);
      testPassed = tr.passed; await safeWriteFile(target.path, target.fullText);
      if (!tr.passed) cr.warnings.push(`Test failed (exit ${tr.exitCode})`);
    }
    const constraintsPass = cr.results.every((r) => r.passed); const composite = evaluation.aggregate.composite;
    const scoreDelta = composite - comparisonValidationScore;
    const validationExecutorClean = (evaluation.executorFailureCount ?? 0) === 0;
    if (!validationExecutorClean) cr.warnings.push(`Executor failed on ${evaluation.executorFailureCount} validation example(s); candidate blocked fail-closed.`);
    // "Accepted" means "cleared every hard gate on a clean, fully executor-grounded validation
    // pass" — it is added to the Pareto pool regardless of whether its aggregate beats the
    // parent, because the pool wants diverse per-instance strengths to sample from, not just a
    // chain of monotone wins. An evaluation with executor failures is not a valid measurement
    // and can neither accept the candidate nor enter the pool.
    const accepted = constraintsPass && validationExecutorClean && (testPassed === undefined ? true : testPassed);
    const candidateRecord: CandidateRecord = { ...draft, candidateFullText: fullText, evaluation, executionTraces: cTraces, constraints: cr.results, warnings: cr.warnings, semanticDriftScore: driftScore, testPassed, gateResults, parentCandidates: parentNames, selectionMethod, minibatchScore };
    const iterRecord: IterationRecord = { iteration: iter, parentCandidate: parentNames[0], parentCandidates: parentNames, mutationRationale: draft.rationale, reflectionPrompt: recordedReflection, candidate: { name: draft.name, rationale: draft.rationale, candidateBody: draft.candidateBody }, evaluation, traces: cTraces, scoreDelta, accepted, gateResults, selectionMethod, paretoFrontierSize: reportedFrontierSize, minibatchFiltered: false };
    iterations.push(iterRecord); await safeWriteFile(path.join(runDir, "iterations", `${iter}.json`), JSON.stringify({ ...iterRecord, candidateFullText: fullText, constraints: cr.results, warnings: cr.warnings, semanticDriftScore: driftScore, testPassed, gateResults }, null, 2));
    if (accepted) {
      candidates.push(candidateRecord);
      // Pool entries feed future mutation/merge prompts via buildReflectionPrompt/summarizeWeaknesses
      // (recordedReflection above, and generateMergeCandidateDraft's per-candidate weakness
      // summaries). Store the MINIBATCH traces/evaluation here, not the validation-pass ones —
      // validation is the selection split (computeParetoFrontier, winner ranking); if descendants
      // are reflected against the exact instances used to select and rank them, they're optimized
      // directly against the metric that judges them, inflating validation scores by construction.
      // `validationScores` (used only for Pareto bookkeeping, never shown to the model) still
      // comes from the real validation pass.
      pool.push({ name: draft.name, body: draft.candidateBody, fullText, validationScores: evaluation.examples.map((e) => e.composite), minibatchScore, traces: minibatchTraces, evaluation: minibatchEval });
    } else {
      options.onProgress?.("iteration", `${iter} rejected: constraints=${constraintsPass} test=${testPassed ?? "n/a"}`);
    }
  }
  if (candidates.length === 0) {
    // No candidate passed every hard gate this run. There is NO recovery path around hard
    // constraints: a draft that failed (or never completed) constraints, the tiered gate, drift,
    // tests, or a clean executor-grounded validation pass is never promoted, regardless of how
    // well it scored. The only fallback outcome is retaining the baseline itself — its
    // `baselineValidation` evaluation is genuine — reported as a no-safe-improvement run.
    const cr = validateConstraints(target, target.body, target.fullText, constraintConfig);
    options.onProgress?.("iterations", "No candidate passed every hard gate; retaining baseline (no safe improvement).");
    candidates.push({
      name: "baseline",
      rationale: "No mutation or merge this run passed every hard gate (constraints, tiered gate, drift, tests, clean executor measurement); baseline retained unchanged.",
      candidateBody: target.body,
      candidateFullText: target.fullText,
      evaluation: baselineValidation,
      executionTraces: [],
      constraints: cr.results,
      warnings: [...cr.warnings, "No safe improvement: no iteration passed every hard gate this run."],
      wasFallbackPromoted: true,
    });
  }
  candidates.sort((a, b) => b.evaluation.aggregate.composite - a.evaluation.aggregate.composite);
  const bestCandidate = candidates[0]!;
  const finalFrontier = computeParetoFrontier(pool).frontier.map((p) => p.name);
  let improvement: number;
  if (bestCandidate.wasFallbackPromoted) {
    // The retained "winner" IS the baseline text: re-running the executor on the identical
    // artifact would only measure evaluation noise and report it as improvement. Reuse the
    // baseline's own holdout measurement and report exactly zero.
    options.onProgress?.("confirm", "baseline retained; reusing baseline holdout measurement");
    bestCandidate.holdoutEvaluation = baselineHoldout;
    improvement = 0;
  } else {
    options.onProgress?.("confirm", `${bestCandidate.name} on holdout`);
    // Executor-grounded, matching baselineHoldout's regime above — `improvement` (below) is a
    // difference of these two, and comparing a judged-real-behavior score against a judged-prose
    // one would make the headline number meaningless.
    const { evaluation: bestHoldoutEvaluation, traces: bestHoldoutTraces } = await evaluateArtifact({
      cwd: options.cwd,
      model: options.model,
      thinkingLevel: options.thinkingLevel,
      target,
      artifactText: bestCandidate.candidateFullText,
      objective: options.objective,
      examples: holdout,
      maxBytes,
      signal: options.signal,
      useRealExecutor: true,
      executorLogDir: path.join(runDir, "executor", "confirm"),
      artifactName: slugify(bestCandidate.name) || "best-candidate",
    });
    bestCandidate.holdoutEvaluation = bestHoldoutEvaluation;
    bestCandidate.executionTraces.push(...bestHoldoutTraces);
    improvement = bestHoldoutEvaluation.aggregate.composite - baselineHoldout.aggregate.composite;
  }

  // PR. Promotion to a branch/PR requires ALL of: a candidate that passed every hard gate (never
  // the fallback-retained baseline), a positive holdout improvement, and clean executor-grounded
  // measurements on both sides of that holdout comparison.
  const holdoutMeasurementClean = (baselineHoldout.executorFailureCount ?? 0) === 0
    && ((bestCandidate.holdoutEvaluation?.executorFailureCount) ?? 0) === 0;
  const promotionBlockedReason = bestCandidate.wasFallbackPromoted
    ? "no candidate passed every hard gate"
    : !holdoutMeasurementClean
      ? "executor failures during holdout measurement"
      : improvement <= 0
        ? "no positive holdout improvement"
        : undefined;
  let prResult: PRAutomationResult | undefined;
  if (options.createPR) {
    if (promotionBlockedReason === undefined) { options.onProgress?.("pr", "Creating branch"); prResult = await createGitBranchWithCandidate(target, bestCandidate, improvement, runDir, path.join(runDir, "report.md"), options.objective, modelLabel, baselineTraces, candidates, options.cwd); }
    else options.onProgress?.("pr", `Skipped: ${promotionBlockedReason}`);
  }

  const reportPath = path.join(runDir, "report.md"); const originalPath = path.join(runDir, "original.md");
  const bestCandidatePath = path.join(runDir, "best-candidate.md"); const datasetPath = path.join(runDir, "dataset.json");
  const manifestPath = path.join(runDir, "manifest.json"); const tracesDir = path.join(runDir, "traces");

  const result: EvolutionRunResult = {
    target,
    objective: options.objective,
    evalSource: options.evalSource,
    modelLabel,
    selectionSplit: "validation",
    confirmationSplit: "holdout",
    paths: { runDir, reportPath, originalPath, bestCandidatePath, datasetPath, manifestPath },
    sessionSnippets,
    trainExamples: train,
    validationExamples: validation,
    holdoutExamples: holdout,
    golden,
    baselineTrain,
    baselineValidation,
    baselineHoldout,
    candidates,
    bestCandidate,
    improvement,
    maxBytes,
    baselineTraces,
    prResult,
    iterations,
    paretoFrontier: finalFrontier,
    mergeAttempts,
    minibatchFilteredCount,
  };

  options.onProgress?.("write", "Writing artifacts");
  await safeWriteFile(originalPath, target.fullText); await safeWriteFile(bestCandidatePath, bestCandidate.candidateFullText);
  await safeWriteFile(datasetPath, JSON.stringify({ train, validation, holdout, golden: golden ? { id: golden.id, description: golden.description, exampleCount: golden.examples.length } : null, sessionSnippets }, null, 2));
  await safeWriteFile(manifestPath, JSON.stringify({
    targetPath: target.path,
    objective: options.objective,
    evalSource: options.evalSource,
    modelLabel,
    selectionSplit: "validation",
    confirmationSplit: "holdout",
    maxBytes,
    splits: { train: train.length, validation: validation.length, holdout: holdout.length },
    goldenTaskId: options.goldenTaskId || null,
    usedPersistedGolden,
    baselineValidation: baselineValidation.aggregate,
    baselineHoldout: baselineHoldout.aggregate,
    bestCandidateName: bestCandidate.name,
    bestCandidate: {
      name: bestCandidate.name,
      rationale: bestCandidate.rationale,
      validationScore: bestCandidate.evaluation.aggregate,
      holdoutScore: bestCandidate.holdoutEvaluation?.aggregate ?? null,
      warnings: bestCandidate.warnings,
      constraints: bestCandidate.constraints,
      semanticDriftScore: bestCandidate.semanticDriftScore,
      testPassed: bestCandidate.testPassed,
      acceptanceMode: bestCandidate.wasFallbackPromoted ? "fallback" : "strict",
      wasFallbackPromoted: bestCandidate.wasFallbackPromoted === true,
    },
    candidates: candidates.map((c) => ({
      name: c.name,
      rationale: c.rationale,
      warnings: c.warnings,
      validationScore: c.evaluation.aggregate,
      holdoutScore: c.holdoutEvaluation?.aggregate ?? null,
      semanticDriftScore: c.semanticDriftScore,
      testPassed: c.testPassed,
      constraintsPassed: c.constraints.every((x) => x.passed),
      parentCandidates: c.parentCandidates ?? [],
      selectionMethod: c.selectionMethod ?? "mutation",
    })),
    optimization: {
      strategy: "gepa-pareto",
      paretoFrontier: finalFrontier,
      mergeAttempts,
      minibatchFilteredCount,
    },
    promotion: {
      eligible: promotionBlockedReason === undefined,
      blockedReason: promotionBlockedReason ?? null,
      baselineHoldoutExecutorFailures: baselineHoldout.executorFailureCount ?? 0,
      bestHoldoutExecutorFailures: bestCandidate.holdoutEvaluation?.executorFailureCount ?? 0,
    },
    traces: { baselineCount: baselineTraces.length },
    prBranch: prResult?.branch ?? null,
    createdAt: new Date().toISOString(),
  }, null, 2));
  await safeWriteFile(reportPath, buildReportMarkdown(result));
  await safeWriteFile(path.join(runDir, "diff.patch"), computeUnifiedDiff(target.name, target.body, bestCandidate.candidateBody));
  for (const c of candidates) { const p = slugify(c.name) || "candidate"; await safeWriteFile(path.join(runDir, "candidates", `${p}.md`), c.candidateFullText); await safeWriteFile(path.join(runDir, "candidates", `${p}.json`), JSON.stringify({ rationale: c.rationale, warnings: c.warnings, evaluation: c.evaluation, holdoutEvaluation: c.holdoutEvaluation, constraints: c.constraints, semanticDriftScore: c.semanticDriftScore, testPassed: c.testPassed }, null, 2)); }
  const allTraces = [...baselineTraces.map((t) => ({ ...t, phase: "baseline" as const })), ...candidates.flatMap((c) => c.executionTraces.map((t) => ({ ...t, phase: `candidate/${c.name}` as const })))];
  await safeWriteFile(path.join(tracesDir, "all-traces.json"), JSON.stringify(allTraces, null, 2));
  const failureOnly = allTraces.filter((t) => t.isFailure);
  if (failureOnly.length > 0) await safeWriteFile(path.join(tracesDir, "failure-traces.json"), JSON.stringify(failureOnly, null, 2));
  // Top-level gate.json with best candidate's tier results (always written, even when empty).
  await safeWriteFile(path.join(runDir, "gate.json"), JSON.stringify(bestCandidate.gateResults ?? [], null, 2));
  // Lineage: append entry linking this run to its ancestor (sibling lane provides module).
  try {
    const bestHoldoutComposite = bestCandidate.holdoutEvaluation?.aggregate.composite ?? bestCandidate.evaluation.aggregate.composite;
    const artifactHash = crypto.createHash("sha256").update(bestCandidate.candidateFullText).digest("hex").slice(0, 16);
    // parentArtifactHash should reference the prior run's winning artifactHash (cross-run
    // chaining), falling back to the pre-mutation source hash only when there is no ancestor.
    const ancestorEntry = ancestor as LineageEntry | undefined;
    const parentArtifactHash = ancestorEntry?.artifactHash
      ?? crypto.createHash("sha256").update(target.fullText).digest("hex").slice(0, 16);
    const runId = path.basename(runDir);
    const entry: LineageEntry = {
      runId,
      parentRunId: ancestorEntry?.runId,
      artifactPath: path.relative(options.cwd, target.path),
      artifactHash,
      parentArtifactHash,
      score: bestHoldoutComposite,
      mutationRationale: bestCandidate.rationale,
      createdAt: new Date().toISOString(),
    };
    await appendLineageEntry(options.cwd, entry);
  } catch { /* sibling module unavailable */ }
  return result;
}

export async function runEvolution(options: {
  cwd: string; targetPath: string; objective: string; evalSource: EvalSource; model?: string; thinkingLevel?: string;
  candidateCount: number; maxExamples: number; sessionQuery?: string; backend?: "auto" | "typescript" | "python";
  goldenTaskId?: string; testCommand?: string; testTimeout?: number; createPR?: boolean; persistGolden?: boolean;
  seed?: number; cohortExamples?: EvalExample[];
  cohortJudgeFunc?: (artifactText: string, examples: EvalExample[]) => Promise<{ composite: number }>;
  coherenceCheck?: (candidateText: string) => Promise<{ passed: boolean; detail: string }>;
  tsConfigPath?: string;
  signal?: AbortSignal; onProgress?: (phase: string, detail?: string) => void;
}): Promise<EvolutionSummaryDetails> {
  const preferred = options.backend ?? "auto";
  if (preferred !== "typescript") {
    const pb = await detectPythonBackend();
    if (pb) { options.onProgress?.("backend", `Using Python backend (${pb.doctor.gepa ? "GEPA" : "DSPy"})`); const pySummary = await runPythonBackend(pb.python, { cwd: options.cwd, targetPath: options.targetPath, objective: options.objective, evalSource: options.evalSource, model: options.model, candidateCount: options.candidateCount, maxExamples: options.maxExamples, sessionQuery: options.sessionQuery, goldenTaskId: options.goldenTaskId, testCommand: options.testCommand, testTimeout: options.testTimeout, createPR: options.createPR, persistGolden: options.persistGolden }); return { runDir: pySummary.runDir, reportPath: pySummary.reportPath, targetPath: pySummary.targetPath, objective: pySummary.objective, evalSource: pySummary.evalSource as EvalSource, modelLabel: pySummary.modelLabel, selectionSplit: pySummary.selectionSplit ?? "validation", confirmationSplit: pySummary.confirmationSplit ?? "holdout", trainExamples: pySummary.trainExamples, validationExamples: pySummary.validationExamples, holdoutExamples: pySummary.holdoutExamples, goldenTaskId: pySummary.goldenTaskId, candidateCount: pySummary.candidateCount, baselineValidationScore: pySummary.baselineValidationScore ?? pySummary.baselineHoldoutScore, bestValidationScore: pySummary.bestValidationScore ?? pySummary.bestHoldoutScore, baselineHoldoutScore: pySummary.baselineHoldoutScore, bestHoldoutScore: pySummary.bestHoldoutScore, improvement: pySummary.improvement, bestCandidateName: pySummary.bestCandidateName, tracesCaptured: pySummary.tracesCaptured ?? 0, constraintsPassed: pySummary.constraintsPassed ?? true, testGatePassed: pySummary.testGatePassed, semanticDriftScore: pySummary.semanticDriftScore, prBranch: pySummary.prBranch, backend: "python", optimizerUsed: pySummary.optimizer_used ?? "dspy" }; }
    if (preferred === "python") throw new Error("Python backend requested but unavailable.");
  }
  options.onProgress?.("backend", "Using TypeScript backend");
  return toToolSummaryDetails(await runTypeScriptEvolution({ ...options }));
}

export function buildToolSummary(r: EvolutionSummaryDetails): string {
  const s = r.improvement >= 0 ? "+" : "";
  return [`Self-evolution completed for ${r.targetPath}`, `Backend: ${r.backend ?? "typescript"}${r.optimizerUsed ? ` (${r.optimizerUsed})` : ""}`, `Best: ${r.bestCandidateName}`, `Splits: ${r.trainExamples}/${r.validationExamples}/${r.holdoutExamples} (train/val/holdout)`, `Selection (${r.selectionSplit}): ${r.baselineValidationScore.toFixed(3)} → ${r.bestValidationScore.toFixed(3)}`, `Confirmation (${r.confirmationSplit}): ${r.baselineHoldoutScore.toFixed(3)} → ${r.bestHoldoutScore.toFixed(3)} (${s}${r.improvement.toFixed(3)})`, `Traces: ${r.tracesCaptured}`, `Constraints: ${r.constraintsPassed ? "all passed" : "some failed"}`, r.testGatePassed !== undefined ? `Test gate: ${r.testGatePassed ? "passed" : "failed"}` : "", r.semanticDriftScore !== undefined ? `Drift: ${r.semanticDriftScore.toFixed(3)}` : "", r.prBranch ? `PR: ${r.prBranch}` : "", `Report: ${r.reportPath}`].filter(Boolean).join("\n");
}

export function toToolSummaryDetails(result: EvolutionRunResult): EvolutionSummaryDetails {
  const allTraces = [...result.baselineTraces, ...result.candidates.flatMap((c) => c.executionTraces)];
  return { runDir: result.paths.runDir, reportPath: result.paths.reportPath, targetPath: result.target.path, objective: result.objective, evalSource: result.evalSource, modelLabel: result.modelLabel, selectionSplit: result.selectionSplit, confirmationSplit: result.confirmationSplit, trainExamples: result.trainExamples.length, validationExamples: result.validationExamples.length, holdoutExamples: result.holdoutExamples.length, goldenTaskId: result.golden?.id ?? null, candidateCount: result.candidates.length, baselineValidationScore: result.baselineValidation.aggregate.composite, bestValidationScore: result.bestCandidate.evaluation.aggregate.composite, baselineHoldoutScore: result.baselineHoldout.aggregate.composite, bestHoldoutScore: result.bestCandidate.holdoutEvaluation?.aggregate.composite ?? result.bestCandidate.evaluation.aggregate.composite, improvement: result.improvement, bestCandidateName: result.bestCandidate.name, tracesCaptured: allTraces.length, constraintsPassed: result.bestCandidate.constraints.length > 0 ? result.bestCandidate.constraints.every((c) => c.passed) : true, testGatePassed: result.bestCandidate.testPassed, semanticDriftScore: result.bestCandidate.semanticDriftScore, prBranch: result.prResult?.branch, backend: "typescript", optimizerUsed: "gepa-pareto" };
}

export interface ApplyManifest {
  targetPath: string;
  /** Top-level field written by both TypeScript and Python backends. */
  bestCandidateName: string;
  improvement: number;
  /** Nested object written by the TypeScript backend — used as fallback when bestCandidateName is absent. */
  bestCandidate?: { name?: string };
}

function extractCandidateName(manifest: ApplyManifest): string {
  return manifest.bestCandidateName || manifest.bestCandidate?.name || "best-candidate";
}

export async function loadApplyManifest(runDir: string): Promise<ApplyManifest | null> {
  try {
    const raw = await fs.readFile(path.join(runDir, "manifest.json"), "utf8");
    return JSON.parse(raw) as ApplyManifest;
  } catch {
    return null;
  }
}

export async function applyBestCandidate(runDir: string): Promise<{ targetPath: string; candidateName: string; improvement: number }> {
  const manifestPath = path.join(runDir, "manifest.json");
  const bestCandidatePath = path.join(runDir, "best-candidate.md");
  let manifest: ApplyManifest;
  try {
    manifest = JSON.parse(await fs.readFile(manifestPath, "utf8")) as ApplyManifest;
  } catch {
    throw new Error(`Cannot read manifest at ${manifestPath}. Is the run directory correct?`);
  }
  let candidateContent: string;
  try {
    candidateContent = await fs.readFile(bestCandidatePath, "utf8");
  } catch {
    throw new Error(`Cannot read best candidate at ${bestCandidatePath}.`);
  }
  const targetPath = manifest.targetPath;
  if (!targetPath) throw new Error("manifest.json missing targetPath.");
  await withFileMutationQueue(targetPath, async () => { await fs.writeFile(targetPath, candidateContent, "utf8"); });
  return { targetPath, candidateName: extractCandidateName(manifest), improvement: manifest.improvement ?? 0 };
}

export { computeUnifiedDiff };

export type { EvolutionSummaryDetails, ToolSummaryDetails } from "./types.js";

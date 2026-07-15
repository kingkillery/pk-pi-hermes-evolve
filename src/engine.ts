import { spawn } from "node:child_process";
import crypto from "node:crypto";
import fs from "node:fs/promises";
import path from "node:path";
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

const CANDIDATE_SYSTEM_PROMPT = `You improve instruction artifacts using reflective search.
Return strict JSON only. Do not include markdown fences or commentary outside the JSON.`;

const DRIFT_SYSTEM_PROMPT = `You compare two versions of an instruction artifact and score their semantic similarity.
A lower drift score means the evolved version preserves the original meaning.
Return strict JSON only. Do not include markdown fences.`;

const SECRET_PATTERNS: Array<{ name: string; pattern: RegExp }> = [
  { name: "anthropic-key", pattern: /\bsk-ant-api\S{10,}\b/ },
  { name: "openrouter-key", pattern: /\bsk-or-v1-\S{10,}\b/ },
  { name: "openai-key", pattern: /\bsk-\S{20,}\b/ },
  { name: "github-token", pattern: /\bghp_\S{10,}\b/ },
  { name: "github-user-token", pattern: /\bghu_\S{10,}\b/ },
  { name: "slack-bot-token", pattern: /\bxoxb-\S{10,}\b/ },
  { name: "slack-app-token", pattern: /\bxapp-\S{10,}\b/ },
  { name: "notion-token", pattern: /\bntn_\S{10,}\b/ },
  { name: "aws-key", pattern: /\bAKIA[0-9A-Z]{16}\b/ },
  { name: "bearer-auth", pattern: /\bBearer\s+\S{20,}\b/ },
  { name: "private-key", pattern: /-----BEGIN\s+(?:RSA\s+)?PRIVATE\sKEY-----/ },
  { name: "env-anthropic", pattern: /\bANTHROPIC_API_KEY\b/ },
  { name: "env-openai", pattern: /\bOPENAI_API_KEY\b/ },
  { name: "env-openrouter", pattern: /\bOPENROUTER_API_KEY\b/ },
  { name: "env-github", pattern: /\bGITHUB_TOKEN\b/ },
  { name: "env-aws-secret", pattern: /\bAWS_SECRET_ACCESS_KEY\b/ },
  { name: "env-database", pattern: /\bDATABASE_URL\b/ },
  { name: "password-assignment", pattern: /\bpassword\s*[=:]\s*\S{6,}\b/ },
  { name: "secret-assignment", pattern: /\bsecret\s*[=:]\s*\S{6,}\b/ },
  { name: "token-assignment", pattern: /\btoken\s*[=:]\s*\S{10,}\b/ },
];

export function scanForSecrets(text: string): SecretScanResult {
  const found: SecretScanResult["patterns"] = [];
  for (const { name, pattern } of SECRET_PATTERNS) {
    const match = text.match(pattern);
    if (match && match[0]) {
      const index = match.index ?? 0;
      const location = index < 200 ? text.slice(0, Math.min(80, text.length)) : `offset ${index}`;
      found.push({ pattern: name, match: match[0].slice(0, 20) + "…", location });
    }
  }
  return { found: found.length > 0, patterns: found };
}

function stripSecretsFromExamples(examples: EvalExample[]): { clean: EvalExample[]; stripped: number } {
  let stripped = 0;
  const clean = examples.map((ex) => {
    const taskScan = scanForSecrets(ex.taskInput);
    const behaviorScan = scanForSecrets(ex.expectedBehavior);
    if (taskScan.found || behaviorScan.found) {
      stripped += 1;
      let taskInput = ex.taskInput;
      let expectedBehavior = ex.expectedBehavior;
      for (const p of taskScan.patterns) taskInput = taskInput.replace(p.match, "[REDACTED]");
      for (const p of behaviorScan.patterns) expectedBehavior = expectedBehavior.replace(p.match, "[REDACTED]");
      return { ...ex, taskInput, expectedBehavior };
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

async function computeSemanticDrift(cwd: string, originalBody: string, evolvedBody: string, objective: string, model?: string, thinkingLevel?: string, signal?: AbortSignal): Promise<{ score: number; feedback: string }> {
  const prompt = [`Original body (first 3000 chars):`, "```", originalBody.slice(0, 3000), "```", "", `Evolved body (first 3000 chars):`, "```", evolvedBody.slice(0, 3000), "```", "", `Objective: ${objective}`, "", "Score SEMANTIC DRIFT: 0.0 = identical meaning, 1.0 = different purpose.", 'Return JSON: {"driftScore": 0.0, "feedback": "explanation"}'].join("\n");
  try { const raw = await runPiTextTask({ cwd, model, thinkingLevel, systemPrompt: DRIFT_SYSTEM_PROMPT, prompt, signal }); const p = extractJsonPayload(raw) as { driftScore?: unknown; feedback?: unknown }; return { score: clampScore(p.driftScore), feedback: String(p.feedback ?? "").trim() }; } catch { return { score: 0.2, feedback: "Drift detection failed." }; }
}

async function createGitBranchWithCandidate(target: ArtifactTarget, bestCandidate: CandidateRecord, improvement: number, runDir: string, reportPath: string, objective: string, modelLabel: string, baselineTraces: ExecutionTrace[], candidates: CandidateRecord[], cwd: string): Promise<PRAutomationResult | undefined> {
  const branch = `evolve/${slugify(target.name)}-${formatTimestamp()}`;
  try {
    await fs.writeFile(target.path, bestCandidate.candidateFullText, "utf8");
    const git = async (...args: string[]): Promise<{ stdout: string; code: number }> => new Promise((resolve) => { const child = spawn("git", args, { cwd, stdio: ["pipe", "pipe", "pipe"] }); let stdout = ""; let stderr = ""; child.stdout.on("data", (c: Buffer) => { stdout += String(c); }); child.stderr.on("data", (c: Buffer) => { stderr += String(c); }); child.on("close", (code) => resolve({ stdout: (stdout || stderr).trim(), code: code ?? 1 })); });
    await git("checkout", "-b", branch); await git("add", target.path);
    const sign = improvement >= 0 ? "+" : "";
    const msg = `evolve: ${target.name} — ${sign}${improvement.toFixed(3)}\n\nObjective: ${objective}\nModel: ${modelLabel}\nTraces: ${baselineTraces.length}`;
    await git("commit", "-m", msg);
    let commitSha = ""; const sha = await git("rev-parse", "HEAD"); if (sha.code === 0) commitSha = sha.stdout.trim();
    let prUrl: string | undefined; let prNumber: number | undefined;
    const push = await git("push", "-u", "origin", branch);
    if (push.code === 0) { try { const pr = await git("pr", "create", "--title", `evolve: ${target.name}`, "--body", `Report: ${reportPath}`); if (pr.code === 0) { prUrl = pr.stdout.match(/https:\/\/\S+/)?.[0]; const nm = pr.stdout.match(/#(\d+)/); if (nm) prNumber = parseInt(nm[1], 10); } } catch { /* gh unavailable */ } }
    const diff = await git("diff", "--stat", "HEAD~1"); const diffStat = diff.stdout.trim() || "no stat";
    await git("checkout", "-");
    return { branch, commitSha, prUrl, prNumber, diffStat };
  } catch { try { spawn("git", ["checkout", "-"], { cwd, stdio: "pipe" }); await fs.writeFile(target.path, target.fullText, "utf8"); } catch { /* best effort */ } return undefined; }
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
  for (let i = 0; i < options.examples.length; i += 1) {
    const ex = options.examples[i]!; options.onProgress?.(`Judging ${i + 1}/${options.examples.length}`);
    let executorContext = ""; let observation: ExecutionObservation | undefined;
    if (options.useRealExecutor) {
      try {
        observation = await executeCandidateInPi({ cwd: options.cwd, candidateFullText: options.artifactText, taskInput: ex.taskInput, artifactName: options.artifactName || slugify(options.target.name) || "candidate", model: options.model, thinkingLevel: options.thinkingLevel, signal: options.signal });
        observations.push(observation);
        executorContext = ["", "Observed agent stdout (actual pi run):", "```", observation.stdout.slice(0, 4000), "```", `Exit code: ${observation.exitCode}; duration: ${observation.durationMs}ms.`].join("\n");
        if (options.executorLogDir) { const dir = path.join(options.executorLogDir, String(i)); await safeWriteFile(path.join(dir, "stdout.log"), observation.stdout); await safeWriteFile(path.join(dir, "stderr.log"), observation.stderr); await safeWriteFile(path.join(dir, "meta.json"), JSON.stringify({ exitCode: observation.exitCode, durationMs: observation.durationMs, taskInput: ex.taskInput }, null, 2)); }
      } catch (err) { executorContext = `\nExecutor unavailable: ${err instanceof Error ? err.message : String(err)}`; }
    }
    // Blind the judge to the artifact's prose whenever we have a real observed transcript: score
    // what the agent actually did, not how well the instructions read. Without a real observation
    // (useRealExecutor off, or the executor call failed) fall back to judging the text itself.
    const textBlock = observation ? [] : ["", "Artifact text:", "```", options.artifactText.trim(), "```"];
    const scoringInstruction = observation
      ? "Score the OBSERVED agent transcript above against the rubric — you do not have the artifact's instruction text, only what the agent actually did."
      : "Score how well an agent following the artifact text above would likely satisfy the rubric.";
    const prompt = [`Artifact type: ${options.target.type}`, `Objective: ${options.objective}`, `Path: ${options.target.path}`, ...textBlock, "", `Task: ${ex.taskInput}`, `Rubric: ${ex.expectedBehavior}`, `Difficulty: ${ex.difficulty}`, `Category: ${ex.category}`, executorContext, "", scoringInstruction, 'Return JSON: {"responsePreview":"...","correctness":0.0,"procedureFollowing":0.0,"conciseness":0.0,"feedback":"...","confidence":0.0}'].join("\n");
    const raw = await runPiTextTask({ cwd: options.cwd, model: options.model, thinkingLevel: options.thinkingLevel, systemPrompt: JUDGE_SYSTEM_PROMPT, prompt, signal: options.signal });
    const j = normalizeJudgeResult(extractJsonPayload(raw));
    if (observation && !j.responsePreview) j.responsePreview = observation.stdout.slice(0, 500);
    const c = 0.5 * j.correctness + 0.3 * j.procedureFollowing + 0.2 * j.conciseness;
    evals.push({ example: ex, composite: c, ...j }); traces.push(buildTrace(options.artifactText, ex, j, c, observation ? observation.stdout : raw, Boolean(observation)));
  }
  const n = Math.max(1, evals.length);
  const raw: AggregateScore = { correctness: evals.reduce((s, e) => s + e.correctness, 0) / n, procedureFollowing: evals.reduce((s, e) => s + e.procedureFollowing, 0) / n, conciseness: evals.reduce((s, e) => s + e.conciseness, 0) / n, confidence: evals.reduce((s, e) => s + e.confidence, 0) / n, lengthPenalty: 0, composite: evals.reduce((s, e) => s + e.composite, 0) / n };
  const sr = Buffer.byteLength(options.artifactText, "utf8") / Math.max(1, options.maxBytes); const lp = sr > 0.9 ? Math.min(0.3, (sr - 0.9) * 3) : 0;
  return { evaluation: { aggregate: { ...raw, lengthPenalty: lp, composite: Math.max(0, raw.composite - lp) }, examples: evals }, traces, executorObservations: observations.length > 0 ? observations : undefined };
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

function buildReportMarkdown(result: EvolutionRunResult): string {
  const baselineValidation = result.baselineValidation.aggregate.composite;
  const bestValidation = result.bestCandidate.evaluation.aggregate.composite;
  const baselineHoldout = result.baselineHoldout.aggregate.composite;
  const bestHoldout = result.bestCandidate.holdoutEvaluation?.aggregate.composite ?? bestValidation;
  const totalTraces = result.baselineTraces.length + result.candidates.reduce((s, c) => s + c.executionTraces.length, 0);
  const failures = [...result.baselineTraces, ...result.candidates.flatMap((c) => c.executionTraces)].filter((t) => t.isFailure);
  return [
    "# Hermes-style Self-Evolution Report", "",
    `- **Target:** ${result.target.path}`, `- **Type:** ${result.target.type}`, `- **Objective:** ${result.objective}`,
    `- **Source:** ${result.evalSource}`, `- **Model:** ${result.modelLabel}`, `- **Run dir:** ${result.paths.runDir}`,
    `- **Selection split:** ${result.selectionSplit}`, `- **Confirmation split:** ${result.confirmationSplit}`,
    `- **Baseline validation:** ${baselineValidation.toFixed(3)}`, `- **Best validation:** ${bestValidation.toFixed(3)}`,
    `- **Baseline holdout:** ${baselineHoldout.toFixed(3)}`, `- **Confirmed holdout:** ${bestHoldout.toFixed(3)}`, `- **Improvement:** ${result.improvement >= 0 ? "+" : ""}${result.improvement.toFixed(3)}`,
    `- **Traces:** ${totalTraces} captured, ${failures.length} failures`, "",
    "## Guardrails", "- Original preserved, never auto-overwritten.", "- Frontmatter preserved verbatim.", "- Placeholders required to survive.", "- Size budget enforced.", "- Growth limited to 20%.", "- Semantic drift checked (threshold 0.4).", "- Secret scanning on datasets.", "",
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
    "## Selected winner confirmation",
    `- **Winner chosen on:** ${result.selectionSplit}`,
    `- **Validation score:** ${bestValidation.toFixed(3)}`,
    `- **Holdout confirmation:** ${bestHoldout.toFixed(3)}`,
    "",
    "### Holdout weaknesses",
    summarizeWeaknesses(result.bestCandidate.holdoutEvaluation ?? result.bestCandidate.evaluation, 3),
    result.prResult ? `\n## PR\n- **Branch:** ${result.prResult.branch}\n- **Commit:** ${result.prResult.commitSha.slice(0, 12)}\n- **URL:** ${result.prResult.prUrl ?? "not created"}` : "",
    "", "## Files", `- Original: ${result.paths.originalPath}`, `- Best: ${result.paths.bestCandidatePath}`, `- Dataset: ${result.paths.datasetPath}`, `- Manifest: ${result.paths.manifestPath}`, `- Traces: ${result.paths.runDir}/traces/`, `- Report: ${result.paths.reportPath}`,
  ].join("\n");
}

async function runTypeScriptEvolution(options: {
  cwd: string; targetPath: string; objective: string; evalSource: EvalSource; model?: string; thinkingLevel?: string;
  candidateCount: number; maxExamples: number; sessionQuery?: string; goldenTaskId?: string;
  testCommand?: string; testTimeout?: number; createPR?: boolean; persistGolden?: boolean;
  seed?: number; cohortExamples?: EvalExample[];
  cohortJudgeFunc?: (examples: EvalExample[]) => Promise<{ composite: number }>;
  coherenceCheck?: () => Promise<{ passed: boolean; detail: string }>;
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
  options.onProgress?.("baseline", "Holdout"); const { evaluation: baselineHoldout } = await evaluateArtifact({ cwd: options.cwd, model: options.model, thinkingLevel: options.thinkingLevel, target, artifactText: target.fullText, objective: options.objective, examples: holdout, maxBytes, signal: options.signal, useRealExecutor: true, executorLogDir: path.join(runDir, "executor", "baseline-holdout"), artifactName: baselineArtifactName });
  options.onProgress?.("baseline", "Validation"); const { evaluation: baselineValidation } = await evaluateArtifact({ cwd: options.cwd, model: options.model, thinkingLevel: options.thinkingLevel, target, artifactText: target.fullText, objective: options.objective, examples: validation, maxBytes, signal: options.signal, useRealExecutor: true, executorLogDir: path.join(runDir, "executor", "baseline-validation"), artifactName: baselineArtifactName });
  const baselineTraces = [...btt];

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
  const pool: ParetoPoolEntry[] = [{
    name: "baseline",
    body: target.body,
    fullText: target.fullText,
    validationScores: baselineValidation.examples.map((e) => e.composite),
    minibatchScore: mean(baselineTrain.examples.slice(0, minibatch.length).map((e) => e.composite)),
    traces: baselineTraces,
    evaluation: baselineTrain,
  }];
  if (ancestorSeed) {
    // Same evaluation regime as every other pool entry: executor-grounded validation score for
    // Pareto bookkeeping, minibatch score/traces (judge-only, matching the mutation-parent
    // filter reference and reflection source) for everything else.
    options.onProgress?.("lineage", "Evaluating ancestor seed on validation");
    const ancestorArtifactName = slugify(`ancestor-${ancestorSeed.entry.runId}`) || "ancestor";
    const { evaluation: ancestorValidation } = await evaluateArtifact({ cwd: options.cwd, model: options.model, thinkingLevel: options.thinkingLevel, target, artifactText: ancestorSeed.fullText, objective: options.objective, examples: validation, maxBytes, signal: options.signal, useRealExecutor: true, executorLogDir: path.join(runDir, "executor", "ancestor-seed"), artifactName: ancestorArtifactName });
    const { evaluation: ancestorMinibatch, traces: ancestorMinibatchTraces } = await evaluateArtifact({ cwd: options.cwd, model: options.model, thinkingLevel: options.thinkingLevel, target, artifactText: ancestorSeed.fullText, objective: options.objective, examples: minibatch, maxBytes, signal: options.signal });
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
        baselineScore: baselineHoldout.aggregate.composite,
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

    // Minibatch pre-filter: judge on a cheap 1-2 example train subset before paying for a full
    // validation pass with the real pi executor. Only reject a CLEAR regression vs. the comparison
    // baseline on the same subset — ties proceed, since a 1-2 example judge score is noisy and a
    // tie carries no real signal to reject on. This is GEPA's main lever for the "35x fewer
    // rollouts" result. The comparison baseline is the sampled parent for a mutation, or the
    // stronger of the two merge inputs for a merge (see comparisonMinibatchScore above).
    options.onProgress?.("iteration", `${iter} minibatch pre-filter`);
    const { evaluation: minibatchEval, traces: minibatchTraces } = await evaluateArtifact({ cwd: options.cwd, model: options.model, thinkingLevel: options.thinkingLevel, target, artifactText: fullText, objective: options.objective, examples: minibatch, maxBytes, signal: options.signal });
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
    // Drift
    let driftScore: number | undefined;
    if (constraintConfig.checkSemanticDrift) {
      options.onProgress?.("drift", draft.name);
      const d = await computeSemanticDrift(options.cwd, target.body, draft.candidateBody, options.objective, options.model, options.thinkingLevel, options.signal);
      driftScore = d.score;
      cr.results.push({ name: "semantic_drift" as ConstraintName, passed: d.score <= constraintConfig.maxDriftScore, message: `Drift: ${d.score.toFixed(3)} (max ${constraintConfig.maxDriftScore}). ${d.feedback}` });
      if (d.score > constraintConfig.maxDriftScore) cr.warnings.push(`Semantic drift too high: ${d.score.toFixed(3)}`);
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
    // "Accepted" now means "cleared constraints/test gate on a full validation pass" — it is
    // added to the Pareto pool regardless of whether its aggregate beats the parent, because the
    // pool wants diverse per-instance strengths to sample from, not just a chain of monotone wins.
    const accepted = constraintsPass && (testPassed === undefined ? true : testPassed);
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
    // Fallback: take the best fully-validated iteration even if not strictly accepted. Only
    // `minibatchFiltered === false` records carry a genuine evaluation of the DRAFT ITSELF —
    // constraint/gate rejections and minibatch-filtered records store the PARENT's (or a
    // partial) evaluation on the iterRecord, so promoting one of those would present a body
    // under a score that was never actually measured for it.
    const fullyEvaluated = iterations.filter((it) => it.minibatchFiltered === false);
    if (fullyEvaluated.length > 0) {
      const best = [...fullyEvaluated].sort((a, b) => b.evaluation.aggregate.composite - a.evaluation.aggregate.composite)[0]!;
      const fullText = reassembleArtifact(target.frontmatter, best.candidate.candidateBody);
      const cr = validateConstraints(target, best.candidate.candidateBody, fullText, constraintConfig);
      options.onProgress?.("iterations", `Fallback acceptance: no iteration met strict criteria; promoting "${best.candidate.name}" (composite=${best.evaluation.aggregate.composite.toFixed(3)})`);
      candidates.push({ ...best.candidate, candidateFullText: fullText, evaluation: best.evaluation, executionTraces: best.traces, constraints: cr.results, warnings: [...cr.warnings, "Fallback acceptance: no iteration met strict acceptance criteria."], semanticDriftScore: undefined, testPassed: undefined, wasFallbackPromoted: true, gateResults: best.gateResults, parentCandidates: best.parentCandidates, selectionMethod: best.selectionMethod });
    } else {
      // Nothing this run ever cleared constraints, the tiered gate, or the minibatch filter for a
      // full validation pass — every candidate() would be a body that was never actually measured
      // at the score we'd have to borrow from its parent to report. Rather than promote a
      // mislabeled draft (or throw and abort the whole multi-iteration run over one bad artifact),
      // retain the baseline itself: its `baselineValidation` evaluation is genuine, and the
      // holdout confirmation step below will naturally report ~0 improvement.
      const cr = validateConstraints(target, target.body, target.fullText, constraintConfig);
      options.onProgress?.("iterations", "Fallback: no iteration passed constraints, the tiered gate, or the minibatch filter; retaining baseline.");
      candidates.push({
        name: "baseline",
        rationale: "No mutation or merge this run cleared constraints, the tiered gate, or the minibatch pre-filter; baseline retained unchanged.",
        candidateBody: target.body,
        candidateFullText: target.fullText,
        evaluation: baselineValidation,
        executionTraces: [],
        constraints: cr.results,
        warnings: [...cr.warnings, "Fallback acceptance: no iteration passed constraints/gates/minibatch filter this run."],
        wasFallbackPromoted: true,
      });
    }
  }
  candidates.sort((a, b) => b.evaluation.aggregate.composite - a.evaluation.aggregate.composite);
  const bestCandidate = candidates[0]!;
  const finalFrontier = computeParetoFrontier(pool).frontier.map((p) => p.name);
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
  const improvement = bestHoldoutEvaluation.aggregate.composite - baselineHoldout.aggregate.composite;

  // PR
  let prResult: PRAutomationResult | undefined;
  if (options.createPR && improvement > 0) { options.onProgress?.("pr", "Creating branch"); prResult = await createGitBranchWithCandidate(target, bestCandidate, improvement, runDir, path.join(runDir, "report.md"), options.objective, modelLabel, baselineTraces, candidates, options.cwd); }

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
    traces: { baselineCount: baselineTraces.length },
    prBranch: prResult?.branch ?? null,
    createdAt: new Date().toISOString(),
  }, null, 2));
  await safeWriteFile(reportPath, buildReportMarkdown(result));
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
  cohortJudgeFunc?: (examples: EvalExample[]) => Promise<{ composite: number }>;
  coherenceCheck?: () => Promise<{ passed: boolean; detail: string }>;
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

export type { EvolutionSummaryDetails, ToolSummaryDetails } from "./types.js";

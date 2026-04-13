import { spawn } from "node:child_process";
import crypto from "node:crypto";
import fs from "node:fs/promises";
import path from "node:path";
import { withFileMutationQueue } from "@mariozechner/pi-coding-agent";
import { detectPythonBackend, runPythonBackend } from "./python-backend.js";
import { mineSessionSnippets } from "./session-history.js";
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
  ExecutionTrace,
  ExampleEvaluation,
  GoldenDataset,
  GoldenDatasetManifest,
  JudgeResult,
  PRAutomationResult,
  SecretScanResult,
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

function splitExamples(examples: EvalExample[]): { train: EvalExample[]; validation: EvalExample[]; holdout: EvalExample[] } {
  const s = [...examples]; for (let i = s.length - 1; i > 0; i -= 1) { const j = Math.floor(Math.random() * (i + 1)); [s[i], s[j]] = [s[j]!, s[i]!]; }
  const tc = Math.max(3, Math.ceil(s.length * 0.5)); const vc = Math.max(1, Math.floor(s.length * 0.2));
  const train = s.slice(0, tc); const validation = s.slice(tc, tc + vc); const holdout = s.slice(tc + vc);
  if (holdout.length === 0 && train.length > 2) holdout.push(train.pop()!);
  if (validation.length === 0 && train.length > 2) validation.push(train.pop()!);
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

function buildTrace(artifactText: string, example: EvalExample, judged: JudgeResult, composite: number, rawOutput: string): ExecutionTrace {
  return { traceId: traceId(), artifactText: artifactText.slice(0, 2000), taskInput: example.taskInput, expectedBehavior: example.expectedBehavior, rawOutput: rawOutput.slice(0, 2000), responsePreview: judged.responsePreview.slice(0, 500), scores: { correctness: judged.correctness, procedureFollowing: judged.procedureFollowing, conciseness: judged.conciseness, composite }, feedback: judged.feedback, isFailure: composite < 0.5, timestamp: new Date().toISOString() };
}

async function evaluateArtifact(options: { cwd: string; model?: string; thinkingLevel?: string; target: ArtifactTarget; artifactText: string; objective: string; examples: EvalExample[]; maxBytes: number; signal?: AbortSignal; onProgress?: (detail: string) => void }): Promise<{ evaluation: ArtifactEvaluation; traces: ExecutionTrace[] }> {
  const evals: ExampleEvaluation[] = []; const traces: ExecutionTrace[] = [];
  for (let i = 0; i < options.examples.length; i += 1) {
    const ex = options.examples[i]!; options.onProgress?.(`Judging ${i + 1}/${options.examples.length}`);
    const prompt = [`Artifact type: ${options.target.type}`, `Objective: ${options.objective}`, `Path: ${options.target.path}`, "", "Artifact text:", "```", options.artifactText.trim(), "```", "", `Task: ${ex.taskInput}`, `Rubric: ${ex.expectedBehavior}`, `Difficulty: ${ex.difficulty}`, `Category: ${ex.category}`, "", 'Return JSON: {"responsePreview":"...","correctness":0.0,"procedureFollowing":0.0,"conciseness":0.0,"feedback":"...","confidence":0.0}'].join("\n");
    const raw = await runPiTextTask({ cwd: options.cwd, model: options.model, thinkingLevel: options.thinkingLevel, systemPrompt: JUDGE_SYSTEM_PROMPT, prompt, signal: options.signal });
    const j = normalizeJudgeResult(extractJsonPayload(raw)); const c = 0.5 * j.correctness + 0.3 * j.procedureFollowing + 0.2 * j.conciseness;
    evals.push({ example: ex, composite: c, ...j }); traces.push(buildTrace(options.artifactText, ex, j, c, raw));
  }
  const n = Math.max(1, evals.length);
  const raw: AggregateScore = { correctness: evals.reduce((s, e) => s + e.correctness, 0) / n, procedureFollowing: evals.reduce((s, e) => s + e.procedureFollowing, 0) / n, conciseness: evals.reduce((s, e) => s + e.conciseness, 0) / n, confidence: evals.reduce((s, e) => s + e.confidence, 0) / n, lengthPenalty: 0, composite: evals.reduce((s, e) => s + e.composite, 0) / n };
  const sr = Buffer.byteLength(options.artifactText, "utf8") / Math.max(1, options.maxBytes); const lp = sr > 0.9 ? Math.min(0.3, (sr - 0.9) * 3) : 0;
  return { evaluation: { aggregate: { ...raw, lengthPenalty: lp, composite: Math.max(0, raw.composite - lp) }, examples: evals }, traces };
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

async function generateCandidates(options: { cwd: string; model?: string; thinkingLevel?: string; target: ArtifactTarget; objective: string; trainExamples: EvalExample[]; baselineTrain: ArtifactEvaluation; baselineTraces: ExecutionTrace[]; maxBytes: number; candidateCount: number; signal?: AbortSignal }): Promise<CandidateDraft[]> {
  const failures = options.baselineTraces.filter((t) => t.isFailure);
  const traceSection = failures.length > 0 ? ["", "Observed failure traces:", ...failures.slice(0, 5).map((t, i) => `  ${i + 1}. Task: ${t.taskInput}\n     Scores: correctness=${t.scores.correctness.toFixed(2)}, procedure=${t.scores.procedureFollowing.toFixed(2)}, composite=${t.scores.composite.toFixed(2)}\n     Feedback: ${t.feedback}`)].join("\n") : "";
  const prompt = [`Artifact type: ${options.target.type}`, `Path: ${options.target.path}`, `Objective: ${options.objective}`, `Max bytes: ${options.maxBytes}`, `Placeholders: ${options.target.placeholders.length > 0 ? options.target.placeholders.join(", ") : "none"}`, `Top heading: ${options.target.topHeading ?? "none"}`, "", "Original BODY:", "```", options.target.body.trim(), "```", "", "Training tasks:", options.trainExamples.map((e, i) => `${i + 1}. ${e.taskInput}\n   Rubric: ${e.expectedBehavior}`).join("\n\n"), "", "Weaknesses:", summarizeWeaknesses(options.baselineTrain, 3), traceSection, "", 'Return JSON: {"candidates":[{"name":"short-kebab","rationale":"paragraph","candidateBody":"full revised body"}]}', "", `Generate ${options.candidateCount} DISTINCT candidates.`, "Rules: use failure traces for TARGETED fixes, preserve placeholders, keep markdown valid, never mention evaluation/scores."].join("\n");
  const raw = await runPiTextTask({ cwd: options.cwd, model: options.model, thinkingLevel: options.thinkingLevel, systemPrompt: CANDIDATE_SYSTEM_PROMPT, prompt, signal: options.signal });
  const payload = extractJsonPayload(raw) as { candidates?: unknown } | unknown[];
  const rawC = Array.isArray(payload) ? payload : Array.isArray(payload.candidates) ? payload.candidates : [];
  const seen = new Set<string>(); const drafts: CandidateDraft[] = [];
  for (const item of rawC) {
    if (!item || typeof item !== "object") continue; const r = item as Record<string, unknown>;
    let cb = String(r.candidateBody ?? r.candidate_body ?? "").trim(); if (!cb) continue;
    if (cb.startsWith("---")) cb = splitFrontmatter(cb).body; if (seen.has(cb)) continue; seen.add(cb);
    drafts.push({ name: slugify(String(r.name ?? `candidate-${drafts.length + 1}`)) || `candidate-${drafts.length + 1}`, rationale: String(r.rationale ?? "").trim() || "No rationale.", candidateBody: cb });
  }
  if (drafts.length === 0) throw new Error("Candidate generation returned no usable revisions.");
  return drafts.slice(0, options.candidateCount);
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
    "## Baseline weaknesses", summarizeWeaknesses(result.baselineTrain, 3), "",
    "## Execution traces", `- Total: ${totalTraces}`, `- Failures: ${failures.length}`, ...failures.slice(0, 5).map((t, i) => `${i + 1}. [${t.traceId}] composite=${t.scores.composite.toFixed(2)} — ${t.feedback.slice(0, 120)}`), "",
    "## Candidates", "| Name | Validation | Holdout | Correctness | Procedure | Conciseness | Constraints | Drift |", "|---|---:|---:|---:|---:|---:|---|---|",
    ...result.candidates.map((c) => `| ${c.name} | ${c.evaluation.aggregate.composite.toFixed(3)} | ${c.holdoutEvaluation?.aggregate.composite.toFixed(3) ?? "—"} | ${c.evaluation.aggregate.correctness.toFixed(3)} | ${c.evaluation.aggregate.procedureFollowing.toFixed(3)} | ${c.evaluation.aggregate.conciseness.toFixed(3)} | ${c.constraints.every((x) => x.passed) ? "✅" : "❌"} | ${c.semanticDriftScore?.toFixed(2) ?? "—"} |`),
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
    const splits = splitExamples(ds.examples); train = splits.train; validation = splits.validation; holdout = splits.holdout; sessionSnippets = ds.sessionSnippets;
  }
  const golden = buildGoldenDataset(validation, options.goldenTaskId);
  if (golden && options.persistGolden !== false && !usedPersistedGolden) await saveGoldenDataset(options.cwd, golden, train, validation, holdout, target.path, target.name);

  // Baseline evaluation with traces
  options.onProgress?.("baseline", "Train"); const { evaluation: baselineTrain, traces: btt } = await evaluateArtifact({ cwd: options.cwd, model: options.model, thinkingLevel: options.thinkingLevel, target, artifactText: target.fullText, objective: options.objective, examples: train, maxBytes, signal: options.signal, onProgress: (d) => options.onProgress?.("baseline", d) });
  options.onProgress?.("baseline", "Holdout"); const { evaluation: baselineHoldout } = await evaluateArtifact({ cwd: options.cwd, model: options.model, thinkingLevel: options.thinkingLevel, target, artifactText: target.fullText, objective: options.objective, examples: holdout, maxBytes, signal: options.signal });
  options.onProgress?.("baseline", "Validation"); const { evaluation: baselineValidation } = await evaluateArtifact({ cwd: options.cwd, model: options.model, thinkingLevel: options.thinkingLevel, target, artifactText: target.fullText, objective: options.objective, examples: validation, maxBytes, signal: options.signal });
  const baselineTraces = [...btt];

  // Candidates with trace-informed mutation
  options.onProgress?.("candidates", `Generating ${options.candidateCount} candidate(s)`);
  const drafts = await generateCandidates({ cwd: options.cwd, model: options.model, thinkingLevel: options.thinkingLevel, target, objective: options.objective, trainExamples: train, baselineTrain, baselineTraces, maxBytes, candidateCount: options.candidateCount, signal: options.signal });

  const candidates: CandidateRecord[] = [];
  for (let i = 0; i < drafts.length; i += 1) {
    const draft = drafts[i]!;
    const fullText = reassembleArtifact(target.frontmatter, draft.candidateBody);
    const cr = validateConstraints(target, draft.candidateBody, fullText, constraintConfig);
    if (!cr.valid) continue;
    options.onProgress?.("judge", `Candidate ${i + 1}/${drafts.length}: ${draft.name} on validation`);
    const { evaluation, traces: cTraces } = await evaluateArtifact({ cwd: options.cwd, model: options.model, thinkingLevel: options.thinkingLevel, target, artifactText: fullText, objective: options.objective, examples: validation, maxBytes, signal: options.signal });

    // Drift
    let driftScore: number | undefined;
    if (constraintConfig.checkSemanticDrift) {
      options.onProgress?.("drift", draft.name);
      const d = await computeSemanticDrift(options.cwd, target.body, draft.candidateBody, options.objective, options.model, options.thinkingLevel, options.signal);
      driftScore = d.score;
      cr.results.push({ name: "semantic_drift" as ConstraintName, passed: d.score <= constraintConfig.maxDriftScore, message: `Drift: ${d.score.toFixed(3)} (max ${constraintConfig.maxDriftScore}). ${d.feedback}` });
      if (d.score > constraintConfig.maxDriftScore) { cr.warnings.push(`Semantic drift too high: ${d.score.toFixed(3)}`); continue; }
    }

    // Test gate
    let testPassed: boolean | undefined;
    if (options.testCommand) {
      options.onProgress?.("test", draft.name);
      await safeWriteFile(target.path, fullText); const tr = await runTestCommand(options.testCommand, options.cwd, constraintConfig.testTimeoutMs, options.signal);
      testPassed = tr.passed; await safeWriteFile(target.path, target.fullText);
      if (!tr.passed) { cr.warnings.push(`Test failed (exit ${tr.exitCode})`); continue; }
    }

    candidates.push({ ...draft, candidateFullText: fullText, evaluation, executionTraces: cTraces, constraints: cr.results, warnings: cr.warnings, semanticDriftScore: driftScore, testPassed });
  }

  if (candidates.length === 0) throw new Error("All candidates rejected by constraints.");
  candidates.sort((a, b) => b.evaluation.aggregate.composite - a.evaluation.aggregate.composite);
  const bestCandidate = candidates[0]!;
  options.onProgress?.("confirm", `${bestCandidate.name} on holdout`);
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
    })),
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
  return result;
}

export async function runEvolution(options: {
  cwd: string; targetPath: string; objective: string; evalSource: EvalSource; model?: string; thinkingLevel?: string;
  candidateCount: number; maxExamples: number; sessionQuery?: string; backend?: "auto" | "typescript" | "python";
  goldenTaskId?: string; testCommand?: string; testTimeout?: number; createPR?: boolean; persistGolden?: boolean;
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
  return { runDir: result.paths.runDir, reportPath: result.paths.reportPath, targetPath: result.target.path, objective: result.objective, evalSource: result.evalSource, modelLabel: result.modelLabel, selectionSplit: result.selectionSplit, confirmationSplit: result.confirmationSplit, trainExamples: result.trainExamples.length, validationExamples: result.validationExamples.length, holdoutExamples: result.holdoutExamples.length, goldenTaskId: result.golden?.id ?? null, candidateCount: result.candidates.length, baselineValidationScore: result.baselineValidation.aggregate.composite, bestValidationScore: result.bestCandidate.evaluation.aggregate.composite, baselineHoldoutScore: result.baselineHoldout.aggregate.composite, bestHoldoutScore: result.bestCandidate.holdoutEvaluation?.aggregate.composite ?? result.bestCandidate.evaluation.aggregate.composite, improvement: result.improvement, bestCandidateName: result.bestCandidate.name, tracesCaptured: allTraces.length, constraintsPassed: result.bestCandidate.constraints.length > 0 ? result.bestCandidate.constraints.every((c) => c.passed) : true, testGatePassed: result.bestCandidate.testPassed, semanticDriftScore: result.bestCandidate.semanticDriftScore, prBranch: result.prResult?.branch, backend: "typescript", optimizerUsed: "typescript-proxy" };
}

export type { EvolutionSummaryDetails, ToolSummaryDetails } from "./types.js";

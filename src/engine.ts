import { spawn } from "node:child_process";
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
  EvalExample,
  EvalSource,
  EvolutionOptions,
  EvolutionRunResult,
  ExampleEvaluation,
  JudgeResult,
} from "./types.js";

const DATASET_SYSTEM_PROMPT = `You create compact evaluation datasets for agent instructions.
Return strict JSON only. No markdown fences, no prose before or after the JSON.`;

const JUDGE_SYSTEM_PROMPT = `You are a strict evaluator for agent instruction artifacts.
Estimate how an agent following the provided artifact would likely respond to a task.
Return strict JSON only. Be conservative, concrete, and terse.`;

const CANDIDATE_SYSTEM_PROMPT = `You improve instruction artifacts using reflective search.
Return strict JSON only. Do not include markdown fences or commentary outside the JSON.`;

function slugify(value: string): string {
  return value
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 48);
}

function formatTimestamp(date = new Date()): string {
  return date.toISOString().replace(/[:.]/g, "-").replace("T", "_").slice(0, 19);
}

function getPiInvocation(args: string[]): { command: string; args: string[] } {
  const currentScript = process.argv[1];
  if (currentScript) {
    return { command: process.execPath, args: [currentScript, ...args] };
  }
  return { command: "pi", args };
}

async function runPiTextTask(options: {
  cwd: string;
  model?: string;
  thinkingLevel?: string;
  systemPrompt: string;
  prompt: string;
  signal?: AbortSignal;
}): Promise<string> {
  const args = [
    "-p",
    "--no-session",
    "--no-extensions",
    "--no-skills",
    "--no-prompt-templates",
    "--no-themes",
    "--no-tools",
    "--system-prompt",
    options.systemPrompt,
    "Use the piped instructions. Return only the requested JSON.",
  ];

  if (options.model) {
    args.splice(args.length - 1, 0, "--model", options.model);
  }
  if (options.thinkingLevel && options.thinkingLevel !== "off") {
    args.splice(args.length - 1, 0, "--thinking", options.thinkingLevel);
  }

  const invocation = getPiInvocation(args);

  return await new Promise<string>((resolve, reject) => {
    const child = spawn(invocation.command, invocation.args, {
      cwd: options.cwd,
      env: { ...process.env, PI_SKIP_VERSION_CHECK: "1" },
      stdio: ["pipe", "pipe", "pipe"],
    });

    let stdout = "";
    let stderr = "";
    let settled = false;

    const cleanup = () => {
      options.signal?.removeEventListener("abort", onAbort);
    };

    const finishReject = (error: Error) => {
      if (settled) return;
      settled = true;
      cleanup();
      reject(error);
    };

    const finishResolve = (value: string) => {
      if (settled) return;
      settled = true;
      cleanup();
      resolve(value);
    };

    const onAbort = () => {
      child.kill();
      finishReject(new Error("Evolution run aborted."));
    };

    options.signal?.addEventListener("abort", onAbort, { once: true });

    child.stdout.on("data", (chunk) => {
      stdout += String(chunk);
    });
    child.stderr.on("data", (chunk) => {
      stderr += String(chunk);
    });
    child.on("error", (error) => {
      finishReject(error instanceof Error ? error : new Error(String(error)));
    });
    child.on("close", (code) => {
      if (code !== 0) {
        finishReject(new Error(`pi subprocess failed (exit ${code}): ${stderr || stdout}`.trim()));
        return;
      }
      finishResolve(stdout.trim());
    });

    child.stdin.write(options.prompt);
    child.stdin.end();
  });
}

function extractJsonPayload(text: string): unknown {
  const trimmed = text.trim();
  if (!trimmed) throw new Error("Model returned empty output.");

  try {
    return JSON.parse(trimmed);
  } catch {
    // continue
  }

  const candidates = ["{", "["];
  for (const opener of candidates) {
    const start = trimmed.indexOf(opener);
    if (start < 0) continue;

    let depth = 0;
    let inString = false;
    let escapeNext = false;
    const closer = opener === "{" ? "}" : "]";

    for (let i = start; i < trimmed.length; i += 1) {
      const ch = trimmed[i]!;
      if (escapeNext) {
        escapeNext = false;
        continue;
      }
      if (ch === "\\" && inString) {
        escapeNext = true;
        continue;
      }
      if (ch === '"') {
        inString = !inString;
        continue;
      }
      if (inString) continue;
      if (ch === opener) depth += 1;
      if (ch === closer) depth -= 1;
      if (depth === 0) {
        const slice = trimmed.slice(start, i + 1);
        return JSON.parse(slice);
      }
    }
  }

  throw new Error(`Could not parse JSON from model output:\n${text}`);
}

function clampScore(value: unknown): number {
  const parsed = typeof value === "number" ? value : Number(String(value ?? "0.5").trim());
  if (Number.isNaN(parsed)) return 0.5;
  return Math.max(0, Math.min(1, parsed));
}

function splitFrontmatter(fullText: string): { frontmatter?: string; body: string } {
  const match = fullText.match(/^---\r?\n([\s\S]*?)\r?\n---\r?\n?/);
  if (!match) return { body: fullText.trim() };
  const frontmatter = match[1]?.trimEnd();
  const body = fullText.slice(match[0].length).trim();
  return { frontmatter, body };
}

function reassembleArtifact(frontmatter: string | undefined, body: string): string {
  const normalizedBody = body.trimEnd();
  if (!frontmatter) return `${normalizedBody}\n`;
  return `---\n${frontmatter.trimEnd()}\n---\n\n${normalizedBody}\n`;
}

function extractPlaceholders(text: string): string[] {
  return Array.from(new Set(text.match(/{{[^}]+}}/g) ?? []));
}

function detectArtifactType(resolvedPath: string): ArtifactTarget["type"] {
  const normalized = resolvedPath.replace(/\\/g, "/").toLowerCase();
  if (normalized.endsWith("/skill.md") || path.basename(normalized) === "skill.md") return "skill";
  if (normalized.includes("/.pi/prompts/") || normalized.includes("/.agents/prompts/") || normalized.endsWith(".prompt.md")) {
    return "prompt";
  }
  if (normalized.endsWith("agents.md") || normalized.endsWith("system.md") || normalized.endsWith("append_system.md")) {
    return "instructions";
  }
  return normalized.endsWith(".md") ? "prompt" : "instructions";
}

export async function resolveArtifactTarget(inputPath: string, cwd: string): Promise<ArtifactTarget> {
  const cleaned = inputPath.startsWith("@") ? inputPath.slice(1) : inputPath;
  const resolvedPath = path.isAbsolute(cleaned) ? cleaned : path.resolve(cwd, cleaned);
  const fullText = await fs.readFile(resolvedPath, "utf8");
  const { frontmatter, body } = splitFrontmatter(fullText);
  const headingMatch = body.match(/^#\s+(.+)$/m);

  return {
    path: resolvedPath,
    name: path.basename(path.dirname(resolvedPath)) === ".pi"
      ? path.basename(resolvedPath)
      : path.basename(resolvedPath, path.extname(resolvedPath)) || path.basename(resolvedPath),
    type: detectArtifactType(resolvedPath),
    fullText,
    body,
    frontmatter,
    originalBytes: Buffer.byteLength(fullText, "utf8"),
    placeholders: extractPlaceholders(fullText),
    topHeading: headingMatch?.[1]?.trim(),
  };
}

function computeMaxBytes(originalBytes: number): number {
  return Math.max(originalBytes + 400, Math.ceil(originalBytes * 1.2));
}

function normalizeExamples(payload: unknown, evalSource: EvalSource): EvalExample[] {
  const root = payload as { examples?: unknown } | unknown[];
  const rawExamples = Array.isArray(root) ? root : Array.isArray(root.examples) ? root.examples : [];

  const normalized = rawExamples
    .map((item): EvalExample | null => {
      if (!item || typeof item !== "object") return null;
      const record = item as Record<string, unknown>;
      const taskInput = String(record.taskInput ?? record.task_input ?? "").trim();
      const expectedBehavior = String(record.expectedBehavior ?? record.expected_behavior ?? "").trim();
      if (!taskInput || !expectedBehavior) return null;
      const difficultyRaw = String(record.difficulty ?? "medium").toLowerCase();
      const difficulty = difficultyRaw === "easy" || difficultyRaw === "hard" ? difficultyRaw : "medium";
      const category = String(record.category ?? "general").trim() || "general";
      const sourceRaw = String(record.source ?? (evalSource === "session" ? "session" : "synthetic")).toLowerCase();
      const source = sourceRaw === "session" ? "session" : "synthetic";
      return {
        taskInput: taskInput.slice(0, 1800),
        expectedBehavior: expectedBehavior.slice(0, 1800),
        difficulty,
        category,
        source,
      };
    })
    .filter((item): item is EvalExample => Boolean(item));

  return normalized;
}

function splitExamples(examples: EvalExample[]): { train: EvalExample[]; holdout: EvalExample[] } {
  const shuffled = [...examples];
  for (let i = shuffled.length - 1; i > 0; i -= 1) {
    const j = Math.floor(Math.random() * (i + 1));
    [shuffled[i], shuffled[j]] = [shuffled[j]!, shuffled[i]!];
  }
  const trainCount = Math.max(3, Math.ceil(shuffled.length * 0.6));
  const holdout = shuffled.slice(trainCount);
  const train = shuffled.slice(0, trainCount);
  if (holdout.length === 0 && train.length > 1) {
    holdout.push(train.pop()!);
  }
  return { train, holdout };
}

function summarizeWeaknesses(evaluation: ArtifactEvaluation, limit = 3): string {
  return [...evaluation.examples]
    .sort((a, b) => a.composite - b.composite)
    .slice(0, limit)
    .map((item, index) => {
      return [
        `${index + 1}. Task: ${item.example.taskInput}`,
        `   Rubric: ${item.example.expectedBehavior}`,
        `   Scores: correctness=${item.correctness.toFixed(2)}, procedure=${item.procedureFollowing.toFixed(2)}, concision=${item.conciseness.toFixed(2)}`,
        `   Feedback: ${item.feedback}`,
      ].join("\n");
    })
    .join("\n\n");
}

function validateCandidate(target: ArtifactTarget, candidateBody: string, maxBytes: number): { valid: boolean; warnings: string[] } {
  const warnings: string[] = [];
  const normalizedBody = candidateBody.trim();
  if (!normalizedBody) {
    warnings.push("Candidate body was empty.");
    return { valid: false, warnings };
  }

  const candidateFullText = reassembleArtifact(target.frontmatter, normalizedBody);
  if (Buffer.byteLength(candidateFullText, "utf8") > maxBytes) {
    warnings.push(`Candidate exceeded size budget (${Buffer.byteLength(candidateFullText, "utf8")}/${maxBytes} bytes).`);
    return { valid: false, warnings };
  }

  const missingPlaceholders = target.placeholders.filter((placeholder) => !candidateFullText.includes(placeholder));
  if (missingPlaceholders.length > 0) {
    warnings.push(`Candidate dropped placeholders: ${missingPlaceholders.join(", ")}`);
    return { valid: false, warnings };
  }

  if (target.topHeading && !normalizedBody.match(/^#\s+.+$/m)) {
    warnings.push("Candidate lost the top-level markdown heading.");
  }
  if (normalizedBody === target.body.trim()) {
    warnings.push("Candidate body was identical to the baseline body.");
  }

  return { valid: true, warnings };
}

function normalizeJudgeResult(payload: unknown): JudgeResult {
  const record = (payload as Record<string, unknown>) ?? {};
  return {
    responsePreview: String(record.responsePreview ?? record.response_preview ?? "").trim(),
    correctness: clampScore(record.correctness),
    procedureFollowing: clampScore(record.procedureFollowing ?? record.procedure_following),
    conciseness: clampScore(record.conciseness),
    feedback: String(record.feedback ?? "").trim(),
    confidence: clampScore(record.confidence ?? 0.6),
  };
}

async function evaluateArtifact(options: {
  cwd: string;
  model?: string;
  thinkingLevel?: string;
  target: ArtifactTarget;
  artifactText: string;
  objective: string;
  examples: EvalExample[];
  maxBytes: number;
  signal?: AbortSignal;
  onProgress?: (detail: string) => void;
}): Promise<ArtifactEvaluation> {
  const exampleEvaluations: ExampleEvaluation[] = [];

  for (let index = 0; index < options.examples.length; index += 1) {
    const example = options.examples[index]!;
    options.onProgress?.(`Judging example ${index + 1}/${options.examples.length}`);

    const prompt = [
      `Artifact type: ${options.target.type}`,
      `Improvement objective: ${options.objective}`,
      `Artifact path: ${options.target.path}`,
      "",
      "Artifact text:",
      "```",
      options.artifactText.trim(),
      "```",
      "",
      `Task input: ${example.taskInput}`,
      `Expected behavior rubric: ${example.expectedBehavior}`,
      `Difficulty: ${example.difficulty}`,
      `Category: ${example.category}`,
      "",
      "Return a JSON object with exactly these keys:",
      '{"responsePreview":"...","correctness":0.0,"procedureFollowing":0.0,"conciseness":0.0,"feedback":"...","confidence":0.0}',
      "",
      "Scoring guidance:",
      "- correctness: did the likely response solve the task?",
      "- procedureFollowing: would the artifact steer the agent through the right workflow?",
      "- conciseness: would the response stay focused without becoming vague?",
      "- feedback: short, concrete, revision-oriented.",
      "- confidence: how reliable your estimate is.",
    ].join("\n");

    const raw = await runPiTextTask({
      cwd: options.cwd,
      model: options.model,
      thinkingLevel: options.thinkingLevel,
      systemPrompt: JUDGE_SYSTEM_PROMPT,
      prompt,
      signal: options.signal,
    });

    const judged = normalizeJudgeResult(extractJsonPayload(raw));
    const composite = 0.5 * judged.correctness + 0.3 * judged.procedureFollowing + 0.2 * judged.conciseness;
    exampleEvaluations.push({ example, composite, ...judged });
  }

  const count = Math.max(1, exampleEvaluations.length);
  const rawAggregate: AggregateScore = {
    correctness: exampleEvaluations.reduce((sum, item) => sum + item.correctness, 0) / count,
    procedureFollowing: exampleEvaluations.reduce((sum, item) => sum + item.procedureFollowing, 0) / count,
    conciseness: exampleEvaluations.reduce((sum, item) => sum + item.conciseness, 0) / count,
    confidence: exampleEvaluations.reduce((sum, item) => sum + item.confidence, 0) / count,
    lengthPenalty: 0,
    composite: exampleEvaluations.reduce((sum, item) => sum + item.composite, 0) / count,
  };

  const sizeRatio = Buffer.byteLength(options.artifactText, "utf8") / Math.max(1, options.maxBytes);
  const lengthPenalty = sizeRatio > 0.9 ? Math.min(0.3, (sizeRatio - 0.9) * 3) : 0;

  return {
    aggregate: {
      ...rawAggregate,
      lengthPenalty,
      composite: Math.max(0, rawAggregate.composite - lengthPenalty),
    },
    examples: exampleEvaluations,
  };
}

async function generateDataset(options: {
  cwd: string;
  model?: string;
  thinkingLevel?: string;
  target: ArtifactTarget;
  objective: string;
  evalSource: EvalSource;
  maxExamples: number;
  sessionQuery?: string;
  signal?: AbortSignal;
}): Promise<{ examples: EvalExample[]; sessionSnippets: ReturnType<typeof mineSessionSnippets> }> {
  const sessionSnippets = options.evalSource === "synthetic"
    ? []
    : mineSessionSnippets({
        cwd: options.cwd,
        targetName: options.target.name,
        objective: options.objective,
        artifactBody: options.target.body,
        sessionQuery: options.sessionQuery,
        maxSnippets: 6,
      });

  const sourceNote = options.evalSource === "session" && sessionSnippets.length === 0
    ? "No relevant pi session snippets were found for this cwd; fall back to synthetic tasks that still reflect the artifact's intent."
    : "Blend synthetic tasks with any relevant recent session patterns you see below.";

  const prompt = [
    `Artifact type: ${options.target.type}`,
    `Artifact path: ${options.target.path}`,
    `Improvement objective: ${options.objective}`,
    `Requested example count: ${options.maxExamples}`,
    `Dataset mode: ${options.evalSource}`,
    "",
    "Artifact text:",
    "```",
    options.target.fullText.trim(),
    "```",
    "",
    "Session snippets (optional realism hints):",
    sessionSnippets.length === 0
      ? "- none found"
      : sessionSnippets.map((snippet, index) => [
          `- Snippet ${index + 1} (score ${snippet.score})`,
          `  User: ${snippet.userText}`,
          snippet.assistantText ? `  Assistant: ${snippet.assistantText}` : "  Assistant: <none>",
        ].join("\n")).join("\n"),
    "",
    sourceNote,
    "",
    "Return a JSON object with this exact shape:",
    '{"examples":[{"taskInput":"...","expectedBehavior":"...","difficulty":"easy|medium|hard","category":"...","source":"synthetic|session"}]}',
    "",
    "Rules:",
    "- expectedBehavior must be a rubric, not exact output text.",
    "- Prefer short, realistic, concrete tasks.",
    "- Avoid secrets, credentials, or machine-specific paths unless the artifact clearly requires them.",
    "- Cover easy, medium, and hard cases when possible.",
    "- Keep categories informative (e.g. trigger-detection, safety, formatting, sequencing).",
  ].join("\n");

  const raw = await runPiTextTask({
    cwd: options.cwd,
    model: options.model,
    thinkingLevel: options.thinkingLevel,
    systemPrompt: DATASET_SYSTEM_PROMPT,
    prompt,
    signal: options.signal,
  });

  const examples = normalizeExamples(extractJsonPayload(raw), options.evalSource).slice(0, options.maxExamples);
  if (examples.length < 4) {
    throw new Error(`Dataset generation only produced ${examples.length} usable example(s); need at least 4.`);
  }

  return { examples, sessionSnippets };
}

async function generateCandidates(options: {
  cwd: string;
  model?: string;
  thinkingLevel?: string;
  target: ArtifactTarget;
  objective: string;
  trainExamples: EvalExample[];
  baselineTrain: ArtifactEvaluation;
  maxBytes: number;
  candidateCount: number;
  signal?: AbortSignal;
}): Promise<CandidateDraft[]> {
  const prompt = [
    `Artifact type: ${options.target.type}`,
    `Artifact path: ${options.target.path}`,
    `Improvement objective: ${options.objective}`,
    `Target maximum bytes after reassembly: ${options.maxBytes}`,
    `Required placeholder tokens to preserve: ${options.target.placeholders.length > 0 ? options.target.placeholders.join(", ") : "none"}`,
    `Top heading to preserve conceptually: ${options.target.topHeading ?? "none"}`,
    "",
    "Original artifact BODY only:",
    "```",
    options.target.body.trim(),
    "```",
    "",
    "Representative training tasks:",
    options.trainExamples.map((example, index) => {
      return `${index + 1}. Task: ${example.taskInput}\n   Rubric: ${example.expectedBehavior}`;
    }).join("\n\n"),
    "",
    "Weakest baseline behavior:",
    summarizeWeaknesses(options.baselineTrain, 3),
    "",
    "Return a JSON object with this exact shape:",
    '{"candidates":[{"name":"short-kebab-name","rationale":"one paragraph","candidateBody":"full revised body only"}]}',
    "",
    `Generate ${options.candidateCount} DISTINCT candidates with different strategies.`,
    "Rules:",
    "- Keep the artifact's core job the same.",
    "- Improve trigger clarity, ordering, constraints, and examples when that helps.",
    "- Do NOT mention evaluation, scores, rubrics, GEPA, self-evolution, or hidden reasoning.",
    "- candidateBody must be the revised BODY only, not the frontmatter.",
    "- Keep markdown valid and practical.",
    "- Preserve all existing placeholder tokens verbatim.",
  ].join("\n");

  const raw = await runPiTextTask({
    cwd: options.cwd,
    model: options.model,
    thinkingLevel: options.thinkingLevel,
    systemPrompt: CANDIDATE_SYSTEM_PROMPT,
    prompt,
    signal: options.signal,
  });

  const payload = extractJsonPayload(raw) as { candidates?: unknown } | unknown[];
  const rawCandidates = Array.isArray(payload) ? payload : Array.isArray(payload.candidates) ? payload.candidates : [];

  const seenBodies = new Set<string>();
  const drafts: CandidateDraft[] = [];

  for (const item of rawCandidates) {
    if (!item || typeof item !== "object") continue;
    const record = item as Record<string, unknown>;
    let candidateBody = String(record.candidateBody ?? record.candidate_body ?? "").trim();
    if (!candidateBody) continue;

    if (candidateBody.startsWith("---")) {
      candidateBody = splitFrontmatter(candidateBody).body;
    }

    if (seenBodies.has(candidateBody)) continue;
    seenBodies.add(candidateBody);

    drafts.push({
      name: slugify(String(record.name ?? `candidate-${drafts.length + 1}`)) || `candidate-${drafts.length + 1}`,
      rationale: String(record.rationale ?? "").trim() || "No rationale provided.",
      candidateBody,
    });
  }

  if (drafts.length === 0) {
    throw new Error("Candidate generation returned no usable revisions.");
  }

  return drafts.slice(0, options.candidateCount);
}

async function safeWriteFile(filePath: string, content: string): Promise<void> {
  await fs.mkdir(path.dirname(filePath), { recursive: true });
  await withFileMutationQueue(filePath, async () => {
    await fs.writeFile(filePath, content, "utf8");
  });
}

function buildReportMarkdown(result: EvolutionRunResult): string {
  const baseline = result.baselineHoldout.aggregate.composite;
  const best = result.bestCandidate.evaluation.aggregate.composite;

  const candidateTable = [
    "| Candidate | Holdout composite | Correctness | Procedure | Conciseness | Notes |",
    "|---|---:|---:|---:|---:|---|",
    ...result.candidates.map((candidate) => {
      return `| ${candidate.name} | ${candidate.evaluation.aggregate.composite.toFixed(3)} | ${candidate.evaluation.aggregate.correctness.toFixed(3)} | ${candidate.evaluation.aggregate.procedureFollowing.toFixed(3)} | ${candidate.evaluation.aggregate.conciseness.toFixed(3)} | ${candidate.rationale.replace(/\|/g, "\\|")} |`;
    }),
  ].join("\n");

  const trainList = result.trainExamples
    .map((example, index) => `1. **${example.category}** (${example.difficulty}, ${example.source})\n   - Task: ${example.taskInput}\n   - Rubric: ${example.expectedBehavior}`.replace(/^1\./, `${index + 1}.`))
    .join("\n");

  const holdoutList = result.holdoutExamples
    .map((example, index) => `1. **${example.category}** (${example.difficulty}, ${example.source})\n   - Task: ${example.taskInput}\n   - Rubric: ${example.expectedBehavior}`.replace(/^1\./, `${index + 1}.`))
    .join("\n");

  const weaknessSummary = summarizeWeaknesses(result.baselineTrain, 3) || "No weakness summary available.";
  const bestWarnings = result.bestCandidate.warnings.length > 0 ? result.bestCandidate.warnings.map((w) => `- ${w}`).join("\n") : "- none";
  const snippetSection = result.sessionSnippets.length > 0
    ? result.sessionSnippets.map((snippet, index) => `### Snippet ${index + 1}\n- Session: ${snippet.sessionFile}\n- Score: ${snippet.score}\n- User: ${snippet.userText}\n- Assistant: ${snippet.assistantText || "<none>"}`).join("\n\n")
    : "No relevant historical pi session snippets were found for this cwd.";

  return [
    `# Hermes-style Self-Evolution Report`,
    "",
    `- **Target:** ${result.target.path}`,
    `- **Artifact type:** ${result.target.type}`,
    `- **Objective:** ${result.objective}`,
    `- **Eval source:** ${result.evalSource}`,
    `- **Model:** ${result.modelLabel}`,
    `- **Run directory:** ${result.paths.runDir}`,
    `- **Baseline holdout composite:** ${baseline.toFixed(3)}`,
    `- **Best holdout composite:** ${best.toFixed(3)}`,
    `- **Improvement:** ${result.improvement >= 0 ? "+" : ""}${result.improvement.toFixed(3)}`,
    `- **Size budget:** ${result.maxBytes} bytes`,
    "",
    `## Guardrails`,
    "",
    `- Original target was preserved; nothing was overwritten automatically.`,
    `- Frontmatter was preserved verbatim when present.`,
    `- Existing placeholders were required to survive every candidate.`,
    `- Candidates that exceeded the size budget were rejected.`,
    "",
    `## Baseline weaknesses used for mutation`,
    "",
    weaknessSummary,
    "",
    `## Dataset`,
    "",
    `### Train examples`,
    trainList,
    "",
    `### Holdout examples`,
    holdoutList,
    "",
    `## Session snippets`,
    "",
    snippetSection,
    "",
    `## Candidate comparison`,
    "",
    candidateTable,
    "",
    `## Best candidate`,
    "",
    `- **Name:** ${result.bestCandidate.name}`,
    `- **Rationale:** ${result.bestCandidate.rationale}`,
    `- **Warnings:**`,
    bestWarnings,
    "",
    `### Best candidate weakest holdout feedback`,
    "",
    summarizeWeaknesses(result.bestCandidate.evaluation, 3) || "No weakness summary available.",
    "",
    `## Output files`,
    "",
    `- Original snapshot: ${result.paths.originalPath}`,
    `- Best candidate: ${result.paths.bestCandidatePath}`,
    `- Dataset JSON: ${result.paths.datasetPath}`,
    `- Manifest JSON: ${result.paths.manifestPath}`,
    `- This report: ${result.paths.reportPath}`,
    "",
    `## Suggested next steps`,
    "",
    `1. Review the markdown diff between the original snapshot and the best candidate.`,
    `2. If the candidate looks right, manually apply it or copy selected sections.`,
    `3. Re-run the evolution loop with a narrower objective if the candidate is too broad.`,
    `4. Add real validation steps outside this proxy judge if the artifact controls production behavior.`,
    "",
  ].join("\n");
}

async function runTypeScriptEvolution(options: {
  cwd: string;
  targetPath: string;
  objective: string;
  evalSource: EvalSource;
  model?: string;
  thinkingLevel?: string;
  candidateCount: number;
  maxExamples: number;
  sessionQuery?: string;
  signal?: AbortSignal;
  onProgress?: (phase: string, detail?: string) => void;
}): Promise<EvolutionRunResult> {
  const target = await resolveArtifactTarget(options.targetPath, options.cwd);
  const modelLabel = options.model ?? "current-session-model";
  const runDir = path.join(options.cwd, ".pi", "hermes-self-evolution", "runs", `${formatTimestamp()}-${slugify(target.name || "artifact")}`);
  const maxBytes = computeMaxBytes(target.originalBytes);

  options.onProgress?.("dataset", "Generating evaluation set");
  const dataset = await generateDataset({
    cwd: options.cwd,
    model: options.model,
    thinkingLevel: options.thinkingLevel,
    target,
    objective: options.objective,
    evalSource: options.evalSource,
    maxExamples: Math.max(4, options.maxExamples),
    sessionQuery: options.sessionQuery,
    signal: options.signal,
  });
  const { train, holdout } = splitExamples(dataset.examples);

  options.onProgress?.("baseline", "Evaluating baseline on train examples");
  const baselineTrain = await evaluateArtifact({
    cwd: options.cwd,
    model: options.model,
    thinkingLevel: options.thinkingLevel,
    target,
    artifactText: target.fullText,
    objective: options.objective,
    examples: train,
    maxBytes,
    signal: options.signal,
  });

  options.onProgress?.("baseline", "Evaluating baseline on holdout examples");
  const baselineHoldout = await evaluateArtifact({
    cwd: options.cwd,
    model: options.model,
    thinkingLevel: options.thinkingLevel,
    target,
    artifactText: target.fullText,
    objective: options.objective,
    examples: holdout,
    maxBytes,
    signal: options.signal,
  });

  options.onProgress?.("candidates", `Generating ${options.candidateCount} candidate(s)`);
  const drafts = await generateCandidates({
    cwd: options.cwd,
    model: options.model,
    thinkingLevel: options.thinkingLevel,
    target,
    objective: options.objective,
    trainExamples: train,
    baselineTrain,
    maxBytes,
    candidateCount: options.candidateCount,
    signal: options.signal,
  });

  const candidates: CandidateRecord[] = [];
  for (let index = 0; index < drafts.length; index += 1) {
    const draft = drafts[index]!;
    const validation = validateCandidate(target, draft.candidateBody, maxBytes);
    if (!validation.valid) continue;

    const candidateFullText = reassembleArtifact(target.frontmatter, draft.candidateBody);
    options.onProgress?.("judge", `Evaluating candidate ${index + 1}/${drafts.length}: ${draft.name}`);
    const evaluation = await evaluateArtifact({
      cwd: options.cwd,
      model: options.model,
      thinkingLevel: options.thinkingLevel,
      target,
      artifactText: candidateFullText,
      objective: options.objective,
      examples: holdout,
      maxBytes,
      signal: options.signal,
    });

    candidates.push({
      ...draft,
      candidateFullText,
      evaluation,
      warnings: validation.warnings,
    });
  }

  if (candidates.length === 0) {
    throw new Error("All generated candidates were rejected by local guardrails.");
  }

  candidates.sort((a, b) => b.evaluation.aggregate.composite - a.evaluation.aggregate.composite);
  const bestCandidate = candidates[0]!;
  const improvement = bestCandidate.evaluation.aggregate.composite - baselineHoldout.aggregate.composite;

  const reportPath = path.join(runDir, "report.md");
  const originalPath = path.join(runDir, "original.md");
  const bestCandidatePath = path.join(runDir, "best-candidate.md");
  const datasetPath = path.join(runDir, "dataset.json");
  const manifestPath = path.join(runDir, "manifest.json");

  const result: EvolutionRunResult = {
    target,
    objective: options.objective,
    evalSource: options.evalSource,
    modelLabel,
    paths: {
      runDir,
      reportPath,
      originalPath,
      bestCandidatePath,
      datasetPath,
      manifestPath,
    },
    sessionSnippets: dataset.sessionSnippets,
    trainExamples: train,
    holdoutExamples: holdout,
    baselineTrain,
    baselineHoldout,
    candidates,
    bestCandidate,
    improvement,
    maxBytes,
  };

  const report = buildReportMarkdown(result);

  options.onProgress?.("write", "Writing run artifacts");
  await safeWriteFile(originalPath, target.fullText);
  await safeWriteFile(bestCandidatePath, bestCandidate.candidateFullText);
  await safeWriteFile(datasetPath, JSON.stringify({
    train,
    holdout,
    sessionSnippets: dataset.sessionSnippets,
  }, null, 2));
  await safeWriteFile(manifestPath, JSON.stringify({
    targetPath: target.path,
    objective: options.objective,
    evalSource: options.evalSource,
    modelLabel,
    maxBytes,
    baselineHoldout: baselineHoldout.aggregate,
    bestCandidate: {
      name: bestCandidate.name,
      rationale: bestCandidate.rationale,
      score: bestCandidate.evaluation.aggregate,
      warnings: bestCandidate.warnings,
      path: bestCandidatePath,
    },
    candidates: candidates.map((candidate) => ({
      name: candidate.name,
      rationale: candidate.rationale,
      warnings: candidate.warnings,
      score: candidate.evaluation.aggregate,
    })),
    createdAt: new Date().toISOString(),
  }, null, 2));
  await safeWriteFile(reportPath, report);

  for (const candidate of candidates) {
    const prefix = slugify(candidate.name) || "candidate";
    await safeWriteFile(path.join(runDir, "candidates", `${prefix}.md`), candidate.candidateFullText);
    await safeWriteFile(path.join(runDir, "candidates", `${prefix}.json`), JSON.stringify({
      rationale: candidate.rationale,
      warnings: candidate.warnings,
      evaluation: candidate.evaluation,
    }, null, 2));
  }

  return result;
}

export interface EvolutionSummaryDetails {
  runDir: string;
  reportPath: string;
  targetPath: string;
  objective: string;
  evalSource: string;
  modelLabel: string;
  trainExamples: number;
  holdoutExamples: number;
  candidateCount: number;
  baselineHoldoutScore: number;
  bestHoldoutScore: number;
  improvement: number;
  bestCandidateName: string;
  backend?: "typescript" | "python";
  optimizerUsed?: string;
}

export async function runEvolution(options: {
  cwd: string;
  targetPath: string;
  objective: string;
  evalSource: EvalSource;
  model?: string;
  thinkingLevel?: string;
  candidateCount: number;
  maxExamples: number;
  sessionQuery?: string;
  backend?: "auto" | "typescript" | "python";
  signal?: AbortSignal;
  onProgress?: (phase: string, detail?: string) => void;
}): Promise<EvolutionSummaryDetails> {
  const preferredBackend = options.backend ?? "auto";

  if (preferredBackend !== "typescript") {
    const pythonBackend = await detectPythonBackend();
    if (pythonBackend) {
      options.onProgress?.("backend", `Using Python backend (${pythonBackend.doctor.gepa ? "GEPA" : "DSPy"})`);
      const summary = await runPythonBackend(pythonBackend.python, {
        cwd: options.cwd,
        targetPath: options.targetPath,
        objective: options.objective,
        evalSource: options.evalSource,
        model: options.model,
        candidateCount: options.candidateCount,
        maxExamples: options.maxExamples,
        sessionQuery: options.sessionQuery,
      });
      return summary;
    }
    if (preferredBackend === "python") {
      throw new Error("Python backend requested, but DSPy/GEPA backend is unavailable. Install python_backend dependencies first.");
    }
  }

  options.onProgress?.("backend", "Using TypeScript backend");
  const result = await runTypeScriptEvolution(options);
  return toToolSummaryDetails(result);
}

export function buildToolSummary(result: EvolutionSummaryDetails): string {
  const sign = result.improvement >= 0 ? "+" : "";
  return [
    `Self-evolution completed for ${result.targetPath}`,
    `Backend: ${result.backend ?? "typescript"}${result.optimizerUsed ? ` (${result.optimizerUsed})` : ""}`,
    `Best candidate: ${result.bestCandidateName}`,
    `Holdout score: ${result.baselineHoldoutScore.toFixed(3)} → ${result.bestHoldoutScore.toFixed(3)} (${sign}${result.improvement.toFixed(3)})`,
    `Report: ${result.reportPath}`,
  ].join("\n");
}

export function toToolSummaryDetails(result: EvolutionRunResult): EvolutionSummaryDetails {
  return {
    runDir: result.paths.runDir,
    reportPath: result.paths.reportPath,
    targetPath: result.target.path,
    objective: result.objective,
    evalSource: result.evalSource,
    modelLabel: result.modelLabel,
    trainExamples: result.trainExamples.length,
    holdoutExamples: result.holdoutExamples.length,
    candidateCount: result.candidates.length,
    baselineHoldoutScore: result.baselineHoldout.aggregate.composite,
    bestHoldoutScore: result.bestCandidate.evaluation.aggregate.composite,
    improvement: result.improvement,
    bestCandidateName: result.bestCandidate.name,
    backend: "typescript",
    optimizerUsed: "typescript-proxy",
  };
}

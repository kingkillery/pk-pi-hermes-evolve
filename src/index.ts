import fs from "node:fs";
import path from "node:path";
import { StringEnum } from "@mariozechner/pi-ai";
import type { ExtensionAPI, ExtensionCommandContext, ExtensionContext } from "@mariozechner/pi-coding-agent";
import { getMarkdownTheme } from "@mariozechner/pi-coding-agent";
import { Markdown, Text } from "@mariozechner/pi-tui";
import { Type, type Static } from "@sinclair/typebox";
import { runEvolution, type EvolutionSummaryDetails } from "./engine.js";
import type { EvalSource, EvolutionOptions } from "./types.js";

const DEFAULT_OBJECTIVE = "Improve trigger clarity, execution quality, and practical usefulness while preserving the artifact's intent.";
const DEFAULT_CANDIDATES = 3;
const DEFAULT_MAX_EXAMPLES = 8;
const STATUS_KEY = "hermes-self-evolution";
const LAST_RUN_ENTRY = "hermes-self-evolution:last-run";

const ToolParams = Type.Object({
  targetPath: Type.String({ description: "Path to a local markdown or instruction file to evolve" }),
  objective: Type.Optional(Type.String({ description: "What to improve. Defaults to clarity + success rate without changing intent." })),
  evalSource: Type.Optional(StringEnum(["synthetic", "session", "mixed"] as const)),
  backend: Type.Optional(StringEnum(["auto", "typescript", "python"] as const)),
  candidateCount: Type.Optional(Type.Number({ description: "How many candidate revisions to generate (default 3, max 5)" })),
  maxExamples: Type.Optional(Type.Number({ description: "How many evaluation examples to generate (default 8, max 12)" })),
  sessionQuery: Type.Optional(Type.String({ description: "Optional hint phrase for mining relevant pi session history" })),
  model: Type.Optional(Type.String({ description: "Optional model override, e.g. anthropic/claude-sonnet-4-5" })),
  goldenTaskId: Type.Optional(Type.String({ description: "Optional golden task identifier for reproducible validation split evaluation" })),
  testCommand: Type.Optional(Type.String({ description: "Shell command to run as a validation gate before accepting candidates (e.g. 'npm test')" })),
  testTimeout: Type.Optional(Type.Number({ description: "Timeout in seconds for the test command (default 60)" })),
  createPR: Type.Optional(Type.Boolean({ description: "Create a git branch and optionally PR with the best candidate (default false)" })),
  persistGolden: Type.Optional(Type.Boolean({ description: "Persist golden dataset for reuse across runs (default true when goldenTaskId is set)" })),
});

type ToolInput = Static<typeof ToolParams>;

type LastRunData = Partial<EvolutionSummaryDetails>;

function getContentAsString(content: unknown): string {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  return content
    .map((item) => {
      if (!item || typeof item !== "object") return "";
      const block = item as { type?: string; text?: string };
      return block.type === "text" ? block.text ?? "" : "";
    })
    .filter(Boolean)
    .join("\n");
}

function normalizeEvalSource(label: string): EvalSource {
  const value = label.toLowerCase();
  if (value.includes("synthetic")) return "synthetic";
  if (value.includes("session")) return "session";
  return "mixed";
}

function clampCount(value: number | undefined, min: number, max: number, fallback: number): number {
  if (typeof value !== "number" || Number.isNaN(value)) return fallback;
  return Math.max(min, Math.min(max, Math.round(value)));
}

function formatModelLabel(ctx: ExtensionContext, override?: string): string | undefined {
  if (override?.trim()) return override.trim();
  if (!ctx.model) return undefined;
  return `${ctx.model.provider}/${ctx.model.id}`;
}

function getThinkingLevelSafe(pi: ExtensionAPI): string | undefined {
  try {
    return pi.getThinkingLevel();
  } catch {
    return undefined;
  }
}

function updateStatus(ctx: ExtensionContext, phase?: string, detail?: string): void {
  if (!phase) {
    ctx.ui.setStatus(STATUS_KEY, undefined);
    ctx.ui.setWidget(STATUS_KEY, undefined);
    return;
  }

  const statusText = `${ctx.ui.theme.fg("accent", "🧬 evolve")}${detail ? ` ${ctx.ui.theme.fg("muted", detail)}` : ""}`;
  ctx.ui.setStatus(STATUS_KEY, statusText);
  ctx.ui.setWidget(STATUS_KEY, [`${ctx.ui.theme.fg("accent", "Phase:")} ${phase}${detail ? ` — ${detail}` : ""}`]);
}

function shouldSkipDir(name: string): boolean {
  return [".git", "node_modules", "dist", "build", "coverage"].includes(name);
}

function collectArtifacts(root: string, depth = 0, acc: string[] = []): string[] {
  if (depth > 5) return acc;

  let entries: fs.Dirent[] = [];
  try {
    entries = fs.readdirSync(root, { withFileTypes: true });
  } catch {
    return acc;
  }

  for (const entry of entries) {
    const fullPath = path.join(root, entry.name);
    if (entry.isDirectory()) {
      if (shouldSkipDir(entry.name)) continue;
      if (fullPath.includes(`${path.sep}.pi${path.sep}hermes-self-evolution${path.sep}`)) continue;
      collectArtifacts(fullPath, depth + 1, acc);
      continue;
    }

    if (!entry.isFile()) continue;
    const lower = entry.name.toLowerCase();
    if (
      lower === "skill.md"
      || lower === "agents.md"
      || lower === "system.md"
      || lower === "append_system.md"
      || (lower.endsWith(".md") && fullPath.includes(`${path.sep}.pi${path.sep}prompts${path.sep}`))
      || (lower.endsWith(".md") && fullPath.includes(`${path.sep}.agents${path.sep}prompts${path.sep}`))
    ) {
      acc.push(fullPath);
    }
  }

  return acc;
}

function discoverArtifacts(cwd: string): string[] {
  const found = collectArtifacts(cwd);
  return Array.from(new Set(found)).sort();
}

function getLastRun(ctx: ExtensionContext): LastRunData | null {
  const entries = [...ctx.sessionManager.getEntries()].reverse();
  for (const entry of entries) {
    const custom = entry as { type?: string; customType?: string; data?: LastRunData };
    if (custom.type === "custom" && custom.customType === LAST_RUN_ENTRY && custom.data) {
      return custom.data;
    }
  }
  return null;
}

function buildSummaryMarkdown(details: EvolutionSummaryDetails): string {
  const sign = details.improvement >= 0 ? "+" : "";
  const goldenLine = details.goldenTaskId ? `\n- **Golden task:** ${details.goldenTaskId}` : "";
  const testLine = details.testGatePassed !== undefined ? `\n- **Test gate:** ${details.testGatePassed ? "passed ✅" : "failed ❌"}` : "";
  const driftLine = details.semanticDriftScore !== undefined ? `\n- **Semantic drift:** ${details.semanticDriftScore.toFixed(3)}` : "";
  const prLine = details.prBranch ? `\n- **PR branch:** ${details.prBranch}` : "";
  const tracesLine = `\n- **Traces captured:** ${details.tracesCaptured}`;
  const constraintsLine = `\n- **Constraints:** ${details.constraintsPassed ? "all passed ✅" : "some failed ❌"}`;
  return [
    "## Hermes-style self-evolution finished",
    "",
    `- **Target:** ${details.targetPath}`,
    `- **Objective:** ${details.objective}`,
    `- **Eval source:** ${details.evalSource}`,
    `- **Model:** ${details.modelLabel}`,
    `- **Splits:** ${details.trainExamples} train / ${details.validationExamples} validation / ${details.holdoutExamples} holdout`,
    `- **Backend:** ${details.backend ?? "typescript"}${details.optimizerUsed ? ` (${details.optimizerUsed})` : ""}`,
    `- **Selection (${details.selectionSplit}):** ${details.baselineValidationScore.toFixed(3)} → ${details.bestValidationScore.toFixed(3)}`,
    `- **Confirmation (${details.confirmationSplit}):** ${details.baselineHoldoutScore.toFixed(3)} → ${details.bestHoldoutScore.toFixed(3)} (${sign}${details.improvement.toFixed(3)})`,
    `- **Best candidate:** ${details.bestCandidateName}`,
    `- **Report:** ${details.reportPath}`,
    goldenLine,
    testLine,
    driftLine,
    prLine,
    tracesLine,
    constraintsLine,
  ].join("\n");
}

function usageText(cwd: string): string {
  return [
    "Usage:",
    "- /evolve                → interactive picker",
    "- /evolve path/to/file   → evolve a specific artifact",
    "- /evolve last           → show the last report saved in this session",
    "",
    `Artifacts are discovered under ${cwd} (.pi/skills, .pi/prompts, AGENTS.md, SYSTEM.md, etc.).`,
  ].join("\n");
}

async function chooseArtifactInteractively(ctx: ExtensionCommandContext): Promise<string | null> {
  const artifacts = discoverArtifacts(ctx.cwd).map((file) => path.relative(ctx.cwd, file) || path.basename(file));
  if (artifacts.length === 0) {
    ctx.ui.notify("No project-local pi artifacts were found to evolve.", "warning");
    return null;
  }
  const choice = await ctx.ui.select("Choose an artifact to evolve", artifacts.slice(0, 40));
  return choice ?? null;
}

async function resolveCommandOptions(args: string, ctx: ExtensionCommandContext): Promise<EvolutionOptions | null> {
  const trimmed = args.trim();
  if (trimmed === "help" || trimmed === "--help") {
    ctx.ui.notify(usageText(ctx.cwd), "info");
    return null;
  }

  if (trimmed === "last") {
    const last = getLastRun(ctx);
    if (!last?.reportPath) {
      ctx.ui.notify("No self-evolution report has been recorded in this session yet.", "warning");
      return null;
    }
    ctx.ui.notify(`Last report: ${last.reportPath}`, "info");
    return null;
  }

  let targetPath = trimmed;
  if (!targetPath) {
    if (!ctx.hasUI) {
      ctx.ui.notify(usageText(ctx.cwd), "info");
      return null;
    }
    const chosen = await chooseArtifactInteractively(ctx);
    if (!chosen) return null;
    targetPath = chosen;
  }

  const objective = (await ctx.ui.input("Evolution objective", DEFAULT_OBJECTIVE))?.trim() || DEFAULT_OBJECTIVE;
  const sourceChoice = await ctx.ui.select("Evaluation source", [
    "mixed (recommended)",
    "synthetic only",
    "session-heavy",
  ]);
  const evalSource = normalizeEvalSource(sourceChoice ?? "mixed");

  return {
    targetPath,
    objective,
    evalSource,
    candidateCount: DEFAULT_CANDIDATES,
    maxExamples: DEFAULT_MAX_EXAMPLES,
  };
}

async function executeEvolution(
  pi: ExtensionAPI,
  input: EvolutionOptions,
  ctx: ExtensionContext,
  onToolUpdate?: (partial: { content: Array<{ type: "text"; text: string }>; details: Record<string, unknown> }) => void,
): Promise<EvolutionSummaryDetails> {
  const resolvedModel = formatModelLabel(ctx, input.model);
  const thinkingLevel = getThinkingLevelSafe(pi);
  const candidateCount = clampCount(input.candidateCount, 1, 5, DEFAULT_CANDIDATES);
  const maxExamples = clampCount(input.maxExamples, 4, 12, DEFAULT_MAX_EXAMPLES);

  const result = await runEvolution({
    cwd: ctx.cwd,
    targetPath: input.targetPath,
    objective: input.objective?.trim() || DEFAULT_OBJECTIVE,
    evalSource: input.evalSource ?? "mixed",
    model: resolvedModel,
    thinkingLevel,
    candidateCount,
    maxExamples,
    sessionQuery: input.sessionQuery,
    backend: input.backend,
    goldenTaskId: input.goldenTaskId,
    testCommand: input.testCommand,
    testTimeout: input.testTimeout,
    createPR: input.createPR,
    persistGolden: input.persistGolden,
    signal: ctx.signal,
    onProgress: (phase, detail) => {
      updateStatus(ctx, phase, detail);
      onToolUpdate?.({
        content: [{ type: "text", text: `Phase: ${phase}${detail ? ` — ${detail}` : ""}` }],
        details: {},
      });
    },
  });

  pi.appendEntry(LAST_RUN_ENTRY, result);
  return result;
}

export default function hermesSelfEvolutionExtension(pi: ExtensionAPI): void {
  pi.registerMessageRenderer("hermes-self-evolution", (message) => {
    return new Markdown(getContentAsString(message.content), 0, 0, getMarkdownTheme());
  });

  pi.registerCommand("evolve", {
    description: "Run a Hermes-style self-evolution loop for a local skill, prompt, or instruction file",
    handler: async (args, ctx) => {
      const options = await resolveCommandOptions(args, ctx);
      if (!options) return;

      try {
        updateStatus(ctx, "start", "Preparing run");
        const details = await executeEvolution(pi, options, ctx);
        updateStatus(ctx, "done", `Best ${details.bestCandidateName}`);
        ctx.ui.notify(`Self-evolution report saved to ${details.reportPath}`, "info");
        pi.sendMessage({
          customType: "hermes-self-evolution",
          content: buildSummaryMarkdown(details),
          display: true,
          details,
        });
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        ctx.ui.notify(`Self-evolution failed: ${message}`, "error");
      } finally {
        updateStatus(ctx);
      }
    },
  });

  pi.registerTool({
    name: "self_evolve_artifact",
    label: "Self Evolve Artifact",
    description: "Run a Hermes-style reflective evolution loop over a local instruction artifact. It never overwrites the target; it writes a report and candidate files under .pi/hermes-self-evolution/.",
    promptSnippet: "Reflectively improve a local skill, prompt, or instruction file using synthetic/session-derived evaluation examples.",
    promptGuidelines: [
      "Use this tool when the user explicitly asks to improve or evolve a local pi skill, prompt, AGENTS.md, or SYSTEM.md file.",
      "Do not use this tool for arbitrary source code refactors; it is optimized for text instructions and prompts.",
      "Tell the user where the report and candidate files were written so they can review before applying changes.",
      "Optionally specify testCommand to run a validation gate (e.g. 'npm run typecheck') before accepting candidates.",
      "Optionally set createPR to true to auto-create a git branch with the best candidate.",
    ],
    parameters: ToolParams,
    async execute(_toolCallId, params: ToolInput, _signal, onUpdate, ctx) {
      try {
        updateStatus(ctx, "start", "Preparing run");
        const details = await executeEvolution(pi, params, ctx, onUpdate);
        const sign = details.improvement >= 0 ? "+" : "";
        const tracesNote = details.tracesCaptured > 0 ? `\nTraces: ${details.tracesCaptured}` : "";
        return {
          content: [{
            type: "text",
            text: [
              `Self-evolution completed for ${details.targetPath}`,
              `Best candidate: ${details.bestCandidateName}`,
              `Selection (${details.selectionSplit}): ${details.baselineValidationScore.toFixed(3)} → ${details.bestValidationScore.toFixed(3)}`,
              `Confirmation (${details.confirmationSplit}): ${details.baselineHoldoutScore.toFixed(3)} → ${details.bestHoldoutScore.toFixed(3)} (${sign}${details.improvement.toFixed(3)})`,
              details.constraintsPassed ? "Constraints: all passed ✅" : "Constraints: some failed ❌",
              details.testGatePassed !== undefined ? `Test gate: ${details.testGatePassed ? "passed" : "failed"}` : "",
              details.semanticDriftScore !== undefined ? `Drift: ${details.semanticDriftScore.toFixed(3)}` : "",
              details.prBranch ? `PR branch: ${details.prBranch}` : "",
              `Report: ${details.reportPath}`,
              tracesNote,
            ].filter(Boolean).join("\n"),
          }],
          details,
        };
      } finally {
        updateStatus(ctx);
      }
    },
    renderCall(args, theme) {
      const label = theme.fg("toolTitle", theme.bold("self_evolve_artifact "));
      const target = theme.fg("accent", String(args.targetPath ?? "<missing>"));
      const source = args.evalSource ? theme.fg("muted", ` ${String(args.evalSource)}`) : "";
      const test = args.testCommand ? theme.fg("muted", ` test="${String(args.testCommand)}"`) : "";
      const pr = args.createPR ? theme.fg("muted", " pr") : "";
      return new Text(`${label}${target}${source}${test}${pr}`, 0, 0);
    },
    renderResult(result, { expanded }, theme) {
      const details = result.details as EvolutionSummaryDetails | undefined;
      if (!details) {
        const textBlock = result.content.find((block) => block.type === "text");
        return new Text(textBlock?.type === "text" ? textBlock.text : "Self-evolution completed.", 0, 0);
      }

      const lines = [
        `${theme.fg("success", "✓ Evolution complete")}`,
        `${theme.fg("muted", "target:")} ${String(details.targetPath ?? "")}`,
        `${theme.fg("muted", "report:")} ${String(details.reportPath ?? "")}`,
      ];

      if (expanded) {
        const baselineValidation = typeof details.baselineValidationScore === "number" ? details.baselineValidationScore.toFixed(3) : "?";
        const bestValidation = typeof details.bestValidationScore === "number" ? details.bestValidationScore.toFixed(3) : "?";
        const baselineHoldout = typeof details.baselineHoldoutScore === "number" ? details.baselineHoldoutScore.toFixed(3) : "?";
        const bestHoldout = typeof details.bestHoldoutScore === "number" ? details.bestHoldoutScore.toFixed(3) : "?";
        const improvement = typeof details.improvement === "number" ? `${details.improvement >= 0 ? "+" : ""}${details.improvement.toFixed(3)}` : "?";
        lines.push(`${theme.fg("muted", "splits:")} ${details.trainExamples}/${details.validationExamples}/${details.holdoutExamples} (train/val/holdout)`);
        lines.push(`${theme.fg("muted", "selection:")} ${details.selectionSplit} ${baselineValidation} → ${bestValidation}`);
        lines.push(`${theme.fg("muted", "confirmation:")} ${details.confirmationSplit} ${baselineHoldout} → ${bestHoldout} (${improvement})`);
        lines.push(`${theme.fg("muted", "best candidate:")} ${String(details.bestCandidateName ?? "")}`);
        lines.push(`${theme.fg("muted", "backend:")} ${String(details.backend ?? "typescript")}${details.optimizerUsed ? ` (${details.optimizerUsed})` : ""}`);
        if (details.goldenTaskId) {
          lines.push(`${theme.fg("muted", "golden:")} ${details.goldenTaskId}`);
        }
        if (details.tracesCaptured > 0) {
          lines.push(`${theme.fg("muted", "traces:")} ${details.tracesCaptured}`);
        }
        if (details.constraintsPassed !== undefined) {
          lines.push(`${theme.fg("muted", "constraints:")} ${details.constraintsPassed ? "✅ all pass" : "❌ some fail"}`);
        }
        if (details.testGatePassed !== undefined) {
          lines.push(`${theme.fg("muted", "test gate:")} ${details.testGatePassed ? "✅ passed" : "❌ failed"}`);
        }
        if (details.semanticDriftScore !== undefined) {
          lines.push(`${theme.fg("muted", "drift:")} ${details.semanticDriftScore.toFixed(3)}`);
        }
        if (details.prBranch) {
          lines.push(`${theme.fg("muted", "PR:")} ${details.prBranch}`);
        }
      }

      return new Text(lines.join("\n"), 0, 0);
    },
  });
}

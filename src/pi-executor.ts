import { spawn } from "node:child_process";
import crypto from "node:crypto";
import fs from "node:fs/promises";
import path from "node:path";
import type { ExecutionObservation } from "./types.js";

const DEFAULT_TIMEOUT_MS = 120_000;

function getPiInvocation(args: string[]): { command: string; args: string[] } {
  const currentScript = process.argv[1];
  if (currentScript) return { command: process.execPath, args: [currentScript, ...args] };
  return { command: "pi", args };
}

function splitFrontmatter(fullText: string): { frontmatter?: string; body: string } {
  const match = fullText.match(/^---\r?\n([\s\S]*?)\r?\n---\r?\n?/);
  if (!match) return { body: fullText.trim() };
  return { frontmatter: match[1]?.trimEnd(), body: fullText.slice(match[0].length).trim() };
}

async function rmrf(target: string): Promise<void> {
  try {
    await fs.rm(target, { recursive: true, force: true });
  } catch {
    /* best effort */
  }
}

export interface ExecuteCandidateOptions {
  cwd: string;
  candidateFullText: string;
  taskInput: string;
  artifactName: string;
  model?: string;
  thinkingLevel?: string;
  timeoutMs?: number;
  signal?: AbortSignal;
}

/**
 * Spawns a real `pi` subprocess that consumes a candidate skill artifact
 * (typically a SKILL.md) and a task input. Returns the captured stdout/stderr,
 * exit code, and duration as an ExecutionObservation. The candidate text is
 * written to a temp skills directory so the spawned pi process picks it up
 * via --system-prompt (body) and an isolated skills root.
 *
 * SOFT-SPOT(meta-shape): the engine's executor log layout (engine.ts safeWriteFile
 *   for executor/<iter>/<ex>/meta.json) intentionally stores a compact subset
 *   {exitCode, durationMs, taskInput}; stdout and stderr live alongside as
 *   sibling stdout.log / stderr.log files rather than being inlined into meta.json.
 *   The full ExecutionObservation shape is rehydrated by reading the trio together.
 *   The smoke verifier accepts this split-file layout. see tests/smoke-test-report.md §Soft-spot dispositions.
 */
export async function executeCandidateInPi(options: ExecuteCandidateOptions): Promise<ExecutionObservation> {
  const tmpRoot = path.join(
    options.cwd,
    ".pi",
    "hermes-self-evolution",
    ".exec-tmp",
    crypto.randomUUID().slice(0, 8),
  );
  const skillsDir = path.join(tmpRoot, "skills");
  const skillDir = path.join(skillsDir, options.artifactName || "candidate");
  const skillPath = path.join(skillDir, "SKILL.md");

  await fs.mkdir(skillDir, { recursive: true });
  await fs.writeFile(skillPath, options.candidateFullText, "utf8");

  const { body } = splitFrontmatter(options.candidateFullText);
  const systemPrompt = body || options.candidateFullText;

  const args: string[] = [
    "-p",
    "--no-session",
    "--no-extensions",
    "--no-themes",
    "--system-prompt",
    systemPrompt,
    options.taskInput.slice(0, 8000),
  ];
  if (options.model) args.splice(args.length - 1, 0, "--model", options.model);
  if (options.thinkingLevel && options.thinkingLevel !== "off") args.splice(args.length - 1, 0, "--thinking", options.thinkingLevel);

  const invocation = getPiInvocation(args);
  const timeoutMs = options.timeoutMs ?? DEFAULT_TIMEOUT_MS;
  const startedAt = Date.now();

  const observation = await new Promise<ExecutionObservation>((resolve) => {
    let stdout = "";
    let stderr = "";
    let settled = false;
    const child = spawn(invocation.command, invocation.args, {
      cwd: options.cwd,
      env: {
        ...process.env,
        PI_SKIP_VERSION_CHECK: "1",
        PI_SKILLS_DIR: skillsDir,
      },
      stdio: ["pipe", "pipe", "pipe"],
    });

    const finish = (exitCode: number, durationMs: number, extraStderr?: string) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      options.signal?.removeEventListener("abort", onAbort);
      resolve({
        stdout: stdout.trim(),
        stderr: (extraStderr ? `${stderr}\n${extraStderr}` : stderr).trim(),
        exitCode,
        durationMs,
      });
    };

    const timer = setTimeout(() => {
      try {
        child.kill();
      } catch {
        /* ignore */
      }
      finish(124, Date.now() - startedAt, `Timed out after ${timeoutMs}ms`);
    }, timeoutMs);

    const onAbort = () => {
      try {
        child.kill();
      } catch {
        /* ignore */
      }
      finish(-1, Date.now() - startedAt, "Aborted by signal");
    };
    options.signal?.addEventListener("abort", onAbort, { once: true });

    child.stdout.on("data", (chunk: Buffer) => {
      stdout += String(chunk);
    });
    child.stderr.on("data", (chunk: Buffer) => {
      stderr += String(chunk);
    });
    child.on("error", (err: Error) => {
      finish(-1, Date.now() - startedAt, err.message);
    });
    child.on("close", (code) => {
      finish(code ?? 1, Date.now() - startedAt);
    });

    try {
      child.stdin.write(options.taskInput);
    } catch {
      /* ignore EPIPE */
    }
    try {
      child.stdin.end();
    } catch {
      /* ignore */
    }
  });

  await rmrf(tmpRoot);
  return observation;
}

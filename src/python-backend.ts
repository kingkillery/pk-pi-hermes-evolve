import { spawn } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

export interface PythonBackendDoctor {
  ok: boolean;
  python?: string;
  dspy?: boolean;
  gepa?: boolean;
  entrypoint?: string;
  error?: string;
}

export interface PythonBackendRunInput {
  cwd: string;
  targetPath: string;
  objective: string;
  evalSource: string;
  model?: string;
  candidateCount: number;
  maxExamples: number;
  sessionQuery?: string;
}

export interface PythonBackendRunSummary {
  backend: "python";
  optimizerUsed?: string;
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
  error?: string;
}

function getPackageRoot(): string {
  return path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
}

function getRunnerPath(): string {
  return path.join(getPackageRoot(), "python_backend", "run_backend.py");
}

function pythonCandidates(): string[] {
  const configured = process.env.PI_HERMES_EVOLVE_PYTHON?.trim();
  return [configured, "python3", "python"].filter((value): value is string => Boolean(value));
}

async function runProcess(command: string, args: string[], stdinText?: string): Promise<{ code: number; stdout: string; stderr: string }> {
  return await new Promise((resolve, reject) => {
    const child = spawn(command, args, {
      cwd: getPackageRoot(),
      env: { ...process.env, PYTHONUTF8: "1" },
      stdio: ["pipe", "pipe", "pipe"],
    });

    let stdout = "";
    let stderr = "";
    child.stdout.on("data", (chunk) => {
      stdout += String(chunk);
    });
    child.stderr.on("data", (chunk) => {
      stderr += String(chunk);
    });
    child.on("error", (error) => reject(error));
    child.on("close", (code) => resolve({ code: code ?? 1, stdout: stdout.trim(), stderr: stderr.trim() }));

    if (stdinText) child.stdin.write(stdinText);
    child.stdin.end();
  });
}

export async function detectPythonBackend(): Promise<{ python: string; doctor: PythonBackendDoctor } | null> {
  const runner = getRunnerPath();
  if (!fs.existsSync(runner)) return null;

  for (const python of pythonCandidates()) {
    try {
      const result = await runProcess(python, [runner, "doctor"]);
      if (result.code !== 0) continue;
      const doctor = JSON.parse(result.stdout) as PythonBackendDoctor;
      if (doctor.ok) return { python, doctor };
    } catch {
      // try next interpreter
    }
  }
  return null;
}

export async function runPythonBackend(python: string, input: PythonBackendRunInput): Promise<PythonBackendRunSummary> {
  const runner = getRunnerPath();
  const result = await runProcess(python, [runner, "run"], JSON.stringify(input));
  let payload: PythonBackendRunSummary | { error?: string };
  try {
    payload = JSON.parse(result.stdout || "{}") as PythonBackendRunSummary | { error?: string };
  } catch {
    throw new Error(`Python backend returned invalid JSON: ${result.stdout || result.stderr}`);
  }

  if (result.code !== 0 || "error" in payload) {
    throw new Error(payload.error || result.stderr || `Python backend failed with exit ${result.code}`);
  }

  return payload as PythonBackendRunSummary;
}

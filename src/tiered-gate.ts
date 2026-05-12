import { spawn } from "node:child_process";
import type { EvalExample, TieredGateResult } from "./types.js";

export interface TieredGateOptions {
	cwd: string;
	candidateText: string;
	tsConfigPath?: string;
	cohortExamples?: EvalExample[];
	baselineScore?: number;
	maxRegressionPct?: number;
	judgeFunc?: (examples: EvalExample[]) => Promise<{ composite: number }>;
	coherenceCheck?: () => Promise<{ passed: boolean; detail: string }>;
	signal?: AbortSignal;
}

export async function runTieredGate(options: TieredGateOptions): Promise<TieredGateResult[]> {
	const results: TieredGateResult[] = [];

	const tsResult = await runTypecheckTier(options);
	results.push(tsResult);
	if (!tsResult.passed) return results;

	const cohortResult = await runCohortTier(options);
	results.push(cohortResult);
	if (!cohortResult.passed) return results;

	const coherenceResult = await runCoherenceTier(options);
	results.push(coherenceResult);
	return results;
}

async function runTypecheckTier(options: TieredGateOptions): Promise<TieredGateResult> {
	const start = Date.now();
	const tsConfigPath = options.tsConfigPath ?? "tsconfig.json";
	// On Windows npx is a .cmd shim, so spawning it directly without a shell fails with ENOENT.
	const invocation = process.platform === "win32"
		? { cmd: process.env.ComSpec ?? "cmd.exe", args: ["/c", "npx", "tsc", "--noEmit", "-p", tsConfigPath] }
		: { cmd: "npx", args: ["tsc", "--noEmit", "-p", tsConfigPath] };

	return await new Promise<TieredGateResult>((resolve) => {
		const child = spawn(invocation.cmd, invocation.args, {
			cwd: options.cwd,
			env: { ...process.env },
			stdio: ["pipe", "pipe", "pipe"],
		});
		let stdout = "";
		let stderr = "";
		const onAbort = () => {
			child.kill();
		};
		options.signal?.addEventListener("abort", onAbort, { once: true });
		child.stdout.on("data", (c: Buffer) => {
			stdout += String(c);
		});
		child.stderr.on("data", (c: Buffer) => {
			stderr += String(c);
		});
		child.on("error", (err) => {
			options.signal?.removeEventListener("abort", onAbort);
			resolve({
				tier: "typecheck",
				passed: false,
				reasonCode: "typecheck_failed",
				detail: truncateTail(err.message, 500),
				durationMs: Date.now() - start,
			});
		});
		child.on("close", (code) => {
			options.signal?.removeEventListener("abort", onAbort);
			const durationMs = Date.now() - start;
			if (options.signal?.aborted) {
				resolve({
					tier: "typecheck",
					passed: false,
					reasonCode: "typecheck_failed",
					detail: "aborted",
					durationMs,
				});
				return;
			}
			if (code === 0) {
				resolve({
					tier: "typecheck",
					passed: true,
					reasonCode: "ok",
					detail: "typecheck clean",
					durationMs,
				});
				return;
			}
			const combined = `${stderr}${stderr && stdout ? "\n" : ""}${stdout}`;
			resolve({
				tier: "typecheck",
				passed: false,
				reasonCode: "typecheck_failed",
				detail: truncateTail(combined || `exit code ${code ?? -1}`, 500),
				durationMs,
			});
		});
		child.stdin.end();
	});
}

async function runCohortTier(options: TieredGateOptions): Promise<TieredGateResult> {
	const start = Date.now();
	const examples = options.cohortExamples;
	if (!examples || examples.length === 0 || !options.judgeFunc || options.baselineScore === undefined) {
		return {
			tier: "cohort",
			passed: true,
			reasonCode: "skipped_no_cohort",
			detail: "no cohort configured",
			durationMs: Date.now() - start,
		};
	}
	const threshold = options.maxRegressionPct ?? 0.02;
	try {
		const result = await options.judgeFunc(examples);
		const delta = result.composite - options.baselineScore;
		const durationMs = Date.now() - start;
		if (delta < -threshold) {
			return {
				tier: "cohort",
				passed: false,
				reasonCode: "cohort_regression",
				detail: `delta=${delta.toFixed(4)}, threshold=${(-threshold).toFixed(4)}`,
				durationMs,
			};
		}
		return {
			tier: "cohort",
			passed: true,
			reasonCode: "ok",
			detail: `delta=${delta.toFixed(4)}`,
			durationMs,
		};
	} catch (err) {
		return {
			tier: "cohort",
			passed: false,
			reasonCode: "cohort_regression",
			detail: truncateTail(`judge error: ${err instanceof Error ? err.message : String(err)}`, 500),
			durationMs: Date.now() - start,
		};
	}
}

async function runCoherenceTier(options: TieredGateOptions): Promise<TieredGateResult> {
	const start = Date.now();
	if (!options.coherenceCheck) {
		return {
			tier: "coherence",
			passed: true,
			reasonCode: "skipped_no_check",
			detail: "no coherence check configured",
			durationMs: Date.now() - start,
		};
	}
	try {
		const result = await options.coherenceCheck();
		const durationMs = Date.now() - start;
		if (!result.passed) {
			return {
				tier: "coherence",
				passed: false,
				reasonCode: "coherence_failed",
				detail: result.detail,
				durationMs,
			};
		}
		return {
			tier: "coherence",
			passed: true,
			reasonCode: "ok",
			detail: result.detail,
			durationMs,
		};
	} catch (err) {
		return {
			tier: "coherence",
			passed: false,
			reasonCode: "coherence_failed",
			detail: truncateTail(`coherence error: ${err instanceof Error ? err.message : String(err)}`, 500),
			durationMs: Date.now() - start,
		};
	}
}

function truncateTail(s: string, n: number): string {
	if (s.length <= n) return s;
	return s.slice(s.length - n);
}

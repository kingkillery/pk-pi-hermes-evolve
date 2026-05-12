import type {
  ArtifactEvaluation,
  CandidateRecord,
  EvolutionRunResult,
  ExecutionObservation,
  ExecutionTrace,
  IterationRecord,
  LineageEntry,
  TieredGateResult,
} from "../src/types.js";

export interface ExpectedRunDirShape {
  required: string[];
  iterationFiles: { pattern: string; min: number };
  executorTree: { pattern: string };
  candidateFiles: { mdPattern: string; jsonPattern: string };
}

export const EXPECTED_RUN_DIR_SHAPE: ExpectedRunDirShape = {
  required: [
    "original.md",
    "best-candidate.md",
    "dataset.json",
    "manifest.json",
    "report.md",
    "traces/all-traces.json",
  ],
  iterationFiles: { pattern: "iterations/<n>.json", min: 1 },
  executorTree: { pattern: "executor/<iteration>/<exampleIndex>/{stdout.log,stderr.log,meta.json}" },
  candidateFiles: { mdPattern: "candidates/<name>.md", jsonPattern: "candidates/<name>.json" },
};

export const EXPECTED_LINEAGE_ENTRY_KEYS: Array<keyof LineageEntry> = [
  "runId",
  "parentRunId",
  "artifactHash",
  "parentArtifactHash",
  "score",
  "mutationRationale",
  "createdAt",
];

export const EXPECTED_ITERATION_RECORD_KEYS: Array<keyof IterationRecord> = [
  "iteration",
  "parentCandidate",
  "mutationRationale",
  "reflectionPrompt",
  "candidate",
  "evaluation",
  "traces",
  "scoreDelta",
  "accepted",
];

export const EXPECTED_TIERED_GATE_KEYS: Array<keyof TieredGateResult> = [
  "tier",
  "passed",
  "reasonCode",
  "detail",
  "durationMs",
];

export const EXPECTED_EXECUTION_OBSERVATION_KEYS: Array<keyof ExecutionObservation> = [
  "stdout",
  "stderr",
  "exitCode",
  "durationMs",
];

export function assertEvolutionRunResultShape(result: EvolutionRunResult): void {
  if (!result.target?.path) throw new Error("EvolutionRunResult.target.path missing");
  if (!result.paths?.runDir) throw new Error("EvolutionRunResult.paths.runDir missing");
  if (!Array.isArray(result.candidates)) throw new Error("EvolutionRunResult.candidates not array");
  if (!result.bestCandidate) throw new Error("EvolutionRunResult.bestCandidate missing");
  if (typeof result.improvement !== "number") throw new Error("EvolutionRunResult.improvement not number");
  if (!Array.isArray(result.baselineTraces)) throw new Error("EvolutionRunResult.baselineTraces not array");
  if (result.iterations !== undefined && !Array.isArray(result.iterations)) {
    throw new Error("EvolutionRunResult.iterations must be undefined or array");
  }
}

export function assertCandidateRecordShape(record: CandidateRecord): void {
  if (!record.candidateFullText) throw new Error("CandidateRecord.candidateFullText missing");
  assertArtifactEvaluation(record.evaluation);
  if (!Array.isArray(record.executionTraces)) throw new Error("CandidateRecord.executionTraces not array");
  if (!Array.isArray(record.constraints)) throw new Error("CandidateRecord.constraints not array");
}

function assertArtifactEvaluation(ev: ArtifactEvaluation): void {
  if (typeof ev.aggregate.composite !== "number") throw new Error("aggregate.composite not number");
  if (!Array.isArray(ev.examples)) throw new Error("evaluation.examples not array");
}

void EXPECTED_LINEAGE_ENTRY_KEYS;
void EXPECTED_ITERATION_RECORD_KEYS;
void EXPECTED_TIERED_GATE_KEYS;
void EXPECTED_EXECUTION_OBSERVATION_KEYS;
void ((_e: ExecutionTrace) => undefined);

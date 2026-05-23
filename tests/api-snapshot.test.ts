import * as engine from "../src/engine.js";
import * as indexModule from "../src/index.js";
import * as pythonBackend from "../src/python-backend.js";
import * as piExecutor from "../src/pi-executor.js";
import * as tieredGate from "../src/tiered-gate.js";
import * as constraintsStructure from "../src/constraints-structure.js";
import * as lineage from "../src/lineage.js";
import * as types from "../src/types.js";

const EXPECTED_ENGINE_EXPORTS = [
  "scanForSecrets",
  "resolveArtifactTarget",
  "loadGoldenDataset",
  "runEvolution",
  "buildToolSummary",
  "toToolSummaryDetails",
] as const;

const EXPECTED_INDEX_EXPORTS = ["default"] as const;

const EXPECTED_PI_EXECUTOR_EXPORTS = ["executeCandidateInPi"] as const;

const EXPECTED_TIERED_GATE_EXPORTS = ["runTieredGate"] as const;

const EXPECTED_CONSTRAINTS_STRUCTURE_EXPORTS = [
  "buildSkillStructureReport",
  "checkSkillStructure",
] as const;

const EXPECTED_LINEAGE_EXPORTS = [
  "appendLineageEntry",
  "loadLineage",
  "loadBestAncestor",
] as const;

const EXPECTED_PYTHON_BACKEND_EXPORTS = [
  "detectPythonBackend",
  "runPythonBackend",
] as const;

const EXPECTED_TYPE_EXPORTS = [
  "ArtifactType",
  "EvalSource",
  "Difficulty",
  "ArtifactTarget",
  "SessionSnippet",
  "EvalExample",
  "JudgeResult",
  "ExampleEvaluation",
  "AggregateScore",
  "ArtifactEvaluation",
  "CandidateDraft",
  "ExecutionTrace",
  "ConstraintName",
  "ConstraintResult",
  "ConstraintConfig",
  "PRAutomationResult",
  "SecretScanResult",
  "GoldenDatasetManifest",
  "CandidateRecord",
  "EvolutionOptions",
  "EvolutionPaths",
  "GoldenDataset",
  "EvolutionRunResult",
  "ReflectionPrompt",
  "IterationRecord",
  "ExecutionObservation",
  "TieredGateResult",
  "LineageEntry",
  "SkillStructureReport",
  "BackendMode",
  "EvolutionSummaryDetails",
  "ToolSummaryDetails",
] as const;

function assertHasAll(module: Record<string, unknown>, expected: readonly string[], moduleName: string): void {
  const missing = expected.filter((name) => !(name in module));
  if (missing.length > 0) {
    throw new Error(`API snapshot drift in ${moduleName}: missing ${missing.join(", ")}`);
  }
}

export function runApiSnapshotCheck(): { ok: true; checked: number } {
  assertHasAll(engine as unknown as Record<string, unknown>, EXPECTED_ENGINE_EXPORTS, "engine");
  assertHasAll(indexModule as unknown as Record<string, unknown>, EXPECTED_INDEX_EXPORTS, "index");
  assertHasAll(pythonBackend as unknown as Record<string, unknown>, EXPECTED_PYTHON_BACKEND_EXPORTS, "python-backend");
  assertHasAll(piExecutor as unknown as Record<string, unknown>, EXPECTED_PI_EXECUTOR_EXPORTS, "pi-executor");
  assertHasAll(tieredGate as unknown as Record<string, unknown>, EXPECTED_TIERED_GATE_EXPORTS, "tiered-gate");
  assertHasAll(constraintsStructure as unknown as Record<string, unknown>, EXPECTED_CONSTRAINTS_STRUCTURE_EXPORTS, "constraints-structure");
  assertHasAll(lineage as unknown as Record<string, unknown>, EXPECTED_LINEAGE_EXPORTS, "lineage");
  const checked =
    EXPECTED_ENGINE_EXPORTS.length
    + EXPECTED_INDEX_EXPORTS.length
    + EXPECTED_PYTHON_BACKEND_EXPORTS.length
    + EXPECTED_PI_EXECUTOR_EXPORTS.length
    + EXPECTED_TIERED_GATE_EXPORTS.length
    + EXPECTED_CONSTRAINTS_STRUCTURE_EXPORTS.length
    + EXPECTED_LINEAGE_EXPORTS.length
    + EXPECTED_TYPE_EXPORTS.length;
  void types;
  return { ok: true, checked };
}

if (import.meta.url === `file://${process.argv[1]?.replace(/\\/g, "/")}`) {
  const result = runApiSnapshotCheck();
  console.log(`api-snapshot: ${result.checked} symbols verified`);
}

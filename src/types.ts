export type ArtifactType = "skill" | "prompt" | "instructions";
export type EvalSource = "synthetic" | "session" | "mixed";
export type Difficulty = "easy" | "medium" | "hard";

export interface ArtifactTarget {
  path: string;
  name: string;
  type: ArtifactType;
  fullText: string;
  body: string;
  frontmatter?: string;
  originalBytes: number;
  placeholders: string[];
  topHeading?: string;
}

export interface SessionSnippet {
  sessionFile: string;
  userText: string;
  assistantText: string;
  score: number;
}

export interface EvalExample {
  taskInput: string;
  expectedBehavior: string;
  difficulty: Difficulty;
  category: string;
  source: "synthetic" | "session";
}

export interface JudgeResult {
  responsePreview: string;
  correctness: number;
  procedureFollowing: number;
  conciseness: number;
  feedback: string;
  confidence: number;
}

export interface ExampleEvaluation extends JudgeResult {
  example: EvalExample;
  composite: number;
}

export interface AggregateScore {
  correctness: number;
  procedureFollowing: number;
  conciseness: number;
  confidence: number;
  lengthPenalty: number;
  composite: number;
}

export interface ArtifactEvaluation {
  aggregate: AggregateScore;
  examples: ExampleEvaluation[];
}

export interface CandidateDraft {
  name: string;
  rationale: string;
  candidateBody: string;
}

export interface ExecutionTrace {
  traceId: string;
  artifactText: string;
  taskInput: string;
  expectedBehavior: string;
  rawOutput: string;
  responsePreview: string;
  scores: {
    correctness: number;
    procedureFollowing: number;
    conciseness: number;
    composite: number;
  };
  feedback: string;
  isFailure: boolean;
  timestamp: string;
}

export type ConstraintName =
  | "non_empty"
  | "size_limit"
  | "growth_limit"
  | "placeholder_preservation"
  | "top_heading_preservation"
  | "frontmatter_preservation"
  | "semantic_drift"
  | "skill_structure";

export interface ConstraintResult {
  name: ConstraintName;
  passed: boolean;
  message: string;
  details?: string;
}

export interface ConstraintConfig {
  maxSizeBytes: number;
  maxGrowthRatio: number;
  testCommand?: string;
  testTimeoutMs: number;
  checkSemanticDrift: boolean;
  maxDriftScore: number;
}

export interface PRAutomationResult {
  branch: string;
  commitSha: string;
  prUrl?: string;
  prNumber?: number;
  diffStat: string;
}

export interface SecretScanResult {
  found: boolean;
  patterns: Array<{ pattern: string; match: string; location: string }>;
}

export interface GoldenDatasetManifest {
  id: string;
  artifactPath: string;
  artifactName: string;
  exampleCount: number;
  trainCount: number;
  validationCount: number;
  holdoutCount: number;
  createdAt: string;
  lastUsedAt: string;
}

export interface CandidateRecord extends CandidateDraft {
  candidateFullText: string;
  evaluation: ArtifactEvaluation;
  holdoutEvaluation?: ArtifactEvaluation;
  executionTraces: ExecutionTrace[];
  constraints: ConstraintResult[];
  warnings: string[];
  semanticDriftScore?: number;
  testPassed?: boolean;
  executionObservation?: ExecutionObservation;
  gateResults?: TieredGateResult[];
  /**
   * True when this candidate was promoted by the engine's fallback acceptance
   * path because no iteration met the strict score-delta + constraints gate.
   * Surfaced in manifest.json#bestCandidate so downstream tools can detect
   * degenerate "winner" promotion without spelunking `warnings`.
   */
  wasFallbackPromoted?: boolean;
  /** Name of the pool member this candidate was mutated (or merged) from. Undefined for the baseline itself. */
  parentCandidate?: string;
  /** How this candidate's draft was produced: reflective mutation of a Pareto-selected parent, or a merge of two frontier candidates. */
  selectionMethod?: "mutation" | "merge";
  /** Composite judge score on the cheap train-set minibatch, used as the pre-filter gate before a full validation pass. */
  minibatchScore?: number;
}

export interface EvolutionOptions {
  targetPath: string;
  objective?: string;
  evalSource?: EvalSource;
  backend?: "auto" | "typescript" | "python";
  candidateCount?: number;
  maxExamples?: number;
  sessionQuery?: string;
  model?: string;
  thinkingLevel?: string;
  goldenTaskId?: string;
  testCommand?: string;
  testTimeout?: number;
  createPR?: boolean;
  persistGolden?: boolean;
  /** Deterministic seed for splitExamples and any future RNG consumers. When unset, falls back to unseeded Math.random. */
  seed?: number;
  /** Cohort of EvalExamples used by the tiered gate's cohort-regression tier. */
  cohortExamples?: EvalExample[];
  /** Judge callback invoked by the tiered gate to score the cohort. Required when cohortExamples is supplied. */
  cohortJudgeFunc?: (examples: EvalExample[]) => Promise<{ composite: number }>;
  /** Coherence check callback invoked by the tiered gate's coherence tier. */
  coherenceCheck?: () => Promise<{ passed: boolean; detail: string }>;
  /** Override the tsconfig path the tiered-gate typecheck tier runs against. Useful for forcing typecheck-tier failure in test scenarios. */
  tsConfigPath?: string;
}

export interface EvolutionPaths {
  runDir: string;
  reportPath: string;
  originalPath: string;
  bestCandidatePath: string;
  datasetPath: string;
  manifestPath: string;
}

export interface GoldenDataset {
  id: string;
  examples: EvalExample[];
  description: string;
}

export interface EvolutionRunResult {
  target: ArtifactTarget;
  objective: string;
  evalSource: EvalSource;
  modelLabel: string;
  selectionSplit: "validation";
  confirmationSplit: "holdout";
  paths: EvolutionPaths;
  sessionSnippets: SessionSnippet[];
  trainExamples: EvalExample[];
  validationExamples: EvalExample[];
  holdoutExamples: EvalExample[];
  golden: GoldenDataset | null;
  baselineTrain: ArtifactEvaluation;
  baselineValidation: ArtifactEvaluation;
  baselineHoldout: ArtifactEvaluation;
  candidates: CandidateRecord[];
  bestCandidate: CandidateRecord;
  improvement: number;
  maxBytes: number;
  baselineTraces: ExecutionTrace[];
  prResult?: PRAutomationResult;
  iterations?: IterationRecord[];
  /** Names of the final Pareto-frontier candidates (best-on-at-least-one-validation-instance), for reporting. */
  paretoFrontier?: string[];
  /** Number of system-aware merge candidates attempted during this run. */
  mergeAttempts?: number;
  /** Number of iterations rejected by the minibatch pre-filter before reaching a full validation pass. */
  minibatchFilteredCount?: number;
}

export interface ReflectionPrompt {
  priorTraces: ExecutionTrace[];
  priorJudgeFeedback: string[];
  objective: string;
  weaknessSummary: string;
}

export interface IterationRecord {
  iteration: number;
  parentCandidate?: string;
  mutationRationale: string;
  reflectionPrompt: ReflectionPrompt;
  candidate: CandidateDraft;
  evaluation: ArtifactEvaluation;
  traces: ExecutionTrace[];
  scoreDelta: number;
  accepted: boolean;
  gateResults?: TieredGateResult[];
  /** How the draft evaluated in this iteration was produced. */
  selectionMethod?: "mutation" | "merge";
  /** Size of the Pareto frontier the parent was sampled from (mutation only). */
  paretoFrontierSize?: number;
  /**
   * True when the candidate was rejected by the cheap minibatch pre-filter
   * (composite on a small train subset did not beat its parent) and therefore
   * never went through the expensive full-validation + real-executor pass.
   * When true, `evaluation`/`traces` hold the minibatch-only results, not a
   * validation-set evaluation.
   */
  minibatchFiltered?: boolean;
}

export interface ExecutionObservation {
  stdout: string;
  stderr: string;
  exitCode: number;
  durationMs: number;
  capturedFiles?: Record<string, string>;
}

export interface TieredGateResult {
  tier: "typecheck" | "cohort" | "coherence";
  passed: boolean;
  reasonCode: string;
  detail: string;
  durationMs: number;
}

export interface LineageEntry {
  runId: string;
  parentRunId?: string;
  artifactPath?: string;
  artifactHash: string;
  parentArtifactHash?: string;
  score: number;
  mutationRationale: string;
  createdAt: string;
}

export interface SkillStructureReport {
  hasFrontmatter: boolean;
  hasName: boolean;
  hasDescription: boolean;
  nameInFirst500: boolean;
  descriptionInFirst500: boolean;
  errors: string[];
}

export type BackendMode = "typescript" | "python-accelerate";

export type EvolutionSummaryDetails = ToolSummaryDetails;

export interface ToolSummaryDetails {
  runDir: string;
  reportPath: string;
  targetPath: string;
  objective: string;
  evalSource: EvalSource;
  modelLabel: string;
  selectionSplit: "validation";
  confirmationSplit: "holdout";
  trainExamples: number;
  validationExamples: number;
  holdoutExamples: number;
  goldenTaskId: string | null;
  candidateCount: number;
  baselineValidationScore: number;
  bestValidationScore: number;
  baselineHoldoutScore: number;
  bestHoldoutScore: number;
  improvement: number;
  bestCandidateName: string;
  tracesCaptured: number;
  constraintsPassed: boolean;
  testGatePassed?: boolean;
  semanticDriftScore?: number;
  prBranch?: string;
  backend?: "typescript" | "python";
  optimizerUsed?: string;
}

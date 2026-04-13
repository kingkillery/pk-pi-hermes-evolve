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
  | "semantic_drift";

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
}

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

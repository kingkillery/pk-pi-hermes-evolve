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

export interface CandidateRecord extends CandidateDraft {
  candidateFullText: string;
  evaluation: ArtifactEvaluation;
  warnings: string[];
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
}

export interface EvolutionPaths {
  runDir: string;
  reportPath: string;
  originalPath: string;
  bestCandidatePath: string;
  datasetPath: string;
  manifestPath: string;
}

export interface EvolutionRunResult {
  target: ArtifactTarget;
  objective: string;
  evalSource: EvalSource;
  modelLabel: string;
  paths: EvolutionPaths;
  sessionSnippets: SessionSnippet[];
  trainExamples: EvalExample[];
  holdoutExamples: EvalExample[];
  baselineTrain: ArtifactEvaluation;
  baselineHoldout: ArtifactEvaluation;
  candidates: CandidateRecord[];
  bestCandidate: CandidateRecord;
  improvement: number;
  maxBytes: number;
}

export interface ToolSummaryDetails {
  runDir: string;
  reportPath: string;
  targetPath: string;
  objective: string;
  evalSource: EvalSource;
  modelLabel: string;
  trainExamples: number;
  holdoutExamples: number;
  candidateCount: number;
  baselineHoldoutScore: number;
  bestHoldoutScore: number;
  improvement: number;
  bestCandidateName: string;
}

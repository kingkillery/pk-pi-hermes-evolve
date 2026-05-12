import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(__dirname, "..");

export interface ParityRow {
  capability: string;
  status: "complete" | "optional" | "missing";
  evidence: string;
}

export const EXPECTED_PARITY_ROWS: ParityRow[] = [
  { capability: "3-source dataset", status: "complete", evidence: "src/engine.ts" },
  { capability: "Train / validation / holdout split", status: "complete", evidence: "src/engine.ts" },
  { capability: "Golden dataset persistence", status: "complete", evidence: "src/engine.ts" },
  { capability: "Hermes-weighted judge", status: "complete", evidence: "src/engine.ts" },
  { capability: "7-check constraint validator", status: "complete", evidence: "src/engine.ts + src/constraints-structure.ts" },
  { capability: "Execution traces", status: "complete", evidence: "src/engine.ts" },
  { capability: "Secret scanner", status: "complete", evidence: "src/engine.ts" },
  { capability: "Optional test-command gate", status: "complete", evidence: "src/engine.ts" },
  { capability: "Optional PR automation", status: "complete", evidence: "src/engine.ts" },
  { capability: "Iterative reflective loop", status: "complete", evidence: "src/engine.ts iterations/" },
  { capability: "Pi-native executor", status: "complete", evidence: "src/pi-executor.ts" },
  { capability: "Tiered regression gate", status: "complete", evidence: "src/tiered-gate.ts" },
  { capability: "SKILL.md structural validator", status: "complete", evidence: "src/constraints-structure.ts" },
  { capability: "Cross-run lineage memory", status: "complete", evidence: "src/lineage.ts" },
  { capability: "TS as default, Python as --accelerate", status: "complete", evidence: "README.md + src/python-backend.ts" },
  { capability: "Python DSPy/GEPA acceleration sidecar", status: "optional", evidence: "python_backend/" },
  { capability: "OTel-traced Ralph loop", status: "optional", evidence: "scripts/ralph_otel.py" },
  { capability: "Sokoban benchmark scaffold", status: "optional", evidence: "scripts/sokoban_benchmark.py" },
];

export function readParityTableFromReadme(readmePath: string = path.join(REPO_ROOT, "README.md")): string {
  const text = fs.readFileSync(readmePath, "utf8");
  const headerIdx = text.indexOf("## Hermes Phase 1 parity");
  if (headerIdx < 0) throw new Error("Parity section header missing from README.md");
  const tail = text.slice(headerIdx);
  const nextSection = tail.slice(2).search(/\n## /);
  return nextSection >= 0 ? tail.slice(0, nextSection + 2) : tail;
}

function stripMarkdown(text: string): string {
  return text.replace(/\*\*/g, "").replace(/`/g, "");
}

export function runParityGate(): { ok: true; rowsChecked: number } {
  const section = stripMarkdown(readParityTableFromReadme());
  const missing: string[] = [];
  for (const row of EXPECTED_PARITY_ROWS) {
    if (!section.includes(stripMarkdown(row.capability))) missing.push(row.capability);
  }
  if (missing.length > 0) {
    throw new Error(`README parity table is missing rows: ${missing.join("; ")}`);
  }
  const regressedSurplus: string[] = [];
  if (section.includes("typescript-proxy")) regressedSurplus.push("'typescript-proxy' framing leaked back into README");
  if (section.includes("not a full Hermes reproduction")) regressedSurplus.push("framing regressed to 'not a reproduction'");
  if (regressedSurplus.length > 0) {
    throw new Error(`SURPLUS regression detected: ${regressedSurplus.join("; ")}`);
  }
  return { ok: true, rowsChecked: EXPECTED_PARITY_ROWS.length };
}

if (import.meta.url === `file://${process.argv[1]?.replace(/\\/g, "/")}`) {
  const r = runParityGate();
  console.log(`parity: ${r.rowsChecked} rows verified`);
}

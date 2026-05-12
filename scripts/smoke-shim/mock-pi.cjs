#!/usr/bin/env node
/* eslint-disable */
// Mock pi binary used by scripts/smoke-test.ts. Reads stdin, classifies the
// caller by --system-prompt prefix, and writes a deterministic JSON response
// drawn from tests/fixtures/smoke-skill/mock-llm-responses.json. State (call
// counts per system-prompt class) is kept on disk so successive invocations
// advance through the canned response array.

const fs = require("node:fs");
const path = require("node:path");

const FIXTURE_PATH = process.env.SMOKE_FIXTURE_PATH;
const STATE_DIR = process.env.SMOKE_STATE_DIR;
const MOCK_MODE = process.env.SMOKE_MOCK_MODE || "default";

if (!FIXTURE_PATH || !STATE_DIR) {
  process.stderr.write("mock-pi: SMOKE_FIXTURE_PATH and SMOKE_STATE_DIR must be set.\n");
  process.exit(2);
}

function parseSystemPrompt(argv) {
  for (let i = 0; i < argv.length - 1; i += 1) {
    if (argv[i] === "--system-prompt") return argv[i + 1];
  }
  return "";
}

function classify(systemPrompt) {
  const s = systemPrompt.slice(0, 200);
  if (/^You create compact evaluation datasets/.test(s)) return "dataset";
  if (/^You are a strict evaluator/.test(s)) return "judge";
  if (/^You improve instruction artifacts/.test(s)) return "candidate";
  if (/^You compare two versions/.test(s)) return "drift";
  // Anything else is the pi-executor calling pi with the candidate body as system prompt.
  return "executor";
}

function readStdinSync() {
  try {
    return fs.readFileSync(0, "utf8");
  } catch {
    return "";
  }
}

function loadCounters() {
  const file = path.join(STATE_DIR, "counters.json");
  try {
    return JSON.parse(fs.readFileSync(file, "utf8"));
  } catch {
    return { dataset: 0, judge: 0, candidate: 0, drift: 0, executor: 0 };
  }
}

function saveCounters(counters) {
  fs.mkdirSync(STATE_DIR, { recursive: true });
  fs.writeFileSync(path.join(STATE_DIR, "counters.json"), JSON.stringify(counters), "utf8");
}

function pickResponse(table, kind, index, mode) {
  const modeKey = table[kind] && table[kind][mode] !== undefined ? mode : "default";
  const bucket = table[kind] && table[kind][modeKey];
  if (Array.isArray(bucket)) {
    if (bucket.length === 0) throw new Error(`mock-pi: empty response array for ${kind}/${modeKey}`);
    return bucket[Math.min(index, bucket.length - 1)];
  }
  return bucket;
}

function emit(payload) {
  process.stdout.write(typeof payload === "string" ? payload : JSON.stringify(payload));
}

function main() {
  const fixture = JSON.parse(fs.readFileSync(FIXTURE_PATH, "utf8"));
  const systemPrompt = parseSystemPrompt(process.argv);
  const kind = classify(systemPrompt);
  // Drain stdin so the caller's child.stdin.end() resolves cleanly even when we ignore it.
  readStdinSync();
  const counters = loadCounters();
  const index = counters[kind] || 0;
  counters[kind] = index + 1;
  saveCounters(counters);

  if (kind === "executor") {
    const template = pickResponse(fixture, "executor", index, MOCK_MODE) || "MOCK_EXECUTOR_STDOUT";
    emit(typeof template === "string" ? template.replace(/\{\{counter\}\}/g, String(index)) : template);
    process.exit(0);
  }

  const response = pickResponse(fixture, kind, index, MOCK_MODE);
  if (response === undefined) {
    process.stderr.write(`mock-pi: no canned response for kind=${kind} mode=${MOCK_MODE} index=${index}\n`);
    process.exit(3);
  }
  emit(response);
  process.exit(0);
}

main();

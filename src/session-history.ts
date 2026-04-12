import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import type { SessionSnippet } from "./types.js";

interface SessionHeader {
  cwd?: string;
}

interface SessionEntry {
  type?: string;
  message?: {
    role?: string;
    content?: unknown;
  };
}

function getAgentDir(): string {
  return process.env.PI_CODING_AGENT_DIR
    ? path.resolve(process.env.PI_CODING_AGENT_DIR)
    : path.join(os.homedir(), ".pi", "agent");
}

function getSessionsDir(): string {
  return path.join(getAgentDir(), "sessions");
}

function collectJsonlFiles(root: string, acc: string[] = []): string[] {
  let entries: fs.Dirent[] = [];
  try {
    entries = fs.readdirSync(root, { withFileTypes: true });
  } catch {
    return acc;
  }

  for (const entry of entries) {
    const fullPath = path.join(root, entry.name);
    if (entry.isDirectory()) {
      collectJsonlFiles(fullPath, acc);
      continue;
    }
    if (entry.isFile() && entry.name.endsWith(".jsonl")) {
      acc.push(fullPath);
    }
  }
  return acc;
}

function extractText(content: unknown): string {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  return content
    .map((block) => {
      if (!block || typeof block !== "object") return "";
      const maybeText = (block as { type?: string; text?: string }).type === "text"
        ? (block as { text?: string }).text
        : undefined;
      return typeof maybeText === "string" ? maybeText : "";
    })
    .filter(Boolean)
    .join("\n")
    .trim();
}

function normalizeKeyword(word: string): string {
  return word.toLowerCase().replace(/[^a-z0-9_-]/g, "").trim();
}

export function buildKeywordSet(targetName: string, objective: string, artifactBody: string, sessionQuery?: string): string[] {
  const base = new Set<string>();
  const addWords = (text: string, minLength: number) => {
    for (const raw of text.split(/\s+/g)) {
      const normalized = normalizeKeyword(raw);
      if (normalized.length >= minLength) base.add(normalized);
    }
  };

  addWords(targetName.replace(/[-_.]/g, " "), 3);
  addWords(objective, 4);
  addWords(sessionQuery ?? "", 3);

  const headingMatch = artifactBody.match(/^#\s+(.+)$/m);
  if (headingMatch?.[1]) addWords(headingMatch[1], 3);

  const firstChunk = artifactBody.slice(0, 900);
  addWords(firstChunk, 5);

  return Array.from(base).slice(0, 24);
}

function scoreTextAgainstKeywords(text: string, keywords: string[]): number {
  if (keywords.length === 0) return 0;
  const lower = text.toLowerCase();
  let score = 0;
  for (const keyword of keywords) {
    if (keyword && lower.includes(keyword)) score += 1;
  }
  return score;
}

function parseSessionPairs(sessionFile: string): Array<{ userText: string; assistantText: string }> {
  let raw = "";
  try {
    raw = fs.readFileSync(sessionFile, "utf8");
  } catch {
    return [];
  }
  const lines = raw.split(/\r?\n/g).filter(Boolean);
  if (lines.length < 2) return [];

  const entries: SessionEntry[] = [];
  for (const line of lines.slice(1)) {
    try {
      entries.push(JSON.parse(line) as SessionEntry);
    } catch {
      // ignore malformed lines
    }
  }

  const pairs: Array<{ userText: string; assistantText: string }> = [];
  for (let i = 0; i < entries.length; i += 1) {
    const entry = entries[i];
    if (entry.type !== "message" || entry.message?.role !== "user") continue;
    const userText = extractText(entry.message.content).trim();
    if (!userText) continue;

    let assistantText = "";
    for (let j = i + 1; j < entries.length; j += 1) {
      const next = entries[j];
      if (next.type !== "message") continue;
      if (next.message?.role === "assistant") {
        assistantText = extractText(next.message.content).trim();
        if (assistantText) break;
      }
      if (next.message?.role === "user") break;
    }

    pairs.push({ userText, assistantText });
  }

  return pairs;
}

export function mineSessionSnippets(options: {
  cwd: string;
  targetName: string;
  objective: string;
  artifactBody: string;
  sessionQuery?: string;
  maxSnippets?: number;
}): SessionSnippet[] {
  const sessionsDir = getSessionsDir();
  if (!fs.existsSync(sessionsDir)) return [];

  const keywords = buildKeywordSet(options.targetName, options.objective, options.artifactBody, options.sessionQuery);
  const files = collectJsonlFiles(sessionsDir)
    .map((file) => {
      let header: SessionHeader | undefined;
      try {
        const firstLine = fs.readFileSync(file, "utf8").split(/\r?\n/g, 1)[0];
        header = JSON.parse(firstLine) as SessionHeader;
      } catch {
        header = undefined;
      }
      let stat: fs.Stats | undefined;
      try {
        stat = fs.statSync(file);
      } catch {
        stat = undefined;
      }
      return { file, header, mtime: stat?.mtimeMs ?? 0 };
    })
    .filter((entry) => entry.header?.cwd === options.cwd)
    .sort((a, b) => b.mtime - a.mtime)
    .slice(0, 18);

  const ranked: SessionSnippet[] = [];
  for (const file of files) {
    const pairs = parseSessionPairs(file.file);
    for (const pair of pairs) {
      const combined = `${pair.userText}\n${pair.assistantText}`.trim();
      if (!combined) continue;
      const score = scoreTextAgainstKeywords(combined, keywords);
      ranked.push({
        sessionFile: file.file,
        userText: pair.userText,
        assistantText: pair.assistantText,
        score,
      });
    }
  }

  ranked.sort((a, b) => {
    if (b.score !== a.score) return b.score - a.score;
    return b.userText.length - a.userText.length;
  });

  const filtered = ranked.filter((snippet, index) => {
    if (index < 6 && snippet.score > 0) return true;
    return snippet.score > 1;
  });

  const shortlist = (filtered.length > 0 ? filtered : ranked).slice(0, options.maxSnippets ?? 6);
  return shortlist.map((snippet) => ({
    ...snippet,
    userText: snippet.userText.slice(0, 1200),
    assistantText: snippet.assistantText.slice(0, 1200),
  }));
}

from __future__ import annotations

import json
import os
import random
import re
import subprocess
import sys
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import dspy  # type: ignore
except Exception:  # pragma: no cover - handled by doctor/run path
    dspy = None


# ── Secret detection (P1) ──

SECRET_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("anthropic-key", re.compile(r"\bsk-ant-api\S{10,}\b")),
    ("openrouter-key", re.compile(r"\bsk-or-v1-\S{10,}\b")),
    ("openai-key", re.compile(r"\bsk-\S{20,}\b")),
    ("github-token", re.compile(r"\bghp_\S{10,}\b")),
    ("slack-bot-token", re.compile(r"\bxoxb-\S{10,}\b")),
    ("bearer-auth", re.compile(r"\bBearer\s+\S{20,}\b")),
    ("private-key", re.compile(r"-----BEGIN\s+(?:RSA\s+)?PRIVATE\sKEY-----")),
    ("env-anthropic", re.compile(r"\bANTHROPIC_API_KEY\b")),
    ("env-openai", re.compile(r"\bOPENAI_API_KEY\b")),
    ("env-openrouter", re.compile(r"\bOPENROUTER_API_KEY\b")),
    ("env-github", re.compile(r"\bGITHUB_TOKEN\b")),
    ("env-aws", re.compile(r"\bAWS_SECRET_ACCESS_KEY\b")),
    ("env-database", re.compile(r"\bDATABASE_URL\b")),
    ("password", re.compile(r"\bpassword\s*[=:]\s*\S{6,}\b")),
    ("secret", re.compile(r"\bsecret\s*[=:]\s*\S{6,}\b")),
    ("token", re.compile(r"\btoken\s*[=:]\s*\S{10,}\b")),
]


def scan_for_secrets(text: str) -> list[dict[str, str]]:
    found: list[dict[str, str]] = []
    for name, pattern in SECRET_PATTERNS:
        m = pattern.search(text)
        if m and m.group(0):
            found.append({"pattern": name, "match": m.group(0)[:20] + "…"})
    return found


def strip_secrets(text: str) -> str:
    for _, pattern in SECRET_PATTERNS:
        text = pattern.sub("[REDACTED]", text)
    return text


# ── Data classes ──

@dataclass
class ExecutionTrace:
    trace_id: str
    artifact_text: str
    task_input: str
    expected_behavior: str
    raw_output: str
    response_preview: str
    scores: dict[str, float]
    feedback: str
    is_failure: bool
    timestamp: str


@dataclass
class ConstraintResult:
    name: str
    passed: bool
    message: str


DATASET_SYSTEM_PROMPT = (
    "You create compact evaluation datasets for agent instructions. "
    "Return strict JSON only. No markdown fences, no prose before or after the JSON."
)
JUDGE_SYSTEM_PROMPT = (
    "You are a strict evaluator for agent instruction artifacts. "
    "Estimate how an agent following the provided artifact would likely respond to a task. "
    "Return strict JSON only. Be conservative, concrete, and terse."
)
CANDIDATE_SYSTEM_PROMPT = (
    "You improve instruction artifacts using reflective search. "
    "Return strict JSON only. Do not include markdown fences or commentary outside the JSON."
)
DRIFT_SYSTEM_PROMPT = (
    "You compare two versions of an instruction artifact and score their semantic similarity. "
    "A lower drift score means the evolved version preserves the original meaning. "
    "Return strict JSON only."
)


@dataclass
class SessionSnippet:
    session_file: str
    user_text: str
    assistant_text: str
    score: int


@dataclass
class EvalExample:
    task_input: str
    expected_behavior: str
    difficulty: str
    category: str
    source: str


@dataclass
class AggregateScore:
    correctness: float
    procedure_following: float
    conciseness: float
    confidence: float
    length_penalty: float
    composite: float


@dataclass
class ExampleEvaluation:
    example: EvalExample
    response_preview: str
    correctness: float
    procedure_following: float
    conciseness: float
    feedback: str
    confidence: float
    composite: float


@dataclass
class GoldenDataset:
    id: str
    examples: list[EvalExample]
    description: str


@dataclass
class CandidateRecord:
    name: str
    rationale: str
    candidate_body: str
    candidate_full_text: str
    warnings: list[str]
    aggregate: AggregateScore
    examples: list[ExampleEvaluation]
    holdout_aggregate: AggregateScore | None = None
    holdout_examples: list[ExampleEvaluation] = field(default_factory=list)
    execution_traces: list[ExecutionTrace] = field(default_factory=list)
    constraints: list[ConstraintResult] = field(default_factory=list)
    semantic_drift_score: float | None = None
    test_passed: bool | None = None


def _require_dspy() -> None:
    if dspy is None:
        raise RuntimeError(
            "DSPy is not installed for the Python backend. "
            "Run `pip install -e python_backend` or install `dspy>=3.0.0`."
        )


def _slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")[:48] or "artifact"


def _timestamp() -> str:
    return datetime.utcnow().isoformat(timespec="seconds").replace(":", "-").replace("T", "_")


def _trace_id() -> str:
    return uuid.uuid4().hex[:12]


def _extract_json_payload(text: str) -> Any:
    text = text.strip()
    if not text:
        raise ValueError("Empty model output")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        if start < 0:
            continue
        depth = 0
        in_string = False
        escape_next = False
        for index, ch in enumerate(text[start:], start=start):
            if escape_next:
                escape_next = False
                continue
            if ch == "\\" and in_string:
                escape_next = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    return json.loads(text[start : index + 1])
    raise ValueError(f"Could not parse JSON from model output:\n{text}")


def _clamp_score(value: Any) -> float:
    try:
        parsed = float(value)
    except Exception:
        parsed = 0.5
    return max(0.0, min(1.0, parsed))


def _split_frontmatter(full_text: str) -> tuple[str | None, str]:
    match = re.match(r"^---\r?\n([\s\S]*?)\r?\n---\r?\n?", full_text)
    if not match:
        return None, full_text.strip()
    return match.group(1).rstrip(), full_text[match.end() :].strip()


def _reassemble(frontmatter: str | None, body: str) -> str:
    body = body.rstrip()
    if not frontmatter:
        return body + "\n"
    return f"---\n{frontmatter.rstrip()}\n---\n\n{body}\n"


def _extract_placeholders(text: str) -> list[str]:
    return sorted(set(re.findall(r"{{[^}]+}}", text)))


def _detect_type(resolved_path: Path) -> str:
    normalized = resolved_path.as_posix().lower()
    if normalized.endswith("/skill.md"):
        return "skill"
    if "/.pi/prompts/" in normalized or "/.agents/prompts/" in normalized:
        return "prompt"
    if normalized.endswith("agents.md") or normalized.endswith("system.md") or normalized.endswith("append_system.md"):
        return "instructions"
    return "prompt" if normalized.endswith(".md") else "instructions"


def _resolve_target(path_value: str, cwd: str) -> dict[str, Any]:
    raw = path_value[1:] if path_value.startswith("@") else path_value
    resolved = Path(raw) if os.path.isabs(raw) else Path(cwd) / raw
    resolved = resolved.resolve()
    full_text = resolved.read_text(encoding="utf-8")
    frontmatter, body = _split_frontmatter(full_text)
    heading_match = re.search(r"^#\s+(.+)$", body, re.MULTILINE)
    return {
        "path": str(resolved),
        "name": resolved.stem,
        "type": _detect_type(resolved),
        "full_text": full_text,
        "body": body,
        "frontmatter": frontmatter,
        "original_bytes": len(full_text.encode("utf-8")),
        "placeholders": _extract_placeholders(full_text),
        "top_heading": heading_match.group(1).strip() if heading_match else None,
    }


def _max_bytes(original_bytes: int) -> int:
    return max(original_bytes + 400, int(original_bytes * 1.2 + 0.9999))


# ── Golden dataset persistence (P1) ──

def _golden_dir(cwd: str) -> Path:
    return Path(cwd) / ".pi" / "hermes-self-evolution" / "golden"


def _save_golden_dataset(
    cwd: str,
    golden: GoldenDataset,
    train: list[EvalExample],
    validation: list[EvalExample],
    holdout: list[EvalExample],
    artifact_path: str,
    artifact_name: str,
) -> None:
    gd = _golden_dir(cwd) / golden.id
    gd.mkdir(parents=True, exist_ok=True)

    for split_name, split_data in [("train", train), ("validation", validation), ("holdout", holdout)]:
        with open(gd / f"{split_name}.jsonl", "w", encoding="utf-8") as f:
            for ex in split_data:
                f.write(json.dumps(asdict(ex)) + "\n")

    manifest = {
        "id": golden.id,
        "artifactPath": artifact_path,
        "artifactName": artifact_name,
        "exampleCount": len(train) + len(validation) + len(holdout),
        "trainCount": len(train),
        "validationCount": len(validation),
        "holdoutCount": len(holdout),
        "createdAt": datetime.utcnow().isoformat(),
        "lastUsedAt": datetime.utcnow().isoformat(),
    }
    (gd / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _load_golden_dataset(cwd: str, golden_task_id: str) -> dict[str, list[EvalExample]] | None:
    gd = _golden_dir(cwd) / golden_task_id
    manifest_path = gd / "manifest.json"
    if not manifest_path.exists():
        return None

    def read_jsonl(filename: str) -> list[EvalExample]:
        examples: list[EvalExample] = []
        fpath = gd / filename
        if not fpath.exists():
            return examples
        for line in fpath.read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = json.loads(line)
                examples.append(EvalExample(
                    task_input=d.get("taskInput", d.get("task_input", "")),
                    expected_behavior=d.get("expectedBehavior", d.get("expected_behavior", "")),
                    difficulty=d.get("difficulty", "medium"),
                    category=d.get("category", "general"),
                    source=d.get("source", "synthetic"),
                ))
        return examples

    train = read_jsonl("train.jsonl")
    validation = read_jsonl("validation.jsonl")
    holdout = read_jsonl("holdout.jsonl")

    if not train and not validation and not holdout:
        return None

    # Update lastUsedAt
    try:
        m = json.loads(manifest_path.read_text(encoding="utf-8"))
        m["lastUsedAt"] = datetime.utcnow().isoformat()
        manifest_path.write_text(json.dumps(m, indent=2), encoding="utf-8")
    except Exception:
        pass

    return {"train": train, "validation": validation, "holdout": holdout}


# ── Formal constraint validation (P1) ──

def _validate_constraints(
    target: dict[str, Any],
    candidate_body: str,
    candidate_full_text: str,
    max_bytes: int,
    max_growth_ratio: float = 0.2,
) -> tuple[list[ConstraintResult], bool, list[str]]:
    results: list[ConstraintResult] = []
    warnings: list[str] = []
    normalized_body = candidate_body.strip()

    # 1. Non-empty
    if not normalized_body:
        results.append(ConstraintResult("non_empty", False, "Candidate body was empty."))
    else:
        results.append(ConstraintResult("non_empty", True, "Non-empty."))

    # 2. Size limit
    size_bytes = len(candidate_full_text.encode("utf-8"))
    if size_bytes > max_bytes:
        results.append(ConstraintResult("size_limit", False, f"Exceeded size budget ({size_bytes}/{max_bytes} bytes)."))
    else:
        results.append(ConstraintResult("size_limit", True, f"Size OK: {size_bytes}/{max_bytes} bytes."))

    # 3. Growth limit
    baseline_bytes = int(target.get("original_bytes", 0))
    growth_ratio = (size_bytes - baseline_bytes) / max(1, baseline_bytes)
    if growth_ratio > max_growth_ratio:
        results.append(ConstraintResult("growth_limit", False, f"Growth {growth_ratio:.1%} exceeds max {max_growth_ratio:.1%}."))
    else:
        results.append(ConstraintResult("growth_limit", True, f"Growth {growth_ratio:.1%} within budget."))

    # 4. Placeholder preservation
    placeholders = target.get("placeholders", [])
    missing = [t for t in placeholders if t not in candidate_full_text]
    if missing:
        results.append(ConstraintResult("placeholder_preservation", False, f"Dropped placeholders: {', '.join(missing)}"))
    elif placeholders:
        results.append(ConstraintResult("placeholder_preservation", True, f"All {len(placeholders)} placeholder(s) preserved."))
    else:
        results.append(ConstraintResult("placeholder_preservation", True, "No placeholders to check."))

    # 5. Top heading
    if target.get("top_heading") and not re.search(r"^#\s+.+$", normalized_body, re.MULTILINE):
        results.append(ConstraintResult("top_heading_preservation", False, "Lost the top-level markdown heading."))
        warnings.append("Candidate lost the top-level markdown heading.")
    elif target.get("top_heading"):
        results.append(ConstraintResult("top_heading_preservation", True, "Top heading preserved."))

    # 6. Frontmatter
    if target.get("frontmatter"):
        candidate_fm = _split_frontmatter(candidate_full_text)[0]
        if candidate_fm != target["frontmatter"]:
            results.append(ConstraintResult("frontmatter_preservation", False, "Frontmatter was modified."))
        else:
            results.append(ConstraintResult("frontmatter_preservation", True, "Frontmatter preserved verbatim."))

    valid = all(r.passed for r in results)
    return results, valid, warnings


# ── Semantic drift detection (P1) ──

def _compute_semantic_drift(
    model: str,
    original_body: str,
    evolved_body: str,
    objective: str,
) -> tuple[float, str]:
    _require_dspy()
    lm = dspy.LM(model)

    class DriftSignature(dspy.Signature):
        original_text: str = dspy.InputField(desc="Original artifact body")
        evolved_text: str = dspy.InputField(desc="Evolved artifact body")
        objective: str = dspy.InputField(desc="Improvement objective")
        drift_score: float = dspy.OutputField(desc="Semantic drift score 0.0-1.0 (0=identical meaning, 1=completely different purpose)")
        feedback: str = dspy.OutputField(desc="Explanation of what changed semantically")

    judge = dspy.ChainOfThought(DriftSignature)
    with dspy.context(lm=lm):
        try:
            result = judge(
                original_text=original_body[:3000],
                evolved_text=evolved_body[:3000],
                objective=objective,
            )
            score = _clamp_score(getattr(result, "drift_score", 0.2))
            feedback = str(getattr(result, "feedback", "")).strip()
            return score, feedback
        except Exception:
            return 0.2, "Drift detection failed; assuming moderate drift."


# ── Test command gate (P2) ──

def _run_test_command(
    test_command: str,
    cwd: str,
    timeout_ms: int = 60_000,
) -> tuple[bool, int, str]:
    try:
        shell_cmd = [os.environ.get("ComSpec", "cmd.exe"), "/d", "/s", "/c", test_command] if os.name == "nt" else ["sh", "-c", test_command]
        proc = subprocess.run(
            shell_cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout_ms / 1000,
        )
        return proc.returncode == 0, proc.returncode, proc.stderr or proc.stdout
    except subprocess.TimeoutExpired:
        return False, 124, f"Timed out after {timeout_ms}ms"
    except Exception as exc:
        return False, -1, str(exc)


# ── Agent dir helpers ──

def _get_agent_dir() -> Path:
    env = os.getenv("PI_CODING_AGENT_DIR")
    return Path(env).resolve() if env else (Path.home() / ".pi" / "agent")


def _build_keyword_set(target_name: str, objective: str, artifact_body: str, session_query: str | None) -> list[str]:
    keywords: set[str] = set()

    def add_words(text: str, min_len: int) -> None:
        for raw in re.split(r"\s+", text):
            normalized = re.sub(r"[^a-z0-9_-]", "", raw.lower()).strip()
            if len(normalized) >= min_len:
                keywords.add(normalized)

    add_words(target_name.replace("-", " ").replace("_", " "), 3)
    add_words(objective, 4)
    add_words(session_query or "", 3)

    heading = re.search(r"^#\s+(.+)$", artifact_body, re.MULTILINE)
    if heading:
        add_words(heading.group(1), 3)
    add_words(artifact_body[:900], 5)
    return list(sorted(keywords))[:24]


def _extract_text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(str(block.get("text", "")))
    return "\n".join(p for p in parts if p).strip()


def _score_text(text: str, keywords: list[str]) -> int:
    lower = text.lower()
    return sum(1 for keyword in keywords if keyword and keyword in lower)


def _collect_session_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(root.rglob("*.jsonl"), key=lambda item: item.stat().st_mtime, reverse=True)


def _parse_session_pairs(session_file: Path) -> list[tuple[str, str]]:
    try:
        lines = [line for line in session_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    except Exception:
        return []
    if len(lines) < 2:
        return []

    entries: list[dict[str, Any]] = []
    for line in lines[1:]:
        try:
            entries.append(json.loads(line))
        except Exception:
            continue

    pairs: list[tuple[str, str]] = []
    for index, entry in enumerate(entries):
        message = entry.get("message") if entry.get("type") == "message" else None
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        user_text = _extract_text_content(message.get("content"))
        if not user_text:
            continue
        assistant_text = ""
        for next_entry in entries[index + 1 :]:
            next_message = next_entry.get("message") if next_entry.get("type") == "message" else None
            if not isinstance(next_message, dict):
                continue
            role = next_message.get("role")
            if role == "assistant":
                assistant_text = _extract_text_content(next_message.get("content"))
                if assistant_text:
                    break
            if role == "user":
                break
        pairs.append((user_text, assistant_text))
    return pairs


def _mine_session_snippets(cwd: str, target_name: str, objective: str, artifact_body: str, session_query: str | None, max_snippets: int = 6) -> list[SessionSnippet]:
    sessions_dir = _get_agent_dir() / "sessions"
    keywords = _build_keyword_set(target_name, objective, artifact_body, session_query)
    ranked: list[SessionSnippet] = []

    for session_file in _collect_session_files(sessions_dir)[:18]:
        try:
            header = json.loads(session_file.read_text(encoding="utf-8").splitlines()[0])
        except Exception:
            continue
        if header.get("cwd") != cwd:
            continue
        for user_text, assistant_text in _parse_session_pairs(session_file):
            combined = (user_text + "\n" + assistant_text).strip()
            if not combined:
                continue
            score = _score_text(combined, keywords)
            ranked.append(
                SessionSnippet(
                    session_file=str(session_file),
                    user_text=user_text[:1200],
                    assistant_text=assistant_text[:1200],
                    score=score,
                )
            )

    ranked.sort(key=lambda item: (item.score, len(item.user_text)), reverse=True)
    filtered = [item for idx, item in enumerate(ranked) if (idx < 6 and item.score > 0) or item.score > 1]
    return (filtered or ranked)[:max_snippets]


def _configure_model(model: str):
    _require_dspy()
    lm = dspy.LM(model)
    dspy.configure(lm=lm)
    return lm


class GenerateDatasetSignature(dspy.Signature):
    artifact_type: str = dspy.InputField(desc="Artifact type: skill, prompt, or instructions")
    artifact_path: str = dspy.InputField(desc="Path to the artifact")
    objective: str = dspy.InputField(desc="What to improve")
    example_count: int = dspy.InputField(desc="How many examples to produce")
    artifact_text: str = dspy.InputField(desc="Full artifact text")
    session_snippets: str = dspy.InputField(desc="Recent session snippets that may hint at realistic tasks")
    examples_json: str = dspy.OutputField(desc='Strict JSON: {"examples":[{"taskInput":...,"expectedBehavior":...,"difficulty":...,"category":...,"source":...}]}')


class JudgeSignature(dspy.Signature):
    artifact_type: str = dspy.InputField(desc="Artifact type")
    objective: str = dspy.InputField(desc="Improvement objective")
    artifact_text: str = dspy.InputField(desc="Artifact text")
    task_input: str = dspy.InputField(desc="The user's task")
    expected_behavior: str = dspy.InputField(desc="Rubric describing good behavior")
    difficulty: str = dspy.InputField(desc="Difficulty label")
    category: str = dspy.InputField(desc="Category label")
    result_json: str = dspy.OutputField(desc='Strict JSON: {"responsePreview":...,"correctness":0-1,"procedureFollowing":0-1,"conciseness":0-1,"feedback":...,"confidence":0-1}')


class CandidateBodySignature(dspy.Signature):
    artifact_type: str = dspy.InputField(desc="Artifact type")
    objective: str = dspy.InputField(desc="Improvement objective")
    artifact_body: str = dspy.InputField(desc="Artifact body only")
    preserved_tokens: str = dspy.InputField(desc="Comma-separated placeholders that must survive unchanged")
    top_heading: str = dspy.InputField(desc="Top heading to preserve conceptually")
    weak_points: str = dspy.InputField(desc="Observed weaknesses from baseline evaluation")
    failure_traces: str = dspy.InputField(desc="Execution traces showing why the baseline fails on specific tasks")
    style_hint: str = dspy.InputField(desc="A concrete revision strategy to emphasize")
    max_bytes: int = dspy.InputField(desc="Maximum allowed bytes after reassembly")
    candidate_body: str = dspy.OutputField(desc="The full revised BODY only")
    rationale: str = dspy.OutputField(desc="One short paragraph explaining why this revision helps")


class CandidateBodyModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generator = dspy.ChainOfThought(CandidateBodySignature)

    def forward(
        self,
        artifact_type: str,
        objective: str,
        artifact_body: str,
        preserved_tokens: str,
        top_heading: str,
        weak_points: str,
        failure_traces: str,
        style_hint: str,
        max_bytes: int,
    ):
        return self.generator(
            artifact_type=artifact_type,
            objective=objective,
            artifact_body=artifact_body,
            preserved_tokens=preserved_tokens,
            top_heading=top_heading,
            weak_points=weak_points,
            failure_traces=failure_traces,
            style_hint=style_hint,
            max_bytes=max_bytes,
        )


def _normalize_examples(payload: Any, eval_source: str) -> list[EvalExample]:
    root = payload if isinstance(payload, list) else payload.get("examples", []) if isinstance(payload, dict) else []
    examples: list[EvalExample] = []
    for item in root:
        if not isinstance(item, dict):
            continue
        task_input = str(item.get("taskInput") or item.get("task_input") or "").strip()
        expected_behavior = str(item.get("expectedBehavior") or item.get("expected_behavior") or "").strip()
        if not task_input or not expected_behavior:
            continue
        difficulty = str(item.get("difficulty") or "medium").lower().strip()
        if difficulty not in {"easy", "medium", "hard"}:
            difficulty = "medium"
        category = str(item.get("category") or "general").strip() or "general"
        source = str(item.get("source") or ("session" if eval_source == "session" else "synthetic")).strip().lower()
        if source not in {"synthetic", "session"}:
            source = "synthetic"
        # P1: Strip secrets
        task_input = strip_secrets(task_input)
        expected_behavior = strip_secrets(expected_behavior)
        examples.append(
            EvalExample(
                task_input=task_input[:1800],
                expected_behavior=expected_behavior[:1800],
                difficulty=difficulty,
                category=category,
                source=source,
            )
        )
    return examples


def _split_examples(examples: list[EvalExample]) -> tuple[list[EvalExample], list[EvalExample], list[EvalExample]]:
    shuffled = list(examples)
    random.shuffle(shuffled)
    train_count = max(3, int(len(shuffled) * 0.5 + 0.9999))
    validation_count = max(1, int(len(shuffled) * 0.2 + 0.9999))
    train = shuffled[:train_count]
    validation = shuffled[train_count : train_count + validation_count]
    holdout = shuffled[train_count + validation_count :]
    if not holdout and len(train) > 2:
        holdout = [train.pop()]
    if not validation and len(train) > 2:
        validation = [train.pop()]
    return train, validation, holdout


def _weak_points_text(example_evaluations: list[ExampleEvaluation]) -> str:
    weakest = sorted(example_evaluations, key=lambda item: item.composite)[:3]
    if not weakest:
        return "No weakness summary available."
    parts: list[str] = []
    for idx, item in enumerate(weakest, start=1):
        parts.append(
            f"{idx}. Task: {item.example.task_input}\n"
            f"   Rubric: {item.example.expected_behavior}\n"
            f"   Scores: correctness={item.correctness:.2f}, procedure={item.procedure_following:.2f}, conciseness={item.conciseness:.2f}\n"
            f"   Feedback: {item.feedback}"
        )
    return "\n\n".join(parts)


def _failure_traces_text(traces: list[ExecutionTrace]) -> str:
    failures = [t for t in traces if t.is_failure]
    if not failures:
        return "No failure traces."
    parts: list[str] = []
    for idx, t in enumerate(failures[:5], start=1):
        parts.append(
            f"{idx}. [trace={t.trace_id}] composite={t.scores.get('composite', 0):.2f}\n"
            f"   Task: {t.task_input}\n"
            f"   Expected: {t.expected_behavior}\n"
            f"   Feedback: {t.feedback}"
        )
    return "\n\n".join(parts)


def _evaluate_artifact_with_traces(
    target: dict[str, Any],
    artifact_text: str,
    objective: str,
    examples: list[EvalExample],
    max_bytes: int,
    model: str,
) -> tuple[AggregateScore, list[ExampleEvaluation], list[ExecutionTrace]]:
    _require_dspy()
    _configure_model(model)
    judge = dspy.ChainOfThought(JudgeSignature)
    evaluations: list[ExampleEvaluation] = []
    traces: list[ExecutionTrace] = []

    for example in examples:
        prompt_result = judge(
            artifact_type=target["type"],
            objective=objective,
            artifact_text=artifact_text.strip(),
            task_input=example.task_input,
            expected_behavior=example.expected_behavior,
            difficulty=example.difficulty,
            category=example.category,
        )
        raw_json = str(prompt_result.result_json)
        judged = _extract_json_payload(raw_json)
        response_preview = str(judged.get("responsePreview") or judged.get("response_preview") or "").strip()
        correctness = _clamp_score(judged.get("correctness"))
        procedure = _clamp_score(judged.get("procedureFollowing") or judged.get("procedure_following"))
        conciseness = _clamp_score(judged.get("conciseness"))
        confidence = _clamp_score(judged.get("confidence"))
        feedback = str(judged.get("feedback") or "").strip()
        composite = 0.5 * correctness + 0.3 * procedure + 0.2 * conciseness

        example_eval = ExampleEvaluation(
            example=example,
            response_preview=response_preview,
            correctness=correctness,
            procedure_following=procedure,
            conciseness=conciseness,
            feedback=feedback,
            confidence=confidence,
            composite=composite,
        )
        evaluations.append(example_eval)

        # P0: Capture execution trace
        trace = ExecutionTrace(
            trace_id=_trace_id(),
            artifact_text=artifact_text[:2000],
            task_input=example.task_input,
            expected_behavior=example.expected_behavior,
            raw_output=raw_json[:2000],
            response_preview=response_preview[:500],
            scores={"correctness": correctness, "procedure_following": procedure, "conciseness": conciseness, "composite": composite},
            feedback=feedback,
            is_failure=composite < 0.5,
            timestamp=datetime.utcnow().isoformat(),
        )
        traces.append(trace)

    count = max(1, len(evaluations))
    size_ratio = len(artifact_text.encode("utf-8")) / max(1, max_bytes)
    length_penalty = min(0.3, (size_ratio - 0.9) * 3) if size_ratio > 0.9 else 0.0
    aggregate = AggregateScore(
        correctness=sum(item.correctness for item in evaluations) / count,
        procedure_following=sum(item.procedure_following for item in evaluations) / count,
        conciseness=sum(item.conciseness for item in evaluations) / count,
        confidence=sum(item.confidence for item in evaluations) / count,
        length_penalty=length_penalty,
        composite=max(0.0, sum(item.composite for item in evaluations) / count - length_penalty),
    )
    return aggregate, evaluations, traces


def _style_hints(weak_points: str) -> list[str]:
    hints = [
        "Tighten trigger conditions and when-to-use guidance.",
        "Improve sequencing, checkpoints, and step ordering.",
        "Make examples, pitfalls, and output constraints more concrete while staying concise.",
    ]
    lowered = weak_points.lower()
    if "concise" in lowered or "verbose" in lowered:
        hints.append("Shorten instructions and remove repetition without losing important constraints.")
    if "format" in lowered or "output" in lowered:
        hints.append("Clarify the expected output format and make it easier for the agent to follow reliably.")
    return hints[:4]


def _optimize_candidates(
    target: dict[str, Any],
    objective: str,
    train_examples: list[EvalExample],
    baseline_train_examples: list[ExampleEvaluation],
    baseline_traces: list[ExecutionTrace],
    max_bytes: int,
    model: str,
    candidate_count: int,
) -> tuple[str, list[dict[str, str]]]:
    _require_dspy()
    _configure_model(model)
    module = CandidateBodyModule()
    weak_points = _weak_points_text(baseline_train_examples)
    failure_traces = _failure_traces_text(baseline_traces)
    hints = _style_hints(weak_points)
    example_inputs = []
    for hint in hints:
        example_inputs.append(
            dspy.Example(
                artifact_type=target["type"],
                objective=objective,
                artifact_body=target["body"].strip(),
                preserved_tokens=", ".join(target.get("placeholders", [])) or "none",
                top_heading=target.get("top_heading") or "none",
                weak_points=weak_points,
                failure_traces=failure_traces,
                style_hint=hint,
                max_bytes=max_bytes,
            ).with_inputs(
                "artifact_type",
                "objective",
                "artifact_body",
                "preserved_tokens",
                "top_heading",
                "weak_points",
                "failure_traces",
                "style_hint",
                "max_bytes",
            )
        )

    eval_subset = train_examples[: min(3, len(train_examples))]

    def metric(example, prediction, trace=None):
        candidate_body = str(getattr(prediction, "candidate_body", "")).strip()
        constraints, valid, _ = _validate_constraints(target, candidate_body, _reassemble(target.get("frontmatter"), candidate_body), max_bytes)
        if not valid:
            return 0.0
        aggregate, _, _ = _evaluate_artifact_with_traces(target, _reassemble(target.get("frontmatter"), candidate_body), objective, eval_subset, max_bytes, model)
        return aggregate.composite

    optimizer_used = "chain-of-thought"
    optimized = module
    try:
        if hasattr(dspy, "GEPA"):
            optimizer = dspy.GEPA(metric=metric, max_steps=max(2, int(os.getenv("PI_HERMES_EVOLVE_GEPA_STEPS", "4"))))
            optimized = optimizer.compile(module, trainset=example_inputs, valset=example_inputs[:1])
            optimizer_used = "gepa"
        elif hasattr(dspy, "MIPROv2"):
            optimizer = dspy.MIPROv2(metric=metric, auto="light")
            optimized = optimizer.compile(module, trainset=example_inputs)
            optimizer_used = "miprov2"
    except Exception:
        optimizer_used = "chain-of-thought"
        optimized = module

    raw_candidates: list[dict[str, str]] = []
    for index, hint in enumerate(hints[: max(1, candidate_count)]):
        prediction = optimized(
            artifact_type=target["type"],
            objective=objective,
            artifact_body=target["body"].strip(),
            preserved_tokens=", ".join(target.get("placeholders", [])) or "none",
            top_heading=target.get("top_heading") or "none",
            weak_points=weak_points,
            failure_traces=failure_traces,
            style_hint=hint,
            max_bytes=max_bytes,
        )
        raw_candidates.append(
            {
                "name": _slugify(f"{optimizer_used}-{index + 1}"),
                "rationale": str(getattr(prediction, "rationale", "")).strip() or hint,
                "candidate_body": str(getattr(prediction, "candidate_body", "")).strip(),
            }
        )
    return optimizer_used, raw_candidates


def _write_text(path_value: Path, content: str) -> None:
    path_value.parent.mkdir(parents=True, exist_ok=True)
    path_value.write_text(content, encoding="utf-8")


def _write_json(path_value: Path, payload: Any) -> None:
    path_value.parent.mkdir(parents=True, exist_ok=True)
    path_value.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_backend(payload: dict[str, Any]) -> dict[str, Any]:
    _require_dspy()
    cwd = str(payload["cwd"])
    target = _resolve_target(str(payload["targetPath"]), cwd)
    objective = str(payload.get("objective") or "Improve trigger clarity, execution quality, and practical usefulness while preserving the artifact's intent.")
    eval_source = str(payload.get("evalSource") or "mixed")
    model = str(payload.get("model") or os.getenv("PI_HERMES_EVOLVE_MODEL") or "openai/gpt-4.1-mini")
    max_examples = max(4, min(12, int(payload.get("maxExamples") or 8)))
    candidate_count = max(1, min(5, int(payload.get("candidateCount") or 3)))
    session_query = payload.get("sessionQuery")
    test_command = payload.get("testCommand")
    test_timeout_ms = int(payload.get("testTimeout", 60)) * 1000
    create_pr = bool(payload.get("createPR", False))
    persist_golden = payload.get("persistGolden", True)

    snippets = [] if eval_source == "synthetic" else _mine_session_snippets(cwd, target["name"], objective, target["body"], session_query)
    snippet_text = (
        "- none found"
        if not snippets
        else "\n".join(
            f"- Snippet {idx + 1} (score {snippet.score})\n  User: {snippet.user_text}\n  Assistant: {snippet.assistant_text or '<none>'}"
            for idx, snippet in enumerate(snippets)
        )
    )

    _configure_model(model)
    dataset_generator = dspy.ChainOfThought(GenerateDatasetSignature)
    dataset_result = dataset_generator(
        artifact_type=target["type"],
        artifact_path=target["path"],
        objective=objective,
        example_count=max_examples,
        artifact_text=target["full_text"].strip(),
        session_snippets=snippet_text,
    )
    examples = _normalize_examples(_extract_json_payload(str(dataset_result.examples_json)), eval_source)
    if len(examples) < 4:
        raise RuntimeError(f"Dataset generation produced only {len(examples)} usable examples.")

    train_examples, validation_examples, holdout_examples = _split_examples(examples)
    max_bytes = _max_bytes(int(target["original_bytes"]))

    # P1: Load persisted golden dataset
    golden_task_id = str(payload.get("goldenTaskId") or "").strip() or None
    used_persisted_golden = False
    if golden_task_id and persist_golden is not False:
        loaded = _load_golden_dataset(cwd, golden_task_id)
        if loaded and loaded["train"]:
            train_examples = loaded["train"]
            validation_examples = loaded["validation"]
            holdout_examples = loaded["holdout"]
            used_persisted_golden = True

    golden_dataset: GoldenDataset | None = None
    if golden_task_id and validation_examples:
        golden_dataset = GoldenDataset(
            id=golden_task_id,
            examples=validation_examples,
            description=f"Golden dataset derived from validation split for task {golden_task_id}",
        )

    # P1: Persist golden dataset
    if golden_dataset and persist_golden is not False and not used_persisted_golden:
        _save_golden_dataset(cwd, golden_dataset, train_examples, validation_examples, holdout_examples, target["path"], target["name"])

    # Evaluate baseline with traces (P0)
    baseline_train_aggregate, baseline_train_details, baseline_train_traces = _evaluate_artifact_with_traces(target, target["full_text"], objective, train_examples, max_bytes, model)
    baseline_validation_aggregate, _, _ = _evaluate_artifact_with_traces(target, target["full_text"], objective, validation_examples, max_bytes, model)
    baseline_holdout_aggregate, _, baseline_holdout_traces = _evaluate_artifact_with_traces(target, target["full_text"], objective, holdout_examples, max_bytes, model)

    all_baseline_traces = baseline_train_traces + baseline_holdout_traces

    optimizer_used, draft_candidates = _optimize_candidates(
        target=target,
        objective=objective,
        train_examples=train_examples,
        baseline_train_examples=baseline_train_details,
        baseline_traces=all_baseline_traces,
        max_bytes=max_bytes,
        model=model,
        candidate_count=candidate_count,
    )

    candidates: list[CandidateRecord] = []
    seen_bodies: set[str] = set()
    for index, draft in enumerate(draft_candidates, start=1):
        body = draft["candidate_body"].strip()
        if not body or body in seen_bodies:
            continue
        seen_bodies.add(body)

        # P1: Formal constraint validation
        constraints, valid, constraint_warnings = _validate_constraints(target, body, _reassemble(target.get("frontmatter"), body), max_bytes)
        if not valid:
            continue

        full_text = _reassemble(target.get("frontmatter"), body)
        aggregate, example_details, candidate_traces = _evaluate_artifact_with_traces(target, full_text, objective, validation_examples, max_bytes, model)

        # P1: Semantic drift detection
        drift_score = None
        if True:  # Always check drift
            drift_score, drift_feedback = _compute_semantic_drift(model, target["body"], body, objective)
            drift_constraint = ConstraintResult(
                "semantic_drift",
                drift_score <= 0.4,
                f"Drift score: {drift_score:.3f} (max: 0.4). {drift_feedback}",
            )
            constraints.append(drift_constraint)
            if not drift_constraint.passed:
                constraint_warnings.append(f"Semantic drift too high: {drift_score:.3f}")
                continue

        # P2: Test command gate
        test_passed = None
        if test_command:
            # Write candidate temporarily
            target_path = Path(target["path"])
            original_content = target_path.read_text(encoding="utf-8")
            try:
                _write_text(target_path, full_text)
                test_passed, exit_code, test_stderr = _run_test_command(test_command, cwd, test_timeout_ms)
            finally:
                _write_text(target_path, original_content)
            if not test_passed:
                constraint_warnings.append(f"Test command failed (exit {exit_code})")
                continue

        candidates.append(
            CandidateRecord(
                name=draft["name"] or f"candidate-{index}",
                rationale=draft["rationale"],
                candidate_body=body,
                candidate_full_text=full_text,
                warnings=constraint_warnings,
                aggregate=aggregate,
                examples=example_details,
                execution_traces=candidate_traces,
                constraints=constraints,
                semantic_drift_score=drift_score,
                test_passed=test_passed,
            )
        )

    if not candidates:
        raise RuntimeError("All Python backend candidates were rejected by constraints.")

    candidates.sort(key=lambda item: item.aggregate.composite, reverse=True)
    best = candidates[0]
    best_holdout_aggregate, best_holdout_details, best_holdout_traces = _evaluate_artifact_with_traces(
        target,
        best.candidate_full_text,
        objective,
        holdout_examples,
        max_bytes,
        model,
    )
    best.holdout_aggregate = best_holdout_aggregate
    best.holdout_examples = best_holdout_details
    best.execution_traces.extend(best_holdout_traces)
    improvement = best_holdout_aggregate.composite - baseline_holdout_aggregate.composite

    run_dir = Path(cwd) / ".pi" / "hermes-self-evolution" / "runs" / f"{_timestamp()}-{_slugify(target['name'])}"
    original_path = run_dir / "original.md"
    best_path = run_dir / "best-candidate.md"
    report_path = run_dir / "report.md"
    dataset_path = run_dir / "dataset.json"
    manifest_path = run_dir / "manifest.json"
    traces_dir = run_dir / "traces"

    _write_text(original_path, target["full_text"])
    _write_text(best_path, best.candidate_full_text)
    _write_json(
        dataset_path,
        {
            "train": [asdict(example) for example in train_examples],
            "validation": [asdict(example) for example in validation_examples],
            "holdout": [asdict(example) for example in holdout_examples],
            "golden": {"id": golden_dataset.id, "description": golden_dataset.description, "exampleCount": len(golden_dataset.examples)} if golden_dataset else None,
            "session_snippets": [asdict(snippet) for snippet in snippets],
        },
    )

    # P0: Write execution traces
    all_traces = [
        *[asdict(t) | {"phase": "baseline"} for t in all_baseline_traces],
        *[
            item
            for c in candidates
            for item in [asdict(t) | {"phase": f"candidate/{c.name}"} for t in c.execution_traces]
        ],
    ]
    _write_json(traces_dir / "all-traces.json", all_traces)
    failure_traces = [t for t in all_traces if t.get("is_failure")]
    if failure_traces:
        _write_json(traces_dir / "failure-traces.json", failure_traces)

    total_traces = len(all_traces)
    total_failures = len(failure_traces)

    candidate_payloads = []
    for candidate in candidates:
        prefix = _slugify(candidate.name)
        _write_text(run_dir / "candidates" / f"{prefix}.md", candidate.candidate_full_text)
        _write_json(
            run_dir / "candidates" / f"{prefix}.json",
            {
                "rationale": candidate.rationale,
                "warnings": candidate.warnings,
                "validationAggregate": asdict(candidate.aggregate),
                "holdoutAggregate": asdict(candidate.holdout_aggregate) if candidate.holdout_aggregate else None,
                "constraints": [{"name": c.name, "passed": c.passed, "message": c.message} for c in candidate.constraints],
                "semanticDriftScore": candidate.semantic_drift_score,
                "testPassed": candidate.test_passed,
                "executionTraces": len(candidate.execution_traces),
            },
        )
        candidate_payloads.append(
            {
                "name": candidate.name,
                "rationale": candidate.rationale,
                "warnings": candidate.warnings,
                "validationScore": asdict(candidate.aggregate),
                "holdoutScore": asdict(candidate.holdout_aggregate) if candidate.holdout_aggregate else None,
                "semanticDriftScore": candidate.semantic_drift_score,
                "testPassed": candidate.test_passed,
                "constraintsPassed": all(c.passed for c in candidate.constraints),
            }
        )

    constraints_passed = all(c.passed for c in best.constraints) if best.constraints else True

    summary = {
        "backend": "python",
        "optimizer_used": optimizer_used,
        "runDir": str(run_dir),
        "reportPath": str(report_path),
        "targetPath": target["path"],
        "objective": objective,
        "evalSource": eval_source,
        "modelLabel": model,
        "selectionSplit": "validation",
        "confirmationSplit": "holdout",
        "trainExamples": len(train_examples),
        "validationExamples": len(validation_examples),
        "holdoutExamples": len(holdout_examples),
        "goldenTaskId": golden_task_id,
        "candidateCount": len(candidates),
        "baselineValidationScore": baseline_validation_aggregate.composite,
        "bestValidationScore": best.aggregate.composite,
        "baselineHoldoutScore": baseline_holdout_aggregate.composite,
        "bestHoldoutScore": best_holdout_aggregate.composite,
        "improvement": improvement,
        "bestCandidateName": best.name,
        "target_type": target["type"],
        "tracesCaptured": total_traces,
        "constraintsPassed": constraints_passed,
        "testGatePassed": best.test_passed,
        "semanticDriftScore": best.semantic_drift_score,
    }
    _write_json(manifest_path, {**summary, "candidates": candidate_payloads})

    # Minimal report (Python side — full report is in TS backend)
    _write_text(
        report_path,
        (
            "# Hermes-style Self-Evolution Report (Python backend)\n\n"
            f"- Selection (validation): {baseline_validation_aggregate.composite:.3f} → {best.aggregate.composite:.3f}\n"
            f"- Confirmation (holdout): {baseline_holdout_aggregate.composite:.3f} → {best_holdout_aggregate.composite:.3f}\n\n"
            "See manifest.json for full metrics.\n"
        ),
    )

    return summary


def doctor() -> dict[str, Any]:
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    dspy_available = dspy is not None
    gepa_available = bool(dspy_available and hasattr(dspy, "GEPA"))
    return {
        "ok": dspy_available,
        "python": python_version,
        "dspy": dspy_available,
        "gepa": gepa_available,
        "entrypoint": str(Path(__file__).resolve()),
    }

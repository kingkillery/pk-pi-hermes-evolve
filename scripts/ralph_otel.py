from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import shlex
import shutil
import subprocess
import sys
import textwrap
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

try:  # pragma: no cover - import surface depends on local env
    from opentelemetry import metrics, trace
    from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import ConsoleMetricExporter, PeriodicExportingMetricReader
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
except Exception:  # pragma: no cover - handled at runtime with an actionable error
    metrics = None
    trace = None
    OTLPMetricExporter = None
    OTLPSpanExporter = None
    MeterProvider = None
    ConsoleMetricExporter = None
    PeriodicExportingMetricReader = None
    Resource = None
    TracerProvider = None
    BatchSpanProcessor = None
    ConsoleSpanExporter = None


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TASK_FILE = ROOT / "scripts" / "tasks" / "hermes_parity_task.json"
DEFAULT_MODEL = os.getenv("RALPH_PI_MODEL", "zai/glm-5.1")
DEFAULT_THINKING = os.getenv("RALPH_PI_THINKING", "high")
SERVICE_NAME = "pk-pi-hermes-evolve.ralph"
SERVICE_VERSION = "0.2.1"


@dataclass
class RepoCheckSpec:
    check_id: str
    description: str
    kind: str
    base: str = "repo"
    path: str = ""
    glob: str = ""
    needles: list[str] = field(default_factory=list)
    regexes: list[str] = field(default_factory=list)
    min_matches: int = 1

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RepoCheckSpec":
        return cls(
            check_id=str(payload["check_id"]),
            description=str(payload["description"]),
            kind=str(payload["kind"]),
            base=str(payload.get("base") or "repo"),
            path=str(payload.get("path") or ""),
            glob=str(payload.get("glob") or ""),
            needles=[str(item) for item in payload.get("needles", [])],
            regexes=[str(item) for item in payload.get("regexes", [])],
            min_matches=max(1, int(payload.get("min_matches") or 1)),
        )


@dataclass
class RepoCheckResult:
    check_id: str
    description: str
    kind: str
    base: str
    passed: bool
    evidence: str


@dataclass
class TaskSpec:
    task_id: str
    title: str
    objective: str
    acceptable_end_state: str
    deliverables: list[str] = field(default_factory=list)
    validation_commands: list[str] = field(default_factory=list)
    success_signals: list[str] = field(default_factory=list)
    out_of_scope: list[str] = field(default_factory=list)
    repo_notes: list[str] = field(default_factory=list)
    repo_checks: list[RepoCheckSpec] = field(default_factory=list)

    @classmethod
    def from_file(cls, path: Path) -> "TaskSpec":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            task_id=str(payload["task_id"]),
            title=str(payload["title"]),
            objective=str(payload["objective"]),
            acceptable_end_state=str(payload["acceptable_end_state"]),
            deliverables=[str(item) for item in payload.get("deliverables", [])],
            validation_commands=[str(item) for item in payload.get("validation_commands", [])],
            success_signals=[str(item) for item in payload.get("success_signals", [])],
            out_of_scope=[str(item) for item in payload.get("out_of_scope", [])],
            repo_notes=[str(item) for item in payload.get("repo_notes", [])],
            repo_checks=[RepoCheckSpec.from_dict(item) for item in payload.get("repo_checks", []) if isinstance(item, dict)],
        )


@dataclass
class ModelResponse:
    message: str
    final_answer: str
    needs_more_steps: bool
    next_focus: str
    completed: list[str]
    remaining: list[str]
    token_usage: int
    raw_output: str


@dataclass
class ToolResult:
    name: str
    command: str
    success: bool
    exit_code: int
    stdout: str
    stderr: str
    attempts: int
    duration_ms: float


@dataclass
class JudgeResult:
    passed: bool
    reason: str
    score: float
    next_focus: str
    token_usage: int
    raw_output: str
    repo_checks: list[RepoCheckResult] = field(default_factory=list)


@dataclass
class StepRecord:
    step: int
    started_at: float
    duration_ms: float
    model: ModelResponse
    validations: list[ToolResult]
    judge: JudgeResult
    changed_files: list[str]
    diff_stat: str


class TraceLogFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if trace is None:
            record.trace_id = "-"
            record.span_id = "-"
            return True
        span = trace.get_current_span()
        context = span.get_span_context() if span else None
        if context and context.is_valid:
            record.trace_id = f"{context.trace_id:032x}"
            record.span_id = f"{context.span_id:016x}"
        else:
            record.trace_id = "-"
            record.span_id = "-"
        return True


class TelemetryContext:
    def __init__(self, tracer: Any, meter: Any, exporter_name: str):
        self.tracer = tracer
        self.meter = meter
        self.exporter_name = exporter_name
        self.runs_counter = meter.create_counter("ralph.runs")
        self.errors_counter = meter.create_counter("ralph.errors")
        self.tokens_counter = meter.create_counter("ralph.tokens")
        self.tool_calls_counter = meter.create_counter("ralph.tool_calls")
        self.run_latency = meter.create_histogram("ralph.run.latency_ms")
        self.step_latency = meter.create_histogram("ralph.step.latency_ms")
        self.tool_latency = meter.create_histogram("ralph.tool.latency_ms")


def configure_logging(verbose: bool) -> logging.Logger:
    logger = logging.getLogger("ralph_otel")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.handlers.clear()
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    handler.addFilter(TraceLogFilter())
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [trace=%(trace_id)s span=%(span_id)s] %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


class _NullCounter:
    def add(self, *_args: Any, **_kwargs: Any) -> None:
        return None


class _NullHistogram:
    def record(self, *_args: Any, **_kwargs: Any) -> None:
        return None


class _NullMeter:
    def create_counter(self, _name: str) -> _NullCounter:
        return _NullCounter()

    def create_histogram(self, _name: str) -> _NullHistogram:
        return _NullHistogram()


class _NullSpan:
    def __enter__(self) -> "_NullSpan":
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _tb: Any) -> None:
        return None

    def set_attribute(self, _name: str, _value: Any) -> None:
        return None

    def add_event(self, _name: str, _attributes: dict[str, Any] | None = None) -> None:
        return None

    def record_exception(self, _exc: BaseException) -> None:
        return None


class _NullTracer:
    def start_as_current_span(self, _name: str, attributes: dict[str, Any] | None = None) -> _NullSpan:
        return _NullSpan()


def normalize_otlp_endpoint(endpoint: str, signal: str) -> str:
    clean = endpoint.rstrip("/")
    if clean.endswith("/v1/traces") or clean.endswith("/v1/metrics"):
        clean = clean.rsplit("/v1/", 1)[0]
    return f"{clean}/v1/{signal}"


def configure_telemetry(exporter_name: str, otlp_endpoint: str | None) -> TelemetryContext:
    if exporter_name == "none":
        return TelemetryContext(_NullTracer(), _NullMeter(), exporter_name)

    if trace is None or metrics is None:
        raise RuntimeError(
            "OpenTelemetry dependencies are not installed. Install the Python backend with its telemetry dependencies first."
        )

    resource = Resource.create({
        "service.name": SERVICE_NAME,
        "service.version": SERVICE_VERSION,
    })

    tracer_provider = TracerProvider(resource=resource)
    metric_readers: list[Any] = []

    if exporter_name == "console":
        tracer_provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
        metric_readers.append(PeriodicExportingMetricReader(ConsoleMetricExporter()))
    elif exporter_name == "otlp-http":
        if not otlp_endpoint:
            raise RuntimeError("--otlp-endpoint is required when --telemetry-export otlp-http is used.")
        tracer_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=normalize_otlp_endpoint(otlp_endpoint, "traces"))))
        metric_readers.append(PeriodicExportingMetricReader(OTLPMetricExporter(endpoint=normalize_otlp_endpoint(otlp_endpoint, "metrics"))))
    else:
        raise RuntimeError(f"Unsupported telemetry exporter: {exporter_name}")

    meter_provider = MeterProvider(resource=resource, metric_readers=metric_readers)
    trace.set_tracer_provider(tracer_provider)
    metrics.set_meter_provider(meter_provider)

    tracer = trace.get_tracer(SERVICE_NAME, SERVICE_VERSION)
    meter = metrics.get_meter(SERVICE_NAME, SERVICE_VERSION)
    return TelemetryContext(tracer, meter, exporter_name)


def flush_telemetry() -> None:
    if trace is not None:
        provider = trace.get_tracer_provider()
        if hasattr(provider, "force_flush"):
            provider.force_flush()
        if hasattr(provider, "shutdown"):
            provider.shutdown()
    if metrics is not None and hasattr(metrics, "get_meter_provider"):
        provider = metrics.get_meter_provider()
        if hasattr(provider, "force_flush"):
            provider.force_flush()
        if hasattr(provider, "shutdown"):
            provider.shutdown()


def slugify(value: str) -> str:
    return "-".join(part for part in "".join(ch.lower() if ch.isalnum() else "-" for ch in value).split("-") if part)[:64] or "task"


def prompt_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def estimate_tokens(*texts: str) -> int:
    total_chars = sum(len(text) for text in texts)
    return max(1, total_chars // 4)


def trim_text(text: str, limit: int = 6000) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return f"{text[:limit]}\n...[truncated {len(text) - limit} chars]"


def build_pi_invocation(pi_command: str, args: Sequence[str]) -> list[str]:
    resolved = shutil.which(pi_command) or pi_command
    command_path = Path(resolved)
    suffix = command_path.suffix.lower()
    if suffix == ".cmd":
        base_dir = command_path.parent
        cli_js = base_dir / "node_modules" / "@mariozechner" / "pi-coding-agent" / "dist" / "cli.js"
        node_exe = base_dir / "node.exe"
        node_command = str(node_exe if node_exe.exists() else (shutil.which("node") or "node"))
        if cli_js.exists():
            return [node_command, str(cli_js), *args]
    return [resolved, *args]


def extract_json_payload(text: str) -> dict[str, Any]:
    text = text.strip()
    if not text:
        raise ValueError("Empty model output; expected JSON payload.")

    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        end = text.rfind(closer)
        if start == -1 or end == -1 or end <= start:
            continue
        candidate = text[start : end + 1]
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
        if isinstance(payload, list):
            return {"items": payload}
    raise ValueError(f"Could not parse JSON payload from output:\n{text[:500]}")


def _check_root(spec: RepoCheckSpec, repo_root: Path, run_dir: Path) -> Path:
    return run_dir if spec.base == "run" else repo_root


def _read_text_if_file(path: Path) -> str:
    try:
        if path.is_file():
            return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return ""
    except OSError:
        return ""
    return ""


def _aggregate_glob_text(root: Path, pattern: str) -> tuple[list[Path], str]:
    paths = sorted(path for path in root.glob(pattern) if path.is_file())
    chunks: list[str] = []
    for path in paths[:200]:
        text = _read_text_if_file(path)
        if text:
            chunks.append(text)
    return paths, "\n\n".join(chunks)


def evaluate_repo_checks(specs: Sequence[RepoCheckSpec], repo_root: Path, run_dir: Path) -> list[RepoCheckResult]:
    results: list[RepoCheckResult] = []
    for spec in specs:
        root = _check_root(spec, repo_root, run_dir)
        try:
            if spec.kind == "path_exists":
                target = root / spec.path
                passed = target.exists()
                evidence = f"exists: {target}" if passed else f"missing: {target}"
            elif spec.kind == "glob_min_matches":
                matches = sorted(path for path in root.glob(spec.glob) if path.is_file())
                passed = len(matches) >= spec.min_matches
                preview = ", ".join(str(path.relative_to(root)) for path in matches[:5]) or "none"
                evidence = f"matches={len(matches)} min_required={spec.min_matches} preview={preview}"
            elif spec.kind == "file_contains_all":
                target = root / spec.path
                text = _read_text_if_file(target)
                missing = [needle for needle in spec.needles if needle not in text]
                passed = target.exists() and not missing
                evidence = f"missing needles: {missing}" if missing else f"all needles present in {target}"
            elif spec.kind == "file_contains_any":
                target = root / spec.path
                text = _read_text_if_file(target)
                found = [needle for needle in spec.needles if needle in text]
                passed = target.exists() and bool(found)
                evidence = f"found needles: {found}" if found else f"no needles found in {target}"
            elif spec.kind == "glob_contains_all":
                matches, text = _aggregate_glob_text(root, spec.glob)
                missing = [needle for needle in spec.needles if needle not in text]
                passed = bool(matches) and not missing
                evidence = f"files={len(matches)} missing needles={missing}"
            elif spec.kind == "glob_contains_any":
                matches, text = _aggregate_glob_text(root, spec.glob)
                found = [needle for needle in spec.needles if needle in text]
                passed = bool(matches) and bool(found)
                evidence = f"files={len(matches)} found needles={found}"
            elif spec.kind == "file_regex_all":
                target = root / spec.path
                text = _read_text_if_file(target)
                missing = [pattern for pattern in spec.regexes if not re.search(pattern, text, re.MULTILINE)]
                passed = target.exists() and not missing
                evidence = f"missing regexes: {missing}" if missing else f"all regexes matched in {target}"
            elif spec.kind == "glob_regex_all":
                matches, text = _aggregate_glob_text(root, spec.glob)
                missing = [pattern for pattern in spec.regexes if not re.search(pattern, text, re.MULTILINE)]
                passed = bool(matches) and not missing
                evidence = f"files={len(matches)} missing regexes={missing}"
            else:
                passed = False
                evidence = f"unsupported check kind: {spec.kind}"
        except Exception as exc:
            passed = False
            evidence = f"error while evaluating check: {exc}"

        results.append(
            RepoCheckResult(
                check_id=spec.check_id,
                description=spec.description,
                kind=spec.kind,
                base=spec.base,
                passed=passed,
                evidence=evidence,
            )
        )
    return results


def run_subprocess(
    name: str,
    command: Sequence[str],
    cwd: Path,
    timeout: int,
    telemetry: TelemetryContext,
    logger: logging.Logger,
    retries: int = 1,
    env: dict[str, str] | None = None,
) -> ToolResult:
    attempts = 0
    last_stdout = ""
    last_stderr = ""
    last_exit = -1
    started = time.perf_counter()

    for attempt in range(1, retries + 1):
        attempts = attempt
        with telemetry.tracer.start_as_current_span(
            "tool",
            attributes={
                "tool.name": name,
                "tool.command": " ".join(shlex.quote(part) for part in command),
                "tool.attempt": attempt,
            },
        ) as span:
            step_started = time.perf_counter()
            try:
                proc = subprocess.run(
                    command,
                    cwd=str(cwd),
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    env={**os.environ, **(env or {})},
                )
                duration_ms = (time.perf_counter() - step_started) * 1000
                last_stdout = proc.stdout or ""
                last_stderr = proc.stderr or ""
                last_exit = int(proc.returncode)
                span.set_attribute("tool.exit_code", last_exit)
                span.set_attribute("tool.duration_ms", duration_ms)
                span.add_event("tool.stdout", {"output": trim_text(last_stdout, 1500)})
                if last_stderr.strip():
                    span.add_event("tool.stderr", {"output": trim_text(last_stderr, 1500)})
                telemetry.tool_calls_counter.add(1, {"tool.name": name})
                telemetry.tool_latency.record(duration_ms, {"tool.name": name})

                if proc.returncode == 0:
                    return ToolResult(
                        name=name,
                        command=" ".join(command),
                        success=True,
                        exit_code=last_exit,
                        stdout=last_stdout,
                        stderr=last_stderr,
                        attempts=attempts,
                        duration_ms=duration_ms,
                    )

                logger.warning("Tool %s failed on attempt %s/%s with exit code %s", name, attempt, retries, proc.returncode)
            except subprocess.TimeoutExpired as exc:
                duration_ms = (time.perf_counter() - step_started) * 1000
                last_stdout = exc.stdout or ""
                last_stderr = exc.stderr or f"Timed out after {timeout}s"
                last_exit = 124
                span.set_attribute("tool.exit_code", last_exit)
                span.set_attribute("tool.duration_ms", duration_ms)
                span.record_exception(exc)
                telemetry.tool_calls_counter.add(1, {"tool.name": name})
                telemetry.tool_latency.record(duration_ms, {"tool.name": name})
                logger.warning("Tool %s timed out on attempt %s/%s", name, attempt, retries)

    duration_ms = (time.perf_counter() - started) * 1000
    return ToolResult(
        name=name,
        command=" ".join(command),
        success=False,
        exit_code=last_exit,
        stdout=last_stdout,
        stderr=last_stderr,
        attempts=attempts,
        duration_ms=duration_ms,
    )


class PiCliModel:
    def __init__(self, pi_command: str, model: str, thinking: str, timeout: int):
        self.pi_command = pi_command
        self.model = model
        self.thinking = thinking
        self.timeout = timeout

    def infer(
        self,
        *,
        task: TaskSpec,
        repo_root: Path,
        step: int,
        max_steps: int,
        last_judge: str,
        last_summary: str,
        telemetry: TelemetryContext,
        logger: logging.Logger,
        golden_task_id: str | None,
    ) -> ModelResponse:
        system_prompt = textwrap.dedent(
            """
            You are the execution worker inside a Ralph loop for a code repository.
            Make concrete progress on the task using the repository tools available through pi.
            You may inspect files, run commands, and edit files when needed.
            Keep changes focused on the stated objective. Avoid unrelated refactors.
            After finishing your work for this step, return STRICT JSON only with exactly these keys:
            {
              "summary": "short paragraph",
              "completed": ["item", "..."],
              "remaining": ["item", "..."],
              "needs_more_steps": true,
              "next_focus": "next thing to do"
            }
            Do not wrap the JSON in markdown fences.
            """
        ).strip()

        prompt = textwrap.dedent(
            f"""
            Task ID: {task.task_id}
            Title: {task.title}
            Step: {step}/{max_steps}
            Objective:
            {task.objective}

            Acceptable end state:
            {task.acceptable_end_state}

            Deliverables:
            {chr(10).join(f"- {item}" for item in task.deliverables) or '- none provided'}

            Success signals:
            {chr(10).join(f"- {item}" for item in task.success_signals) or '- none provided'}

            Out of scope:
            {chr(10).join(f"- {item}" for item in task.out_of_scope) or '- none provided'}

            Repo notes:
            {chr(10).join(f"- {item}" for item in task.repo_notes) or '- none provided'}

            Previous step summary:
            {last_summary or 'No previous execution step has run yet.'}

            Previous judge result:
            {last_judge or 'No judge result yet.'}

            Work in the current repository. Make bounded, high-leverage changes that move the repository closer to the acceptable end state.
            If the repo already satisfies the task, avoid unnecessary edits and say so in the JSON response.
            """
        ).strip()

        token_usage = estimate_tokens(system_prompt, prompt)
        command = build_pi_invocation(self.pi_command, [
            "--print",
            "--no-session",
            "--model",
            self.model,
            "--thinking",
            self.thinking,
            "--tools",
            "read,bash,edit,write,grep,find,ls",
            "--system-prompt",
            system_prompt,
            prompt,
        ])

        with telemetry.tracer.start_as_current_span(
            "model.infer",
            attributes={
                "genai.model.name": self.model,
                "genai.model.version": "pi-cli",
                "genai.usage.total_tokens": token_usage,
                "ralph.prompt_hash": prompt_hash(prompt),
            },
        ) as span:
            if golden_task_id:
                span.add_event("model.prompt", {"plaintext": trim_text(prompt, 8000)})
            started = time.perf_counter()
            proc = subprocess.run(
                command,
                cwd=str(repo_root),
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env={**os.environ, "PI_SKIP_VERSION_CHECK": "1"},
            )
            duration_ms = (time.perf_counter() - started) * 1000
            span.set_attribute("model.duration_ms", duration_ms)
            if proc.returncode != 0:
                error = RuntimeError(f"pi model call failed: {proc.stderr or proc.stdout}")
                span.record_exception(error)
                raise error
            raw_output = proc.stdout.strip()
            payload = extract_json_payload(raw_output)
            logger.info("Model step %s completed: %s", step, payload.get("summary", "<no summary>"))
            telemetry.tokens_counter.add(token_usage, {"phase": "model.infer"})
            return ModelResponse(
                message=str(payload.get("summary") or "").strip(),
                final_answer=str(payload.get("summary") or "").strip(),
                needs_more_steps=bool(payload.get("needs_more_steps", True)),
                next_focus=str(payload.get("next_focus") or "").strip(),
                completed=[str(item) for item in payload.get("completed", []) if str(item).strip()],
                remaining=[str(item) for item in payload.get("remaining", []) if str(item).strip()],
                token_usage=token_usage,
                raw_output=raw_output,
            )


class PiJudge:
    def __init__(self, pi_command: str, model: str, timeout: int):
        self.pi_command = pi_command
        self.model = model
        self.timeout = timeout

    def evaluate(
        self,
        *,
        task: TaskSpec,
        repo_root: Path,
        run_dir: Path,
        step: int,
        model_response: ModelResponse,
        validations: list[ToolResult],
        changed_files: list[str],
        diff_stat: str,
        telemetry: TelemetryContext,
        logger: logging.Logger,
        golden_task_id: str | None,
    ) -> JudgeResult:
        validation_summary = []
        for result in validations:
            status = "pass" if result.success else f"fail(exit={result.exit_code})"
            snippet = trim_text(result.stdout or result.stderr, 1200) or "<no output>"
            validation_summary.append(f"- {result.name}: {status}\n  {snippet}")

        repo_checks = evaluate_repo_checks(task.repo_checks, repo_root, run_dir)
        repo_check_summary = []
        for check in repo_checks:
            status = "pass" if check.passed else "fail"
            repo_check_summary.append(f"- {check.check_id}: {status} ({check.kind}, base={check.base})\n  {check.description}\n  Evidence: {trim_text(check.evidence, 800)}")

        system_prompt = textwrap.dedent(
            """
            You are the judge inside a Ralph loop for a code repository task.
            Assess whether the current repository state satisfies the stated acceptable end state.
            Use the task objective, validation command results, and changed files as the primary evidence.
            Return STRICT JSON only with exactly these keys:
            {
              "passed": false,
              "reason": "short paragraph",
              "score": 0.0,
              "next_focus": "what should happen next"
            }
            Score must be 0.0 to 1.0.
            Do not include markdown fences.
            """
        ).strip()

        prompt = textwrap.dedent(
            f"""
            Task ID: {task.task_id}
            Title: {task.title}
            Step judged: {step}

            Objective:
            {task.objective}

            Acceptable end state:
            {task.acceptable_end_state}

            Deliverables:
            {chr(10).join(f"- {item}" for item in task.deliverables) or '- none provided'}

            Success signals:
            {chr(10).join(f"- {item}" for item in task.success_signals) or '- none provided'}

            Latest execution summary:
            {model_response.message or '<none>'}

            Latest completed items:
            {chr(10).join(f"- {item}" for item in model_response.completed) or '- none reported'}

            Latest remaining items:
            {chr(10).join(f"- {item}" for item in model_response.remaining) or '- none reported'}

            Changed files:
            {chr(10).join(f"- {item}" for item in changed_files) or '- no changed files detected'}

            Diff stat:
            {diff_stat or '<no diff stat>'}

            Validation results:
            {chr(10).join(validation_summary) or '- no validation commands configured'}

            Repo deliverable checks:
            {chr(10).join(repo_check_summary) or '- no repo checks configured'}
            """
        ).strip()

        token_usage = estimate_tokens(system_prompt, prompt)
        command = build_pi_invocation(self.pi_command, [
            "--print",
            "--no-session",
            "--model",
            self.model,
            "--thinking",
            "medium",
            "--tools",
            "read,grep,find,ls",
            "--system-prompt",
            system_prompt,
            prompt,
        ])

        with telemetry.tracer.start_as_current_span(
            "judge",
            attributes={
                "genai.model.name": self.model,
                "genai.model.version": "pi-cli",
                "genai.usage.total_tokens": token_usage,
                "ralph.prompt_hash": prompt_hash(prompt),
            },
        ) as span:
            if golden_task_id:
                span.add_event("judge.prompt", {"plaintext": trim_text(prompt, 8000)})
            started = time.perf_counter()
            proc = subprocess.run(
                command,
                cwd=str(repo_root),
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env={**os.environ, "PI_SKIP_VERSION_CHECK": "1"},
            )
            duration_ms = (time.perf_counter() - started) * 1000
            span.set_attribute("judge.duration_ms", duration_ms)
            if proc.returncode != 0:
                error = RuntimeError(f"pi judge call failed: {proc.stderr or proc.stdout}")
                span.record_exception(error)
                raise error
            raw_output = proc.stdout.strip()
            payload = extract_json_payload(raw_output)
            score = float(payload.get("score", 0.0) or 0.0)
            score = max(0.0, min(1.0, score))
            validations_passed = all(result.success for result in validations)
            repo_checks_passed = all(check.passed for check in repo_checks)
            passed = bool(payload.get("passed", False)) and validations_passed and repo_checks_passed
            reason = str(payload.get("reason") or "").strip()
            if not validations_passed:
                failed = ", ".join(result.name for result in validations if not result.success)
                reason = f"Validation commands failed ({failed}). {reason}".strip()
            if not repo_checks_passed:
                failed_checks = ", ".join(check.check_id for check in repo_checks if not check.passed)
                reason = f"Repo deliverable checks failed ({failed_checks}). {reason}".strip()
            next_focus = str(payload.get("next_focus") or "").strip()
            span.set_attribute("ralph.pass_fail", passed)
            span.set_attribute("ralph.repo_checks_total", len(repo_checks))
            span.set_attribute("ralph.repo_checks_failed", sum(1 for check in repo_checks if not check.passed))
            telemetry.tokens_counter.add(token_usage, {"phase": "judge"})
            logger.info("Judge step %s => passed=%s score=%.2f repo_checks_failed=%s", step, passed, score, sum(1 for check in repo_checks if not check.passed))
            return JudgeResult(
                passed=passed,
                reason=reason,
                score=score,
                next_focus=next_focus,
                token_usage=token_usage,
                raw_output=raw_output,
                repo_checks=repo_checks,
            )


def git_metadata(repo_root: Path, telemetry: TelemetryContext, logger: logging.Logger) -> tuple[str, str]:
    branch = run_subprocess("git.branch", ["git", "rev-parse", "--abbrev-ref", "HEAD"], repo_root, 30, telemetry, logger)
    commit = run_subprocess("git.commit", ["git", "rev-parse", "HEAD"], repo_root, 30, telemetry, logger)
    branch_name = branch.stdout.strip() if branch.success else "unknown"
    commit_sha = commit.stdout.strip() if commit.success else "unknown"
    return branch_name, commit_sha


def changed_files(repo_root: Path, telemetry: TelemetryContext, logger: logging.Logger) -> list[str]:
    result = run_subprocess("git.diff.name_only", ["git", "diff", "--name-only"], repo_root, 30, telemetry, logger)
    if not result.success:
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def diff_stat(repo_root: Path, telemetry: TelemetryContext, logger: logging.Logger) -> str:
    result = run_subprocess("git.diff.stat", ["git", "diff", "--stat"], repo_root, 30, telemetry, logger)
    if not result.success:
        return trim_text(result.stderr or result.stdout, 2000)
    return trim_text(result.stdout, 4000)


def run_validation_commands(
    commands: Sequence[str],
    repo_root: Path,
    telemetry: TelemetryContext,
    logger: logging.Logger,
    timeout: int,
    shell_command: str,
) -> list[ToolResult]:
    results: list[ToolResult] = []
    for index, command in enumerate(commands, start=1):
        results.append(
            run_subprocess(
                name=f"validation.{index}",
                command=[shell_command, "-lc", command],
                cwd=repo_root,
                timeout=timeout,
                telemetry=telemetry,
                logger=logger,
                retries=1,
            )
        )
    return results


def build_run_dir(repo_root: Path, task_id: str) -> Path:
    stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = repo_root / ".pi" / "hermes-self-evolution" / "ralph-runs" / f"{stamp}-{slugify(task_id)}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def persist_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_loop(args: argparse.Namespace) -> int:
    logger = configure_logging(args.verbose)
    telemetry = configure_telemetry(args.telemetry_export, args.otlp_endpoint)

    repo_root = Path(args.repo).resolve()
    shell_command = args.shell or shutil.which("bash") or shutil.which("sh") or "bash"
    task_file = Path(args.task_file).resolve()
    task = TaskSpec.from_file(task_file)
    run_dir = build_run_dir(repo_root, task.task_id)

    branch_name, commit_sha = git_metadata(repo_root, telemetry, logger)
    model = PiCliModel(pi_command=args.pi_command, model=args.model, thinking=args.thinking, timeout=args.model_timeout)
    judge = PiJudge(pi_command=args.pi_command, model=args.judge_model or args.model, timeout=args.judge_timeout)

    logger.info("Starting Ralph loop for task '%s' in %s", task.title, repo_root)
    logger.info("Run artifacts will be written to %s", run_dir)

    persist_json(
        run_dir / "config.json",
        {
            "task": asdict(task),
            "repo": str(repo_root),
            "task_file": str(task_file),
            "model": args.model,
            "judge_model": args.judge_model or args.model,
            "max_steps": args.max_steps,
            "telemetry_export": args.telemetry_export,
            "branch": branch_name,
            "commit_sha": commit_sha,
            "golden_task_id": args.golden_task_id,
        },
    )

    last_summary = ""
    last_judge = ""
    step_records: list[StepRecord] = []
    overall_passed = False

    telemetry.runs_counter.add(1, {"task_id": task.task_id})

    root_span_name = f"ralph.run/{slugify(task.task_id)}"
    run_started = time.perf_counter()
    with telemetry.tracer.start_as_current_span(
        root_span_name,
        attributes={
            "ralph.task_id": task.task_id,
            "ralph.prompt_hash": prompt_hash(task.objective + task.acceptable_end_state),
            "ralph.branch": branch_name,
            "ralph.commit_sha": commit_sha,
            "ralph.golden_task_id": args.golden_task_id or "",
            "genai.model.name": args.model,
            "genai.model.version": "pi-cli",
            "ralph.tool_calls": 0,
        },
    ) as root_span:
        try:
            for step in range(1, args.max_steps + 1):
                logger.info("--- Ralph step %s/%s ---", step, args.max_steps)
                step_started = time.perf_counter()
                with telemetry.tracer.start_as_current_span(
                    "loop.step",
                    attributes={
                        "ralph.step": step,
                        "ralph.max_steps": args.max_steps,
                    },
                ):
                    response = model.infer(
                        task=task,
                        repo_root=repo_root,
                        step=step,
                        max_steps=args.max_steps,
                        last_judge=last_judge,
                        last_summary=last_summary,
                        telemetry=telemetry,
                        logger=logger,
                        golden_task_id=args.golden_task_id,
                    )
                    validations = run_validation_commands(task.validation_commands, repo_root, telemetry, logger, args.validation_timeout, shell_command)
                    files = changed_files(repo_root, telemetry, logger)
                    diff = diff_stat(repo_root, telemetry, logger)
                    judge_result = judge.evaluate(
                        task=task,
                        repo_root=repo_root,
                        run_dir=run_dir,
                        step=step,
                        model_response=response,
                        validations=validations,
                        changed_files=files,
                        diff_stat=diff,
                        telemetry=telemetry,
                        logger=logger,
                        golden_task_id=args.golden_task_id,
                    )
                    duration_ms = (time.perf_counter() - step_started) * 1000
                    telemetry.step_latency.record(duration_ms, {"task_id": task.task_id})
                    record = StepRecord(
                        step=step,
                        started_at=time.time(),
                        duration_ms=duration_ms,
                        model=response,
                        validations=validations,
                        judge=judge_result,
                        changed_files=files,
                        diff_stat=diff,
                    )
                    step_records.append(record)
                    persist_json(run_dir / "steps" / f"step-{step}.json", asdict(record))
                    last_summary = response.message or response.final_answer
                    last_judge = judge_result.reason
                    root_span.set_attribute("ralph.tool_calls", sum(len(record.validations) for record in step_records))

                    if judge_result.passed:
                        overall_passed = True
                        logger.info("Task passed at step %s", step)
                        break
                    if not response.needs_more_steps:
                        logger.info("Model indicated no more steps are needed; stopping after step %s", step)
                        break

            run_duration_ms = (time.perf_counter() - run_started) * 1000
            telemetry.run_latency.record(run_duration_ms, {"task_id": task.task_id})
            root_span.set_attribute("ralph.pass_fail", overall_passed)
            root_span.set_attribute("genai.usage.total_tokens", sum(item.model.token_usage + item.judge.token_usage for item in step_records))
            latest_repo_checks = step_records[-1].judge.repo_checks if step_records else []
            configured_repo_checks = [spec.check_id for spec in task.repo_checks]
            summary = {
                "passed": overall_passed,
                "task_id": task.task_id,
                "title": task.title,
                "steps_executed": len(step_records),
                "branch": branch_name,
                "commit_sha": commit_sha,
                "run_dir": str(run_dir),
                "latest_summary": last_summary,
                "latest_judge": last_judge,
                "changed_files": step_records[-1].changed_files if step_records else [],
                "diff_stat": step_records[-1].diff_stat if step_records else "",
                "repo_checks_configured": configured_repo_checks,
                "repo_checks_ran": len(latest_repo_checks),
                "repo_checks_passed": all(check.passed for check in latest_repo_checks) if latest_repo_checks else (False if configured_repo_checks else True),
                "failed_repo_checks": [check.check_id for check in latest_repo_checks if not check.passed] if latest_repo_checks else configured_repo_checks,
            }
            persist_json(run_dir / "summary.json", summary)
            print(json.dumps(summary, indent=2))
            return 0 if overall_passed else 1
        except Exception as exc:
            telemetry.errors_counter.add(1, {"task_id": task.task_id})
            root_span.record_exception(exc)
            root_span.set_attribute("ralph.pass_fail", False)
            persist_json(
                run_dir / "summary.json",
                {
                    "passed": False,
                    "task_id": task.task_id,
                    "title": task.title,
                    "steps_executed": len(step_records),
                    "error": str(exc),
                    "run_dir": str(run_dir),
                },
            )
            logger.exception("Ralph loop failed")
            print(json.dumps({"passed": False, "error": str(exc), "run_dir": str(run_dir)}, indent=2))
            return 2
        finally:
            flush_telemetry()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Traced Ralph loop for pk-pi-hermes-evolve parity work.")
    parser.add_argument("--repo", default=str(ROOT), help="Repository root to operate on.")
    parser.add_argument("--task-file", default=str(DEFAULT_TASK_FILE), help="Path to a JSON task specification.")
    parser.add_argument("--pi-command", default="pi", help="pi executable to invoke.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="pi model used for execution steps.")
    parser.add_argument("--judge-model", default="", help="Optional separate model for judge calls.")
    parser.add_argument("--thinking", default=DEFAULT_THINKING, help="Thinking level for execution steps.")
    parser.add_argument("--max-steps", type=int, default=4, help="Maximum Ralph steps to run.")
    parser.add_argument("--shell", default="", help="Shell executable used for validation commands (defaults to bash/sh detection).")
    parser.add_argument("--model-timeout", type=int, default=900, help="Timeout in seconds for each execution step.")
    parser.add_argument("--judge-timeout", type=int, default=600, help="Timeout in seconds for each judge call.")
    parser.add_argument("--validation-timeout", type=int, default=300, help="Timeout in seconds for each validation command.")
    parser.add_argument("--telemetry-export", choices=["console", "otlp-http", "none"], default="console", help="Telemetry exporter.")
    parser.add_argument("--otlp-endpoint", default="", help="OTLP HTTP collector endpoint.")
    parser.add_argument("--golden-task-id", default="", help="Optional golden task identifier; enables plaintext prompt attachment.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return run_loop(args)


if __name__ == "__main__":
    raise SystemExit(main())

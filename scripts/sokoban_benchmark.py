from __future__ import annotations

import argparse
import csv
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from typing import Any


CSV_COLUMNS = [
    "run_id",
    "arm",
    "attempt",
    "level_id",
    "held_out",
    "solved",
    "wall_clock_seconds",
    "total_actions",
    "movement_actions",
    "undo_count",
    "reset_count",
    "irrecoverable_state_count",
    "dead_box_events",
    "notes",
]

REQUIRED_RESULT_FIELDS = {
    "solved": bool,
    "wall_clock_seconds": (int, float),
    "total_actions": int,
    "movement_actions": int,
    "undo_count": int,
    "reset_count": int,
    "irrecoverable_state_count": int,
    "dead_box_events": int,
}

DEFAULT_INITIAL_SKILL = """name: sokoban_solver
version: 1
principles:
  - Never push a box into a corner unless that corner is a goal.
  - Before any push, check whether the box will still have a path to a goal.
  - Prefer moves that preserve walking space around boxes.
  - Clear boxes from walls early unless the wall path ends at a goal.
  - Use undo immediately after a push that reduces future mobility.
  - Reset when multiple boxes become frozen and recovery cost exceeds restart cost.
pre_push_checklist:
  - Is the destination square legal?
  - Does the push trap the box against a wall or corner?
  - Will the player still be able to reach the useful side of the box?
  - Does this push reduce the number of reachable goals?
failure_signals:
  - Box frozen off-goal
  - Two boxes mutually blocking a corridor
  - Goal lane blocked by wrong box order
update_policy:
  - Add only rules supported by at least one observed failure.
  - Merge duplicate rules.
  - Keep the total rule set short enough to apply during play.
"""

BASE_PROMPT = """You are being evaluated on your ability to improve a reusable game-playing skill across repeated attempts.

Game: Sokoban
Goal: Complete the assigned level as quickly as possible.
Controls: Use only the game's supported controls (movement keys, and reset/undo if available).

Protocol:
- You will play 5 total attempts.
- Attempts 1-4 are training attempts.
- After each of attempts 1-4, you may update your persistent skill/instructions/memory for how to play Sokoban better.
- Attempt 5 is the evaluation attempt on a fresh unseen level from the same game family.
- On attempt 5, you must use the improved skill but you may not inspect or solve the evaluation level in advance.

Your job each attempt:
1. Observe the board.
2. Make a plan before acting.
3. Play to completion as efficiently as possible.
4. After the run, write a short postmortem:
   - what caused wasted moves or resets,
   - what heuristic would have helped,
   - what rule should be added to the skill.
5. Update the skill with concise reusable heuristics, not level-specific move sequences.

Important constraints:
- Do not overfit to one board.
- Prefer general heuristics that transfer to unseen levels.
- Track: success/failure, completion time, total actions, resets, undos, and whether the board became irrecoverable.
- On attempt 5, prioritize applying the learned skill cleanly.

Deliverables after each attempt:
- result.json with metrics,
- postmortem.md,
- updated skill file.
"""


@dataclass
class AttemptManifest:
    run_id: str
    arm: str
    attempt: int
    level_id: str
    held_out: bool
    skill_updates_allowed: bool
    benchmark_pack: str
    result_path: str
    postmortem_path: str
    prompt_path: str
    input_skill_path: str
    updated_skill_path: str
    attempt_dir: str
    status: str = "prepared"


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def benchmark_asset_dir() -> Path:
    return repo_root() / "benchmarks" / "sokoban"


def default_results_template() -> Path:
    return benchmark_asset_dir() / "sokoban-results-template.csv"


def default_benchmark_pack() -> Path:
    return benchmark_asset_dir() / "sokoban-skill-benchmark-pack.md"


def benchmark_runs_root(root_override: str | None = None) -> Path:
    base = Path(root_override).resolve() if root_override else repo_root()
    return base / ".pi" / "hermes-self-evolution" / "benchmarks" / "sokoban"


def write_text(path_value: Path, content: str) -> None:
    path_value.parent.mkdir(parents=True, exist_ok=True)
    path_value.write_text(content, encoding="utf-8")


def write_json(path_value: Path, payload: Any) -> None:
    write_text(path_value, json.dumps(payload, indent=2) + "\n")


def read_json(path_value: Path) -> dict[str, Any]:
    return json.loads(path_value.read_text(encoding="utf-8"))


def ensure_columns(template_file: Path) -> list[str]:
    header = template_file.read_text(encoding="utf-8").splitlines()[0].strip()
    columns = [item.strip() for item in header.split(",") if item.strip()]
    if columns != CSV_COLUMNS:
        raise ValueError(f"Unexpected CSV columns in {template_file}")
    return columns


def load_run(run_dir: Path) -> dict[str, Any]:
    manifest = run_dir / "run.json"
    if not manifest.exists():
        raise FileNotFoundError(f"Missing run manifest: {manifest}")
    return read_json(manifest)


def copy_file(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dest)


def current_skill_path(run_dir: Path, arm: str) -> Path:
    return run_dir / arm / "current-skill.md"


def skill_history_path(run_dir: Path, arm: str, attempt_index: int) -> Path:
    return run_dir / arm / "skill-history" / f"attempt-{attempt_index}.md"


def level_for_attempt(run_manifest: dict[str, Any], attempt: int) -> str:
    if attempt < 1 or attempt > 5:
        raise ValueError("Attempt must be between 1 and 5.")
    if attempt <= 4:
        return str(run_manifest["training_levels"][attempt - 1])
    return str(run_manifest["heldout_level"])


def build_attempt_prompt(manifest: AttemptManifest) -> str:
    update_rule = (
        "You may update the skill after this attempt. Keep only transferable heuristics."
        if manifest.skill_updates_allowed
        else "Skill updates are not allowed for this attempt. Keep the updated skill file unchanged except for formatting if needed."
    )
    held_out_rule = (
        "This is the held-out evaluation attempt. Do not inspect or solve this level in advance."
        if manifest.held_out
        else "This is a training attempt. Use it to refine reusable heuristics."
    )
    return (
        f"{BASE_PROMPT}\n"
        f"Attempt metadata:\n"
        f"- arm: {manifest.arm}\n"
        f"- attempt: {manifest.attempt}\n"
        f"- level_id: {manifest.level_id}\n"
        f"- held_out: {str(manifest.held_out).lower()}\n"
        f"- benchmark pack: {manifest.benchmark_pack}\n"
        f"- input skill: {manifest.input_skill_path}\n"
        f"- updated skill target: {manifest.updated_skill_path}\n"
        f"- result file: {manifest.result_path}\n"
        f"- postmortem file: {manifest.postmortem_path}\n\n"
        f"Attempt-specific rules:\n"
        f"- {held_out_rule}\n"
        f"- {update_rule}\n"
    )


def build_result_template(manifest: AttemptManifest) -> dict[str, Any]:
    return {
        "run_id": manifest.run_id,
        "arm": manifest.arm,
        "attempt": manifest.attempt,
        "level_id": manifest.level_id,
        "held_out": manifest.held_out,
        "solved": False,
        "wall_clock_seconds": 0,
        "total_actions": 0,
        "movement_actions": 0,
        "undo_count": 0,
        "reset_count": 0,
        "irrecoverable_state_count": 0,
        "dead_box_events": 0,
        "notes": "",
    }


def build_postmortem_template() -> str:
    return (
        "# Postmortem\n\n"
        "- What mistake caused the most wasted actions?\n"
        "- Which boxes became dangerous and why?\n"
        "- Which heuristic would have prevented that?\n"
        "- What concise reusable rule should be added to the skill?\n"
    )


def command_init(args: argparse.Namespace) -> None:
    run_dir = benchmark_runs_root(args.root) / args.run_id
    if run_dir.exists():
        raise FileExistsError(f"Run already exists: {run_dir}")

    if len(args.training_levels) != 4:
        raise ValueError("Provide exactly 4 training level IDs.")

    template_file = Path(args.results_template).resolve() if args.results_template else default_results_template()
    pack_file = Path(args.benchmark_pack).resolve() if args.benchmark_pack else default_benchmark_pack()
    ensure_columns(template_file)

    initial_skill_text = (
        Path(args.initial_skill_file).read_text(encoding="utf-8")
        if args.initial_skill_file
        else DEFAULT_INITIAL_SKILL
    )

    run_dir.mkdir(parents=True, exist_ok=False)
    copy_file(pack_file, run_dir / "benchmark-pack.md")
    write_text(run_dir / "results.csv", ",".join(CSV_COLUMNS) + "\n")
    write_json(
        run_dir / "run.json",
        {
            "run_id": args.run_id,
            "training_levels": args.training_levels,
            "heldout_level": args.heldout_level,
            "benchmark_pack": str((run_dir / "benchmark-pack.md").resolve()),
            "results_csv": str((run_dir / "results.csv").resolve()),
            "results_template_source": str(template_file),
            "benchmark_pack_source": str(pack_file),
        },
    )

    for arm in ("baseline", "improvement"):
        write_text(current_skill_path(run_dir, arm), initial_skill_text)
        write_text(skill_history_path(run_dir, arm, 0), initial_skill_text)

    write_text(
        run_dir / "README.md",
        (
            f"# Sokoban Benchmark Run {args.run_id}\n\n"
            "Workflow:\n"
            f"1. Prepare an attempt: `python scripts/sokoban_benchmark.py prepare-attempt --run-id {args.run_id} --arm improvement --attempt 1`\n"
            "2. Play the attempt using the generated prompt and files.\n"
            "3. Record the attempt: `python scripts/sokoban_benchmark.py record-attempt --run-id ... --arm ... --attempt ...`\n"
            f"4. Analyze the run: `python scripts/sokoban_benchmark.py analyze --run-id {args.run_id}`\n"
        ),
    )

    print(f"Initialized Sokoban benchmark run at {run_dir}")


def command_prepare_attempt(args: argparse.Namespace) -> None:
    run_dir = benchmark_runs_root(args.root) / args.run_id
    run_manifest = load_run(run_dir)
    arm = args.arm
    attempt = args.attempt
    attempt_dir = run_dir / arm / f"attempt-{attempt}"
    if attempt_dir.exists() and not args.force:
        raise FileExistsError(f"Attempt directory already exists: {attempt_dir}")

    level_id = level_for_attempt(run_manifest, attempt)
    held_out = attempt == 5
    skill_updates_allowed = arm == "improvement" and attempt <= 4
    input_skill = current_skill_path(run_dir, arm)
    if not input_skill.exists():
        raise FileNotFoundError(f"Missing current skill file: {input_skill}")

    if attempt_dir.exists() and args.force:
        shutil.rmtree(attempt_dir)
    attempt_dir.mkdir(parents=True, exist_ok=True)

    input_skill_path = attempt_dir / "input-skill.md"
    updated_skill_path = attempt_dir / "updated-skill.md"
    result_path = attempt_dir / "result.json"
    postmortem_path = attempt_dir / "postmortem.md"
    prompt_path = attempt_dir / "prompt.md"

    skill_text = input_skill.read_text(encoding="utf-8")
    write_text(input_skill_path, skill_text)
    write_text(updated_skill_path, skill_text)

    manifest = AttemptManifest(
        run_id=args.run_id,
        arm=arm,
        attempt=attempt,
        level_id=level_id,
        held_out=held_out,
        skill_updates_allowed=skill_updates_allowed,
        benchmark_pack=run_manifest["benchmark_pack"],
        result_path=str(result_path.resolve()),
        postmortem_path=str(postmortem_path.resolve()),
        prompt_path=str(prompt_path.resolve()),
        input_skill_path=str(input_skill_path.resolve()),
        updated_skill_path=str(updated_skill_path.resolve()),
        attempt_dir=str(attempt_dir.resolve()),
    )

    write_json(attempt_dir / "attempt.json", asdict(manifest))
    write_json(result_path, build_result_template(manifest))
    write_text(postmortem_path, build_postmortem_template())
    write_text(prompt_path, build_attempt_prompt(manifest))

    print(f"Prepared {arm} attempt {attempt} at {attempt_dir}")


def validate_result_payload(payload: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    validated: dict[str, Any] = {
        "run_id": manifest["run_id"],
        "arm": manifest["arm"],
        "attempt": int(manifest["attempt"]),
        "level_id": manifest["level_id"],
        "held_out": bool(manifest["held_out"]),
    }
    for field, expected_type in REQUIRED_RESULT_FIELDS.items():
        value = payload.get(field)
        if not isinstance(value, expected_type):
            raise ValueError(f"Invalid or missing field `{field}` in result.json")
        validated[field] = value
    notes = payload.get("notes", "")
    if not isinstance(notes, str):
        raise ValueError("`notes` must be a string.")
    validated["notes"] = notes
    return validated


def upsert_csv_row(csv_path: Path, row: dict[str, Any]) -> None:
    rows: list[dict[str, Any]] = []
    if csv_path.exists():
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))

    replaced = False
    for index, existing in enumerate(rows):
        if existing["run_id"] == str(row["run_id"]) and existing["arm"] == str(row["arm"]) and existing["attempt"] == str(row["attempt"]):
            rows[index] = {column: str(row[column]) for column in CSV_COLUMNS}
            replaced = True
            break

    if not replaced:
        rows.append({column: str(row[column]) for column in CSV_COLUMNS})

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def command_record_attempt(args: argparse.Namespace) -> None:
    run_dir = benchmark_runs_root(args.root) / args.run_id
    run_manifest = load_run(run_dir)
    attempt_dir = run_dir / args.arm / f"attempt-{args.attempt}"
    attempt_manifest = read_json(attempt_dir / "attempt.json")
    result_path = attempt_dir / "result.json"
    updated_skill_path = attempt_dir / "updated-skill.md"
    postmortem_path = attempt_dir / "postmortem.md"

    if not result_path.exists():
        raise FileNotFoundError(f"Missing result file: {result_path}")
    if not postmortem_path.exists():
        raise FileNotFoundError(f"Missing postmortem file: {postmortem_path}")
    if not updated_skill_path.exists():
        raise FileNotFoundError(f"Missing updated skill file: {updated_skill_path}")

    validated = validate_result_payload(read_json(result_path), attempt_manifest)
    upsert_csv_row(Path(run_manifest["results_csv"]), validated)

    history_dest = skill_history_path(run_dir, args.arm, args.attempt)
    copy_file(updated_skill_path, history_dest)
    if attempt_manifest["skill_updates_allowed"]:
        copy_file(updated_skill_path, current_skill_path(run_dir, args.arm))

    attempt_manifest["status"] = "recorded"
    write_json(attempt_dir / "attempt.json", attempt_manifest)
    print(f"Recorded {args.arm} attempt {args.attempt} into {run_manifest['results_csv']}")


def load_rows(csv_path: Path, run_id: str) -> list[dict[str, Any]]:
    if not csv_path.exists():
        return []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("run_id") == run_id]
    parsed: list[dict[str, Any]] = []
    for row in rows:
        parsed.append(
            {
                **row,
                "attempt": int(row["attempt"]),
                "held_out": row["held_out"].lower() == "true",
                "solved": row["solved"].lower() == "true",
                "wall_clock_seconds": float(row["wall_clock_seconds"]),
                "total_actions": int(row["total_actions"]),
                "movement_actions": int(row["movement_actions"]),
                "undo_count": int(row["undo_count"]),
                "reset_count": int(row["reset_count"]),
                "irrecoverable_state_count": int(row["irrecoverable_state_count"]),
                "dead_box_events": int(row["dead_box_events"]),
            }
        )
    return parsed


def find_attempt(rows: list[dict[str, Any]], arm: str, attempt: int) -> dict[str, Any] | None:
    for row in rows:
        if row["arm"] == arm and row["attempt"] == attempt:
            return row
    return None


def safe_delta(before: float | int | None, after: float | int | None) -> float | None:
    if before is None or after is None:
        return None
    return float(after) - float(before)


def summarize_run(rows: list[dict[str, Any]]) -> dict[str, Any]:
    improvement_attempt_1 = find_attempt(rows, "improvement", 1)
    improvement_attempt_5 = find_attempt(rows, "improvement", 5)
    baseline_attempt_5 = find_attempt(rows, "baseline", 5)
    held_out_rows = [row for row in rows if row["held_out"]]
    held_out_times = [row["wall_clock_seconds"] for row in held_out_rows if row["solved"]]

    return {
        "recorded_attempts": len(rows),
        "held_out_attempts": len(held_out_rows),
        "held_out_solve_rate": (sum(1 for row in held_out_rows if row["solved"]) / len(held_out_rows)) if held_out_rows else None,
        "held_out_median_time": median(held_out_times) if held_out_times else None,
        "improvement_arm_attempt_1_vs_5": {
            "solve_delta": safe_delta(
                1 if improvement_attempt_1 and improvement_attempt_1["solved"] else 0 if improvement_attempt_1 else None,
                1 if improvement_attempt_5 and improvement_attempt_5["solved"] else 0 if improvement_attempt_5 else None,
            ),
            "wall_clock_delta": safe_delta(
                improvement_attempt_1["wall_clock_seconds"] if improvement_attempt_1 else None,
                improvement_attempt_5["wall_clock_seconds"] if improvement_attempt_5 else None,
            ),
            "total_actions_delta": safe_delta(
                improvement_attempt_1["total_actions"] if improvement_attempt_1 else None,
                improvement_attempt_5["total_actions"] if improvement_attempt_5 else None,
            ),
        },
        "baseline_vs_improvement_attempt_5": {
            "solve_delta": safe_delta(
                1 if baseline_attempt_5 and baseline_attempt_5["solved"] else 0 if baseline_attempt_5 else None,
                1 if improvement_attempt_5 and improvement_attempt_5["solved"] else 0 if improvement_attempt_5 else None,
            ),
            "wall_clock_delta": safe_delta(
                baseline_attempt_5["wall_clock_seconds"] if baseline_attempt_5 else None,
                improvement_attempt_5["wall_clock_seconds"] if improvement_attempt_5 else None,
            ),
            "total_actions_delta": safe_delta(
                baseline_attempt_5["total_actions"] if baseline_attempt_5 else None,
                improvement_attempt_5["total_actions"] if improvement_attempt_5 else None,
            ),
            "reset_delta": safe_delta(
                baseline_attempt_5["reset_count"] if baseline_attempt_5 else None,
                improvement_attempt_5["reset_count"] if improvement_attempt_5 else None,
            ),
            "undo_delta": safe_delta(
                baseline_attempt_5["undo_count"] if baseline_attempt_5 else None,
                improvement_attempt_5["undo_count"] if improvement_attempt_5 else None,
            ),
        },
    }


def build_summary_markdown(summary: dict[str, Any]) -> str:
    def fmt(value: Any) -> str:
        if value is None:
            return "n/a"
        if isinstance(value, float):
            return f"{value:.3f}"
        return str(value)

    within = summary["improvement_arm_attempt_1_vs_5"]
    arms = summary["baseline_vs_improvement_attempt_5"]
    return (
        "# Sokoban Benchmark Summary\n\n"
        f"- Recorded attempts: {fmt(summary['recorded_attempts'])}\n"
        f"- Held-out attempts: {fmt(summary['held_out_attempts'])}\n"
        f"- Held-out solve rate: {fmt(summary['held_out_solve_rate'])}\n"
        f"- Held-out median time: {fmt(summary['held_out_median_time'])}\n\n"
        "## Improvement arm attempt 1 vs attempt 5\n"
        f"- Solve delta: {fmt(within['solve_delta'])}\n"
        f"- Wall clock delta: {fmt(within['wall_clock_delta'])}\n"
        f"- Total actions delta: {fmt(within['total_actions_delta'])}\n\n"
        "## Baseline attempt 5 vs improvement attempt 5\n"
        f"- Solve delta: {fmt(arms['solve_delta'])}\n"
        f"- Wall clock delta: {fmt(arms['wall_clock_delta'])}\n"
        f"- Total actions delta: {fmt(arms['total_actions_delta'])}\n"
        f"- Reset delta: {fmt(arms['reset_delta'])}\n"
        f"- Undo delta: {fmt(arms['undo_delta'])}\n"
    )


def command_analyze(args: argparse.Namespace) -> None:
    run_dir = benchmark_runs_root(args.root) / args.run_id
    run_manifest = load_run(run_dir)
    rows = load_rows(Path(run_manifest["results_csv"]), args.run_id)
    summary = summarize_run(rows)
    write_json(run_dir / "summary.json", summary)
    write_text(run_dir / "summary.md", build_summary_markdown(summary))
    print(f"Wrote benchmark summary to {run_dir / 'summary.md'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Initialize and manage a Sokoban skill-improvement benchmark run.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="Create a new Sokoban benchmark run.")
    init_parser.add_argument("--run-id", required=True)
    init_parser.add_argument("--training-levels", nargs=4, required=True, metavar=("A", "B", "C", "D"))
    init_parser.add_argument("--heldout-level", required=True)
    init_parser.add_argument("--initial-skill-file")
    init_parser.add_argument("--benchmark-pack")
    init_parser.add_argument("--results-template")
    init_parser.add_argument("--root")
    init_parser.set_defaults(func=command_init)

    prepare_parser = subparsers.add_parser("prepare-attempt", help="Prepare files for a specific arm/attempt.")
    prepare_parser.add_argument("--run-id", required=True)
    prepare_parser.add_argument("--arm", choices=["baseline", "improvement"], required=True)
    prepare_parser.add_argument("--attempt", type=int, required=True)
    prepare_parser.add_argument("--root")
    prepare_parser.add_argument("--force", action="store_true")
    prepare_parser.set_defaults(func=command_prepare_attempt)

    record_parser = subparsers.add_parser("record-attempt", help="Validate and record a completed attempt.")
    record_parser.add_argument("--run-id", required=True)
    record_parser.add_argument("--arm", choices=["baseline", "improvement"], required=True)
    record_parser.add_argument("--attempt", type=int, required=True)
    record_parser.add_argument("--root")
    record_parser.set_defaults(func=command_record_attempt)

    analyze_parser = subparsers.add_parser("analyze", help="Summarize recorded benchmark results.")
    analyze_parser.add_argument("--run-id", required=True)
    analyze_parser.add_argument("--root")
    analyze_parser.set_defaults(func=command_analyze)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

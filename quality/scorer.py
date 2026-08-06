"""Quality scorer: evaluates model responses against the task pack."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from agentperf.models import CheckType, ScoreResult, TaskItem


# ── helpers ────────────────────────────────────────────────────────────────────

_CODE_FENCE_RE = re.compile(r"^```[a-zA-Z]*\n?(.*?)\n?```$", re.DOTALL)


def _strip_code_fences(text: str) -> str:
    """Remove a single wrapping markdown code fence block, if present."""
    stripped = text.strip()
    m = _CODE_FENCE_RE.match(stripped)
    return m.group(1).strip() if m else stripped


# ── scorers ────────────────────────────────────────────────────────────────────

def score_code_exec(task: TaskItem, model_output: str) -> ScoreResult:
    """Execute model_output + unit_tests in a subprocess; pass on returncode 0."""
    if task.unit_tests is None:
        return ScoreResult(task_id=task.id, passed=False, error="missing_unit_tests")

    source = model_output + "\n\n" + task.unit_tests

    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            delete=False,
            encoding="utf-8",
        ) as fh:
            tmp_path = Path(fh.name)
            fh.write(source)

        result = subprocess.run(
            [sys.executable, str(tmp_path)],
            capture_output=True,
            timeout=10,
        )
        if result.returncode == 0:
            return ScoreResult(task_id=task.id, passed=True)
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        return ScoreResult(task_id=task.id, passed=False, error=stderr or "nonzero_exit")
    except subprocess.TimeoutExpired:
        return ScoreResult(task_id=task.id, passed=False, error="timeout")
    finally:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink()


def score_function_call(task: TaskItem, model_output: str) -> ScoreResult:
    """Compare parsed JSON function call against expected_output."""
    cleaned = _strip_code_fences(model_output)
    try:
        actual: dict[str, Any] = json.loads(cleaned)
    except json.JSONDecodeError:
        return ScoreResult(task_id=task.id, passed=False, error="json_parse_error")

    try:
        expected: dict[str, Any] = json.loads(task.expected_output)
    except json.JSONDecodeError:
        return ScoreResult(task_id=task.id, passed=False, error="expected_output_parse_error")

    names_match = actual.get("name") == expected.get("name")
    # dict compare on arguments; missing key treated as empty dict
    args_match = actual.get("arguments", {}) == expected.get("arguments", {})
    passed = names_match and args_match
    return ScoreResult(task_id=task.id, passed=passed)


def score_exact_match(task: TaskItem, model_output: str) -> ScoreResult:
    """Case-insensitive, whitespace-trimmed equality check."""
    passed = model_output.strip().lower() == task.expected_output.strip().lower()
    return ScoreResult(task_id=task.id, passed=passed)


def score_task(task: TaskItem, model_output: str) -> ScoreResult:
    """Dispatch to the appropriate scorer based on task.check_type."""
    if task.check_type == CheckType.CODE_EXEC:
        return score_code_exec(task, model_output)
    if task.check_type == CheckType.FUNCTION_CALL:
        return score_function_call(task, model_output)
    if task.check_type == CheckType.EXACT_MATCH:
        return score_exact_match(task, model_output)
    return ScoreResult(task_id=task.id, passed=False, error="unknown_check_type")


# ── reporting ──────────────────────────────────────────────────────────────────

def _build_report(
    results: list[ScoreResult],
    tasks_by_id: dict[str, TaskItem],
) -> dict[str, Any]:
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    success_rate = round(passed / total, 4) if total else 0.0

    by_category: dict[str, dict[str, Any]] = {}
    for result in results:
        task = tasks_by_id.get(result.task_id)
        cat = task.category if task else "unknown"
        bucket = by_category.setdefault(cat, {"total": 0, "passed": 0})
        bucket["total"] += 1
        if result.passed:
            bucket["passed"] += 1

    for cat, bucket in by_category.items():
        n = bucket["total"]
        bucket["success_rate"] = round(bucket["passed"] / n, 4) if n else 0.0

    return {
        "total": total,
        "passed": passed,
        "success_rate": success_rate,
        "by_category": by_category,
        "results": [r.model_dump() for r in results],
    }


# ── entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score model responses against a quality task pack."
    )
    parser.add_argument(
        "--responses",
        required=True,
        metavar="PATH",
        help="JSONL file; each line: {\"id\": str, \"output\": str}",
    )
    parser.add_argument(
        "--task-pack",
        required=True,
        metavar="PATH",
        help="JSON file containing the list of TaskItem objects",
    )
    parser.add_argument(
        "--output",
        required=True,
        metavar="PATH",
        help="Destination path for the JSON score report",
    )
    args = parser.parse_args()

    # Load task pack
    with open(args.task_pack, "r", encoding="utf-8") as fh:
        raw_tasks: list[dict[str, Any]] = json.load(fh)
    tasks_by_id: dict[str, TaskItem] = {
        t["id"]: TaskItem(**t) for t in raw_tasks
    }

    # Load responses
    responses: dict[str, str] = {}
    with open(args.responses, "r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                responses[entry["id"]] = entry["output"]
            except (json.JSONDecodeError, KeyError) as exc:
                # Malformed lines are silently skipped; score for that task
                # will appear as an error in the results.
                sys.stderr.write(f"Warning: skipping line {line_no}: {exc}\n")

    # Score every task that has a corresponding response
    results: list[ScoreResult] = []
    for task_id, task in tasks_by_id.items():
        if task_id not in responses:
            results.append(
                ScoreResult(task_id=task_id, passed=False, error="missing_response")
            )
            continue
        try:
            result = score_task(task, responses[task_id])
        except Exception as exc:  # noqa: BLE001
            result = ScoreResult(task_id=task_id, passed=False, error=str(exc))
        results.append(result)

    report = _build_report(results, tasks_by_id)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)


if __name__ == "__main__":
    main()

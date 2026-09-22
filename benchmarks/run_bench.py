#!/usr/bin/env python3
"""Run ThothBench-30 against a configured real provider."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

from thoth.exceptions import ThothError
from thoth.orchestrator import Orchestrator
from thoth.server import default_adapter_factory

HERE = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the 30-task Project Thoth benchmark")
    parser.add_argument("--provider", choices=("groq", "gemini", "ollama"), default="groq")
    parser.add_argument("--model", default="llama-3.1-8b-instant")
    parser.add_argument("--max-turns", type=int, choices=range(1, 6), default=3)
    parser.add_argument("--tasks", type=Path, default=HERE / "thothbench30.json")
    parser.add_argument("--output", type=Path, default=HERE / "results.json")
    parser.add_argument("--limit", type=int, help="run only the first N tasks for a smoke test")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tasks: list[dict[str, Any]] = json.loads(args.tasks.read_text(encoding="utf-8"))
    if args.limit:
        tasks = tasks[: args.limit]
    results: list[dict[str, Any]] = []

    for index, task in enumerate(tasks, 1):
        print(f"[{index}/{len(tasks)}] {task['id']}: {task['category']}", flush=True)
        coder, verifier = default_adapter_factory(args.provider, args.model)
        orchestrator = Orchestrator(
            coder,
            verifier,
            max_turns=args.max_turns,
            allow_filesystem=bool(task.get("requires_filesystem")),
        )
        started = time.monotonic()
        try:
            outcome = orchestrator.run(task["prompt"])
            record = {
                "task_id": task["id"],
                "category": task["category"],
                "status": "APPROVED",
                "turns": outcome.turns,
                "tokens": outcome.total_tokens,
                "duration_seconds": round(time.monotonic() - started, 2),
            }
        except Exception as exc:
            record = {
                "task_id": task["id"],
                "category": task["category"],
                "status": type(exc).__name__,
                "turns": args.max_turns if isinstance(exc, ThothError) else 0,
                "tokens": 0,
                "duration_seconds": round(time.monotonic() - started, 2),
                "error": str(exc),
            }
        results.append(record)
        args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")

    approved = [item for item in results if item["status"] == "APPROVED"]
    total = len(results) or 1
    first_pass = sum(item["turns"] == 1 for item in approved) / total
    two_turn = sum(item["turns"] <= 2 for item in approved) / total
    categories = Counter(item["category"] for item in approved)
    summary = {
        "provider": args.provider,
        "model": args.model,
        "tasks": len(results),
        "first_pass_approval_rate": round(first_pass, 4),
        "two_turn_convergence_rate": round(two_turn, 4),
        "approved_by_category": dict(categories),
        "results": results,
    }
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "results"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

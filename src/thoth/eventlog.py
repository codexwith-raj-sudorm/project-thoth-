"""Append-only JSONL session event logging and retention."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class SessionLogger:
    def __init__(self, session_id: str, directory: Path | None = None) -> None:
        self.session_id = session_id
        self.directory = directory or Path.home() / ".thoth" / "logs"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / f"session_{session_id}.jsonl"
        self.rotate()

    def event(self, *, turn_index: int, agent_role: str, **values: Any) -> None:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session_id": self.session_id,
            "turn_index": turn_index,
            "agent_role": agent_role,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cumulative_session_tokens": 0,
            "ast_scan_result": "pass",
            "tautological_tests_count": 0,
            "execution_exit_code": None,
            "execution_stdout": "",
            "execution_stderr": "",
            "verifier_status": None,
            "verifier_implementation_critique": None,
            "verifier_test_critique": None,
            **values,
        }
        record["execution_stdout"] = record["execution_stdout"][:2000]
        record["execution_stderr"] = record["execution_stderr"][:2000]
        with self.path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    def rotate(self) -> None:
        files = sorted(self.directory.glob("session_*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
        cutoff = time.time() - 7 * 86400
        # Keep a file if it is among the newest 50 OR younger than seven days.
        for index, path in enumerate(files):
            if index >= 50 and path.stat().st_mtime < cutoff:
                path.unlink(missing_ok=True)

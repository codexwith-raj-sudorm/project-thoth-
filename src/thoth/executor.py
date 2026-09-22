"""Best-effort isolated candidate execution."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

from .schemas import ExecutionResult

_SAFE_ENV_NAMES = {"SYSTEMROOT", "WINDIR", "COMSPEC", "PATHEXT", "TMP", "TEMP", "LANG", "LC_ALL"}


def _sanitized_environment(temp_dir: str) -> dict[str, str]:
    env = {key: value for key, value in os.environ.items() if key in _SAFE_ENV_NAMES}
    python_dir = str(Path(sys.executable).resolve().parent)
    env.update({
        "PATH": python_dir,
        "PYTHONPATH": "",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "HOME": temp_dir,
    })
    return env


def execute_code(code: str, timeout: float = 5.0) -> ExecutionResult:
    with tempfile.TemporaryDirectory(prefix="thoth-") as temp_dir:
        script = Path(temp_dir) / "candidate.py"
        script.write_text(code, encoding="utf-8")
        try:
            completed = subprocess.run(
                [sys.executable, "-I", script.name],
                cwd=temp_dir,
                env=_sanitized_environment(temp_dir),
                shell=False,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
            # Tracebacks resolve the script to the random temp path. Normalize it so
            # identical failures have identical stderr hashes for stagnation checks.
            stderr = completed.stderr.replace(str(script), script.name).replace(temp_dir + os.sep, "")
            return ExecutionResult(
                exit_code=completed.returncode,
                stdout=completed.stdout,
                stderr=stderr,
            )
        except subprocess.TimeoutExpired as exc:
            return ExecutionResult(
                exit_code=None,
                stdout=(exc.stdout or "") if isinstance(exc.stdout, str) else "",
                stderr=((exc.stderr or "") if isinstance(exc.stderr, str) else "") + "\nExecution timed out.",
                timed_out=True,
            )

"""Cross-platform single-run advisory file lock."""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import IO

from .exceptions import ConcurrentRunError


class SingleInstanceLock:
    def __init__(self, path: Path | None = None, stale_after: float = 7200) -> None:
        self.path = path or Path.home() / ".thoth" / "thoth.lock"
        self.stale_after = stale_after
        self._file: IO[str] | None = None

    @staticmethod
    def _alive(pid: int) -> bool:
        if pid <= 0:
            return False
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except OSError:
            return False

    def _try_lock(self, file: IO[str]) -> bool:
        try:
            if os.name == "nt":
                import msvcrt
                file.seek(0)
                msvcrt.locking(file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except (OSError, BlockingIOError):
            return False

    def acquire(self) -> "SingleInstanceLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        file = self.path.open("a+", encoding="utf-8")
        if not self._try_lock(file):
            file.seek(0)
            try:
                metadata = json.load(file)
            except (json.JSONDecodeError, ValueError):
                metadata = {}
            pid = int(metadata.get("pid", 0))
            timestamp = float(metadata.get("unix_timestamp", time.time()))
            file.close()
            age = time.time() - timestamp
            detail = f"PID {pid}" if self._alive(pid) and age <= self.stale_after else "a lock holder"
            raise ConcurrentRunError(f"another Thoth run is active ({detail})")
        self._file = file
        file.seek(0)
        file.truncate()
        json.dump({
            "pid": os.getpid(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "unix_timestamp": time.time(),
        }, file)
        file.flush()
        os.fsync(file.fileno())
        return self

    def release(self) -> None:
        if not self._file:
            return
        try:
            self._file.seek(0)
            self._file.truncate()
            if os.name == "nt":
                import msvcrt
                self._file.seek(0)
                msvcrt.locking(self._file.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(self._file.fileno(), fcntl.LOCK_UN)
        finally:
            self._file.close()
            self._file = None
            self.path.unlink(missing_ok=True)

    def __enter__(self) -> "SingleInstanceLock":
        return self.acquire()

    def __exit__(self, *_: object) -> None:
        self.release()

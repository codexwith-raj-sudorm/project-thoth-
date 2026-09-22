"""Thread-safe in-memory token-bucket limiter and retry utility."""

from __future__ import annotations

import random
import threading
import time
from collections.abc import Callable
from typing import TypeVar

from .exceptions import RateLimitExceeded

T = TypeVar("T")


class TokenBucket:
    def __init__(self, requests_per_minute: float = 12, *, clock: Callable[[], float] = time.monotonic) -> None:
        if requests_per_minute <= 0:
            raise ValueError("requests_per_minute must be positive")
        self.capacity = float(requests_per_minute)
        self.tokens = self.capacity
        self.refill_rate = self.capacity / 60.0
        self.clock = clock
        self.updated = clock()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        while True:
            with self._lock:
                now = self.clock()
                self.tokens = min(self.capacity, self.tokens + (now - self.updated) * self.refill_rate)
                self.updated = now
                if self.tokens >= 1:
                    self.tokens -= 1
                    return
                delay = (1 - self.tokens) / self.refill_rate
            time.sleep(delay)


def retry_transient(
    operation: Callable[[], T],
    *,
    is_transient: Callable[[Exception], bool],
    max_retries: int = 3,
    base_delay: float = 1.0,
) -> T:
    for retry in range(max_retries + 1):
        try:
            return operation()
        except Exception as exc:
            if not is_transient(exc):
                raise
            if retry == max_retries:
                if getattr(exc, "code", None) == 429:
                    raise RateLimitExceeded(
                        f"provider rate limit persisted after {max_retries} retries: {exc}"
                    ) from exc
                raise
            time.sleep(random.uniform(0, base_delay * (2**retry)))
    raise AssertionError("unreachable")

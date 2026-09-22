"""Deterministic mediator state machine for dual-agent synthesis."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from .adapters.base import ModelAdapter
from .eventlog import SessionLogger
from .exceptions import (
    MaxTurnsExceeded,
    SecurityViolationException,
    StagnantLoopException,
    TokenBudgetExceeded,
)
from .executor import execute_code
from .prompts import CODER_SYSTEM, VERIFIER_SYSTEM
from .schemas import CoderOutput, ExecutionResult, VerifierOutput
from .security import count_tautological_assertions, scan_code


@dataclass(frozen=True)
class RunResult:
    session_id: str
    code: str
    turns: int
    execution: ExecutionResult
    verifier: VerifierOutput
    log_path: Path
    total_tokens: int = 0


class Orchestrator:
    def __init__(
        self,
        coder: ModelAdapter,
        verifier: ModelAdapter,
        *,
        max_turns: int = 3,
        token_budget: int = 20_000,
        allow_filesystem: bool = False,
        allow_network: bool = False,
        execution_timeout: float = 5.0,
        log_directory: Path | None = None,
        executor: Callable[[str, float], ExecutionResult] = execute_code,
        event_sink: Callable[[str, int | None, dict[str, Any]], None] | None = None,
    ) -> None:
        if not 1 <= max_turns <= 5:
            raise ValueError("max_turns must be between 1 and 5")
        self.coder = coder
        self.verifier = verifier
        self.max_turns = max_turns
        self.token_budget = token_budget
        self.allow_filesystem = allow_filesystem
        self.allow_network = allow_network
        self.execution_timeout = execution_timeout
        self.log_directory = log_directory
        self.executor = executor
        self.event_sink = event_sink

    def _emit(self, event_type: str, turn_index: int | None = None, **data: Any) -> None:
        if self.event_sink is not None:
            try:
                self.event_sink(event_type, turn_index, data)
            except Exception:
                # Observability must never alter deterministic engine control flow.
                pass

    @staticmethod
    def _outcome(execution: ExecutionResult) -> tuple[int | None, str]:
        return execution.exit_code, hashlib.sha256(execution.stderr.encode()).hexdigest()

    def run(self, requirement: str) -> RunResult:
        if not requirement.strip():
            raise ValueError("requirement cannot be empty")
        session_id = str(uuid4())
        logger = SessionLogger(session_id, self.log_directory)
        critiques: list[VerifierOutput] = []
        latest_code = ""
        latest_execution = ExecutionResult(exit_code=None)
        previous_code: str | None = None
        previous_outcome: tuple[int | None, str] | None = None
        cumulative = 0
        logger.event(turn_index=0, agent_role="orchestrator", prompt=requirement)

        for turn in range(1, self.max_turns + 1):
            self._emit("turn_start", turn)
            self._emit("coder_start", turn)
            coder_prompt = self._coder_prompt(requirement, critiques, latest_code, latest_execution, turn)
            candidate = self.coder.generate_structured(CODER_SYSTEM, coder_prompt, CoderOutput)
            cumulative = self._add_usage(cumulative, self.coder)
            latest_code = candidate.implementation_code
            thought = candidate.thought_process.strip().replace("\n", " ")[:240]
            self._emit("coder_done", turn, thought_summary=thought)

            try:
                scan = scan_code(
                    latest_code,
                    allow_filesystem=self.allow_filesystem,
                    allow_network=self.allow_network,
                )
            except SecurityViolationException as exc:
                logger.event(
                    turn_index=turn,
                    agent_role="orchestrator",
                    prompt_tokens=self.coder.last_usage.prompt_tokens,
                    completion_tokens=self.coder.last_usage.completion_tokens,
                    cumulative_session_tokens=cumulative,
                    ast_scan_result=str(exc),
                )
                raise
            tautologies = count_tautological_assertions(latest_code)
            self._emit("execution_start", turn)
            execution_started = time.monotonic()
            latest_execution = self.executor(latest_code, self.execution_timeout)
            duration_ms = round((time.monotonic() - execution_started) * 1000)
            self._emit(
                "execution_done",
                turn,
                exit_code=latest_execution.exit_code,
                duration_ms=duration_ms,
                tests_passed=latest_execution.exit_code == 0,
            )
            outcome = self._outcome(latest_execution)

            logger.event(
                turn_index=turn,
                agent_role="coder",
                prompt_tokens=self.coder.last_usage.prompt_tokens,
                completion_tokens=self.coder.last_usage.completion_tokens,
                cumulative_session_tokens=cumulative,
                ast_scan_result=scan.log_value,
                tautological_tests_count=tautologies,
                execution_exit_code=latest_execution.exit_code,
                execution_stdout=latest_execution.stdout,
                execution_stderr=latest_execution.stderr,
            )

            if previous_code is not None:
                similarity = SequenceMatcher(None, previous_code, latest_code).ratio()
                if similarity > 0.95 and outcome == previous_outcome:
                    raise StagnantLoopException(
                        f"revision stagnated at turn {turn} (similarity {similarity:.3f})",
                        latest_code,
                    )
            previous_code, previous_outcome = latest_code, outcome

            if tautologies:
                verdict = VerifierOutput(
                    status="NEEDS_REVISION",
                    implementation_critique="Static checks completed; verifier call skipped.",
                    test_critique=f"{tautologies} tautological assertions detected by static analysis.",
                    suggested_fix="Replace tautological assertions with expected-output and edge-case checks.",
                )
            else:
                self._emit("verifier_start", turn)
                verifier_prompt = self._verifier_prompt(requirement, latest_code, latest_execution)
                verdict = self.verifier.generate_structured(VERIFIER_SYSTEM, verifier_prompt, VerifierOutput)
                cumulative = self._add_usage(cumulative, self.verifier)
                # LLM status cannot overrule deterministic runtime integrity.
                if verdict.status == "APPROVED" and latest_execution.exit_code != 0:
                    verdict = VerifierOutput(
                        status="NEEDS_REVISION",
                        implementation_critique=(
                            verdict.implementation_critique + " Runtime exited non-zero; approval rejected."
                        ).strip(),
                        test_critique=verdict.test_critique,
                        suggested_fix="Fix the runtime failure shown in stderr before approval.",
                    )

            critiques.append(verdict)
            critique_summary = " ".join(
                part for part in (verdict.implementation_critique, verdict.test_critique) if part
            )[:400]
            self._emit(
                "verifier_done",
                turn,
                status=verdict.status,
                critique_summary=critique_summary,
            )
            logger.event(
                turn_index=turn,
                agent_role="verifier",
                prompt_tokens=0 if tautologies else self.verifier.last_usage.prompt_tokens,
                completion_tokens=0 if tautologies else self.verifier.last_usage.completion_tokens,
                cumulative_session_tokens=cumulative,
                ast_scan_result=scan.log_value,
                tautological_tests_count=tautologies,
                execution_exit_code=latest_execution.exit_code,
                execution_stdout=latest_execution.stdout,
                execution_stderr=latest_execution.stderr,
                verifier_status=verdict.status,
                verifier_implementation_critique=verdict.implementation_critique,
                verifier_test_critique=verdict.test_critique,
            )
            if verdict.status == "APPROVED":
                return RunResult(
                    session_id,
                    latest_code,
                    turn,
                    latest_execution,
                    verdict,
                    logger.path,
                    cumulative,
                )

        raise MaxTurnsExceeded(
            f"no candidate was approved within {self.max_turns} turns",
            latest_code,
        )

    def _add_usage(self, cumulative: int, adapter: ModelAdapter) -> int:
        cumulative += adapter.last_usage.prompt_tokens + adapter.last_usage.completion_tokens
        if cumulative > self.token_budget:
            raise TokenBudgetExceeded(
                f"session token budget exceeded ({cumulative} > {self.token_budget})"
            )
        return cumulative

    @staticmethod
    def _coder_prompt(
        requirement: str,
        critiques: list[VerifierOutput],
        latest_code: str,
        execution: ExecutionResult,
        turn: int,
    ) -> str:
        sections = [f"ORIGINAL REQUIREMENT (permanent):\n{requirement}", f"TURN: {turn}"]
        if critiques:
            history = "\n".join(
                f"Turn {index}: implementation={item.implementation_critique!r}; "
                f"tests={item.test_critique!r}; fix={item.suggested_fix!r}"
                for index, item in enumerate(critiques, 1)
            )
            sections.append(f"VERIFIER CRITIQUE HISTORY (permanent):\n{history}")
            sections.append(f"LATEST CANDIDATE ONLY:\n{latest_code}")
            sections.append(
                "LATEST EXECUTION ONLY:\n"
                f"exit_code={execution.exit_code}\nstdout={execution.stdout[:2000]}\nstderr={execution.stderr[:2000]}"
            )
        return "\n\n".join(sections)

    @staticmethod
    def _verifier_prompt(requirement: str, code: str, execution: ExecutionResult) -> str:
        return (
            f"ORIGINAL REQUIREMENT:\n{requirement}\n\nCANDIDATE CODE:\n{code}\n\n"
            f"ACTUAL EXECUTION:\nexit_code={execution.exit_code}\n"
            f"stdout={execution.stdout[:2000]}\nstderr={execution.stderr[:2000]}"
        )

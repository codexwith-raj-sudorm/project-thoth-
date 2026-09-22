"""Parallel multi-model synthesis and verifier-as-judge Arena mode."""

from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass
from typing import Any, Callable
from uuid import uuid4

import httpx

from .adapters.base import ModelAdapter
from .exceptions import SecurityViolationException, TokenBudgetExceeded
from .eventlog import SessionLogger
from .executor import execute_code
from .orchestrator import Orchestrator, RunResult
from .prompts import CODER_SYSTEM
from .schemas import (
    ArenaCandidateConfig,
    ArenaVerifierOutput,
    CoderOutput,
    ExecutionResult,
)
from .security import count_tautological_assertions, scan_code

ARENA_JUDGE_SYSTEM = """You are Thoth's Arena Judge. Compare every labeled candidate independently.
Rank only the candidates provided. Score correctness, completeness, and code quality from 1 to 3,
set total to the exact sum, and return strict JSON matching ArenaVerifierOutput. Runtime evidence is
authoritative: a non-zero exit or tautological test cannot receive correctness 3. Prefer meaningful
edge-case tests, typed readable code, and complete requirement coverage. Do not use markdown."""

AdapterFactory = Callable[[str, str], tuple[ModelAdapter, ModelAdapter]]
EventSink = Callable[[str, int | None, dict[str, Any]], None]


@dataclass
class CandidateResult:
    candidate_id: str
    provider: str
    model: str
    status: str
    code: str = ""
    thought_summary: str = ""
    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    tests_passed: bool = False
    tautological_tests: int = 0
    tokens: int = 0
    error: str = ""

    def public_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["stdout"] = self.stdout[:2000]
        value["stderr"] = self.stderr[:2000]
        return value


@dataclass
class ArenaResult:
    winner: str
    ranking: list[str]
    scores: list[dict[str, Any]]
    winner_rationale: str
    candidates: list[CandidateResult]
    total_tokens: int
    fallback: bool = False

    def public_dict(self) -> dict[str, Any]:
        return {
            "winner": self.winner,
            "ranking": self.ranking,
            "scores": self.scores,
            "winner_rationale": self.winner_rationale,
            "all_results": [item.public_dict() for item in self.candidates],
            "total_tokens": self.total_tokens,
            "fallback": self.fallback,
        }


async def default_roster() -> list[ArenaCandidateConfig]:
    """Build the roster from configured cloud keys and a reachable local Ollama."""
    import os

    roster: list[ArenaCandidateConfig] = []
    if os.getenv("GROQ_API_KEY"):
        roster.append(
            ArenaCandidateConfig(
                candidate_id="A", provider="groq", model="llama-3.1-8b-instant"
            )
        )
    if os.getenv("GEMINI_API_KEY"):
        roster.append(
            ArenaCandidateConfig(
                candidate_id="B", provider="gemini", model="gemini-2.5-flash"
            )
        )
    base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    try:
        async with httpx.AsyncClient(timeout=1.5) as client:
            response = await client.get(f"{base}/api/tags")
        if response.is_success:
            roster.append(
                ArenaCandidateConfig(
                    candidate_id="C", provider="ollama", model="qwen2.5-coder:7b"
                )
            )
    except httpx.HTTPError:
        pass
    return roster


def _generate_candidate(
    config: ArenaCandidateConfig,
    prompt: str,
    adapter_factory: AdapterFactory,
    allow_filesystem: bool,
    allow_network: bool,
) -> CandidateResult:
    try:
        coder, _ = adapter_factory(config.provider, config.model)
        output = coder.generate_structured(CODER_SYSTEM, prompt, CoderOutput)
        code = output.implementation_code
        scan_code(code, allow_filesystem=allow_filesystem, allow_network=allow_network)
        tautologies = count_tautological_assertions(code)
        execution: ExecutionResult = execute_code(code, 5.0)
        tokens = coder.last_usage.prompt_tokens + coder.last_usage.completion_tokens
        return CandidateResult(
            candidate_id=config.candidate_id,
            provider=config.provider,
            model=config.model,
            status="completed",
            code=code,
            thought_summary=output.thought_process.strip().replace("\n", " ")[:240],
            exit_code=execution.exit_code,
            stdout=execution.stdout,
            stderr=execution.stderr,
            tests_passed=execution.exit_code == 0 and tautologies == 0,
            tautological_tests=tautologies,
            tokens=tokens,
        )
    except SecurityViolationException as exc:
        return CandidateResult(
            candidate_id=config.candidate_id,
            provider=config.provider,
            model=config.model,
            status="blocked",
            error=str(exc),
        )
    except Exception as exc:
        return CandidateResult(
            candidate_id=config.candidate_id,
            provider=config.provider,
            model=config.model,
            status="error",
            error=str(exc),
        )


def _judge_prompt(prompt: str, candidates: list[CandidateResult]) -> str:
    sections = [f"ORIGINAL REQUIREMENT:\n{prompt}"]
    for item in candidates:
        sections.append(
            f"CANDIDATE {item.candidate_id} ({item.provider}/{item.model})\n"
            f"CODE:\n{item.code}\n"
            f"EXECUTION: exit_code={item.exit_code}; tests_passed={item.tests_passed}; "
            f"tautological_assertions={item.tautological_tests}\n"
            f"STDOUT:\n{item.stdout[:2000]}\nSTDERR:\n{item.stderr[:2000]}"
        )
    return "\n\n".join(sections)


async def run_arena(
    prompt: str,
    candidates: list[ArenaCandidateConfig],
    adapter_factory: AdapterFactory,
    *,
    allow_filesystem: bool = False,
    allow_network: bool = False,
    token_budget: int = 40_000,
    event_sink: EventSink | None = None,
    log_directory=None,
) -> ArenaResult:
    if not candidates:
        raise ValueError("no Arena providers are configured")
    logger = SessionLogger(str(uuid4()), log_directory)
    logger.event(turn_index=0, agent_role="orchestrator", prompt=prompt, mode="arena")

    def emit(kind: str, **data: Any) -> None:
        if event_sink:
            event_sink(kind, None, data)

    emit(
        "arena_start",
        candidates=[f"{item.candidate_id}:{item.provider}" for item in candidates],
    )
    tasks = [
        asyncio.create_task(
            asyncio.to_thread(
                _generate_candidate,
                config,
                prompt,
                adapter_factory,
                allow_filesystem,
                allow_network,
            )
        )
        for config in candidates
    ]
    results: list[CandidateResult] = []
    for pending in asyncio.as_completed(tasks):
        result = await pending
        results.append(result)
        emit(
            "candidate_done",
            candidate_id=result.candidate_id,
            provider=result.provider,
            model=result.model,
            status=result.status,
            tests_passed=result.tests_passed,
            error=result.error,
        )
    order = {config.candidate_id: index for index, config in enumerate(candidates)}
    results.sort(key=lambda item: order[item.candidate_id])
    for index, item in enumerate(results, 1):
        logger.event(
            turn_index=index,
            agent_role="coder",
            prompt_tokens=0,
            completion_tokens=item.tokens,
            ast_scan_result="pass" if item.status == "completed" else (item.error or item.status),
            tautological_tests_count=item.tautological_tests,
            execution_exit_code=item.exit_code,
            execution_stdout=item.stdout,
            execution_stderr=item.stderr or item.error,
            candidate_id=item.candidate_id,
            provider=item.provider,
            model=item.model,
        )
    completed = [item for item in results if item.status == "completed"]
    total_tokens = sum(item.tokens for item in results)
    if total_tokens > token_budget:
        raise TokenBudgetExceeded(f"Arena token budget exceeded ({total_tokens} > {token_budget})")

    if len(completed) < 2:
        if not completed:
            errors = "; ".join(f"{item.candidate_id}: {item.error}" for item in results)
            raise RuntimeError(f"all Arena candidates failed: {errors}")
        # A one-provider Arena has no meaningful ranking. Run the full standard
        # Coder/Verifier loop with that provider so normal approval rules still apply.
        sole = completed[0]
        coder, verifier = adapter_factory(sole.provider, sole.model)
        orchestrator = Orchestrator(
            coder,
            verifier,
            max_turns=3,
            token_budget=token_budget,
            allow_filesystem=allow_filesystem,
            allow_network=allow_network,
            log_directory=log_directory,
            event_sink=event_sink,
        )
        standard: RunResult = await asyncio.to_thread(orchestrator.run, prompt)
        sole.code = standard.code
        sole.exit_code = standard.execution.exit_code
        sole.stdout = standard.execution.stdout
        sole.stderr = standard.execution.stderr
        sole.tests_passed = standard.execution.exit_code == 0
        total_tokens += standard.total_tokens
        if total_tokens > token_budget:
            raise TokenBudgetExceeded(
                f"Arena token budget exceeded ({total_tokens} > {token_budget})"
            )
        return ArenaResult(
            winner=sole.candidate_id,
            ranking=[sole.candidate_id],
            scores=[],
            winner_rationale="Only one provider remained; completed the standard Coder/Verifier workflow.",
            candidates=results,
            total_tokens=total_tokens,
            fallback=True,
        )

    emit("judge_start")
    judgment: ArenaVerifierOutput | None = None
    judge: ModelAdapter | None = None
    judge_errors: list[str] = []
    for judge_candidate in completed:
        try:
            _, judge_adapter = adapter_factory(judge_candidate.provider, judge_candidate.model)
            judgment = await asyncio.to_thread(
                judge_adapter.generate_structured,
                ARENA_JUDGE_SYSTEM,
                _judge_prompt(prompt, completed),
                ArenaVerifierOutput,
            )
            expected_ids = {item.candidate_id for item in completed}
            if set(judgment.ranking) != expected_ids:
                raise ValueError("judge ranking does not match the completed candidate set")
            judge = judge_adapter
            break
        except Exception as exc:
            judge_errors.append(f"{judge_candidate.provider}: {exc}")
    if judgment is None or judge is None:
        raise RuntimeError(f"all Arena judge providers failed: {'; '.join(judge_errors)}")
    total_tokens += judge.last_usage.prompt_tokens + judge.last_usage.completion_tokens
    if total_tokens > token_budget:
        raise TokenBudgetExceeded(f"Arena token budget exceeded ({total_tokens} > {token_budget})")
    score_data = [score.model_dump() for score in judgment.scores]
    emit(
        "judge_done",
        ranking=list(judgment.ranking),
        scores=score_data,
        winner_rationale=judgment.winner_rationale,
    )
    logger.event(
        turn_index=len(results),
        agent_role="verifier",
        prompt_tokens=judge.last_usage.prompt_tokens,
        completion_tokens=judge.last_usage.completion_tokens,
        cumulative_session_tokens=total_tokens,
        verifier_status="APPROVED",
        verifier_implementation_critique=judgment.winner_rationale,
        verifier_test_critique=f"Arena ranking: {', '.join(judgment.ranking)}",
        arena_winner=judgment.ranking[0],
    )
    return ArenaResult(
        winner=judgment.ranking[0],
        ranking=list(judgment.ranking),
        scores=score_data,
        winner_rationale=judgment.winner_rationale,
        candidates=results,
        total_tokens=total_tokens,
    )

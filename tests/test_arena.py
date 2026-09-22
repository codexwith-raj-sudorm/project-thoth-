import asyncio
from pathlib import Path

import pytest
from pydantic import BaseModel

from thoth.adapters.base import ModelAdapter
from thoth.arena import run_arena
from thoth.schemas import (
    ArenaCandidateConfig,
    ArenaVerifierOutput,
    CandidateScore,
    CoderOutput,
    Usage,
    VerifierOutput,
)


class ArenaStub(ModelAdapter):
    def __init__(self, label: str):
        self.label = label
        self.last_usage = Usage(prompt_tokens=5, completion_tokens=7)

    def _generate_text(self, system_prompt: str, user_prompt: str, schema: type[BaseModel]) -> str:
        raise NotImplementedError

    def generate_structured(self, system_prompt, user_prompt, schema):
        if schema is CoderOutput:
            value = ord(self.label[0])
            return CoderOutput(
                thought_process=f"Candidate {self.label}",
                implementation_code=(
                    f"def candidate() -> int:\n    return {value}\n\n"
                    f"assert candidate() == {value}"
                ),
            )
        if schema is ArenaVerifierOutput:
            return ArenaVerifierOutput(
                ranking=["B", "A"],
                scores=[
                    CandidateScore(candidate_id="A", correctness=3, completeness=2, code_quality=2, total=7, notes="Good"),
                    CandidateScore(candidate_id="B", correctness=3, completeness=3, code_quality=3, total=9, notes="Best"),
                ],
                winner_rationale="Candidate B has the strongest edge-case coverage.",
            )
        raise AssertionError(schema)


def adapter_factory(provider: str, model: str):
    return ArenaStub(provider), ArenaStub("judge")


def test_arena_runs_candidates_and_judges(tmp_path: Path):
    events = []
    roster = [
        ArenaCandidateConfig(candidate_id="A", provider="groq", model="a"),
        ArenaCandidateConfig(candidate_id="B", provider="gemini", model="b"),
    ]
    result = asyncio.run(
        run_arena(
            "Build a deterministic candidate function",
            roster,
            adapter_factory,
            log_directory=tmp_path,
            event_sink=lambda kind, turn, data: events.append((kind, data)),
        )
    )
    assert result.winner == "B"
    assert result.ranking == ["B", "A"]
    assert len(result.candidates) == 2
    assert all(candidate.tests_passed for candidate in result.candidates)
    assert result.total_tokens == 36
    kinds = [kind for kind, _ in events]
    assert kinds.count("candidate_done") == 2
    assert kinds[-2:] == ["judge_start", "judge_done"]
    assert list(tmp_path.glob("session_*.jsonl"))


def test_arena_verifier_rejects_inconsistent_total_and_ranking():
    with pytest.raises(ValueError, match="total"):
        CandidateScore(
            candidate_id="A", correctness=3, completeness=3, code_quality=3, total=8, notes="wrong"
        )
    with pytest.raises(ValueError, match="ranking"):
        ArenaVerifierOutput(
            ranking=["A"],
            scores=[
                CandidateScore(
                    candidate_id="B", correctness=1, completeness=1, code_quality=1, total=3, notes="bad"
                )
            ],
            winner_rationale="none",
        )


class FallbackStub(ModelAdapter):
    def __init__(self):
        self.last_usage = Usage(prompt_tokens=1, completion_tokens=1)

    def _generate_text(self, system_prompt, user_prompt, schema):
        raise NotImplementedError

    def generate_structured(self, system_prompt, user_prompt, schema):
        if schema is CoderOutput:
            return CoderOutput(
                thought_process="fallback",
                implementation_code="def ok() -> bool:\n    return True\n\nassert ok() is True",
            )
        if schema is VerifierOutput:
            return VerifierOutput(
                status="APPROVED",
                implementation_critique="Correct.",
                test_critique="Meaningful behavior check.",
            )
        raise AssertionError(schema)


def test_arena_falls_back_when_only_one_provider_survives(tmp_path: Path):
    def factory(provider, model):
        if provider == "gemini":
            raise ConnectionError("provider offline")
        return FallbackStub(), FallbackStub()

    roster = [
        ArenaCandidateConfig(candidate_id="A", provider="groq", model="a"),
        ArenaCandidateConfig(candidate_id="B", provider="gemini", model="b"),
    ]
    result = asyncio.run(
        run_arena("Build a typed boolean helper function", roster, factory, log_directory=tmp_path)
    )
    assert result.fallback is True
    assert result.winner == "A"
    assert result.candidates[1].status == "error"
    assert result.candidates[0].tests_passed is True

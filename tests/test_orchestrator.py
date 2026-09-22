from pathlib import Path

import pytest
from pydantic import BaseModel

from thoth.adapters.base import ModelAdapter
from thoth.exceptions import MaxTurnsExceeded, StagnantLoopException
from thoth.orchestrator import Orchestrator
from thoth.schemas import CoderOutput, ExecutionResult, Usage, VerifierOutput


class QueueAdapter(ModelAdapter):
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.last_usage = Usage(prompt_tokens=2, completion_tokens=3)
        self.prompts = []

    def _generate_text(self, system_prompt: str, user_prompt: str, schema: type[BaseModel]) -> str:
        raise NotImplementedError

    def generate_structured(self, system_prompt, user_prompt, schema):
        self.prompts.append(user_prompt)
        return next(self.outputs)


def approved():
    return VerifierOutput(
        status="APPROVED", implementation_critique="correct", test_critique="meaningful"
    )


def test_happy_path(tmp_path: Path):
    coder = QueueAdapter([CoderOutput(thought_process="p", implementation_code="assert 2 + 2 == 4")])
    verifier = QueueAdapter([approved()])
    result = Orchestrator(coder, verifier, log_directory=tmp_path).run("check arithmetic")
    assert result.turns == 1
    assert result.execution.exit_code == 0
    assert result.log_path.exists()
    assert len(result.log_path.read_text().splitlines()) == 3


def test_runtime_overrules_approval(tmp_path: Path):
    code1 = "raise RuntimeError('x')\nassert 1 == 2"
    code2 = "raise RuntimeError('y')\nassert 1 == 2"
    coder = QueueAdapter([
        CoderOutput(thought_process="p", implementation_code=code1),
        CoderOutput(thought_process="p", implementation_code=code2),
    ])
    verifier = QueueAdapter([approved(), approved()])
    with pytest.raises(MaxTurnsExceeded):
        Orchestrator(coder, verifier, max_turns=2, log_directory=tmp_path).run("work")
    assert "Runtime exited non-zero" in coder.prompts[1]


def test_tautology_skips_verifier(tmp_path: Path):
    coder = QueueAdapter([CoderOutput(thought_process="p", implementation_code="assert True")])
    verifier = QueueAdapter([])
    with pytest.raises(MaxTurnsExceeded):
        Orchestrator(coder, verifier, max_turns=1, log_directory=tmp_path).run("work")
    assert verifier.prompts == []


def test_stagnation(tmp_path: Path):
    output = CoderOutput(thought_process="p", implementation_code="assert 2 + 2 == 5")
    coder = QueueAdapter([output, output])
    revision = VerifierOutput(
        status="NEEDS_REVISION", implementation_critique="wrong", test_critique="fine", suggested_fix="fix"
    )
    verifier = QueueAdapter([revision])
    with pytest.raises(StagnantLoopException) as error:
        Orchestrator(coder, verifier, max_turns=3, log_directory=tmp_path).run("work")
    assert error.value.partial_code

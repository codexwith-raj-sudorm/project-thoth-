import pytest
from pydantic import BaseModel

from thoth.adapters.base import ModelAdapter
from thoth.exceptions import SchemaValidationException
from thoth.schemas import CoderOutput


class Stub(ModelAdapter):
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = 0

    def _generate_text(self, system_prompt: str, user_prompt: str, schema: type[BaseModel]) -> str:
        self.calls += 1
        return next(self.responses)


def test_adapter_repairs_once():
    adapter = Stub(["oops", '{"thought_process":"fixed","implementation_code":"pass"}'])
    assert adapter.generate_structured("s", "u", CoderOutput).thought_process == "fixed"
    assert adapter.calls == 2


def test_adapter_stops_after_repair():
    adapter = Stub(["oops", "still oops"])
    with pytest.raises(SchemaValidationException):
        adapter.generate_structured("s", "u", CoderOutput)
    assert adapter.calls == 2

import pytest

from thoth.exceptions import SchemaValidationException
from thoth.parsing import parse_structured, strip_markdown_fence
from thoth.schemas import CoderOutput, VerifierOutput


def test_fence_sanitization_and_parse():
    raw = '```json\n{"thought_process":"p","implementation_code":"pass"}\n```'
    assert strip_markdown_fence(raw).startswith("{")
    assert parse_structured(raw, CoderOutput).implementation_code == "pass"


def test_root_text_and_extra_fields_rejected():
    with pytest.raises(SchemaValidationException):
        parse_structured('["not", "object"]', CoderOutput)
    with pytest.raises(SchemaValidationException):
        parse_structured('{"thought_process":"p","implementation_code":"x","extra":1}', CoderOutput)


def test_revision_requires_suggested_fix():
    with pytest.raises(ValueError):
        VerifierOutput(
            status="NEEDS_REVISION", implementation_critique="bad", test_critique="bad"
        )

"""Model output sanitizing and strict schema parsing."""

import json
import re
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from .exceptions import SchemaValidationException

T = TypeVar("T", bound=BaseModel)
_FENCE = re.compile(r"^\s*```(?:json)?\s*\n?(.*?)\n?```\s*$", re.DOTALL | re.IGNORECASE)


def strip_markdown_fence(text: str) -> str:
    match = _FENCE.match(text)
    return match.group(1).strip() if match else text.strip()


def parse_structured(text: str, schema: type[T]) -> T:
    cleaned = strip_markdown_fence(text)
    try:
        payload = json.loads(cleaned)
        if not isinstance(payload, dict):
            raise ValueError("root JSON value must be an object")
        return schema.model_validate(payload)
    except (json.JSONDecodeError, ValidationError, ValueError) as exc:
        raise SchemaValidationException(str(exc)) from exc

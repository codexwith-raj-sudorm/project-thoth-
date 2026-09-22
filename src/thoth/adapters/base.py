"""Provider-independent model adapter contract and repair flow."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TypeVar

from pydantic import BaseModel

from ..exceptions import SchemaValidationException
from ..parsing import parse_structured
from ..schemas import Usage

T = TypeVar("T", bound=BaseModel)


class ModelAdapter(ABC):
    """Base interface used by the mediator; agents never call each other."""

    last_usage: Usage = Usage()

    @abstractmethod
    def _generate_text(self, system_prompt: str, user_prompt: str, schema: type[BaseModel]) -> str:
        """Return provider text, preferably a JSON object."""

    def generate_structured(self, system_prompt: str, user_prompt: str, schema: type[T]) -> T:
        self.last_usage = Usage()
        raw = self._generate_text(system_prompt, user_prompt, schema)
        initial_usage = self.last_usage
        try:
            return parse_structured(raw, schema)
        except SchemaValidationException as first_error:
            repair = (
                f"{user_prompt}\n\nYour previous response failed strict schema validation:\n"
                f"{first_error}\nReturn ONLY one valid JSON object matching this JSON schema:\n"
                f"{schema.model_json_schema()}"
            )
            raw = self._generate_text(system_prompt, repair, schema)
            self.last_usage = Usage(
                prompt_tokens=initial_usage.prompt_tokens + self.last_usage.prompt_tokens,
                completion_tokens=initial_usage.completion_tokens + self.last_usage.completion_tokens,
            )
            try:
                return parse_structured(raw, schema)
            except SchemaValidationException as second_error:
                raise SchemaValidationException(
                    f"provider failed initial and repair validation: {second_error}"
                ) from second_error

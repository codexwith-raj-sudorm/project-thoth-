"""Domain exceptions raised by Thoth."""


class ThothError(Exception):
    """Base class for expected Thoth failures."""


class MaxTurnsExceeded(ThothError):
    def __init__(self, message: str, partial_code: str = "") -> None:
        super().__init__(message)
        self.partial_code = partial_code


class StagnantLoopException(ThothError):
    def __init__(self, message: str, partial_code: str = "") -> None:
        super().__init__(message)
        self.partial_code = partial_code


class SecurityViolationException(ThothError):
    pass


class RateLimitExceeded(ThothError):
    pass


class SchemaValidationException(ThothError):
    pass


class ConcurrentRunError(ThothError):
    pass


class TokenBudgetExceeded(ThothError):
    pass

"""Shared Pydantic model configuration for HRRR-VLM models."""

from pydantic import ConfigDict

DEFAULT_MODEL_CONFIG = ConfigDict(
    str_strip_whitespace=True,
    extra="forbid",
    validate_assignment=True,
    arbitrary_types_allowed=False,
    validate_by_name=True,
)

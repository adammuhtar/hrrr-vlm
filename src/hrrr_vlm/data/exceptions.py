"""Custom exceptions for the data module."""


class HRRRVLMError(Exception):
    """Base exception for all HRRR VLM package errors."""


class DataLoadError(HRRRVLMError):
    """Raised when weather data cannot be loaded."""


class WeatherDataError(HRRRVLMError):
    """Raised when there are issues with weather data processing."""


class ValidationError(HRRRVLMError):
    """Raised when data validation fails."""


class ConfigurationError(HRRRVLMError):
    """Raised when there are configuration issues."""

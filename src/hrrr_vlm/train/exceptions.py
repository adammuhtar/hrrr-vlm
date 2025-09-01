"""Custom exceptions for the train module."""


class DataError(Exception):
    """Custom exception for image-caption data-related errors."""


class ModelInitError(Exception):
    """Custom exception for model initialisation-related errors."""


class ModelTrainingError(Exception):
    """Custom exception for model training-related errors."""

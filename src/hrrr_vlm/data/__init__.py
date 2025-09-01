"""Module for generating and loading image-caption data pairs for HRRR-VLM."""

from .caption_generator import CaptionGenerator
from .config import (
    DataGeneratorConfig,
    ModelConfig,
    RegionConfig,
    WeatherVariableConfig,
)
from .constants import (
    LATITUDE_MAX,
    LATITUDE_MIN,
    LONGITUDE_MAX,
    LONGITUDE_MIN,
    MODEL_CONFIGS,
    REGIONS,
    WEATHER_VARIABLES,
)
from .exceptions import (
    ConfigurationError,
    DataLoadError,
    HRRRVLMError,
    ValidationError,
    WeatherDataError,
)
from .generator import WeatherDataGenerator
from .loader import (
    DEFAULT_VARIABLES,
    REGION_MODEL_PAIRS,
    HRRRImageCaptionDataset,
    get_first_mondays,
)
from .models import CaptionMetadata, TrainingRecord, WeatherStatistics
from .weather_data import WeatherDataService

__all__ = [
    "DEFAULT_VARIABLES",
    "LATITUDE_MAX",
    "LATITUDE_MIN",
    "LONGITUDE_MAX",
    "LONGITUDE_MIN",
    "MODEL_CONFIGS",
    "REGIONS",
    "REGION_MODEL_PAIRS",
    "WEATHER_VARIABLES",
    "CaptionGenerator",
    "CaptionMetadata",
    "ConfigurationError",
    "DataGeneratorConfig",
    "DataLoadError",
    "HRRRImageCaptionDataset",
    "HRRRVLMError",
    "ModelConfig",
    "RegionConfig",
    "TrainingRecord",
    "ValidationError",
    "WeatherDataError",
    "WeatherDataGenerator",
    "WeatherDataService",
    "WeatherStatistics",
    "WeatherVariableConfig",
    "get_first_mondays",
]

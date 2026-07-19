"""Data models for the HRRR image-caption dataset components."""

from datetime import datetime
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field, NonNegativeFloat, NonNegativeInt

from hrrr_vlm.utils.model_config import DEFAULT_MODEL_CONFIG


class WeatherStatistics(BaseModel):
    """Statistical summary of weather data."""

    model_config: ClassVar[ConfigDict] = DEFAULT_MODEL_CONFIG
    min: float = Field(default=..., description="Minimum value")
    max: float = Field(default=..., description="Maximum value")
    mean: float = Field(default=..., description="Mean value")
    std: NonNegativeFloat = Field(default=..., description="Standard deviation")
    median: float = Field(default=..., description="Median value")
    range: float = Field(default=..., description="Range (max - min)")
    var: float = Field(default=..., description="Variance")
    skewness: float = Field(default=..., description="Skewness")
    kurtosis: float = Field(default=..., description="Kurtosis")
    percentile_25: float = Field(default=..., description="25th percentile")
    percentile_75: float = Field(default=..., description="75th percentile")
    percentile_90: float = Field(default=..., description="90th percentile")
    percentile_95: float = Field(default=..., description="95th percentile")
    iqr: float = Field(default=..., description="Interquartile range")
    mad: NonNegativeFloat = Field(default=..., description="Median absolute deviation")
    coeff_variation: NonNegativeFloat = Field(
        ..., description="Coefficient of variation"
    )
    count_valid: NonNegativeInt = Field(
        default=..., description="Number of valid data points"
    )
    count_missing: NonNegativeInt = Field(
        ..., description="Number of missing data points"
    )
    variable: str = Field(default=..., description="Variable name")
    unit: str = Field(default=..., description="Unit of measurement")
    description: str = Field(default=..., description="Variable description")
    valid_time: str = Field(default=..., description="Valid time string")
    model: str = Field(default=..., description="Weather model name")
    forecast_hour: NonNegativeInt = Field(default=..., description="Forecast hour")
    grib_name: str = Field(default=..., description="GRIB parameter name")
    domain: str = Field(default=..., description="Model domain")
    region: str = Field(default="", description="Region name")


class CaptionMetadata(BaseModel):
    """Metadata for generated captions."""

    model_config: ClassVar[ConfigDict] = DEFAULT_MODEL_CONFIG
    image_filename: str = Field(default=..., description="Image filename")
    caption_files: list[str] = Field(
        default=..., description="List of caption filenames"
    )
    sample_id: str = Field(default=..., description="Unique sample identifier")
    variable: str = Field(default=..., description="Weather variable")
    model: str = Field(default=..., description="Weather model")
    date: str = Field(default=..., description="Date string")
    region: str | None = Field(default=None, description="Region name")
    domain: str | None = Field(default=None, description="Model domain")
    generation_time: datetime = Field(
        default_factory=datetime.now, description="Caption generation timestamp"
    )
    stats: WeatherStatistics | None = Field(
        default=None, description="Weather statistics for the variable"
    )


class TrainingRecord(BaseModel):
    """Record for image-text training dataset."""

    model_config: ClassVar[ConfigDict] = DEFAULT_MODEL_CONFIG
    image_path: str = Field(default=..., description="Path to image file")
    caption: str = Field(default=..., description="Generated caption")
    sample_id: str = Field(default=..., description="Unique sample identifier")
    variable: str = Field(default=..., description="Weather variable")
    model: str = Field(default=..., description="Weather model")
    date: str = Field(default=..., description="Date string")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )

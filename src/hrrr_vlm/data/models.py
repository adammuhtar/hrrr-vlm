"""Data models for the HRRR image-caption dataset components."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, NonNegativeFloat, NonNegativeInt


class WeatherStatistics(BaseModel):
    """Statistical summary of weather data."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
        validate_assignment=True,
        arbitrary_types_allowed=False,
        validate_by_name=True,
    )
    min: float = Field(..., description="Minimum value")
    max: float = Field(..., description="Maximum value")
    mean: float = Field(..., description="Mean value")
    std: NonNegativeFloat = Field(..., description="Standard deviation")
    median: float = Field(..., description="Median value")
    range: float = Field(..., description="Range (max - min)")
    var: float = Field(..., description="Variance")
    skewness: float = Field(..., description="Skewness")
    kurtosis: float = Field(..., description="Kurtosis")
    percentile_25: float = Field(..., description="25th percentile")
    percentile_75: float = Field(..., description="75th percentile")
    percentile_90: float = Field(..., description="90th percentile")
    percentile_95: float = Field(..., description="95th percentile")
    iqr: float = Field(..., description="Interquartile range")
    mad: NonNegativeFloat = Field(..., description="Median absolute deviation")
    coeff_variation: NonNegativeFloat = Field(
        ..., description="Coefficient of variation"
    )
    count_valid: NonNegativeInt = Field(..., description="Number of valid data points")
    count_missing: NonNegativeInt = Field(
        ..., description="Number of missing data points"
    )
    variable: str = Field(..., description="Variable name")
    unit: str = Field(..., description="Unit of measurement")
    description: str = Field(..., description="Variable description")
    valid_time: str = Field(..., description="Valid time string")
    model: str = Field(..., description="Weather model name")
    forecast_hour: NonNegativeInt = Field(..., description="Forecast hour")
    grib_name: str = Field(..., description="GRIB parameter name")
    domain: str = Field(..., description="Model domain")
    region: str = Field(default="", description="Region name")


class CaptionMetadata(BaseModel):
    """Metadata for generated captions."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
        validate_assignment=True,
        arbitrary_types_allowed=False,
        validate_by_name=True,
    )
    image_filename: str = Field(..., description="Image filename")
    caption_files: list[str] = Field(..., description="List of caption filenames")
    sample_id: str = Field(..., description="Unique sample identifier")
    variable: str = Field(..., description="Weather variable")
    model: str = Field(..., description="Weather model")
    date: str = Field(..., description="Date string")
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

    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
        validate_assignment=True,
        arbitrary_types_allowed=False,
        validate_by_name=True,
    )
    image_path: str = Field(..., description="Path to image file")
    caption: str = Field(..., description="Generated caption")
    sample_id: str = Field(..., description="Unique sample identifier")
    variable: str = Field(..., description="Weather variable")
    model: str = Field(..., description="Weather model")
    date: str = Field(..., description="Date string")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )

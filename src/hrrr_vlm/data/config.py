"""Configuration models for the HRRR VLM package using Pydantic v2."""

import xarray as xr
from pydantic import BaseModel, ConfigDict, Field, PositiveInt
from pydantic.functional_validators import field_validator

# Constants for validation
LATITUDE_MIN = -90
LATITUDE_MAX = 90
LONGITUDE_MIN = -180
LONGITUDE_MAX = 180
BOUNDS_LENGTH = 4


class WeatherVariableConfig(BaseModel):
    """Configuration for a weather variable used in HRRR VLM data generation.

    Attributes:
        variable (`str`): Identifier for the weather variable.
        search_string (`str`): GRIB search string to locate the variable in data
            files.
        unit (`str`): Unit of measurement for the variable.
        description (`str`): Human-readable description of the variable.
        conversion_factor (`float`, optional): Linear conversion factor to apply
            to the data.
        conversion_offset (`float`, optional): Linear conversion offset to apply
            to the data.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
        validate_assignment=True,
        arbitrary_types_allowed=False,
        validate_by_name=True,
    )
    variable: str = Field(default=..., description="Variable identifier")
    search_string: str = Field(default=..., description="GRIB search string")
    unit: str = Field(default=..., description="Unit of measurement")
    description: str = Field(default=..., description="Human-readable description")
    cmap: str = Field(default=..., description="Matplotlib colormap name")
    conversion_factor: float | None = Field(
        default=None, description="Linear conversion factor"
    )
    conversion_offset: float | None = Field(
        default=None, description="Linear conversion offset (applied first)"
    )

    def convert_data(self, data: xr.DataArray) -> xr.DataArray:
        """Apply conversion to data.

        Args:
            data (`xr.DataArray`): Input data array

        Returns:
            `xr.DataArray`: Converted data array
        """
        converted = data
        if self.conversion_offset is not None:
            converted += self.conversion_offset
        if self.conversion_factor is not None:
            converted *= self.conversion_factor
        return converted


class RegionConfig(BaseModel):
    """Configuration for a geographical region used in HRRR VLM data generation.

    Attributes:
        name (`str`): Name of the region.
        bounds (`list[float]`): Region bounds as [lon_min, lon_max, lat_min, lat_max].
        description (`str`): Optional description of the region.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
        validate_assignment=True,
        arbitrary_types_allowed=False,
        validate_by_name=True,
    )
    name: str = Field(..., description="Region name")
    bounds: list[float] = Field(
        ..., description="Region bounds as [lon_min, lon_max, lat_min, lat_max]"
    )
    description: str = Field(default="", description="Region description")

    @field_validator("bounds", mode="after")
    @classmethod
    def validate_bounds(cls, v: list[float]) -> list[float]:
        """Validate region bounds.

        Args:
            v (`list[float]`): List of bounds [lon_min, lon_max, lat_min, lat_max].

        Returns:
            `list[float]`: Validated bounds list.

        Raises:
            ValueError: If bounds do not meet the required criteria.
        """
        if len(v) != BOUNDS_LENGTH:
            msg = (
                "Bounds must contain exactly 4 values: [lon_min, lon_max, lat_min, "
                "lat_max]"
            )
            raise ValueError(msg)

        lon_min, lon_max, lat_min, lat_max = v

        if lon_min >= lon_max:
            msg = "lon_min must be less than lon_max"
            raise ValueError(msg)
        if lat_min >= lat_max:
            msg = "lat_min must be less than lat_max"
            raise ValueError(msg)
        if not (LATITUDE_MIN <= lat_min <= LATITUDE_MAX) or not (
            LATITUDE_MIN <= lat_max <= LATITUDE_MAX
        ):
            msg = f"Latitude values must be between {LATITUDE_MIN} and {LATITUDE_MAX}"
            raise ValueError(msg)
        if not (LONGITUDE_MIN <= lon_min <= LONGITUDE_MAX) or not (
            LONGITUDE_MIN <= lon_max <= LONGITUDE_MAX
        ):
            msg = (
                f"Longitude values must be between {LONGITUDE_MIN} and {LONGITUDE_MAX}"
            )
            raise ValueError(msg)

        return v


class ModelConfig(BaseModel):
    """Configuration for a weather mode used in HRRR VLM data generation.

    Attributes:
        name (`str`): Name of the model.
        product (`str`): Product type of the model.
        domain (`str`): Domain of the model.
        default_region (`str`): Default region name for the model.
        available_regions (`list[str]`): List of available region names.
        map_resolution (`str`): Map resolution for the model, e.g., "10m", "50m",
            "110m".
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
        validate_assignment=True,
        arbitrary_types_allowed=False,
        validate_by_name=True,
    )
    name: str = Field(..., description="Model name")
    product: str = Field(..., description="Model product")
    domain: str = Field(..., description="Model domain")
    default_region: str = Field(..., description="Default region name")
    available_regions: list[str] = Field(..., description="Available region names")
    map_resolution: str = Field(default="10m", description="Map resolution")

    @field_validator("map_resolution", mode="after")
    @classmethod
    def validate_resolution(cls, v: str) -> str:
        """Validate map resolution.

        Args:
            v (`str`): Map resolution string.

        Returns:
            `str`: Validated resolution string.

        Raises:
            ValueError: If the resolution is not one of the valid options.
        """
        valid_resolutions = {"10m", "50m", "110m"}
        if v not in valid_resolutions:
            msg = f"Map resolution must be one of {valid_resolutions}"
            raise ValueError(msg)
        return v


class DataGeneratorConfig(BaseModel):
    """Configuration for the data generator in HRRR VLM.

    Attributes:
        output_dir (`str`): Directory where generated data will be saved.
        cache_dir (`str`): Directory for caching threshold data.
        enable_json_logging (`bool`): Whether to enable JSON logging.
        log_level (`str`): Logging level (e.g., "DEBUG", "INFO", "WARNING",
            "ERROR", "CRITICAL").
        default_dpi (`int`): Default DPI for generated images.
        default_figsize (`tuple[int, int]`): Default figure size for plots.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
        validate_assignment=True,
        arbitrary_types_allowed=False,
        validate_by_name=True,
    )
    output_dir: str = Field(..., description="Output directory path")
    cache_dir: str = Field(default="threshold_cache", description="Cache directory")
    enable_json_logging: bool = Field(default=False, description="Enable JSON logging")
    log_level: str = Field(default="INFO", description="Logging level")
    default_dpi: PositiveInt = Field(default=150, description="Default image DPI")
    default_figsize: tuple[PositiveInt, PositiveInt] = Field(
        default=(12, 8), description="Default figure size"
    )

    @field_validator("log_level", mode="after")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate logging level.

        Args:
            v (`str`): Logging level to validate.

        Returns:
            Validated logging level.

        Raises:
            ValueError: If the logging level is not one of the valid options.
        """
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        v_upper = v.upper()
        if v_upper not in valid_levels:
            msg = f"Log level must be one of {valid_levels}"
            raise ValueError(msg)
        return v_upper

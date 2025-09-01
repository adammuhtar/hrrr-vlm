"""Constants for the HRRR VLM package."""

from hrrr_vlm.data.config import ModelConfig, RegionConfig, WeatherVariableConfig

# Geographical constants
LATITUDE_MIN = -90
LATITUDE_MAX = 90
LONGITUDE_MIN = -180
LONGITUDE_MAX = 180

# US climate regions with lon/lat extents
REGIONS: dict[str, RegionConfig] = {
    "Alaska": RegionConfig(
        name="Alaska", bounds=[-180, -130, 55, 68], description="Alaska region"
    ),
    "Northeast": RegionConfig(
        name="Northeast",
        bounds=[-80, -67, 39, 47.5],
        description="Northeastern United States",
    ),
    "Northern Rockies and Plains": RegionConfig(
        name="Northern Rockies and Plains",
        bounds=[-114.5, -96, 41, 49.5],
        description="Northern Rockies and Plains region",
    ),
    "Northwest": RegionConfig(
        name="Northwest", bounds=[-128, -112, 41.5, 49], description="Pacific Northwest"
    ),
    "Ohio Valley": RegionConfig(
        name="Ohio Valley",
        bounds=[-95.5, -78.5, 35, 41.5],
        description="Ohio River Valley",
    ),
    "South": RegionConfig(
        name="South",
        bounds=[-106, -88.5, 25, 40],
        description="South Central United States",
    ),
    "Southeast": RegionConfig(
        name="Southeast",
        bounds=[-88, -75, 25, 39],
        description="Southeastern United States",
    ),
    "Southwest": RegionConfig(
        name="Southwest",
        bounds=[-115, -102, 31.5, 42],
        description="Southwestern United States",
    ),
    "Upper Midwest": RegionConfig(
        name="Upper Midwest",
        bounds=[-97.5, -82.5, 40, 48.5],
        description="Upper Midwest",
    ),
    "West": RegionConfig(
        name="West",
        bounds=[-125, -113.5, 32.5, 42],
        description="Western United States",
    ),
}

# Model-specific configurations
MODEL_CONFIGS: dict[str, ModelConfig] = {
    "hrrr": ModelConfig(
        name="hrrr",
        product="sfc",
        domain="conus",
        default_region="Continental US",
        available_regions=[
            "Northeast",
            "Upper Midwest",
            "Ohio Valley",
            "Southeast",
            "Northern Rockies and Plains",
            "South",
            "Southwest",
            "Northwest",
            "West",
        ],
        map_resolution="10m",
    ),
    "hrrrak": ModelConfig(
        name="hrrrak",
        product="sfc",
        domain="alaska",
        default_region="Alaska",
        available_regions=["Alaska"],
        map_resolution="50m",
    ),
}

# Weather variable configurations
WEATHER_VARIABLES: dict[str, WeatherVariableConfig] = {
    "temperature": WeatherVariableConfig(
        variable="temperature",
        search_string="TMP:2 m above",
        unit="°C",
        description="2-meter temperature",
        cmap="RdYlBu_r",
        conversion_offset=-273.15,  # Kelvin to Celsius
    ),
    "wind_speed": WeatherVariableConfig(
        variable="wind_speed",
        search_string="WIND:10 m above",
        unit="km/h",
        description="10-meter wind speed",
        cmap="viridis",
        conversion_factor=3.6,  # m/s to km/h
    ),
    "precipitation": WeatherVariableConfig(
        variable="precipitation",
        search_string="APCP:surface",
        unit="mm",
        description="accumulated precipitation",
        cmap="Blues",
        conversion_factor=1.0,
        conversion_offset=0.0,
    ),
    "humidity": WeatherVariableConfig(
        variable="humidity",
        search_string="RH:2 m above",
        unit="%",
        description="2-meter relative humidity",
        cmap="BuGn",
        conversion_factor=1.0,
    ),
}

"""Comprehensive unit tests for weather_data.py module."""

from typing import Any
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from hrrr_vlm.data.config import WeatherVariableConfig
from hrrr_vlm.data.constants import MODEL_CONFIGS
from hrrr_vlm.data.exceptions import DataLoadError, WeatherDataError
from hrrr_vlm.data.models import WeatherStatistics
from hrrr_vlm.data.weather_data import WeatherDataService


@pytest.fixture
def temp_variable_config() -> WeatherVariableConfig:
    """Create a temperature variable configuration for testing.

    Returns:
        `WeatherVariableConfig`: Configuration for 2-meter temperature.
    """
    return WeatherVariableConfig(
        variable="temperature",
        search_string="TMP:2 m above",
        unit="°C",
        description="2-meter temperature",
        cmap="RdYlBu_r",
        conversion_offset=-273.15,
    )


@pytest.fixture
def mock_logger() -> Mock:
    """Create a mock logger for testing.

    Returns:
        `Mock`: A mock logger instance.
    """
    return Mock()


@pytest.fixture
def weather_service(
    temp_variable_config: WeatherVariableConfig, mock_logger: Mock
) -> WeatherDataService:
    """Create a WeatherDataService instance for testing.

    Args:
        temp_variable_config (WeatherVariableConfig): Configuration for temperature
            variable.
        mock_logger (Mock): Mock logger instance.

    Returns:
        `WeatherDataService`: An instance of WeatherDataService.
    """
    return WeatherDataService(temp_variable_config, mock_logger)


@pytest.fixture
def mock_herbie() -> Mock:
    """Create a mock Herbie instance.

    Returns:
        `Mock`: A mock Herbie instance.
    """
    herbie = Mock()
    herbie.date = "2023-06-15"
    herbie.model = "hrrr"
    herbie.product = "sfc"
    herbie.fxx = 0
    herbie.crs = Mock()
    return herbie


@pytest.fixture
def sample_data_array() -> xr.DataArray:
    """Create a sample data array for testing.

    Returns:
        `xr.DataArray`: A sample xarray DataArray with temperature data.
    """
    rng = np.random.default_rng(42)
    data = rng.normal(15, 5, (40, 50))

    return xr.DataArray(
        data,
        dims=["latitude", "longitude"],
        coords={
            "latitude": np.linspace(20, 50, 40),
            "longitude": np.linspace(-130, -60, 50),
        },
        attrs={"units": "°C"},
    )


@pytest.fixture
def sample_dataset() -> xr.Dataset:
    """Create a sample dataset for testing.

    Returns:
        `xr.Dataset`: A sample xarray Dataset with temperature data.
    """
    rng = np.random.default_rng(42)
    lon_data = np.linspace(-130, -60, 50)
    lat_data = np.linspace(20, 50, 40)
    temp_data = rng.normal(288, 10, (40, 50))

    ds = xr.Dataset(
        {"temperature": (["latitude", "longitude"], temp_data)},
        coords={
            "longitude": lon_data,
            "latitude": lat_data,
            "valid_time": pd.Timestamp("2023-06-15T12:00:00"),
        },
        attrs={"model": "HRRR"},
    )

    ds["temperature"].attrs = {"GRIB_name": "TMP", "units": "K"}

    return ds


class TestWeatherDataServiceInit:
    """Test WeatherDataService initialisation."""

    def test_init_with_logger(
        self, temp_variable_config: Any, mock_logger: Mock
    ) -> None:
        """Test initialisation with provided logger."""
        service = WeatherDataService(temp_variable_config, mock_logger)

        assert service.variable_config == temp_variable_config
        assert service.logger == mock_logger
        assert service.herbie is None
        assert service.current_model is None

    @patch("hrrr_vlm.data.weather_data.configure_logger")
    def test_init_without_logger(
        self, mock_configure_logger: Any, temp_variable_config: Any
    ) -> None:
        """Test initialisation without provided logger."""
        mock_logger = Mock()
        mock_configure_logger.return_value = mock_logger

        service = WeatherDataService(temp_variable_config)

        assert service.variable_config == temp_variable_config
        assert service.logger == mock_logger
        assert service.herbie is None
        assert service.current_model is None
        mock_configure_logger.assert_called_once_with(
            enable_json=False, log_level="INFO"
        )


class TestWeatherDataServiceLoadData:
    """Test WeatherDataService load_data method."""

    def test_load_data_unsupported_model(
        self, weather_service: WeatherDataService
    ) -> None:
        """Test loading data with unsupported model."""
        with pytest.raises(DataLoadError) as exc_info:
            weather_service.load_data("20230615", "invalid_model")

        assert "Unsupported model: invalid_model" in str(exc_info.value)
        assert "Available: ['hrrr', 'hrrrak']" in str(exc_info.value)

    @patch("hrrr_vlm.data.weather_data.Herbie")
    def test_load_data_success(
        self,
        mock_herbie_class: Any,
        weather_service: WeatherDataService,
        mock_herbie: Mock,
    ) -> None:
        """Test successful data loading."""
        mock_herbie_class.return_value = mock_herbie

        result = weather_service.load_data("20230615", "hrrr", 0)

        assert result is True
        assert weather_service.herbie == mock_herbie
        assert weather_service.current_model == "hrrr"

        mock_herbie_class.assert_called_once_with(
            "20230615", model="hrrr", product=MODEL_CONFIGS["hrrr"].product, fxx=0
        )

    @patch("hrrr_vlm.data.weather_data.Herbie")
    def test_load_data_herbie_exception(
        self, mock_herbie_class: Any, weather_service: WeatherDataService
    ) -> None:
        """Test data loading when Herbie raises an exception."""
        mock_herbie_class.side_effect = Exception("Connection failed")

        with pytest.raises(DataLoadError) as exc_info:
            weather_service.load_data("20230615", "hrrr")

        assert "Failed to load weather data: Connection failed" in str(exc_info.value)
        assert weather_service.herbie is None
        assert weather_service.current_model is None


class TestWeatherDataServiceGetAvailableVariables:
    """Test WeatherDataService get_available_variables method."""

    def test_get_available_variables_no_data_loaded(
        self, weather_service: WeatherDataService
    ) -> None:
        """Test getting variables when no data is loaded."""
        with pytest.raises(WeatherDataError) as exc_info:
            weather_service.get_available_variables()

        assert "No weather data loaded. Call load_data first." in str(exc_info.value)

    def test_get_available_variables_success(
        self, weather_service: WeatherDataService, mock_herbie: Mock
    ) -> None:
        """Test successful retrieval of available variables."""
        mock_inventory = pd.DataFrame(
            {
                "variable": ["TMP", "UGRD", "VGRD", "RH"],
                "level": [
                    "2 m above ground",
                    "10 m above ground",
                    "10 m above ground",
                    "2 m above ground",
                ],
            }
        )
        mock_herbie.inventory.return_value = mock_inventory
        weather_service.herbie = mock_herbie

        variables = weather_service.get_available_variables()

        assert variables == ["TMP", "UGRD", "VGRD", "RH"]
        mock_herbie.inventory.assert_called_once()

    def test_get_available_variables_inventory_exception(
        self, weather_service: WeatherDataService, mock_herbie: Mock
    ) -> None:
        """Test getting variables when inventory fails."""
        mock_herbie.inventory.side_effect = Exception("Inventory failed")
        weather_service.herbie = mock_herbie

        with pytest.raises(WeatherDataError) as exc_info:
            weather_service.get_available_variables()

        assert "Could not retrieve variable inventory: Inventory failed" in str(
            exc_info.value
        )


class TestWeatherDataServiceGetDataArray:
    """Test WeatherDataService get_data_array method."""

    def test_get_data_array_no_data_loaded(
        self, weather_service: WeatherDataService
    ) -> None:
        """Test getting data array when no data is loaded."""
        with pytest.raises(WeatherDataError) as exc_info:
            weather_service.get_data_array()

        assert "No weather data loaded. Call load_data first." in str(exc_info.value)

    @patch("hrrr_vlm.data.config.WeatherVariableConfig.convert_data")
    def test_get_data_array_success(
        self,
        mock_convert_data: Any,
        weather_service: WeatherDataService,
        mock_herbie: Mock,
        sample_dataset: xr.Dataset,
        sample_data_array: xr.DataArray,
    ) -> None:
        """Test successful data array retrieval."""
        mock_herbie.xarray.return_value = sample_dataset
        weather_service.herbie = mock_herbie
        weather_service.current_model = "hrrr"

        # Mock the variable config convert_data method
        mock_convert_data.return_value = sample_data_array

        result = weather_service.get_data_array()

        assert result is sample_data_array
        mock_herbie.xarray.assert_called_once_with(
            weather_service.variable_config.search_string
        )
        mock_convert_data.assert_called_once()

    def test_get_data_array_no_data_variables(
        self, weather_service: WeatherDataService, mock_herbie: Mock
    ) -> None:
        """Test getting data array when dataset has no data variables."""
        empty_dataset = xr.Dataset()
        mock_herbie.xarray.return_value = empty_dataset
        weather_service.herbie = mock_herbie

        with pytest.raises(WeatherDataError) as exc_info:
            weather_service.get_data_array()

        assert "No data variables found in dataset" in str(exc_info.value)


class TestWeatherDataServiceCalculateStatistics:
    """Test WeatherDataService calculate_statistics method."""

    def test_calculate_statistics_no_data_loaded(
        self, weather_service: WeatherDataService
    ) -> None:
        """Test calculating statistics when no data is loaded."""
        with pytest.raises(WeatherDataError) as exc_info:
            weather_service.calculate_statistics()

        assert "No weather data loaded. Call load_data first." in str(exc_info.value)

    @patch.object(WeatherDataService, "get_data_array")
    def test_calculate_statistics_success(
        self,
        mock_get_data_array: Any,
        weather_service: WeatherDataService,
        mock_herbie: Mock,
        sample_dataset: xr.Dataset,
        sample_data_array: xr.DataArray,
    ) -> None:
        """Test successful statistics calculation."""
        mock_get_data_array.return_value = sample_data_array
        mock_herbie.xarray.return_value = sample_dataset
        mock_herbie.fxx = 0
        weather_service.herbie = mock_herbie
        weather_service.current_model = "hrrr"

        with (
            patch.object(WeatherDataService, "_calculate_skewness") as mock_skew,
            patch.object(WeatherDataService, "_calculate_kurtosis") as mock_kurt,
        ):
            mock_skew.return_value = xr.DataArray(0.1)
            mock_kurt.return_value = xr.DataArray(-0.5)

            stats = weather_service.calculate_statistics()

        assert isinstance(stats, WeatherStatistics)
        assert stats.variable == weather_service.variable_config.variable
        assert stats.unit == weather_service.variable_config.unit
        assert stats.description == weather_service.variable_config.description
        assert stats.model == "HRRR"
        assert stats.forecast_hour == 0
        assert not stats.region  # Empty string


class TestWeatherDataServiceGenerateMap:
    """Test WeatherDataService generate_map method."""

    def test_generate_map_no_data_loaded(
        self, weather_service: WeatherDataService
    ) -> None:
        """Test generating map when no data is loaded."""
        with pytest.raises(WeatherDataError) as exc_info:
            weather_service.generate_map()

        assert "No weather data loaded. Call load_data first." in str(exc_info.value)

    @patch.object(WeatherDataService, "calculate_statistics")
    def test_generate_map_exception(
        self,
        mock_calc_stats: Any,  # noqa: ARG002
        weather_service: WeatherDataService,
        mock_herbie: Mock,
    ) -> None:
        """Test map generation when an exception occurs."""
        # Mock herbie.xarray to return a dataset but cause failure in the generation
        mock_dataset = Mock()
        mock_dataset.__next__ = Mock(
            side_effect=TypeError("'Mock' object is not iterable")
        )
        mock_herbie.xarray.return_value = mock_dataset

        weather_service.herbie = mock_herbie

        with pytest.raises(WeatherDataError) as exc_info:
            weather_service.generate_map()

        assert "Failed to generate weather map:" in str(exc_info.value)


class TestWeatherDataServiceHelperMethods:
    """Test WeatherDataService private helper methods."""

    def test_get_region_display_name_none(
        self, weather_service: WeatherDataService
    ) -> None:
        """Test getting display name for None region."""
        weather_service.current_model = "hrrr"

        result = weather_service._get_region_display_name(None)

        assert result == MODEL_CONFIGS["hrrr"].default_region

    def test_get_region_display_name_alaska(
        self, weather_service: WeatherDataService
    ) -> None:
        """Test getting display name for Alaska."""
        result = weather_service._get_region_display_name("Alaska")

        assert result == "Alaska"

    def test_get_region_display_name_continental_us(
        self, weather_service: WeatherDataService
    ) -> None:
        """Test getting display name for Continental US."""
        result = weather_service._get_region_display_name("Continental US")

        assert result == "Continental US"

    def test_get_region_display_name_other_region(
        self, weather_service: WeatherDataService
    ) -> None:
        """Test getting display name for other regions."""
        result = weather_service._get_region_display_name("Northeast")

        assert result == "US Northeast region"

    def test_calculate_skewness(self, sample_data_array: xr.DataArray) -> None:
        """Test skewness calculation."""
        result = WeatherDataService._calculate_skewness(sample_data_array)

        assert isinstance(result, xr.DataArray)
        # The actual value will depend on the data, just check it's a reasonable number
        assert -5 < result.to_numpy() < 5

    def test_calculate_kurtosis(self, sample_data_array: xr.DataArray) -> None:
        """Test kurtosis calculation."""
        result = WeatherDataService._calculate_kurtosis(sample_data_array)

        assert isinstance(result, xr.DataArray)
        # The actual value will depend on the data, just check it's a reasonable number
        assert -5 < result.to_numpy() < 5


class TestWeatherDataServiceIntegration:
    """Test WeatherDataService integration scenarios."""

    def test_error_propagation_chain(self, weather_service: WeatherDataService) -> None:
        """Test that errors propagate correctly through method chain."""
        # No data loaded - should fail at each step
        with pytest.raises(WeatherDataError):
            weather_service.get_available_variables()

        with pytest.raises(WeatherDataError):
            weather_service.get_data_array()

        with pytest.raises(WeatherDataError):
            weather_service.calculate_statistics()

        with pytest.raises(WeatherDataError):
            weather_service.generate_map()

    @patch("hrrr_vlm.data.weather_data.Herbie")
    def test_model_switching(
        self,
        mock_herbie_class: Any,
        weather_service: WeatherDataService,
        mock_herbie: Mock,
    ) -> None:
        """Test switching between different weather models."""
        mock_herbie_class.return_value = mock_herbie

        # Load HRRR data
        weather_service.load_data("20230615", "hrrr")
        assert weather_service.current_model == "hrrr"

        # Switch to HRRRAK
        weather_service.load_data("20230615", "hrrrak")
        assert weather_service.current_model == "hrrrak"

        # Verify multiple calls to Herbie
        assert mock_herbie_class.call_count == 2

    @patch("hrrr_vlm.data.weather_data.Herbie")
    def test_different_forecast_hours(
        self,
        mock_herbie_class: Any,
        weather_service: WeatherDataService,
        mock_herbie: Mock,
    ) -> None:
        """Test loading data with different forecast hours."""
        mock_herbie_class.return_value = mock_herbie

        # Test different forecast hours
        for fxx in [0, 6, 12, 18]:
            weather_service.load_data("20230615", "hrrr", fxx)
            mock_herbie_class.assert_called_with(
                "20230615", model="hrrr", product="sfc", fxx=fxx
            )

    def test_variable_config_flexibility(self, mock_logger: Mock) -> None:
        """Test service works with different variable configurations."""
        configs = [
            WeatherVariableConfig(
                variable="temperature",
                search_string="TMP:2 m above",
                unit="°C",
                description="Temperature",
                cmap="RdYlBu_r",
                conversion_offset=-273.15,
            ),
            WeatherVariableConfig(
                variable="wind_speed",
                search_string="WIND:10 m above",
                unit="km/h",
                description="Wind Speed",
                cmap="viridis",
                conversion_factor=3.6,
            ),
            WeatherVariableConfig(
                variable="humidity",
                search_string="RH:2 m above",
                unit="%",
                description="Relative Humidity",
                cmap="Blues",
            ),
        ]

        for config in configs:
            service = WeatherDataService(config, mock_logger)
            assert service.variable_config == config
            assert service.variable_config.variable == config.variable

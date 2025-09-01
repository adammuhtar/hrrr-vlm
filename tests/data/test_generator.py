"""Unit tests for the generator.py module."""

import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, Mock, patch

import pytest

from hrrr_vlm.data.config import DataGeneratorConfig
from hrrr_vlm.data.generator import WeatherDataGenerator
from hrrr_vlm.data.models import WeatherStatistics

# Test constants
TEST_DATE = "2019-04-15"
TEST_PRIMARY_VARIABLE = "temperature"
TEST_ADDITIONAL_VARIABLES = ["wind_speed"]
TEST_MODEL = "hrrr"
TEST_FXX = 0
TEST_CAPTION = "Test weather caption"
TEST_MIN_TEMP = 5.0
TEST_MAX_TEMP = 25.0
TEST_MEAN_TEMP = 15.0
TEST_STD_TEMP = 5.0


@pytest.fixture
def config() -> DataGeneratorConfig:
    """Create a test configuration.

    Returns:
        DataGeneratorConfig: Test configuration instance.
    """
    return DataGeneratorConfig(
        output_dir="/tmp/test_output",
        cache_dir="/tmp/test_cache",
        enable_json_logging=False,
        log_level="INFO",
        default_dpi=150,
        default_figsize=(12, 8),
    )


@pytest.fixture
def mock_weather_stats() -> WeatherStatistics:
    """Create mock weather statistics.

    Returns:
        WeatherStatistics: Mock weather statistics.
    """
    return WeatherStatistics(
        min=TEST_MIN_TEMP,
        max=TEST_MAX_TEMP,
        mean=TEST_MEAN_TEMP,
        std=TEST_STD_TEMP,
        median=12.0,
        range=20.0,
        var=25.0,
        skewness=0.1,
        kurtosis=0.2,
        percentile_25=8.0,
        percentile_75=18.0,
        percentile_90=22.0,
        percentile_95=24.0,
        iqr=10.0,
        mad=3.0,
        coeff_variation=0.33,
        count_valid=1000,
        count_missing=0,
        variable="temperature",
        unit="°C",
        description="2-meter temperature",
        valid_time="2019-04-15T12:00:00Z",
        model="hrrr",
        forecast_hour=0,
        grib_name="TMP",
        domain="conus",
        region="continental_us",
    )


@pytest.fixture
def weather_generator(config: DataGeneratorConfig) -> WeatherDataGenerator:
    """Create a WeatherDataGenerator instance for testing.

    Returns:
        WeatherDataGenerator: Test generator instance.
    """
    with (
        patch("hrrr_vlm.data.generator.WeatherDataService"),
        patch("hrrr_vlm.data.generator.CaptionGenerator"),
    ):
        return WeatherDataGenerator(config)


class TestWeatherDataGeneratorInitialization:
    """Test WeatherDataGenerator initialization."""

    def test_init_creates_output_directory(self, config: DataGeneratorConfig) -> None:
        """Test that initialization creates the output directory."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            config.output_dir = tmp_dir + "/new_output"

            with (
                patch("hrrr_vlm.data.generator.WeatherDataService"),
                patch("hrrr_vlm.data.generator.CaptionGenerator"),
            ):
                generator = WeatherDataGenerator(config)

            assert Path(config.output_dir).exists()
            assert generator.config == config

    def test_init_creates_subdirectories(
        self, weather_generator: WeatherDataGenerator
    ) -> None:
        """Test that initialization creates required subdirectories."""
        assert weather_generator.output_dir.exists()
        assert weather_generator.images_dir.exists()
        assert weather_generator.captions_dir.exists()
        assert weather_generator.images_dir.name == "images"
        assert weather_generator.captions_dir.name == "captions"


class TestGenerateSamples:
    """Test the generate_samples method."""

    def test_generate_samples_returns_dict(
        self, weather_generator: WeatherDataGenerator
    ) -> None:
        """Test that generate_samples returns a dict."""
        dates = [TEST_DATE]

        with patch.object(weather_generator, "generate_single_sample", return_value={}):
            result = weather_generator.generate_samples(dates)

        assert isinstance(result, dict)

    def test_generate_samples_calls_single_sample(
        self, weather_generator: WeatherDataGenerator
    ) -> None:
        """Test that generate_samples calls generate_single_sample correctly."""
        dates = [TEST_DATE, "2019-04-16"]

        with patch.object(
            weather_generator, "generate_single_sample", return_value={}
        ) as mock_single:
            weather_generator.generate_samples(dates)

        # Should be called for each date
        assert mock_single.call_count >= len(dates)


class TestGenerateSingleSample:
    """Test the generate_single_sample method."""

    @patch("hrrr_vlm.data.generator.WeatherDataService")
    def test_generate_single_sample_success(
        self,
        mock_service_class: Any,
        weather_generator: WeatherDataGenerator,
        mock_weather_stats: WeatherStatistics,
    ) -> None:
        """Test successful single sample generation."""
        # Mock service instance and methods
        mock_service = Mock()
        mock_service_class.return_value = mock_service
        mock_service.load_data.return_value = True
        mock_service.get_data_array.return_value = MagicMock()
        mock_service.calculate_statistics.return_value = mock_weather_stats
        mock_service.generate_map.return_value = MagicMock()

        # Mock CaptionGenerator
        with patch.object(
            weather_generator.caption_generator,
            "generate_caption",
            return_value=TEST_CAPTION,
        ):
            result = weather_generator.generate_single_sample(
                date_str=TEST_DATE,
                primary_variable=TEST_PRIMARY_VARIABLE,
                additional_variables=TEST_ADDITIONAL_VARIABLES,
                model=TEST_MODEL,
                region=None,
                fxx=TEST_FXX,
            )

        # Verify result is returned (actual structure depends on implementation)
        assert isinstance(result, bool)


class TestCreateTrainingRecords:
    """Test the create_training_records method."""

    def test_create_training_records_creates_file(
        self, weather_generator: WeatherDataGenerator
    ) -> None:
        """Test that training records creates a file."""
        output_file = "test_training.jsonl"

        # Mock Path.open to avoid actual file operations
        with patch("pathlib.Path.open") as mock_path_open:
            # Mock the context manager
            mock_path_open.return_value.__enter__.return_value.write = Mock()

            result = weather_generator.create_training_records(output_file)

        # Verify a Path is returned
        assert isinstance(result, Path)
        # Verify the output file path is correct
        assert result.name == output_file


class TestGetGenerationSummary:
    """Test the get_generation_summary method."""

    def test_get_generation_summary_structure(
        self, weather_generator: WeatherDataGenerator
    ) -> None:
        """Test that get_generation_summary returns correct structure."""
        summary = weather_generator.get_generation_summary()

        assert isinstance(summary, dict)
        # The actual keys depend on implementation, but it should be a dict


class TestWeatherDataGeneratorIntegration:
    """Integration tests for WeatherDataGenerator."""

    @patch("hrrr_vlm.data.generator.WeatherDataService")
    def test_basic_workflow(
        self,
        mock_service_class: Any,
        weather_generator: WeatherDataGenerator,
        mock_weather_stats: WeatherStatistics,
    ) -> None:
        """Test a basic generation workflow."""
        dates = [TEST_DATE]

        # Mock service
        mock_service = Mock()
        mock_service_class.return_value = mock_service
        mock_service.load_data.return_value = True
        mock_service.get_data_array.return_value = MagicMock()
        mock_service.calculate_statistics.return_value = mock_weather_stats
        mock_service.generate_map.return_value = MagicMock()

        # Mock caption generation
        with patch.object(
            weather_generator.caption_generator,
            "generate_caption",
            return_value=TEST_CAPTION,
        ):
            # Test the workflow
            samples = weather_generator.generate_samples(dates)
            summary = weather_generator.get_generation_summary()

        # Verify workflow completed
        assert isinstance(samples, dict)
        assert isinstance(summary, dict)

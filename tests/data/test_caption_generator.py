"""Unit tests for the CaptionGenerator class."""

from unittest.mock import Mock

import pytest

from hrrr_vlm.data.caption_generator import CaptionGenerator
from hrrr_vlm.data.models import WeatherStatistics


@pytest.fixture
def base_temp_stats() -> WeatherStatistics:
    """Create a base WeatherStatistics object for testing.

    Returns:
        `WeatherStatistics`: A base weather statistics object with typical
            temperature values.
    """
    return WeatherStatistics(
        min=5.0,
        max=25.0,
        mean=15.0,
        std=5.0,
        median=14.0,
        range=20.0,
        var=25.0,
        skewness=0.1,
        kurtosis=2.5,
        percentile_25=10.0,
        percentile_75=20.0,
        percentile_90=23.0,
        percentile_95=24.0,
        iqr=10.0,
        mad=4.0,
        coeff_variation=0.33,
        count_valid=1000,
        count_missing=0,
        variable="temperature",
        unit="°C",
        description="Temperature at 2m above ground",
        valid_time="Mon Apr 15 12:00:00 2019 UTC",
        model="HRRR",
        forecast_hour=0,
        grib_name="TMP:2 m above ground",
        domain="CONUS",
        region="Continental US",
    )


@pytest.fixture
def caption_generator() -> CaptionGenerator:
    """Create a CaptionGenerator instance for testing.

    Returns:
        `CaptionGenerator`: An instance of the CaptionGenerator class.
    """
    return CaptionGenerator()


class TestCaptionGeneratorInitialisation:
    """Test suite for CaptionGenerator initialisation."""

    def test_init_with_default_logger(self) -> None:
        """Test CaptionGenerator initialisation with default logger."""
        generator = CaptionGenerator()
        assert generator.logger is not None

    def test_init_with_custom_logger(self) -> None:
        """Test CaptionGenerator initialisation with custom logger."""
        mock_logger = Mock()
        generator = CaptionGenerator(logger=mock_logger)
        assert generator.logger == mock_logger


class TestDateFormatting:
    """Test suite for date formatting functionality."""

    def test_format_date_valid_input(self) -> None:
        """Test date formatting with valid input."""
        valid_time = "Mon Apr 15 12:00:00 2019 UTC"
        result = CaptionGenerator._format_date(valid_time)
        assert result == "15 12:00:00 2019"

    def test_format_date_invalid_input(self) -> None:
        """Test date formatting with invalid input."""
        invalid_time = "invalid"
        result = CaptionGenerator._format_date(invalid_time)
        assert result == "unknown date"

    def test_format_date_none_input(self) -> None:
        """Test date formatting with None input."""
        result = CaptionGenerator._format_date(None)
        assert result == "unknown date"


class TestOpeningGeneration:
    """Test suite for opening sentence generation."""

    def test_create_opening_continental_us(
        self, caption_generator: CaptionGenerator, base_temp_stats: WeatherStatistics
    ) -> None:
        """Test opening creation for Continental US."""
        date_str = "15 12:00:00 2019"
        result = caption_generator.create_opening(base_temp_stats, date_str)
        # Now expecting correct season (mid spring for April)
        assert result == "Continental US, mid spring, 15 12:00:00 2019"

    def test_create_opening_alaska(
        self, caption_generator: CaptionGenerator, base_temp_stats: WeatherStatistics
    ) -> None:
        """Test opening creation for Alaska."""
        alaska_stats = base_temp_stats.model_copy()
        alaska_stats.region = "Alaska"
        date_str = "15 12:00:00 2019"
        result = caption_generator.create_opening(alaska_stats, date_str)
        assert result == "Alaska, mid spring, 15 12:00:00 2019"

    def test_create_opening_no_region(
        self, caption_generator: CaptionGenerator, base_temp_stats: WeatherStatistics
    ) -> None:
        """Test opening creation with no region specified."""
        no_region_stats = base_temp_stats.model_copy()
        no_region_stats.region = (
            ""  # Use empty string instead of None due to Pydantic validation
        )
        date_str = "15 12:00:00 2019"
        result = caption_generator.create_opening(no_region_stats, date_str)
        assert result == "Continental US, mid spring, 15 12:00:00 2019"


class TestRegionDescriptors:
    """Test suite for region descriptor functionality."""

    def test_get_region_descriptor_known_regions(self) -> None:
        """Test region descriptor for known regions."""
        assert CaptionGenerator._get_region_descriptor("Northeast") == "Northeastern US"
        assert CaptionGenerator._get_region_descriptor("Southwest") == "Southwestern US"
        assert (
            CaptionGenerator._get_region_descriptor("Ohio Valley")
            == "Ohio River Valley"
        )

    def test_get_region_descriptor_unknown_region(self) -> None:
        """Test region descriptor for unknown regions."""
        result = CaptionGenerator._get_region_descriptor("Unknown Region")
        assert result == "Unknown Region region"


class TestSeasonDescriptors:
    """Test suite for season descriptor functionality."""

    def test_get_season_descriptor_winter_months(self) -> None:
        """Test season descriptor for winter months."""
        # Now should correctly parse the month portion
        assert (
            CaptionGenerator._get_season_descriptor("Mon Dec 15 12:00:00 2019 UTC")
            == "early winter"
        )
        assert (
            CaptionGenerator._get_season_descriptor("Mon Jan 15 12:00:00 2019 UTC")
            == "mid winter"
        )
        assert (
            CaptionGenerator._get_season_descriptor("Mon Feb 15 12:00:00 2019 UTC")
            == "late winter"
        )

    def test_get_season_descriptor_spring_months(self) -> None:
        """Test season descriptor for spring months."""
        assert (
            CaptionGenerator._get_season_descriptor("Mon Mar 15 12:00:00 2019 UTC")
            == "early spring"
        )
        assert (
            CaptionGenerator._get_season_descriptor("Mon Apr 15 12:00:00 2019 UTC")
            == "mid spring"
        )
        assert (
            CaptionGenerator._get_season_descriptor("Mon May 15 12:00:00 2019 UTC")
            == "late spring"
        )

    def test_get_season_descriptor_summer_months(self) -> None:
        """Test season descriptor for summer months."""
        assert (
            CaptionGenerator._get_season_descriptor("Mon Jun 15 12:00:00 2019 UTC")
            == "early summer"
        )
        assert (
            CaptionGenerator._get_season_descriptor("Mon Jul 15 12:00:00 2019 UTC")
            == "mid summer"
        )
        assert (
            CaptionGenerator._get_season_descriptor("Mon Aug 15 12:00:00 2019 UTC")
            == "late summer"
        )

    def test_get_season_descriptor_autumn_months(self) -> None:
        """Test season descriptor for autumn months."""
        assert (
            CaptionGenerator._get_season_descriptor("Mon Sep 15 12:00:00 2019 UTC")
            == "early autumn"
        )
        assert (
            CaptionGenerator._get_season_descriptor("Mon Oct 15 12:00:00 2019 UTC")
            == "mid autumn"
        )
        assert (
            CaptionGenerator._get_season_descriptor("Mon Nov 15 12:00:00 2019 UTC")
            == "late autumn"
        )

    def test_get_season_descriptor_invalid_input(self) -> None:
        """Test season descriptor with invalid input."""
        result = CaptionGenerator._get_season_descriptor("invalid")
        assert result == "early summer"  # Default fallback


class TestTemperatureDescriptions:
    """Test suite for temperature description functionality."""

    def test_describe_temperature_cold_simple(
        self, base_temp_stats: WeatherStatistics
    ) -> None:
        """Test simple cold temperature description."""
        cold_stats = base_temp_stats.model_copy()
        cold_stats.mean = 5.0
        result = CaptionGenerator._describe_temperature(cold_stats, detailed=False)
        assert result == "Cold temperatures averaging 5.0°C"

    def test_describe_temperature_hot_simple(
        self, base_temp_stats: WeatherStatistics
    ) -> None:
        """Test simple hot temperature description."""
        hot_stats = base_temp_stats.model_copy()
        hot_stats.mean = 30.0
        result = CaptionGenerator._describe_temperature(hot_stats, detailed=False)
        assert result == "Hot temperatures averaging 30.0°C"

    def test_describe_temperature_mild_simple(
        self, base_temp_stats: WeatherStatistics
    ) -> None:
        """Test simple mild temperature description."""
        result = CaptionGenerator._describe_temperature(base_temp_stats, detailed=False)
        assert result == "Mild temperatures averaging 15.0°C"

    def test_describe_temperature_cold_detailed(
        self, base_temp_stats: WeatherStatistics
    ) -> None:
        """Test detailed cold temperature description."""
        cold_stats = base_temp_stats.model_copy()
        cold_stats.mean = 5.0
        cold_stats.min = -5.0
        cold_stats.max = 10.0
        result = CaptionGenerator._describe_temperature(cold_stats, detailed=True)
        expected = (
            "Temperature ranges from -5°C to 10°C across the region "
            "with cold conditions averaging 5°C"
        )
        assert result == expected

    def test_describe_temperature_hot_detailed(
        self, base_temp_stats: WeatherStatistics
    ) -> None:
        """Test detailed hot temperature description."""
        hot_stats = base_temp_stats.model_copy()
        hot_stats.mean = 30.0
        hot_stats.min = 25.0
        hot_stats.max = 35.0
        result = CaptionGenerator._describe_temperature(hot_stats, detailed=True)
        expected = (
            "Temperature ranges from 25°C to 35°C across the region "
            "with hot conditions averaging 30°C"
        )
        assert result == expected

    def test_describe_temperature_mild_detailed(
        self, base_temp_stats: WeatherStatistics
    ) -> None:
        """Test detailed mild temperature description."""
        result = CaptionGenerator._describe_temperature(base_temp_stats, detailed=True)
        expected = (
            "Temperature ranges from 5°C to 25°C across the region "
            "with mild conditions averaging 15°C"
        )
        assert result == expected


class TestWindDescriptions:
    """Test suite for wind description functionality."""

    @staticmethod
    def create_wind_stats(mean: float, max: float) -> WeatherStatistics:  # noqa: A002
        """Helper method to create wind statistics.

        Args:
            mean (`float`): Mean wind speed value.
            max (`float`): Maximum wind speed value.

        Returns:
            `WeatherStatistics`: Mock wind statistics.
        """
        return WeatherStatistics(
            min=0.0,
            max=max,
            mean=mean,
            std=2.0,
            median=mean,
            range=max,
            var=4.0,
            skewness=0.1,
            kurtosis=2.5,
            percentile_25=mean - 2,
            percentile_75=mean + 2,
            percentile_90=mean + 4,
            percentile_95=mean + 5,
            iqr=4.0,
            mad=2.0,
            coeff_variation=0.25,
            count_valid=1000,
            count_missing=0,
            variable="wind_speed",
            unit="km/h",
            description="Wind speed at 10m above ground",
            valid_time="Mon Apr 15 12:00:00 2019 UTC",
            model="HRRR",
            forecast_hour=0,
            grib_name="WIND:10 m above ground",
            domain="CONUS",
            region="Continental US",
        )

    def test_describe_wind_calm_simple(self) -> None:
        """Test simple calm wind description."""
        wind_stats = self.create_wind_stats(mean=0.0, max=1.0)
        result = CaptionGenerator.describe_wind(wind_stats, detailed=False)
        assert result == "Calm and no winds"

    def test_describe_wind_light_breeze_simple(self) -> None:
        """Test simple light breeze description."""
        wind_stats = self.create_wind_stats(mean=8.0, max=12.0)
        result = CaptionGenerator.describe_wind(wind_stats, detailed=False)
        assert result == "Light breeze averaging 8.0 km/h"

    def test_describe_wind_fresh_breeze_simple(self) -> None:
        """Test fresh breeze wind description for 30.0 km/h."""
        wind_stats = self.create_wind_stats(mean=30.0, max=45.0)
        result = CaptionGenerator.describe_wind(wind_stats, detailed=False)
        assert result == "Fresh breeze averaging 30.0 km/h"

    def test_describe_wind_hurricane_simple(self) -> None:
        """Test simple hurricane wind description."""
        wind_stats = self.create_wind_stats(mean=120.0, max=150.0)
        result = CaptionGenerator.describe_wind(wind_stats, detailed=False)
        assert result == "Hurricane averaging 120.0 km/h "

    def test_describe_wind_detailed(self) -> None:
        """Test detailed wind description."""
        wind_stats = self.create_wind_stats(mean=8.0, max=12.0)
        result = CaptionGenerator.describe_wind(wind_stats, detailed=True)
        expected = (
            "Light breeze conditions with average speeds of 8.0 km/h "
            "and maximum gusts up to 12.0km/h"
        )
        assert result == expected


class TestPrecipitationDescriptions:
    """Test suite for precipitation description functionality."""

    @staticmethod
    def create_precipitation_stats(mean: float, max: float) -> WeatherStatistics:  # noqa: A002
        """Helper method to create precipitation statistics.

        Args:
            mean (`float`): Mean precipitation value.
            max (`float`): Maximum precipitation value.

        Returns:
            `WeatherStatistics`: Mock precipitation statistics.
        """
        return WeatherStatistics(
            min=0.0,
            max=max,
            mean=mean,
            std=0.1,
            median=mean,
            range=max,
            var=0.01,
            skewness=0.5,
            kurtosis=3.0,
            percentile_25=mean - 0.05,
            percentile_75=mean + 0.05,
            percentile_90=mean + 0.1,
            percentile_95=mean + 0.15,
            iqr=0.1,
            mad=0.05,
            coeff_variation=0.5,
            count_valid=1000,
            count_missing=0,
            variable="precipitation",
            unit="mm",
            description="Total precipitation",
            valid_time="Mon Apr 15 12:00:00 2019 UTC",
            model="HRRR",
            forecast_hour=0,
            grib_name="APCP:surface",
            domain="CONUS",
            region="Continental US",
        )

    def test_describe_precipitation_low_simple(self) -> None:
        """Test simple low precipitation description."""
        precip_stats = self.create_precipitation_stats(mean=0.05, max=0.2)
        result = CaptionGenerator.describe_precipitation(precip_stats, detailed=False)
        assert result == "low precipitation"

    def test_describe_precipitation_moderate_simple(self) -> None:
        """Test simple moderate precipitation description."""
        precip_stats = self.create_precipitation_stats(mean=0.2, max=0.4)
        result = CaptionGenerator.describe_precipitation(precip_stats, detailed=False)
        assert result == "moderate precipitation"

    def test_describe_precipitation_heavy_simple(self) -> None:
        """Test simple heavy precipitation description."""
        precip_stats = self.create_precipitation_stats(mean=0.5, max=1.0)
        result = CaptionGenerator.describe_precipitation(precip_stats, detailed=False)
        assert result == "heavy precipitation"

    def test_describe_precipitation_detailed(self) -> None:
        """Test detailed precipitation description."""
        precip_stats = self.create_precipitation_stats(mean=0.2, max=0.4)
        result = CaptionGenerator.describe_precipitation(precip_stats, detailed=True)
        expected = (
            "Precipitation shows moderate activity with accumulations "
            "averaging 0.2mm and maximum 0.4mm"
        )
        assert result == expected


class TestHumidityDescriptions:
    """Test suite for humidity description functionality."""

    @staticmethod
    def create_humidity_stats(
        mean: float, min_val: float, max_val: float
    ) -> WeatherStatistics:
        """Helper method to create humidity statistics.

        Args:
            mean (`float`): Mean humidity value.
            min_val (`float`): Minimum humidity value.
            max_val (`float`): Maximum humidity value.

        Returns:
            `WeatherStatistics`: Mock humidity statistics.
        """
        return WeatherStatistics(
            min=min_val,
            max=max_val,
            mean=mean,
            std=5.0,
            median=mean,
            range=max_val - min_val,
            var=25.0,
            skewness=0.0,
            kurtosis=2.8,
            percentile_25=mean - 5,
            percentile_75=mean + 5,
            percentile_90=mean + 8,
            percentile_95=mean + 10,
            iqr=10.0,
            mad=4.0,
            coeff_variation=0.11,
            count_valid=1000,
            count_missing=0,
            variable="humidity",
            unit="%",
            description="Relative humidity",
            valid_time="Mon Apr 15 12:00:00 2019 UTC",
            model="HRRR",
            forecast_hour=0,
            grib_name="RH:2 m above ground",
            domain="CONUS",
            region="Continental US",
        )

    def test_describe_humidity_low_simple(self) -> None:
        """Test simple low humidity description."""
        humidity_stats = self.create_humidity_stats(
            mean=25.0, min_val=20.0, max_val=30.0
        )
        result = CaptionGenerator.describe_humidity(humidity_stats, detailed=False)
        assert result == "low humidity (25.0%)"

    def test_describe_humidity_moderate_simple(self) -> None:
        """Test simple moderate humidity description."""
        humidity_stats = self.create_humidity_stats(
            mean=45.0, min_val=40.0, max_val=50.0
        )
        result = CaptionGenerator.describe_humidity(humidity_stats, detailed=False)
        assert result == "moderate humidity (45.0%)"

    def test_describe_humidity_high_simple(self) -> None:
        """Test simple high humidity description."""
        humidity_stats = self.create_humidity_stats(
            mean=75.0, min_val=70.0, max_val=80.0
        )
        result = CaptionGenerator.describe_humidity(humidity_stats, detailed=False)
        assert result == "high humidity (75.0%)"

    def test_describe_humidity_detailed(self) -> None:
        """Test detailed humidity description."""
        humidity_stats = self.create_humidity_stats(
            mean=45.0, min_val=40.0, max_val=50.0
        )
        result = CaptionGenerator.describe_humidity(humidity_stats, detailed=True)
        expected = "Humidity levels 40-50% with moderate moisture content"
        assert result == expected


class TestVariableDescriptions:
    """Test suite for variable description functionality."""

    def test_describe_variable_wind_speed(
        self, caption_generator: CaptionGenerator
    ) -> None:
        """Test describing wind speed variable."""
        wind_stats = TestWindDescriptions.create_wind_stats(mean=8.0, max=12.0)
        result = caption_generator._describe_variable(
            "wind_speed", wind_stats, detailed=False
        )
        assert result == "Light breeze averaging 8.0 km/h"

    def test_describe_variable_precipitation(
        self, caption_generator: CaptionGenerator
    ) -> None:
        """Test describing precipitation variable."""
        precip_stats = TestPrecipitationDescriptions.create_precipitation_stats(
            mean=0.2, max=0.4
        )
        result = caption_generator._describe_variable(
            "precipitation", precip_stats, detailed=False
        )
        assert result == "moderate precipitation"

    def test_describe_variable_humidity(
        self, caption_generator: CaptionGenerator
    ) -> None:
        """Test describing humidity variable."""
        humidity_stats = TestHumidityDescriptions.create_humidity_stats(
            mean=45.0, min_val=40.0, max_val=50.0
        )
        result = caption_generator._describe_variable(
            "humidity", humidity_stats, detailed=False
        )
        assert result == "moderate humidity (45.0%)"

    def test_describe_variable_unknown(
        self, caption_generator: CaptionGenerator
    ) -> None:
        """Test describing unknown variable."""
        unknown_stats = TestWindDescriptions.create_wind_stats(mean=8.0, max=12.0)
        result = caption_generator._describe_variable(
            "unknown_var", unknown_stats, detailed=False
        )
        assert not result


class TestCaptionGeneration:
    """Test suite for full caption generation functionality."""

    def test_generate_caption_temperature_only(
        self, caption_generator: CaptionGenerator, base_temp_stats: WeatherStatistics
    ) -> None:
        """Test generating caption with only temperature data."""
        result = caption_generator.generate_caption(base_temp_stats, detailed=False)
        expected = (
            "Continental US, mid spring, 15 12:00:00 2019. "
            "Mild temperatures averaging 15.0°C."
        )
        assert result == expected

    def test_generate_caption_with_multi_variables(
        self, caption_generator: CaptionGenerator, base_temp_stats: WeatherStatistics
    ) -> None:
        """Test generating caption with multiple weather variables."""
        wind_stats = TestWindDescriptions.create_wind_stats(mean=8.0, max=12.0)
        precip_stats = TestPrecipitationDescriptions.create_precipitation_stats(
            mean=0.2, max=0.4
        )
        humidity_stats = TestHumidityDescriptions.create_humidity_stats(
            mean=45.0, min_val=40.0, max_val=50.0
        )

        multi_var_stats = {
            "wind_speed": wind_stats,
            "precipitation": precip_stats,
            "humidity": humidity_stats,
        }

        result = caption_generator.generate_caption(
            base_temp_stats, multi_var_stats=multi_var_stats, detailed=False
        )

        expected = (
            "Continental US, mid spring, 15 12:00:00 2019. "
            "Mild temperatures averaging 15.0°C. "
            "Light breeze averaging 8.0 km/h. "
            "moderate precipitation. "
            "moderate humidity (45.0%)."
        )
        assert result == expected

    def test_generate_caption_detailed(
        self, caption_generator: CaptionGenerator, base_temp_stats: WeatherStatistics
    ) -> None:
        """Test generating detailed caption."""
        result = caption_generator.generate_caption(base_temp_stats, detailed=True)
        expected = (
            "Continental US, mid spring, 15 12:00:00 2019. "
            "Temperature ranges from 5°C to 25°C across the region "
            "with mild conditions averaging 15°C."
        )
        assert result == expected

    def test_generate_caption_partial_multi_vars(
        self, caption_generator: CaptionGenerator, base_temp_stats: WeatherStatistics
    ) -> None:
        """Test generating caption with only some multi-variables."""
        wind_stats = TestWindDescriptions.create_wind_stats(mean=8.0, max=12.0)

        multi_var_stats = {
            "wind_speed": wind_stats
            # Only wind speed, no precipitation or humidity
        }

        result = caption_generator.generate_caption(
            base_temp_stats, multi_var_stats=multi_var_stats, detailed=False
        )

        expected = (
            "Continental US, mid spring, 15 12:00:00 2019. "
            "Mild temperatures averaging 15.0°C. "
            "Light breeze averaging 8.0 km/h."
        )
        assert result == expected


class TestCaptionGeneratorIntegration:
    """Integration tests for the CaptionGenerator class."""

    def test_full_caption_generation_workflow(
        self, caption_generator: CaptionGenerator
    ) -> None:
        """Test the complete caption generation workflow."""
        # Create temperature stats for a cold winter day
        temp_stats = WeatherStatistics(
            min=-10.0,
            max=5.0,
            mean=-2.0,
            std=4.0,
            median=-3.0,
            range=15.0,
            var=16.0,
            skewness=0.2,
            kurtosis=2.3,
            percentile_25=-5.0,
            percentile_75=1.0,
            percentile_90=3.0,
            percentile_95=4.0,
            iqr=6.0,
            mad=3.0,
            coeff_variation=2.0,
            count_valid=1000,
            count_missing=0,
            variable="temperature",
            unit="°C",
            description="Temperature at 2m above ground",
            valid_time="Wed Jan 15 06:00:00 2020 UTC",
            model="HRRR",
            forecast_hour=0,
            grib_name="TMP:2 m above ground",
            domain="CONUS",
            region="Northeast",
        )

        # Create wind stats for strong winds - 30.0 km/h is fresh breeze
        wind_stats = WeatherStatistics(
            min=15.0,
            max=45.0,
            mean=30.0,
            std=8.0,
            median=28.0,
            range=30.0,
            var=64.0,
            skewness=0.3,
            kurtosis=2.1,
            percentile_25=22.0,
            percentile_75=35.0,
            percentile_90=40.0,
            percentile_95=42.0,
            iqr=13.0,
            mad=6.0,
            coeff_variation=0.27,
            count_valid=1000,
            count_missing=0,
            variable="wind_speed",
            unit="km/h",
            description="Wind speed at 10m above ground",
            valid_time="Wed Jan 15 06:00:00 2020 UTC",
            model="HRRR",
            forecast_hour=0,
            grib_name="WIND:10 m above ground",
            domain="CONUS",
            region="Northeast",
        )

        multi_var_stats = {"wind_speed": wind_stats}

        result = caption_generator.generate_caption(
            temp_stats, multi_var_stats=multi_var_stats, detailed=False
        )

        # Verify the complete caption structure - now expecting correct season
        assert "Northeastern US" in result
        assert "mid winter" in result  # January should be mid winter
        assert "15 06:00:00 2020" in result
        assert "Cold temperatures averaging -2.0°C" in result
        assert "Fresh breeze averaging 30.0 km/h" in result  # Changed from Moderate
        assert result.endswith(".")

    def test_edge_case_extreme_weather(
        self, caption_generator: CaptionGenerator
    ) -> None:
        """Test caption generation for extreme weather conditions."""
        # Create stats for hurricane conditions
        extreme_stats = WeatherStatistics(
            min=25.0,
            max=35.0,
            mean=30.0,
            std=3.0,
            median=30.0,
            range=10.0,
            var=9.0,
            skewness=0.0,
            kurtosis=2.5,
            percentile_25=28.0,
            percentile_75=32.0,
            percentile_90=34.0,
            percentile_95=35.0,
            iqr=4.0,
            mad=2.0,
            coeff_variation=0.1,
            count_valid=1000,
            count_missing=0,
            variable="temperature",
            unit="°C",
            description="Temperature at 2m above ground",
            valid_time="Thu Aug 15 15:00:00 2019 UTC",
            model="HRRR",
            forecast_hour=0,
            grib_name="TMP:2 m above ground",
            domain="CONUS",
            region="Southeast",
        )

        hurricane_wind_stats = WeatherStatistics(
            min=100.0,
            max=150.0,
            mean=125.0,
            std=15.0,
            median=120.0,
            range=50.0,
            var=225.0,
            skewness=0.5,
            kurtosis=2.8,
            percentile_25=115.0,
            percentile_75=135.0,
            percentile_90=145.0,
            percentile_95=148.0,
            iqr=20.0,
            mad=12.0,
            coeff_variation=0.12,
            count_valid=1000,
            count_missing=0,
            variable="wind_speed",
            unit="km/h",
            description="Wind speed at 10m above ground",
            valid_time="Thu Aug 15 15:00:00 2019 UTC",
            model="HRRR",
            forecast_hour=0,
            grib_name="WIND:10 m above ground",
            domain="CONUS",
            region="Southeast",
        )

        multi_var_stats = {"wind_speed": hurricane_wind_stats}

        result = caption_generator.generate_caption(
            extreme_stats, multi_var_stats=multi_var_stats, detailed=True
        )

        # Verify extreme weather is properly described - now expecting correct season
        assert "Southeastern US" in result
        # August should be late summer
        assert "late summer" in result
        assert "hot conditions" in result
        hurricane_check = (
            "Hurricane conditions" in result or "Hurricane averaging" in result
        )
        assert hurricane_check


class TestBugFixes:
    """Test cases that verify bug fixes in the implementation."""

    def test_none_multi_var_stats_handling(
        self, caption_generator: CaptionGenerator, base_temp_stats: WeatherStatistics
    ) -> None:
        """Test that the function handles None multi_var_stats correctly."""
        # This should now work correctly after fixing the None handling bug
        result = caption_generator.generate_caption(
            base_temp_stats, multi_var_stats=None, detailed=False
        )
        # Should get a basic caption with just temperature
        expected = (
            "Continental US, mid spring, 15 12:00:00 2019. "
            "Mild temperatures averaging 15.0°C."
        )
        assert result == expected

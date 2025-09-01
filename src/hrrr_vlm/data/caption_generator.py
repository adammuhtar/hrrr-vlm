"""Streamlined multi-variable caption generation for HRRR VLM package."""

import structlog

from hrrr_vlm.data.models import WeatherStatistics
from hrrr_vlm.utils.logger import configure_logger

# Temperature thresholds
COLD_THRESHOLD = 10
HOT_THRESHOLD = 25

# Beaufort wind speed thresholds
CALM_WIND_THRESHOLD = 0
LIGHT_BREEZE_THRESHOLD = 11
GENTLE_BREEZE_THRESHOLD = 19
MODERATE_BREEZE_THRESHOLD = 28
FRESH_BREEZE_THRESHOLD = 38
STRONG_BREEZE_THRESHOLD = 49
NEAR_GALE_THRESHOLD = 61
GALE_THRESHOLD = 74
STRONG_GALE_THRESHOLD = 88
STORM_THRESHOLD = 102
VIOLENT_STORM_THRESHOLD = 117

# Precipitation thresholds
LOW_PRECIPITATION = 0.1
MODERATE_PRECIPITATION = 0.3

# Humidity thresholds
LOW_HUMIDITY = 30
MODERATE_HUMIDITY = 60

# Additional thresholds
WARM_TEMP_THRESHOLD = 20
MODERATE_TEMP_RANGE_ANALYSIS = 15
HIGH_MOISTURE_THRESHOLD = 70


class CaptionGenerator:
    """Multi-variable weather caption generator.

    Attributes:
        logger (structlog.BoundLogger): Logger instance for structured logging.
    """

    def __init__(self, logger: structlog.BoundLogger | None = None) -> None:
        """Initialise the caption generator.

        Args:
            logger (`structlog.BoundLogger`, optional): Logger instance for
                structured logging. If not provided, a default logger is configured.
        """
        self.logger = logger or configure_logger(enable_json=False, log_level="INFO")

    def generate_caption(
        self,
        stats: WeatherStatistics,
        multi_var_stats: dict[str, WeatherStatistics] | None = None,
        *,
        detailed: bool = False,
    ) -> str:
        """Generate a multi-variable caption for weather data.

        Args:
            stats (`WeatherStatistics`): Primary weather statistics (temperature).
            multi_var_stats (`dict[str, WeatherStatistics]`): Additional variable
                statistics.
            detailed (`bool`): Whether to provide a detailed description.

        Returns:
            `str`: Generated multi-variable caption string.
        """
        parts = []

        # Date and opening
        date_str = self._format_date(stats.valid_time)
        opening = self.create_opening(stats, date_str)
        parts.append(opening)

        # Temperature description
        temp_desc = self._describe_temperature(stats, detailed=detailed)
        parts.append(temp_desc)

        # Additional variables
        if multi_var_stats is not None:
            for var_name in ["wind_speed", "precipitation", "humidity"]:
                if var_name in multi_var_stats:
                    var_desc = self._describe_variable(
                        var_name, multi_var_stats[var_name], detailed=detailed
                    )
                    if var_desc:
                        parts.append(var_desc)

        return ". ".join(parts) + "."

    @staticmethod
    def _format_date(valid_time: str) -> str:
        """Format date from valid time string.

        Args:
            valid_time (`str`): Valid time string.

        Returns:
            `str`: Formatted date string.
        """
        try:
            parts = valid_time.split()
            return f"{parts[2]} {parts[3]} {parts[4]}"
        except (IndexError, AttributeError):
            return "unknown date"

    def create_opening(self, stats: WeatherStatistics, date_str: str) -> str:
        """Create the opening sentence for captions.

        Args:
            stats (`WeatherStatistics`): Weather statistics containing region info.
            date_str (`str`): Formatted date string.

        Returns:
            `str`: Opening sentence for the caption.
        """
        region = stats.region or "Continental US"

        region_map = {"Continental US": "Continental US", "Alaska": "Alaska"}

        region_desc = region_map.get(region, f"{self._get_region_descriptor(region)}")
        season_desc = self._get_season_descriptor(stats.valid_time)
        return f"{region_desc}, {season_desc}, {date_str}"

    @staticmethod
    def _get_region_descriptor(region: str) -> str:
        """Get a natural language descriptor for the region.

        Args:
            region (`str`): Region name.

        Returns:
            `str`: Natural language descriptor for the region.
        """
        descriptors = {
            "Northeast": "Northeastern US",
            "Southeast": "Southeastern US",
            "Southwest": "Southwestern US",
            "Northwest": "Pacific Northwest",
            "Upper Midwest": "Upper Midwest",
            "Northern Rockies and Plains": "Rockies and Plains",
            "Ohio Valley": "Ohio River Valley",
            "South": "Southern US",
            "West": "Western US",
        }
        return descriptors.get(region, f"{region} region")

    @staticmethod
    def _get_season_descriptor(valid_time: str) -> str:
        """Get early/mid/late season descriptor based on month within season.

        Args:
            valid_time (`str`): Valid time string to extract month from.

        Returns:
            `str`: Season descriptor like "early spring", "mid summer", etc.
        """
        month_map = {
            "Jan": 1,
            "Feb": 2,
            "Mar": 3,
            "Apr": 4,
            "May": 5,
            "Jun": 6,
            "Jul": 7,
            "Aug": 8,
            "Sep": 9,
            "Oct": 10,
            "Nov": 11,
            "Dec": 12,
        }

        try:
            month_str = valid_time.split()[1]  # Get month (index 1, not 3)
            month = month_map.get(month_str, 1)
        except (IndexError, AttributeError):
            month = 6

        if month in {12, 1, 2}:
            season = "winter"
        elif month in {3, 4, 5}:
            season = "spring"
        elif month in {6, 7, 8}:
            season = "summer"
        else:
            season = "autumn"

        # Map month to season period
        season_month_map = {
            "winter": {12: "early", 1: "mid", 2: "late"},
            "spring": {3: "early", 4: "mid", 5: "late"},
            "summer": {6: "early", 7: "mid", 8: "late"},
            "autumn": {9: "early", 10: "mid", 11: "late"},
        }

        period = season_month_map.get(season).get(month, "mid")
        return f"{period} {season}"

    @staticmethod
    def _describe_temperature(
        stats: WeatherStatistics, *, detailed: bool = False
    ) -> str:
        """Describe temperature conditions.

        Args:
            stats (`WeatherStatistics`): Temperature statistics.
            detailed (`bool`): Whether to provide a detailed description.

        Returns:
            `str`: Temperature description string.
        """
        unit = stats.unit

        # Detailed description
        if detailed:
            if stats.mean <= COLD_THRESHOLD:
                return (
                    f"Temperature ranges from {stats.min:.0f}°C to {stats.max:.0f}°C "
                    "across the region with cold conditions averaging "
                    f"{stats.mean:.0f}{unit}"
                )
            if stats.mean >= HOT_THRESHOLD:
                return (
                    f"Temperature ranges from {stats.min:.0f}°C to {stats.max:.0f}°C "
                    "across the region with hot conditions averaging "
                    f"{stats.mean:.0f}{unit}"
                )
            return (
                f"Temperature ranges from {stats.min:.0f}°C to {stats.max:.0f}°C "
                "across the region with mild conditions averaging "
                f"{stats.mean:.0f}{unit}"
            )

        # Simple description
        if stats.mean <= COLD_THRESHOLD:
            return f"Cold temperatures averaging {stats.mean:.1f}{unit}"
        if stats.mean >= HOT_THRESHOLD:
            return f"Hot temperatures averaging {stats.mean:.1f}{unit}"
        return f"Mild temperatures averaging {stats.mean:.1f}{unit}"

    def _describe_variable(
        self, var_name: str, var_stats: WeatherStatistics, *, detailed: bool = False
    ) -> str:
        """Describe specific weather variable conditions.

        Args:
            var_name (`str`): Name of the weather variable (e.g., "wind_speed").
            var_stats (`WeatherStatistics`): Statistics for the variable.
            detailed (`bool`): Whether to provide a detailed description.

        Returns:
            `str`: Variable description string.
        """
        handlers = {
            "wind_speed": self.describe_wind,
            "precipitation": self.describe_precipitation,
            "humidity": self.describe_humidity,
        }

        handler = handlers.get(var_name)
        return handler(var_stats, detailed=detailed) if handler else ""

    @staticmethod
    def describe_wind(stats: WeatherStatistics, *, detailed: bool = False) -> str:
        """Describe wind conditions.

        Args:
            stats (`WeatherStatistics`): Wind speed statistics.
            detailed (`bool`): Whether to provide a detailed description.

        Returns:
            `str`: Wind description string.
        """
        mean_wind = stats.mean
        unit = stats.unit

        # Detailed description
        if detailed:
            max_wind = stats.max
            if mean_wind <= CALM_WIND_THRESHOLD:
                return (
                    f"Wind speeds are calm with average speeds of {mean_wind:.1f} "
                    f"{unit} and maximum gusts up to {max_wind:.1f}{unit}"
                )
            if mean_wind <= LIGHT_BREEZE_THRESHOLD:
                return (
                    f"Light breeze conditions with average speeds of {mean_wind:.1f} "
                    f"{unit} and maximum gusts up to {max_wind:.1f}{unit}"
                )
            if mean_wind <= GENTLE_BREEZE_THRESHOLD:
                return (
                    f"Gentle breeze conditions with average speeds of {mean_wind:.1f} "
                    f"{unit} and maximum gusts up to {max_wind:.1f}{unit}"
                )
            if mean_wind <= MODERATE_BREEZE_THRESHOLD:
                return (
                    "Moderate breeze conditions with average speeds of "
                    f"{mean_wind:.1f} {unit} and maximum gusts up to "
                    f"{max_wind:.1f}{unit}"
                )
            if mean_wind <= FRESH_BREEZE_THRESHOLD:
                return (
                    f"Fresh breeze conditions with average speeds of {mean_wind:.1f} "
                    f"{unit} and maximum gusts up to {max_wind:.1f}{unit}"
                )
            if mean_wind <= STRONG_BREEZE_THRESHOLD:
                return (
                    f"Strong breeze conditions with average speeds of {mean_wind:.1f} "
                    f"{unit} and maximum gusts up to {max_wind:.1f}{unit}"
                )
            if mean_wind <= NEAR_GALE_THRESHOLD:
                return (
                    f"Near gale conditions with average speeds of {mean_wind:.1f} "
                    f"{unit} and maximum gusts up to {max_wind:.1f}{unit}"
                )
            if mean_wind <= GALE_THRESHOLD:
                return (
                    f"Gale conditions with average speeds of {mean_wind:.1f} "
                    f"{unit} and maximum gusts up to {max_wind:.1f}{unit}"
                )
            if mean_wind <= STRONG_GALE_THRESHOLD:
                return (
                    f"Strong gale conditions with average speeds of {mean_wind:.1f} "
                    f"{unit} and maximum gusts up to {max_wind:.1f}{unit}"
                )
            if mean_wind <= STORM_THRESHOLD:
                return (
                    f"Storm conditions with average speeds of {mean_wind:.1f} "
                    f"{unit} and maximum gusts up to {max_wind:.1f}{unit}"
                )
            if mean_wind <= VIOLENT_STORM_THRESHOLD:
                return (
                    f"Violent storm conditions with average speeds of {mean_wind:.1f} "
                    f"{unit} and maximum gusts up to {max_wind:.1f}{unit}"
                )
            return (
                f"Hurricane conditions with average speeds of {mean_wind:.1f} "
                f"{unit} and maximum gusts up to {max_wind:.1f}{unit}"
            )

        # Simple description
        if mean_wind <= CALM_WIND_THRESHOLD:
            return "Calm and no winds"
        if mean_wind <= LIGHT_BREEZE_THRESHOLD:
            return f"Light breeze averaging {mean_wind:.1f} {unit}"
        if mean_wind <= GENTLE_BREEZE_THRESHOLD:
            return f"Gentle breeze averaging {mean_wind:.1f} {unit}"
        if mean_wind <= MODERATE_BREEZE_THRESHOLD:
            return f"Moderate breeze averaging {mean_wind:.1f} {unit}"
        if mean_wind <= FRESH_BREEZE_THRESHOLD:
            return f"Fresh breeze averaging {mean_wind:.1f} {unit}"
        if mean_wind <= STRONG_BREEZE_THRESHOLD:
            return f"Strong breeze averaging {mean_wind:.1f} {unit}"
        if mean_wind <= NEAR_GALE_THRESHOLD:
            return f"Near gale averaging {mean_wind:.1f} {unit}"
        if mean_wind <= GALE_THRESHOLD:
            return f"Gale winds averaging {mean_wind:.1f} {unit}"
        if mean_wind <= STRONG_GALE_THRESHOLD:
            return f"Strong gale averaging {mean_wind:.1f} {unit}"
        if mean_wind <= STORM_THRESHOLD:
            return f"Storm averaging {mean_wind:.1f} {unit}"
        if mean_wind <= VIOLENT_STORM_THRESHOLD:
            return f"Violent storm averaging {mean_wind:.1f} {unit}"
        return f"Hurricane averaging {mean_wind:.1f} {unit} "

    @staticmethod
    def describe_precipitation(
        stats: WeatherStatistics, *, detailed: bool = False
    ) -> str:
        """Describe precipitation conditions.

        Args:
            stats (`WeatherStatistics`): Precipitation statistics.
            detailed (`bool`): Whether to provide a detailed description.

        Returns:
            `str`: Precipitation description string.
        """
        mean_precip = stats.mean
        unit = stats.unit

        # Detailed description
        if detailed:
            max_precip = stats.max
            if mean_precip <= LOW_PRECIPITATION:
                if max_precip > MODERATE_PRECIPITATION:
                    return (
                        "Precipitation shows scattered activity with accumulations "
                        f"up to {max_precip:.1f}{unit} in some areas"
                    )
                return (
                    "Precipitation shows minimal activity with light amounts up "
                    f"to {max_precip:.1f}{unit}"
                )
            if mean_precip <= MODERATE_PRECIPITATION:
                return (
                    "Precipitation shows moderate activity with accumulations "
                    f"averaging {mean_precip:.1f}{unit} and "
                    f"maximum {max_precip:.1f}{unit}"
                )
            return (
                "Precipitation shows heavy activity with significant accumulations "
                f"averaging {mean_precip:.1f}{unit} and maximum {max_precip:.1f}{unit}"
            )

        # Simple description
        if mean_precip <= LOW_PRECIPITATION:
            return "low precipitation"
        if mean_precip <= MODERATE_PRECIPITATION:
            return "moderate precipitation"
        return "heavy precipitation"

    @staticmethod
    def describe_humidity(stats: WeatherStatistics, *, detailed: bool = False) -> str:
        """Describe humidity conditions.

        Args:
            stats (`WeatherStatistics`): Humidity statistics.
            detailed (`bool`): Whether to provide a detailed description.

        Returns:
            `str`: Humidity description string.
        """
        mean_humidity = stats.mean
        unit = stats.unit

        # Detailed description
        if detailed:
            min_humidity = stats.min
            max_humidity = stats.max
            if mean_humidity < LOW_HUMIDITY:
                return (
                    f"Humidity levels {min_humidity:.0f}-{max_humidity:.0f}{unit} "
                    "with dry conditions throughout the region"
                )
            if mean_humidity < MODERATE_HUMIDITY:
                return (
                    f"Humidity levels {min_humidity:.0f}-{max_humidity:.0f}{unit} "
                    "with moderate moisture content"
                )
            return (
                f"Humidity levels {min_humidity:.0f}-{max_humidity:.0f}{unit} "
                "with high moisture content throughout the region"
            )

        # Simple description
        if mean_humidity < LOW_HUMIDITY:
            return f"low humidity ({mean_humidity:.1f}{unit})"
        if mean_humidity < MODERATE_HUMIDITY:
            return f"moderate humidity ({mean_humidity:.1f}{unit})"
        return f"high humidity ({mean_humidity:.1f}{unit})"

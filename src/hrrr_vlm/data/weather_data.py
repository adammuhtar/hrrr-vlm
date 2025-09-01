"""Weather data service for HRRR VLM package."""

from typing import Literal

import matplotlib.pyplot as plt
import structlog
import xarray as xr
from herbie import Herbie, paint
from herbie.toolbox import EasyMap, pc

from hrrr_vlm.data.config import WeatherVariableConfig
from hrrr_vlm.data.constants import LONGITUDE_MAX, MODEL_CONFIGS, REGIONS
from hrrr_vlm.data.exceptions import DataLoadError, WeatherDataError
from hrrr_vlm.data.models import WeatherStatistics
from hrrr_vlm.utils.logger import configure_logger


class WeatherDataService:
    """Service for loading and processing weather data using Herbie.

    Attributes:
        variable_config (`WeatherVariableConfig`): Configuration for the weather
            variable.
        logger (`structlog.BoundLogger`): Logger instance for structured logging.
        herbie (`Herbie`): Herbie instance for weather data access.
        current_model (`str`): Currently loaded weather model.
    """

    def __init__(
        self,
        variable_config: WeatherVariableConfig,
        logger: structlog.BoundLogger | None = None,
    ) -> None:
        """Initialise the weather data service.

        Args:
            variable_config (`WeatherVariableConfig`): Configuration for the
                weather variable.
            logger (`structlog.BoundLogger`, optional): Logger instance for
                structured logging. If not provided, a default logger is configured.
        """
        self.variable_config = variable_config
        self.logger = logger or configure_logger(enable_json=False, log_level="INFO")
        self.herbie: Herbie | None = None
        self.current_model: str | None = None

    def load_data(
        self, date_str: str, model: Literal["hrrr", "hrrrak"] = "hrrr", fxx: int = 0
    ) -> bool:
        """Load Herbie data for a specific date and configuration.

        Args:
            date_str (`str`): Date in "YYYYMMDD" format.
            model (`str`): Weather model to use (default: "hrrr").
            fxx (`int`): Forecast hour (default: 0).

        Returns:
            `bool`: True if data loaded successfully, False otherwise.

        Raises:
            DataLoadError: If model is unsupported or data loading fails
        """
        if model not in MODEL_CONFIGS:
            available = list(MODEL_CONFIGS.keys())
            msg = f"Unsupported model: {model}. Available: {available}"
            raise DataLoadError(msg)

        try:
            model_config = MODEL_CONFIGS[model]
            self.herbie = Herbie(
                date_str, model=model, product=model_config.product, fxx=fxx
            )
            self.current_model = model

        except Exception as e:
            self.logger.exception(
                "Failed to load weather data", date=date_str, model=model, fxx=fxx
            )
            self.herbie = None
            self.current_model = None
            msg = f"Failed to load weather data: {e}"
            raise DataLoadError(msg) from e

        else:
            self.logger.info(
                "Weather data loaded",
                date=self.herbie.date,
                model=self.herbie.model,
                product=self.herbie.product,
                fxx=self.herbie.fxx,
            )
            return True

    def get_available_variables(self) -> list[str]:
        """List available variables in the current dataset.

        Returns:
            `list[str]`: List of available variable names.

        Raises:
            WeatherDataError: If no data is loaded.
        """
        if self.herbie is None:
            msg = "No weather data loaded. Call load_data first."
            raise WeatherDataError(msg)

        try:
            inventory = self.herbie.inventory()
            variables = inventory["variable"].tolist()

        except Exception as e:
            msg = f"Could not retrieve variable inventory: {e}"
            raise WeatherDataError(msg) from e

        else:
            self.logger.info(
                "Retrieved available variables",
                total_vars=len(inventory),
                retrieved_vars=len(variables),
                date=self.herbie.date,
                model=self.herbie.model,
            )
            return variables

    def get_data_array(self, region: str | None = None) -> xr.DataArray:
        """Get the data array for the configured variable.

        Args:
            region (`str`, optional): Region to filter data.

        Returns:
            `xr.DataArray`: Converted data array.

        Raises:
            WeatherDataError: If no data is loaded or processing fails.
        """
        if self.herbie is None:
            msg = "No weather data loaded. Call load_data first."
            raise WeatherDataError(msg)

        try:
            ds = self.herbie.xarray(self.variable_config.search_string)

            # Get the first data variable
            data_vars = list(ds.data_vars)
            if not data_vars:
                msg = "No data variables found in dataset"
                raise WeatherDataError(msg)  # noqa: TRY301

            data_var = ds[data_vars[0]]
            converted_data = self.variable_config.convert_data(data_var)

            # Apply regional filtering if requested
            if region and region in REGIONS:
                converted_data = self._apply_regional_mask(ds, converted_data, region)

        except Exception as e:
            msg = f"Failed to get data array: {e}"
            raise WeatherDataError(msg) from e

        else:
            return converted_data

    def calculate_statistics(self, region: str | None = None) -> WeatherStatistics:
        """Calculate comprehensive statistics for the weather data.

        Args:
            region (`str`, optional): Region to focus analysis on.

        Returns:
            `WeatherStatistics`: WeatherStatistics object with calculated values.

        Raises:
            WeatherDataError: If no data is loaded or calculation fails.
        """
        if self.herbie is None:
            msg = "No weather data loaded. Call load_data first."
            raise WeatherDataError(msg)

        try:
            data_array = self.get_data_array(region)

            # Get additional metadata
            ds = self.herbie.xarray(self.variable_config.search_string)
            data_var = ds[next(iter(ds.data_vars))]

            valid_time = ds.valid_time.dt.strftime("%H:%M UTC %d %b %Y").item()

            stats = WeatherStatistics(
                min=float(data_array.min().values),
                max=float(data_array.max().values),
                mean=float(data_array.mean().values),
                std=float(data_array.std().values),
                median=float(data_array.median().values),
                range=float(data_array.max().values - data_array.min().values),
                var=float(data_array.var().values),
                skewness=float(self._calculate_skewness(data_array).values),
                kurtosis=float(self._calculate_kurtosis(data_array).values),
                percentile_25=float(data_array.quantile(0.25).values),
                percentile_75=float(data_array.quantile(0.75).values),
                percentile_90=float(data_array.quantile(0.90).values),
                percentile_95=float(data_array.quantile(0.95).values),
                iqr=float(
                    data_array.quantile(0.75).values - data_array.quantile(0.25).values
                ),
                mad=float(abs(data_array - data_array.median()).median().values),
                coeff_variation=float(
                    data_array.std().values / abs(data_array.mean().values)
                )
                if data_array.mean().values != 0
                else 0,
                count_valid=int((~data_array.isnull()).sum().values),
                count_missing=int(data_array.isnull().sum().values),
                variable=self.variable_config.variable,
                unit=self.variable_config.unit,
                description=self.variable_config.description,
                valid_time=valid_time,
                model=ds.model.upper(),
                forecast_hour=self.herbie.fxx,
                grib_name=getattr(data_var, "GRIB_name", "Unknown"),
                domain=MODEL_CONFIGS[self.current_model].domain,
                region=region or "",
            )

        except Exception as e:
            msg = f"Failed to calculate statistics: {e}"
            raise WeatherDataError(msg) from e

        else:
            self.logger.info(
                "Statistics calculated",
                variable=stats.variable,
                min_value=stats.min,
                max_value=stats.max,
                mean_value=stats.mean,
                region=region,
            )

            return stats

    def generate_map(
        self, region: str | None = None, figsize: tuple[int, int] = (12, 8)
    ) -> tuple[plt.Figure, WeatherStatistics]:
        """Generate a weather map with statistics.

        Args:
            region (`str`, optional): Region to focus the map on. If None, use
                default region for the model.
            figsize (`tuple[int, int]`): Size of the figure for the map.

        Returns:
            `tuple`: A tuple containing:
                - `plt.Figure`: The generated matplotlib figure.
                - `WeatherStatistics`: Calculated statistics for the displayed
                    data.

        Raises:
            WeatherDataError: If no data is loaded or map generation fails
        """
        if self.herbie is None:
            msg = "No weather data loaded. Call load_data first."
            raise WeatherDataError(msg)

        try:
            # Get data and calculate statistics
            ds = self.herbie.xarray(self.variable_config.search_string)
            data_var = ds[next(iter(ds.data_vars))]
            converted_data = self.variable_config.convert_data(data_var)

            # Create map
            model_config = MODEL_CONFIGS[self.current_model]
            em = EasyMap(
                scale=model_config.map_resolution, crs=ds.herbie.crs, figsize=figsize
            )
            ax = em.BORDERS().STATES().ax
            fig = ax.get_figure()

            # Plot data
            p = ax.pcolormesh(
                ds.longitude,
                ds.latitude,
                converted_data,
                transform=pc,
                **paint.NWSTemperature.kwargs2,
            )

            # Set region extent if specified
            if region and region in REGIONS:
                region_config = REGIONS[region]
                ax.set_extent(region_config.bounds, crs=pc)

            # Add colour bar
            plt.colorbar(
                p,
                ax=ax,
                orientation="horizontal",
                pad=0.01,
                shrink=0.8,
                label=(
                    f"{self.variable_config.description} ({self.variable_config.unit})"
                ),
                **paint.NWSTemperature.kwargs2,
            )

            # Add titles
            valid_date = ds.valid_time.dt.strftime("%d %b %Y").item()

            region_display = self._get_region_display_name(region)
            ax.set_title(label=region_display, loc="left")
            ax.set_title(label=valid_date, loc="right")

            plt.tight_layout()

            # Calculate statistics
            stats = self.calculate_statistics(region)

        except Exception as e:
            msg = f"Failed to generate weather map: {e}"
            raise WeatherDataError(msg) from e

        else:
            self.logger.info(
                "Weather map generated",
                variable=stats.variable,
                region=region,
                figsize=figsize,
            )

            return fig, stats

    def _apply_regional_mask(
        self, ds: xr.Dataset, data: xr.DataArray, region: str
    ) -> xr.DataArray:
        """Apply regional mask to data.

        Args:
            ds (`xr.Dataset`): Original dataset for coordinate reference.
            data (`xr.DataArray`): Data array to mask.
            region (`str`): Region name.

        Returns:
            `xr.DataArray`: Masked data array.
        """
        region_config = REGIONS[region]
        lon0, lon1, lat0, lat1 = region_config.bounds

        # Handle longitude coordinate system
        if self.current_model == "hrrrak":
            # For Alaska, check data longitude range
            _lon_min, lon_max = float(ds.longitude.min()), float(ds.longitude.max())

            if lon_max > LONGITUDE_MAX:
                # Data is in [0, 360] format
                lon2 = ds.longitude
                # Convert negative region bounds to [0, 360]
                if lon0 < 0:
                    lon0 += 360
                if lon1 < 0:
                    lon1 += 360
            else:
                # Data is in [-180, 180] format
                lon2 = ds.longitude
        else:
            # For CONUS, handle longitude wrapping
            lon2 = xr.where(
                ds.longitude > LONGITUDE_MAX, ds.longitude - 360, ds.longitude
            )

        lat2 = ds.latitude

        # Create mask
        mask = (lon2 >= lon0) & (lon2 <= lon1) & (lat2 >= lat0) & (lat2 <= lat1)
        return data.where(mask)

    def _get_region_display_name(self, region: str | None) -> str:
        """Get display name for region.

        Args:
            region (`str`, optional): Region name.

        Returns:
            `str`: Display name for the region.
        """
        if region is None:
            return MODEL_CONFIGS[self.current_model].default_region

        if region == "Alaska":
            return "Alaska"

        if region == "Continental US":
            return "Continental US"
        return f"US {region} region"

    @staticmethod
    def _calculate_skewness(data: xr.DataArray) -> xr.DataArray:
        """Calculate skewness of data.

        Args:
            data (`xr.DataArray`): Input data array.

        Returns:
            `xr.DataArray`: Skewness of the data.
        """
        mean = data.mean()
        std = data.std()
        return ((data - mean) ** 3).mean() / (std**3)

    @staticmethod
    def _calculate_kurtosis(data: xr.DataArray) -> xr.DataArray:
        """Calculate kurtosis of data (excess kurtosis, minus 3).

        Args:
            data (`xr.DataArray`): Input data array.

        Returns:
            `xr.DataArray`: Kurtosis of the data.
        """
        mean = data.mean()
        std = data.std()
        return ((data - mean) ** 4).mean() / (std**4) - 3

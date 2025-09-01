"""Main data generator for HRRR VLM package."""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from hrrr_vlm.data.caption_generator import CaptionGenerator
from hrrr_vlm.data.config import DataGeneratorConfig
from hrrr_vlm.data.constants import WEATHER_VARIABLES
from hrrr_vlm.data.models import CaptionMetadata, TrainingRecord
from hrrr_vlm.data.weather_data import WeatherDataService
from hrrr_vlm.utils.logger import configure_logger


class WeatherDataGenerator:
    """Main service for generating weather image-caption pairs.

    Attributes:
        config (`DataGeneratorConfig`): Configuration for the data generator.
        output_dir (`Path`): Directory where generated data will be saved.
        images_dir (`Path`): Directory for saving generated images.
        captions_dir (`Path`): Directory for saving generated captions.
        logger (`Logger`): Logger instance for logging events.
        caption_generator (`CaptionGenerator`): Service for generating captions.
        metadata (`list[CaptionMetadata]`): List to store metadata for generated
            samples.
    """

    def __init__(self, config: DataGeneratorConfig) -> None:
        """Initialise the data generator.

        Args:
            config (`DataGeneratorConfig`): Configuration for the data generator.
        """
        self.config = config
        self.logger = configure_logger(
            enable_json=config.enable_json_logging, log_level=config.log_level
        )
        self.caption_generator = CaptionGenerator(self.logger)

        # Create directories
        self.output_dir = Path(config.output_dir)
        self.images_dir = self.output_dir / "images"
        self.captions_dir = self.output_dir / "captions"

        for directory in [self.output_dir, self.images_dir, self.captions_dir]:
            directory.mkdir(parents=True, exist_ok=True)

        self.logger.info(
            "Data generator initialised",
            output=str(self.output_dir),
            images=str(self.images_dir),
            captions=str(self.captions_dir),
        )

        # Metadata storage
        self.metadata: list[CaptionMetadata] = []

    def generate_samples(
        self,
        dates: list[str],
        primary_variable: str = "temperature",
        additional_variables: list[str] | None = None,
        models: list[str] | None = None,
        regions: list[str] | None = None,
        fxx: int = 0,
    ) -> dict[str, Any]:
        """Generate image-caption samples for the given dates.

        Args:
            dates (`list[str]`): List of dates in "YYYY-MM-DD" format.
            primary_variable (`str`): Primary variable for visualisation. Default
                is "temperature".
            additional_variables (`list[str]`, optional): Additional variables for
                caption enhancement.
            models (`list[str]`, optional): List of weather models to use.
            regions (`list[str]`, optional): List of regions to process.
            fxx (`int`, optional): Forecast hour. Default is 0 (current time).

        Returns:
            `dict`: Dictionary with generation statistics
        """
        if additional_variables is None:
            additional_variables = ["wind_speed", "precipitation", "humidity"]
        if models is None:
            models = ["hrrr"]
        if regions is None:
            regions = [None]

        successful_generations = 0
        failed_generations = 0

        self.logger.info(
            "Starting multi-variable generation",
            primary_variable=primary_variable,
            additional_variables=additional_variables,
            dates=len(dates),
            models=models,
            regions=regions,
        )

        for date_str in dates:
            for model in models:
                for region in regions:
                    try:
                        success = self.generate_single_sample(
                            date_str=date_str,
                            primary_variable=primary_variable,
                            additional_variables=additional_variables,
                            model=model,
                            region=region,
                            fxx=fxx,
                        )

                        if success:
                            successful_generations += 1
                        else:
                            failed_generations += 1

                    except Exception:
                        self.logger.exception(
                            "Failed to generate multi-variable sample",
                            date=date_str,
                            model=model,
                            region=region,
                        )
                        failed_generations += 1

        stats = {
            "total_attempted": len(dates) * len(models) * len(regions),
            "successful": successful_generations,
            "failed": failed_generations,
            "success_rate": successful_generations
            / (successful_generations + failed_generations)
            if (successful_generations + failed_generations) > 0
            else 0,
        }

        self.logger.info("Image-caption generation completed", **stats)
        return stats

    def generate_single_sample(
        self,
        date_str: str,
        primary_variable: str,
        additional_variables: list[str],
        model: str,
        region: str | None,
        fxx: int,
    ) -> bool:
        """Generate a single image-caption sample.

        Args:
            date_str (`str`): Date in "YYYY-MM-DD" format.
            primary_variable (`str`): Primary variable for visualisation.
            additional_variables (`list[str]`): Additional variables for caption
                enhancement.
            model (`str`): Weather model to use.
            region (`str`, optional): Region to focus on.
            fxx (`int`): Forecast hour.

        Returns:
            `bool`: True if sample generated successfully, False otherwise
        """
        try:
            # Convert date format
            date_obj = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=UTC)
            herbie_date_str = date_obj.strftime("%Y%m%d")

            # Get primary variable configuration
            if primary_variable not in WEATHER_VARIABLES:
                self.logger.error("Unknown primary variable", variable=primary_variable)
                return False

            primary_config = WEATHER_VARIABLES[primary_variable]

            # Apply precipitation fxx fix for primary variable
            primary_fxx = fxx
            if primary_variable == "precipitation" and fxx == 0:
                primary_fxx = 1
                self.logger.info(
                    "Using fxx=1 for primary precipitation instead of fxx=0"
                )

            # Initialise weather data service for primary variable
            primary_service = WeatherDataService(primary_config, self.logger)

            # Load data with effective fxx - with better error context
            if not primary_service.load_data(herbie_date_str, model, primary_fxx):
                self.logger.warning(
                    "Skipping sample - data load failed",
                    date=date_str,
                    model=model,
                    variable=primary_variable,
                    fxx=primary_fxx,
                )
                return False

            # Generate map and primary statistics - with error context
            try:
                fig, primary_stats = primary_service.generate_map(
                    region=region, figsize=self.config.default_figsize
                )
            except Exception as e:
                self.logger.warning(
                    "Skipping sample - map generation failed",
                    date=date_str,
                    model=model,
                    variable=primary_variable,
                    error=str(e),
                    error_type=type(e).__name__,
                )
                return False

            if fig is None:
                self.logger.warning(
                    "Skipping sample - map generation returned None",
                    date=date_str,
                    model=model,
                    variable=primary_variable,
                )
                return False

            # Collect additional variable statistics
            additional_stats = {}
            for var_name in additional_variables:
                if var_name in WEATHER_VARIABLES:
                    try:
                        var_config = WEATHER_VARIABLES[var_name]
                        var_service = WeatherDataService(var_config, self.logger)

                        # Apply precipitation fxx fix for additional variables too
                        var_fxx = fxx
                        if var_name == "precipitation" and fxx == 0:
                            var_fxx = 1
                            self.logger.debug(
                                "Using fxx=1 for additional precipitation variable"
                            )

                        if var_service.load_data(herbie_date_str, model, var_fxx):
                            var_stats = var_service.calculate_statistics(region)
                            additional_stats[var_name] = var_stats

                    except Exception as e:
                        self.logger.debug(
                            "Failed to get additional variable stats",
                            variable=var_name,
                            date=date_str,
                            error=str(e),
                            error_type=type(e).__name__,
                        )

            # Save image
            region_suffix = f"{region.lower().replace(' ', '_')}" if region else "conus"
            image_filename = (
                f"{date_str}_{primary_variable}_{model}_{region_suffix}.png"
            )
            image_path = self.images_dir / image_filename

            fig.canvas.draw()
            fig.savefig(image_path, dpi=self.config.default_dpi, bbox_inches="tight")
            plt.close(fig)

            # Generate multi-variable caption
            caption = self.caption_generator.generate_caption(
                primary_stats, multi_var_stats=additional_stats
            )

            caption_filename = (
                f"{date_str}_{primary_variable}_{model}_{region_suffix}.txt"
            )
            caption_path = self.captions_dir / caption_filename

            with caption_path.open("w") as f:
                f.write(caption)

            # Store metadata
            metadata = CaptionMetadata(
                image_filename=image_filename,
                caption_files=[caption_filename],
                sample_id=f"{date_str}_{primary_variable}_{model}_{region_suffix}",
                variable=primary_variable,
                model=model,
                date=date_str,
                region=region,
                stats=primary_stats,
            )
            self.metadata.append(metadata)

        except Exception:
            self.logger.exception("Failed to generate multi-variable sample")
            return False

        else:
            self.logger.info(
                "Multi-variable sample generated",
                date=date_str,
                primary_variable=primary_variable,
                additional_variables=list(additional_stats.keys()),
                model=model,
                region=region,
            )

            return True

    def create_training_records(self, output_file: str) -> Path:
        """Create a JSONL file suitable for multimodal training.

        Args:
            output_file (`str`): Name of the output JSONL file.

        Returns:
            `Path`: Path to the created JSONL file.
        """
        clip_data = []

        self.logger.info("Creating CLIP training dataset", output_file=output_file)

        for entry in self.metadata:
            image_path = f"images/{entry.image_filename}"

            for caption_file in entry.caption_files:
                caption_path = self.captions_dir / caption_file

                if caption_path.exists():
                    with caption_path.open() as f:
                        caption = f.read().strip()

                    record = TrainingRecord(
                        image_path=image_path,
                        caption=caption,
                        sample_id=entry.sample_id,
                        variable=entry.variable,
                        model=entry.model,
                        date=entry.date,
                        metadata={
                            "region": entry.region,
                            "domain": entry.domain,
                            "generation_time": entry.generation_time.isoformat(),
                        },
                    )
                    clip_data.append(record)
                else:
                    self.logger.warning(
                        "Caption file not found",
                        caption_path=str(caption_path),
                        sample_id=entry.sample_id,
                    )

        # Save as JSONL
        jsonl_path = self.output_dir / output_file
        with jsonl_path.open("w") as f:
            for record in clip_data:
                f.write(record.model_dump_json() + "\n")

        self.logger.info(
            "CLIP training dataset created",
            output_path=str(jsonl_path),
            total_pairs=len(clip_data),
        )

        return jsonl_path

    def get_generation_summary(self) -> dict[str, Any]:
        """Get summary of generated samples.

        Returns:
            Dictionary with generation summary statistics
        """
        if not self.metadata:
            return {"total_samples": 0}

        # Group by variable, model, region
        by_variable = {}
        by_model = {}
        by_region = {}

        for entry in self.metadata:
            # By variable
            if entry.variable not in by_variable:
                by_variable[entry.variable] = 0
            by_variable[entry.variable] += 1

            # By model
            if entry.model not in by_model:
                by_model[entry.model] = 0
            by_model[entry.model] += 1

            # By region
            region_key = entry.region or "full_domain"
            if region_key not in by_region:
                by_region[region_key] = 0
            by_region[region_key] += 1

        return {
            "total_samples": len(self.metadata),
            "by_variable": by_variable,
            "by_model": by_model,
            "by_region": by_region,
            "output_directory": str(self.output_dir),
            "images_directory": str(self.images_dir),
            "captions_directory": str(self.captions_dir),
        }

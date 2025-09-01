"""Load generated NOAA HRRR image-caption data as PyTorch Datasets."""

import json
from collections.abc import Sequence
from datetime import date, timedelta
from os import PathLike
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download
from PIL import Image
from torch.utils.data import Dataset
from transformers import SiglipProcessor

from hrrr_vlm.data.config import DataGeneratorConfig
from hrrr_vlm.data.generator import WeatherDataGenerator
from hrrr_vlm.data.models import TrainingRecord
from hrrr_vlm.utils.logger import configure_logger

# Configure logger
logger = configure_logger()

# Fixed region-NOAA HRRR model combinations
REGION_MODEL_PAIRS: Sequence[tuple[str, str]] = [
    ("Continental US", "hrrr"),
    ("Alaska", "hrrrak"),
    ("Northeast", "hrrr"),
    ("Northern Rockies and Plains", "hrrr"),
    ("Northwest", "hrrr"),
    ("Ohio Valley", "hrrr"),
    ("South", "hrrr"),
    ("Southeast", "hrrr"),
    ("Southwest", "hrrr"),
    ("Upper Midwest", "hrrr"),
    ("West", "hrrr"),
]

# Default weather variables for multi-variable captions
DEFAULT_VARIABLES: Sequence[str] = [
    "temperature",
    "wind_speed",
    "precipitation",
    "humidity",
]


def get_first_mondays(
    start_year: int, start_month: int, end_year: int, end_month: int
) -> list[str]:
    """Get a list of the first Mondays of each week between two dates.

    Args:
        start_year (`int`): Start year.
        start_month (`int`): Start month (1-12).
        end_year (`int`): End year.
        end_month (`int`): End month (1-12).

    Returns:
        `list[str]`: List of dates as strings in "YYYY-MM-DD" format for each Monday.
    """
    # Start on the 1st of the start month, end on the 30th of the end month
    start = date(start_year, start_month, 1)
    end = date(end_year, end_month, 30)

    # Move forward to the first Monday (weekday() == 0)
    days_to_monday = (0 - start.weekday()) % 7
    current = start + timedelta(days=days_to_monday)

    # Collect each Monday as a formatted string
    dates = []
    while current <= end:
        dates.append(current.strftime("%Y-%m-%d"))
        current += timedelta(weeks=1)
    return dates


class HRRRImageCaptionDataset(Dataset):
    """PyTorch Dataset that integrates with existing HRRR VLM modules.

    This dataset reads from JSONL files produced by WeatherDataGenerator
    and provides the interface expected by HRRRLoRACLIPTrainer.
    """

    def __init__(
        self,
        jsonl_file: PathLike[str],
        processor: SiglipProcessor,
        max_length: int = 77,
        *,
        transform_images: bool = True,
        images_base_path: PathLike[str] | None = None,
    ) -> None:
        """Initialise the HRRR image-caption dataset.

        Args:
            jsonl_file (`PathLike[str]`): Path to JSONL file created by
                WeatherDataGenerator.
            processor (`SiglipProcessor`): SigLIP processor for text and image
                tokenisation.
            max_length (`int`): Maximum length for text tokenization. Defaults
                to SigLIP's limit of 77.
            transform_images (`bool`): Whether to apply CLIP image transformations.
            images_base_path (`PathLike[str]`, optional): Base path for images
                (for Hugging Face datasets).

        Raises:
            FileNotFoundError: If JSONL file doesn't exist
            ValueError: If no valid records found
        """
        self.jsonl_file = Path(jsonl_file)
        self.processor = processor
        self.max_length = min(max_length, 77)  # Enforce CLIP limit
        self.transform_images = transform_images
        self.images_base_path = Path(images_base_path) if images_base_path else None

        if not self.jsonl_file.exists():
            msg = f"JSONL file not found: {self.jsonl_file}"
            raise FileNotFoundError(msg)

        # Load records
        self.records = self._load_records()
        if not self.records:
            msg = "No valid training records found"
            raise ValueError(msg)

        logger.info("Loaded training records", num_samples=len(self.records))

    def _load_records(self) -> list[TrainingRecord]:
        """Load TrainingRecord objects from JSONL file.

        Returns:
            `list[TrainingRecord]`: List of valid TrainingRecord objects.
        """
        records = []

        with self.jsonl_file.open(encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    record = TrainingRecord(**data)

                    # Verify image exists
                    image_path = Path(record.image_path)
                    if image_path.exists():
                        records.append(record)
                    else:
                        logger.warning(
                            "Image not found for record %s: %s",
                            record.sample_id,
                            record.image_path,
                        )

                except json.JSONDecodeError as e:
                    logger.warning("Invalid JSON on line %d: %s", line_num, e)
                except Exception as e:
                    logger.warning("Error processing line %d: %s", line_num, e)

        return records

    def __len__(self) -> int:
        """Return the number of samples in the dataset.

        Returns:
            `int`: Number of samples.
        """
        return len(self.records)

    def _get_image_path(self, record_image_path: str) -> Path:
        """Get the full image path, handling both local and relative paths.

        Args:
            record_image_path (`str`): Image path from the record.

        Returns:
            `Path`: Resolved image path.
        """
        image_path = Path(record_image_path)

        # If it's already an absolute path that exists, use it
        if image_path.is_absolute() and image_path.exists():
            return image_path

        # If we have a base path (e.g., from Hugging Face), use it
        if self.images_base_path:
            full_path = self.images_base_path / image_path
            if full_path.exists():
                return full_path

        # Try relative to JSONL file directory
        jsonl_dir = self.jsonl_file.parent
        relative_path = jsonl_dir / image_path
        if relative_path.exists():
            return relative_path

        # If image is just a filename, try in images subdirectory
        if image_path.name == str(image_path):  # Just a filename
            images_dir = jsonl_dir / "images"
            full_path = images_dir / image_path
            if full_path.exists():
                return full_path

        # Fallback: return the original path
        return image_path

    def __getitem__(self, idx: int) -> tuple[Image.Image, str]:
        """Get weather image and caption pair.

        Args:
            idx (`int`): Index of the sample.

        Returns:
            `tuple`: Tuple of (image, caption).

        Raises:
            IndexError: If index is out of range.
        """
        if idx >= len(self.records):
            msg = f"Index {idx} out of range for dataset of size {len(self.records)}"
            raise IndexError(msg)

        record = self.records[idx]

        # Load image using the flexible path resolution
        try:
            image_path = self._get_image_path(record.image_path)
            image = Image.open(image_path)
            if image.mode != "RGB":
                image = image.convert("RGB")
        except Exception as e:
            logger.warning("Failed to load image %s: %s", record.image_path, e)
            # Create blank image as fallback
            image = Image.new("RGB", (512, 512), color="lightgray")

        # Process caption (ensure within token limit)
        caption = record.caption
        if len(caption) > self.max_length * 4:  # Rough character estimate
            caption = caption[: self.max_length * 4 - 3] + "..."

        return image, caption

    @classmethod
    def from_huggingface(
        cls,
        repo_id: str,
        processor: SiglipProcessor,
        cache_dir: PathLike[str] | None = None,
    ) -> "HRRRImageCaptionDataset":
        """Load dataset directly from Hugging Face Hub.

        Args:
            repo_id (`str`): HF Hub dataset repository ID.
            processor (`SiglipProcessor`): SigLIP processor for text and image
                tokenisation.
            cache_dir (`PathLike[str]`, optional): Cache directory for HF Hub
                downloads.

        Returns:
            `HRRRImageCaptionDataset`: Loaded dataset instance.
        """
        dataset_path = snapshot_download(
            repo_id=repo_id, repo_type="dataset", cache_dir=cache_dir
        )

        dataset_path = Path(dataset_path)
        jsonl_file = dataset_path / "training_dataset.jsonl"

        return cls(
            jsonl_file=jsonl_file,
            processor=processor,
            images_base_path=dataset_path,  # Pass the base path
        )

    def get_record(self, idx: int) -> TrainingRecord:
        """Get the full TrainingRecord for an index.

        Args:
            idx: Index of the sample

        Returns:
            TrainingRecord object

        Raises:
            IndexError: If index is out of range
        """
        if idx >= len(self.records):
            msg = f"Index {idx} out of range"
            raise IndexError(msg)
        return self.records[idx]

    def filter_by_variable(self, variable: str) -> "HRRRImageCaptionDataset":
        """Create a filtered dataset containing only specified weather variable.

        Args:
            variable (`str`): Weather variable to filter by.

        Returns:
            `HRRRImageCaptionDataset`: New HRRRImageCaptionDataset instance with
                filtered records.
        """
        filtered_records = [r for r in self.records if r.variable == variable]

        # Create new dataset instance
        new_dataset = HRRRImageCaptionDataset.__new__(HRRRImageCaptionDataset)
        new_dataset.jsonl_file = self.jsonl_file
        new_dataset.processor = self.processor
        new_dataset.max_length = self.max_length
        new_dataset.transform_images = self.transform_images
        new_dataset.records = filtered_records

        logger.info(
            "Filtered dataset: %d samples for variable '%s'",
            len(filtered_records),
            variable,
        )
        return new_dataset

    def get_statistics(self) -> dict[str, Any]:
        """Get dataset statistics.

        Returns:
            `dict[str, Any]`: Dictionary with dataset statistics
        """
        stats = {
            "total_samples": len(self.records),
            "variables": {},
            "models": {},
            "dates": {"min": None, "max": None},
        }

        dates = []
        for record in self.records:
            # Count variables
            var = record.variable
            stats["variables"][var] = stats["variables"].get(var, 0) + 1

            # Count models
            model = record.model
            stats["models"][model] = stats["models"].get(model, 0) + 1

            # Collect dates
            dates.append(record.date)

        # Date range
        if dates:
            stats["dates"]["min"] = min(dates)
            stats["dates"]["max"] = max(dates)

        return stats

    @classmethod
    def create_dataset(
        cls,
        config: DataGeneratorConfig,
        processor: SiglipProcessor,
        dates: list[str],
        *,
        variables: list[str] | None = None,
        regions: list[str] | None = None,
    ) -> "HRRRImageCaptionDataset":
        """Create a HRRR image-caption dataset by generating samples and returning
        it as a PyTorch Dataset.

        This generates temperature heatmap images with multi-variable captions
        describing all available weather variables.

        Args:
            config (`DataGeneratorConfig`): Data generator configuration.
            processor (`SiglipProcessor`): SigLIP processor.
            dates (`list[str]`): List of dates in YYYY-MM-DD format.
            variables (`list[str]`, optional): Weather variables to include.
            regions (`list[str]`, optional): List of regions.

        Returns:
            `HRRRImageCaptionDataset`: HRRRImageCaptionDataset ready for training.
        """
        if variables is None:
            variables = DEFAULT_VARIABLES

        if regions is None:
            region_model_pairs = REGION_MODEL_PAIRS
        else:
            # Filter to requested regions
            region_model_pairs = [
                (region, model)
                for region, model in REGION_MODEL_PAIRS
                if region in regions
            ]

        logger.info(
            "Generating image-caption datasets",
            total_dates=len(dates),
            total_variables=len(variables),
            total_regions=len(region_model_pairs),
        )

        generator = WeatherDataGenerator(config)

        # Generate multi-variable samples
        primary_variable = "temperature"
        additional_variables = [v for v in variables if v != "temperature"]

        # Use multi-variable generation method for each region-model combination
        total_successful = 0
        for region, model in region_model_pairs:
            stats = generator.generate_samples(
                dates=dates,
                primary_variable=primary_variable,
                additional_variables=additional_variables,
                models=[model],
                regions=[region],
                fxx=0,
            )
            total_successful += stats.get("successful", 0)

        logger.info(
            "Generated multi-variable samples across all regions",
            total_successful=total_successful,
        )

        # Create image-caption dataset file
        jsonl_filename = "training_dataset.jsonl"
        jsonl_path = generator.create_training_records(jsonl_filename)

        # Return PyTorch Dataset
        return cls(jsonl_file=jsonl_path, processor=processor)

    @staticmethod
    def create_from_existing_jsonl(
        jsonl_file: PathLike[str], processor: SiglipProcessor
    ) -> "HRRRImageCaptionDataset":
        """Create a HRRRImageCaptionDataset from an existing JSONL file.

        Args:
            jsonl_file (`PathLike[str]`): Path to existing JSONL file.
            processor (`SiglipProcessor`): SigLIP processor

        Returns:
            HRRRCLIPDataset ready for training
        """
        return HRRRImageCaptionDataset(jsonl_file=jsonl_file, processor=processor)

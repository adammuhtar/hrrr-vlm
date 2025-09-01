"""Module to load HRRR image-caption datasets for training."""

import json
from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import SiglipProcessor

from hrrr_vlm.train.exceptions import DataError
from hrrr_vlm.utils.logger import configure_logger

# Configure logging
logger = configure_logger()


class HRRRImageCaptionDataSetup(Dataset):
    """Dataset class for NOAA HRRR weather image-caption pairs.

    Attributes:
        data_file (`str`): Path to the JSONL file containing weather data.
        images_dir (`str`): Directory containing weather map images.
        processor (`SiglipProcessor`): SigLIP processor for text and image
            preprocessing.
        max_length (`int`): Maximum length for text tokenisation.
    """

    def __init__(
        self,
        data_file: PathLike[str],
        images_dir: PathLike[str],
        processor: SiglipProcessor,
        max_length: int = 256,
    ) -> None:
        """Initialise the HRRR image-caption dataset.

        Args:
            data_file (`PathLike[str]`): Path to JSONL file containing HRRR data.
            images_dir  (`PathLike[str]`): Directory containing HRRR heatmap images.
            processor (`SiglipProcessor`): SigLIP processor for text and image
                preprocessing.
            max_length (`int`): Maximum length for text tokenisation.

        Raises:
            DataError: If data file or images directory is invalid
        """
        self.images_dir = Path(images_dir)
        self.processor = processor
        self.max_length = max_length
        self._rng = np.random.default_rng(42)

        # Validate paths
        if not self.images_dir.exists():
            msg = f"Images directory does not exist: {images_dir}"
            logger.error(msg)
            raise DataError(msg)

        self.data_path = Path(data_file)
        if not self.data_path.exists():
            msg = f"Data file does not exist: {data_file}"
            logger.error(msg)
            raise DataError(msg)

        try:
            self.data = self._load_data(self.data_path)
            logger.info("Weather dataset initialized with %d samples", len(self.data))
        except Exception as e:
            msg = f"Failed to load weather data: {e}"
            logger.exception(msg)
            raise DataError(msg) from e

    def _load_data(self, data_path: Path) -> list[dict[str, dict[str, str | None]]]:
        """Load image-caption data from the JSONL file.

        Args:
            data_path (`Path`): Path to the JSONL file containing weather data.

        Returns:
            `list[dict[str, dict[str, str | None]]]`: List of weather data items,
                each containing image path, caption, sample ID, variable, model,
                date, and metadata.

        Raises:
            DataError: If the data file is invalid or contains missing fields.
        """
        data = []

        try:
            with data_path.open(encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    processed_item = self._process_data_line(line, line_num)
                    if processed_item is not None:
                        data.append(processed_item)

        except Exception as e:
            msg = f"Error reading data file: {e}"
            logger.exception(msg)
            raise DataError(msg) from e

        if not data:
            msg = "No valid image-caption data found"
            logger.error(msg)
            raise DataError(msg)

        logger.info("Loaded image-caption samples", num_samples=len(data))
        return data

    def _process_data_line(
        self, line: str, line_num: int
    ) -> dict[str, dict[str, str | None]] | None:
        """Process a single line from the JSONL data file.

        Args:
            line (`str`): JSON line to process.
            line_num (`int`): Line number for logging.

        Returns:
            `dict[str, dict[str, str | None]]`: Processed item or None if invalid.
        """
        try:
            item = json.loads(line.strip())
        except json.JSONDecodeError as e:
            logger.warning("Line %d: invalid JSON: %s", line_num, e)
            return None
        except Exception as e:
            logger.warning("Line %d: error processing item: %s", line_num, e)
            return None

        # Validate required fields
        required_fields = [
            "image_path",
            "caption",
            "sample_id",
            "variable",
            "model",
            "date",
        ]
        missing_fields = [field for field in required_fields if field not in item]

        if missing_fields:
            logger.warning(
                "Line %d: missing required fields: %s", line_num, missing_fields
            )
            return None

        # Process image path
        if not self._process_image_path(item, line_num):
            return None

        # Add metadata
        item["region"] = item.get("metadata", {}).get("region", "Unknown")
        item["season"] = self._get_season_from_date(item["date"])

        return item

    def _process_image_path(self, item: dict, line_num: int) -> bool:
        """Process and validate image path for an item.

        Args:
            item (`dict`): Data item to process.
            line_num (`int`): Line number for logging.

        Returns:
            `bool`: True if image path is valid, False otherwise.
        """
        image_path = Path(item["image_path"])

        # Resolve image path relative to data file if it's relative
        if not image_path.is_absolute():
            full_image_path = (self.data_path.parent / image_path).resolve()
        else:
            full_image_path = image_path

        # Extract just the filename for backward compatibility
        item["image_filename"] = image_path.name

        # Check if the resolved image path exists
        if not full_image_path.exists():
            # Fall back to checking in images_dir with filename
            fallback_path = self.images_dir / item["image_filename"]
            if not fallback_path.exists():
                logger.warning(
                    "Line %d: image file not found at %s or %s",
                    line_num,
                    full_image_path,
                    fallback_path,
                )
                return False
            # Use the fallback path
            item["resolved_image_path"] = fallback_path
        else:
            item["resolved_image_path"] = full_image_path

        return True

    @staticmethod
    def _get_season_from_date(date_str: str) -> str:
        """Extract season from date string.

        Args:
            date_str (`str`): Date string in format 'YYYY-MM-DD'.

        Returns:
            `str`: Season name ('winter', 'spring', 'summer', 'fall', or 'unknown').
        """
        try:
            month = int(date_str.split("-")[1])
            if month in {12, 1, 2}:
                return "winter"
            if month in {3, 4, 5}:
                return "spring"
            if month in {6, 7, 8}:
                return "summer"
        except (ValueError, IndexError):
            logger.warning("Could not parse date: %s", date_str)
            return "unknown"
        else:
            return "fall"

    def __len__(self) -> int:
        """Return the number of samples in the dataset.

        Returns:
            `int`: Number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[Image.Image, str]:
        """Get a single item from the dataset.

        Args:
            idx (`int`): Index of the item to retrieve.

        Returns:
            `tuple[Image.Image, str]`: Tuple containing the weather image and its
                caption.

        Raises:
            DataError: If the index is out of range or image loading fails.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        try:
            item = self.data[idx]
        except IndexError as e:
            msg = f"Index {idx} out of range for dataset of size {len(self.data)}"
            logger.exception(msg)
            raise DataError(msg) from e

        image_path = item.get(
            "resolved_image_path", self.images_dir / item["image_filename"]
        )

        try:
            img = Image.open(image_path)
            img.verify()
            img = Image.open(image_path)
            if img.mode != "RGB":
                img = img.convert("RGB")
        except (FileNotFoundError, OSError) as e:
            logger.warning(
                "Failed to load weather image %s: %s. Using blank image.",
                item["image_filename"],
                e,
            )
            img = Image.new("RGB", (512, 512), color="lightgray")
        except Exception:
            logger.exception(
                "Unexpected error loading weather image %s", item["image_filename"]
            )
            img = Image.new("RGB", (512, 512), color="lightgray")

        caption = item["caption"]
        enhanced_caption = self._enhance_weather_caption(caption, item)
        return img, enhanced_caption

    def _enhance_weather_caption(self, caption: str, item: dict[str, Any]) -> str:
        """Enhance the caption with additional metadata.

        Args:
            caption (`str`): Original caption text.
            item (`dict[str, Any]`): Weather data item containing metadata.

        Returns:
            `str`: Enhanced caption with model, season, and region information.
        """
        enhanced = caption
        model_info = f" Generated from {item['model'].upper()} model"
        season_info = f" during {item['season']} season"
        region = (item.get("region") or "").strip()
        region_info = (
            f" in the {region} region" if region and region != "Unknown" else ""
        )
        enhanced = f"{enhanced}{model_info}{season_info}{region_info}."
        if len(enhanced) > self.max_length:
            enhanced = enhanced[: self.max_length - 3] + "..."
        return enhanced

    def get_weather_statistics(self) -> dict[str, Any]:
        """Compute basic statistics about the weather dataset.

        Returns:
            `dict[str, Any]`: Dictionary containing statistics like total samples,
                variable counts, model counts, region counts, season counts, and
                date range.
        """
        stats = {
            "total_samples": len(self.data),
            "variables": {},
            "models": {},
            "regions": {},
            "seasons": {},
            "date_range": {"min": None, "max": None},
        }

        dates = []
        for item in self.data:
            var = item.get("variable", "unknown")
            stats["variables"][var] = stats["variables"].get(var, 0) + 1
            model = item.get("model", "unknown")
            stats["models"][model] = stats["models"].get(model, 0) + 1
            region = item.get("region", "unknown")
            stats["regions"][region] = stats["regions"].get(region, 0) + 1
            season = item.get("season", "unknown")
            stats["seasons"][season] = stats["seasons"].get(season, 0) + 1
            dates.append(item.get("date", ""))

        valid_dates = [d for d in dates if d]
        if valid_dates:
            stats["date_range"]["min"] = min(valid_dates)
            stats["date_range"]["max"] = max(valid_dates)
        return stats

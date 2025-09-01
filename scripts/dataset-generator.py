"""Script to generate a CLIP image-caption dataset from NOAA HRRR data.

Usage:
    ```bash
    uv sync --locked --extra dataset
    uv run scripts/dataset-generator.py --help
    ```
"""

import argparse
import sys
from pathlib import Path

from transformers import SiglipProcessor

from hrrr_vlm.data.config import DataGeneratorConfig
from hrrr_vlm.data.constants import REGIONS
from hrrr_vlm.data.loader import (
    DEFAULT_VARIABLES,
    HRRRImageCaptionDataset,
    get_first_mondays,
)
from hrrr_vlm.utils.logger import configure_logger

# Configure logger
logger = configure_logger()

# Globals and defaults
DEFAULT_MODEL_ID = "google/siglip-base-patch16-224"
REGION_NAMES = ["Continental US", *list(REGIONS.keys())]
START_YEAR = 2019
START_MONTH = 1
END_YEAR = 2025
END_MONTH = 6
OUTPUT_DIR = Path.cwd() / "data" / "hrrr_hrrrak_all"


def _validate_range(sy: int, sm: int, ey: int, em: int) -> None:
    """Validate that the end (year, month) is not earlier than the start.

    Args:
        sy (`int`): Start year.
        sm (`int`): Start month (1-12).
        ey (`int`): End year.
        em (`int`): End month (1-12).

    Raises:
        ValueError: If (ey, em) is earlier than (sy, sm).
    """
    if (ey, em) < (sy, sm):
        err_msg = (
            "End date must not be earlier than start date. "
            f"Start: {sy}-{sm}, End: {ey}-{em}"
        )
        logger.error(err_msg)
        raise ValueError(err_msg)


def main(
    start_year: int = START_YEAR,
    start_month: int = START_MONTH,
    end_year: int = END_YEAR,
    end_month: int = END_MONTH,
    *,
    dry_run: bool = False,
) -> int:
    """Main function to generate a CLIP dataset using HRRR data.

    Args:
        start_year (`int`): Start year for the dataset.
        start_month (`int`): Start month for the dataset.
        end_year (`int`): End year for the dataset.
        end_month (`int`): End month for the dataset.
        dry_run (`bool`, optional): If True, only parse inputs and list summary,
            then exit without generating the dataset.

    Returns:
        `int`: Exit code. 0 for success, non-zero for errors.
    """
    _validate_range(start_year, start_month, end_year, end_month)

    dates = list(get_first_mondays(start_year, start_month, end_year, end_month))
    if not dates:
        logger.error(
            "No dates generated. Check your range: %04d-%02d to %04d-%02d",
            start_year,
            start_month,
            end_year,
            end_month,
        )
        return 2

    logger.info(
        "Starting dataset generation",
        model_id=DEFAULT_MODEL_ID,
        variables=DEFAULT_VARIABLES,
        total_dates=len(dates),
        output_dir=str(OUTPUT_DIR),
    )

    if dry_run:
        logger.info("Dry run requested; exiting before any heavy work.")
        return 0

    try:
        processor = SiglipProcessor.from_pretrained(DEFAULT_MODEL_ID)
    except Exception:
        logger.exception(
            "Failed to load SigLIP processor for model", model_id=DEFAULT_MODEL_ID
        )
        return 3

    try:
        config = DataGeneratorConfig(output_dir=str(OUTPUT_DIR))

        # Use the streamlined approach to create dataset
        HRRRImageCaptionDataset.create_dataset(
            config=config,
            processor=processor,
            dates=dates,
            variables=DEFAULT_VARIABLES,
            regions=REGION_NAMES,
        )
    except Exception:
        logger.exception("Dataset generation failed")
        return 4

    logger.info("Dataset generation complete", output_dir=OUTPUT_DIR)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a CLIP image-caption dataset from HRRR data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=START_YEAR,
        help="Start year for the dataset. Start year inclusive.",
    )
    parser.add_argument(
        "--start-month",
        type=int,
        default=START_MONTH,
        help="Start month for the dataset. Start month inclusive.",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=END_YEAR,
        help="End year for the dataset. End year inclusive.",
    )
    parser.add_argument(
        "--end-month",
        type=int,
        default=END_MONTH,
        help="End month for the dataset. End month inclusive.",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Parse inputs, list summary, then exit."
    )

    args = parser.parse_args()
    sys.exit(
        main(
            start_year=args.start_year,
            start_month=args.start_month,
            end_year=args.end_year,
            end_month=args.end_month,
            dry_run=args.dry_run,
        )
    )

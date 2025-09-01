# !/usr/bin/env python3
"""Script to run LoRA fine-tuning of SigLIP model on HRRR dataset.

Usage:
    ```bash
    uv run scripts/model-finetuning.py --help
    ```
"""

import argparse
import sys
from pathlib import Path

from hrrr_vlm.train.train import (
    HRRRLoRASigLIPTrainer,
    ModelTrainingConfig,
    training_init,
)
from hrrr_vlm.utils.logger import configure_logger

# Configure logger
logger = configure_logger()

# Globals and defaults
DEFAULT_MODEL_NAME = "google/siglip-base-patch16-224"
DEFAULT_DATA_FILE = (
    Path.cwd() / "data" / "NOAA-HRRR-HRRRAK-All-ImageCaption" / "training_dataset.jsonl"
)
DEFAULT_IMAGES_DIR = (
    Path.cwd() / "data" / "NOAA-HRRR-HRRRAK-All-ImageCaption" / "images"
)
DEFAULT_SAVE_PATH = Path.cwd() / "models" / "hrrr-all-siglip"
DEFAULT_LOG_FILE = Path.cwd() / "logs" / "hrrr-all-siglip-training-log.csv"

DEFAULT_NUM_EPOCHS = 20
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_BATCH_SIZE = 32
DEFAULT_TRAIN_SPLIT = 0.8
DEFAULT_VAL_SPLIT = 0.15
DEFAULT_TEST_SPLIT = 0.05
DEFAULT_NUM_WORKERS = 0
DEFAULT_MAX_LENGTH = 512

DEFAULT_LORA_R = 32
DEFAULT_LORA_ALPHA = 64
DEFAULT_LORA_DROPOUT = 0.05


def main(
    model_name: str = DEFAULT_MODEL_NAME,
    data_file: str = str(DEFAULT_DATA_FILE),
    images_dir: str = str(DEFAULT_IMAGES_DIR),
    save_path: str = str(DEFAULT_SAVE_PATH),
    log_file: str = str(DEFAULT_LOG_FILE),
    num_epochs: int = DEFAULT_NUM_EPOCHS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    train_split: float = DEFAULT_TRAIN_SPLIT,
    val_split: float = DEFAULT_VAL_SPLIT,
    test_split: float = DEFAULT_TEST_SPLIT,
    num_workers: int = DEFAULT_NUM_WORKERS,
    max_length: int = DEFAULT_MAX_LENGTH,
    lora_r: int = DEFAULT_LORA_R,
    lora_alpha: int = DEFAULT_LORA_ALPHA,
    lora_dropout: float = DEFAULT_LORA_DROPOUT,
    *,
    dry_run: bool = False,
) -> int:
    """Main function to fine-tune SigLIP model with LoRA.

    Args:
        model_name (`str`): Name of the pre-trained model to fine-tune.
        data_file (`str`): Path to the training dataset JSONL file.
        images_dir (`str`): Directory containing the training images.
        save_path (`str`): Path to save the trained model.
        log_file (`str`): Path to save the training log.
        num_epochs (`int`): Number of training epochs.
        learning_rate (`float`): Learning rate for training.
        batch_size (`int`): Batch size for training.
        train_split (`float`): Proportion of data for training.
        val_split (`float`): Proportion of data for validation.
        test_split (`float`): Proportion of data for testing.
        num_workers (`int`): Number of workers for data loading.
        max_length (`int`): Maximum sequence length.
        lora_r (`int`): LoRA rank parameter.
        lora_alpha (`int`): LoRA alpha parameter.
        lora_dropout (`float`): LoRA dropout rate.
        dry_run (`bool`, optional): If True, only validate inputs and show config,
            then exit without training.

    Returns:
        `int`: Exit code. 0 for success, non-zero for errors.
    """
    # Validate input files exist
    data_path = Path(data_file)
    images_path = Path(images_dir)

    if not data_path.exists():
        logger.error("Data file does not exist: %s", data_file)
        return 1

    if not images_path.exists():
        logger.error("Images directory does not exist: %s", images_dir)
        return 2

    # Ensure output directories exist
    save_dir = Path(save_path)
    log_dir = Path(log_file).parent
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Starting model fine-tuning",
        model_name=model_name,
        data_file=data_file,
        images_dir=images_dir,
        save_path=save_path,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
    )

    if dry_run:
        logger.info("Dry run requested; exiting before any heavy work.")
        return 0

    try:
        training_init()

        config = ModelTrainingConfig(
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            train_split=train_split,
            val_split=val_split,
            test_split=test_split,
            num_workers=num_workers,
            max_length=max_length,
        )

        trainer = HRRRLoRASigLIPTrainer(
            model_name=model_name,
            lora_config={
                "r": lora_r,
                "lora_alpha": lora_alpha,
                "target_modules": ["q_proj", "k_proj", "v_proj", "out_proj"],
                "lora_dropout": lora_dropout,
                "bias": "none",
            },
        )

        trainer.setup_model()

        train_loader, val_loader, _ = trainer.create_data_loaders(
            data_file=data_file, images_dir=images_dir, config=config
        )

        trainer.train_model(
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            save_path=save_path,
            log_file=log_file,
        )

    except Exception:
        logger.exception("Model fine-tuning failed")
        return 3

    logger.info("Model fine-tuning complete", save_path=save_path)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune SigLIP model with LoRA on HRRR dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="Name of the pre-trained model to fine-tune.",
    )
    parser.add_argument(
        "--data-file",
        type=str,
        default=str(DEFAULT_DATA_FILE),
        help="Path to the training dataset JSONL file.",
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default=str(DEFAULT_IMAGES_DIR),
        help="Directory containing the training images.",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default=str(DEFAULT_SAVE_PATH),
        help="Path to save the trained model.",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=str(DEFAULT_LOG_FILE),
        help="Path to save the training log.",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=DEFAULT_NUM_EPOCHS,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help="Learning rate for training.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size for training.",
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=DEFAULT_TRAIN_SPLIT,
        help="Proportion of data for training.",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=DEFAULT_VAL_SPLIT,
        help="Proportion of data for validation.",
    )
    parser.add_argument(
        "--test-split",
        type=float,
        default=DEFAULT_TEST_SPLIT,
        help="Proportion of data for testing.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=DEFAULT_NUM_WORKERS,
        help="Number of workers for data loading.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=DEFAULT_MAX_LENGTH,
        help="Maximum sequence length.",
    )
    parser.add_argument(
        "--lora-r", type=int, default=DEFAULT_LORA_R, help="LoRA rank parameter."
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=DEFAULT_LORA_ALPHA,
        help="LoRA alpha parameter.",
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=DEFAULT_LORA_DROPOUT,
        help="LoRA dropout rate.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs, show config, then exit.",
    )

    args = parser.parse_args()
    sys.exit(
        main(
            model_name=args.model_name,
            data_file=args.data_file,
            images_dir=args.images_dir,
            save_path=args.save_path,
            log_file=args.log_file,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            train_split=args.train_split,
            val_split=args.val_split,
            test_split=args.test_split,
            num_workers=args.num_workers,
            max_length=args.max_length,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            dry_run=args.dry_run,
        )
    )

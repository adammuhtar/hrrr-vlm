"""Module for parameter efficient fine-tuning of SigLIP models via low-rank adaptation
(LoRA) on NOAA HRRR Weather Dataset.
"""

import random
import time
from os import PathLike
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
import torch
from peft import LoraConfig, get_peft_model
from PIL import Image
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
)
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import SiglipModel, SiglipProcessor

from hrrr_vlm.train.data_loader import HRRRImageCaptionDataSetup
from hrrr_vlm.train.exceptions import DataError, ModelInitError, ModelTrainingError
from hrrr_vlm.utils.logger import configure_logger

# Configure logging
logger = configure_logger()


def training_init(random_seed: int = 42) -> None:
    """Setup random seeds for reproducible model training.

    Args:
        random_seed (`int`): Seed value for random number generators.
    """
    np.random.default_rng(random_seed)
    torch.manual_seed(random_seed)
    random.seed(random_seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        logger.info("CUDA random seeds set for training")

    logger.info("Random seed initialised for training", random_seed=random_seed)


class ModelTrainingConfig(BaseModel):
    """Configuration for model training parameters.

    Attributes:
        model_name (`str`): Hugging Face model identifier for SigLIP.
        model_config (`ConfigDict`): Configuration for the model.
        num_epochs (`PositiveInt`): Number of training epochs.
        learning_rate (`PositiveFloat`): Learning rate for the optimiser.
        batch_size (`PositiveInt`): Batch size for training.
        train_split (`NonNegativeFloat`): Ratio of data to use for training.
        val_split (`NonNegativeFloat`): Ratio of data to use for validation.
        test_split (`NonNegativeFloat`): Ratio of data to use for testing.
        num_workers (`NonNegativeInt`): Number of workers for data loading.
        max_length (`PositiveInt`): Maximum length for text tokenisation.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
        validate_assignment=True,
        arbitrary_types_allowed=False,
        validate_by_name=True,
    )
    num_epochs: PositiveInt = Field(default=10, description="Number of training epochs")
    learning_rate: PositiveFloat = Field(
        default=5e-5, description="Learning rate for optimiser"
    )
    batch_size: PositiveInt = Field(default=32, description="Batch size for training")
    train_split: NonNegativeFloat = Field(
        default=0.8, le=1, description="Training data split ratio"
    )
    val_split: NonNegativeFloat = Field(
        default=0.1, le=1, description="Validation data split ratio"
    )
    test_split: NonNegativeFloat = Field(
        default=0.1, le=1, description="Test data split ratio"
    )
    num_workers: NonNegativeInt = Field(
        default=0, description="Number of data loader workers"
    )
    max_length: PositiveInt = Field(
        default=1024, gt=0, description="Maximum caption length"
    )


class TrainingEpochResults(NamedTuple):
    """Results from a weather training epoch."""

    epoch: int
    train_loss: float
    val_loss: float
    val_accuracy: float


class HRRRLoRASigLIPTrainer:
    """Trainer class for LoRA fine-tuning of SigLIP model on HRRR weather data.

    Attributes:
        model_name (`str`): Hugging Face model identifier for SigLIP.
        lora_config (`dict[str, Any]`): Configuration for LoRA adaptation.
        device (`torch.device`): Device to run training on (CUDA/MPS/CPU).
        processor (`SiglipProcessor`): SigLIP processor for text and image
            preprocessing.
        model (`torch.nn.Module`, optional): SigLIP model instance.
        criterion (`torch.nn.Module`): Loss function for training.
        logit_scale (`torch.nn.Parameter`, optional): Learnable logit scale
            parameter.
    """

    def __init__(
        self,
        model_name: str = "google/siglip-base-patch16-224",
        lora_config: dict[str, Any] | None = None,
        device: torch.device | None = None,
    ) -> None:
        """Initialise the HRRR SigLIP trainer.

        Args:
            model_name (`str`): Hugging Face model identifier.
            lora_config (`dict[str, Any]`, optional): Configuration for LoRA
                adaptation.
            device (`torch.device`): Device to run training on (CUDA/MPS/CPU).

        Raises:
            ModelInitError: If model or processor initialisation fails.
        """
        self.model_name = model_name
        self.device = device or torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_built()
            else "cpu"
        )
        logger.info("Torch device", device=self.device)

        # LoRA configuration applied to attention projections
        self.lora_config = lora_config or {
            "r": 32,
            "lora_alpha": 32,
            "target_modules": ["q_proj", "k_proj", "v_proj", "out_proj"],
            "lora_dropout": 0.05,
            "bias": "none",
        }

        try:
            self.processor = SiglipProcessor.from_pretrained(model_name)
            logger.info("SigLIP processor loaded successfully for weather data")
        except Exception as e:
            msg = f"Failed to load SigLIP processor: {e}"
            logger.exception(msg)
            raise ModelInitError(msg) from e

        self.model: torch.nn.Module | None = None

        # Learnable logit scale parameter
        self.logit_scale: torch.nn.Parameter | None = None

    def setup_model(self) -> None:
        """Initialise and setup the SigLIP model with LoRA adaptation.

        Raises:
            ModelInitError: If model setup fails.
        """
        try:
            logger.info(
                "Loading base SigLIP model for weather analysis: %s", self.model_name
            )
            base_model = SiglipModel.from_pretrained(self.model_name)
            base_model = base_model.to(self.device)

            logger.info(
                "Applying weather-optimized LoRA configuration: %s", self.lora_config
            )
            config = LoraConfig(**self.lora_config)
            self.model = get_peft_model(base_model, config)

            # Check if logit_scale already exists, if not create it
            if not hasattr(self.model, "logit_scale"):
                logit_scale_value = torch.tensor(np.log(1 / 0.07), dtype=torch.float32)
                self.model.register_parameter(
                    "logit_scale", torch.nn.Parameter(logit_scale_value.to(self.device))
                )
                logger.info("Created new logit_scale parameter")
            else:
                logger.info("Using existing logit_scale parameter from base model")

            # Keep a reference for easy access
            self.logit_scale = self.model.logit_scale

            logger.info("Weather SigLIP model setup complete")
            self.print_trainable_parameters()

        except Exception as e:
            msg = f"Failed to setup SigLIP weather model: {e}"
            logger.exception(msg)
            raise ModelInitError(msg) from e

    def print_trainable_parameters(self) -> None:
        """Print the number of trainable parameters in the model."""
        if self.model is None:
            logger.warning("Weather model not initialised. Call setup_model() first.")
            return
        try:
            trainable_params = 0
            all_param = 0
            for _, param in self.model.named_parameters():
                all_param += param.numel()
                if param.requires_grad:
                    trainable_params += param.numel()
            trainable_percentage = 100 * trainable_params / all_param
            logger.info(
                "SigLIP model trainable parameters",
                trainable_params=trainable_params,
                all_param=all_param,
                trainable_percentage=trainable_percentage,
            )
        except Exception:
            logger.exception("Error calculating trainable parameters for weather model")

    def training_batch_builder(
        self, samples: list[tuple[Image.Image, str]]
    ) -> dict[str, torch.Tensor]:
        """Build a training batch from a list of image-caption pairs.

        Args:
            samples (`list[tuple[Image.Image, str]]`): List of tuples containing
                images and their corresponding captions.

        Returns:
            `dict[str, torch.Tensor]`: Dictionary containing processed images and
                tokenised captions ready for model input.

        Raises:
            DataError: If processing fails for any reason.
        """
        try:
            processor = SiglipProcessor.from_pretrained(self.model_name)
            images, captions = zip(*samples, strict=True)
            return processor(
                text=list(captions),
                images=list(images),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256,
            )
        except Exception as e:
            msg = f"Failed to process weather batch: {e}"
            logger.exception(msg)
            raise DataError(msg) from e

    def create_data_loaders(
        self,
        data_file: PathLike[str],
        images_dir: PathLike[str],
        config: ModelTrainingConfig | None = None,
    ) -> tuple[DataLoader, DataLoader, DataLoader]:
        """Create data loaders for HRRR weather dataset.

        Args:
            data_file (`PathLike[str]`): Path to the JSONL file containing HRRR data.
            images_dir (`PathLike[str]`): Directory containing HRRR heatmap images.
            config (`ModelTrainingConfig`, optional): Configuration for training
                parameters.

        Returns:
            `tuple[DataLoader, DataLoader, DataLoader]`: Tuple containing training,
                validation, and test data loaders.

        Raises:
            DataError: If data loading or processing fails.
        """
        if config is None:
            config = ModelTrainingConfig()

        logger.info("Creating weather data loaders with config: %s", config)

        try:
            full_dataset = HRRRImageCaptionDataSetup(
                data_file=data_file,
                images_dir=images_dir,
                processor=self.processor,
                max_length=config.max_length,
            )

            stats = full_dataset.get_weather_statistics()
            logger.info("Weather dataset statistics: %s", stats)

            total_size = len(full_dataset)
            train_size = int(total_size * config.train_split)
            val_size = int(total_size * config.val_split)
            test_size = total_size - train_size - val_size

            train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
                full_dataset,
                [train_size, val_size, test_size],
                generator=torch.Generator().manual_seed(42),
            )

            logger.info(
                "Weather data split into sets",
                train_set=len(train_dataset),
                val_set=len(val_dataset),
                test_set=len(test_dataset),
            )

            train_loader = DataLoader(
                train_dataset,
                batch_size=config.batch_size,
                shuffle=True,
                num_workers=config.num_workers if torch.cuda.is_available() else 0,
                collate_fn=self.training_batch_builder,
                pin_memory=torch.cuda.is_available(),
                drop_last=True,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=config.num_workers if torch.cuda.is_available() else 0,
                collate_fn=self.training_batch_builder,
                pin_memory=torch.cuda.is_available(),
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=config.num_workers if torch.cuda.is_available() else 0,
                collate_fn=self.training_batch_builder,
                pin_memory=torch.cuda.is_available(),
            )

        except Exception as e:
            if isinstance(e, DataError):
                raise
            err_msg = f"Failed to create data loaders: {e}"
            logger.exception(err_msg)
            raise DataError(err_msg) from e
        else:
            logger.info(
                "Data loaders created successfully",
                train_set=len(train_loader.dataset),
                val_set=len(val_loader.dataset),
                test_set=len(test_loader.dataset),
            )
            return train_loader, val_loader, test_loader

    def _forward_and_similarities(
        self, sample: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run SigLIP forward pass and compute similarity logits.

        Args:
            sample (`dict[str, torch.Tensor]`): A batch sample containing 'input_ids',
                'pixel_values', and optionally 'attention_mask'.

        Returns:
            `tuple[torch.Tensor, torch.Tensor]`: Tuple containing logits for image
                and text embeddings.

        Raises:
            ModelInitError: If the model is not initialised.
        """
        if self.model is None:
            err_msg = "Model not initialised. Call setup_model() first."
            logger.error(err_msg)
            raise ModelInitError(err_msg)

        input_ids = sample["input_ids"].to(self.device)
        pixel_values = sample["pixel_values"].to(self.device)

        # Check if attention_mask exists, if not create it
        if "attention_mask" in sample:
            attention_mask = sample["attention_mask"].to(self.device)
        else:
            # Create attention mask for all tokens
            attention_mask = torch.ones_like(input_ids).to(self.device)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
        )

        # SigLIP returns text_embeds and image_embeds
        text_embeds = outputs.text_embeds  # (B, D)
        image_embeds = outputs.image_embeds  # (B, D)

        # Normalise, scale, and compute pairwise similarities
        text_embeds = nn.functional.normalize(text_embeds, p=2, dim=-1)
        image_embeds = nn.functional.normalize(image_embeds, p=2, dim=-1)

        # Learnable logit scale
        scale = (
            self.model.logit_scale.exp()
            if hasattr(self.model, "logit_scale")
            else self.logit_scale.exp()
        )
        logits_per_image = scale * image_embeds @ text_embeds.t()  # (B, B)
        logits_per_text = logits_per_image.t()  # symmetry

        return logits_per_image, logits_per_text

    def evaluate_model(self, val_loader: DataLoader) -> tuple[float, float]:
        """Evaluate the model on a validation dataset.

        Args:
            val_loader (`DataLoader`): DataLoader for the validation dataset.

        Returns:
            `tuple[float, float]`: Tuple containing validation accuracy and
                average loss.

        Raises:
            ModelInitError: If the model is not initialised.
            ModelTrainingError: If evaluation fails.
        """
        if self.model is None:
            msg = "Model not initialised. Call setup_model() first."
            logger.error(msg)
            raise ModelInitError(msg)

        try:
            self.model.eval()
            running_corrects = 0.0
            running_loss = 0.0
            total_samples = 0

            logger.info("Starting model evaluation")
            with torch.no_grad():
                for sample in tqdm(val_loader, desc="Evaluating Model"):
                    logits_per_image, logits_per_text = self._forward_and_similarities(
                        sample
                    )
                    B = logits_per_image.size(0)  # noqa: N806
                    targets = torch.eye(
                        B, device=self.device, dtype=logits_per_image.dtype
                    )
                    pos_weight = torch.full(
                        (B,), fill_value=(B - 1.0), device=self.device
                    )

                    loss_i = F.binary_cross_entropy_with_logits(
                        logits_per_image, targets, pos_weight=pos_weight
                    )
                    loss_t = F.binary_cross_entropy_with_logits(
                        logits_per_text, targets, pos_weight=pos_weight
                    )
                    loss = 0.5 * (loss_i + loss_t)

                    # Retrieval accuracy@1 (image→text)
                    targets_idx = torch.arange(B, device=self.device)
                    _, preds = torch.max(logits_per_image, dim=1)

                    running_loss += loss.item() * B
                    running_corrects += torch.sum(preds == targets_idx).item()
                    total_samples += B

            accuracy = running_corrects / total_samples
            avg_loss = running_loss / total_samples

        except Exception as e:
            msg = f"Model evaluation failed: {e}"
            logger.exception(msg)
            raise ModelTrainingError(msg) from e

        else:
            logger.info(
                "Weather model evaluation complete",
                accuracy=accuracy,
                avg_loss=avg_loss,
            )
            return accuracy, avg_loss

    def train_model(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: ModelTrainingConfig | None = None,
        save_path: str | None = None,
        log_file: str | None = None,
    ) -> None:
        """Train the SigLIP model on HRRR weather data.

        Args:
            train_loader (`DataLoader`): DataLoader for the training dataset.
            val_loader (`DataLoader`): DataLoader for the validation dataset.
            config (`ModelTrainingConfig`, optional): Configuration for training
                parameters.
            save_path (`str`, optional): Path to save the trained model.
            log_file (`str`, optional): Path to save training logs.

        Raises:
            ModelInitError: If the model is not initialised.
            ModelTrainingError: If training fails.
            RuntimeError: If GPU runs out of memory during training.
        """
        if self.model is None:
            err_msg = "Model not initialised. Call setup_model() first."
            logger.error(err_msg)
            raise ModelInitError(err_msg)

        if config is None:
            config = ModelTrainingConfig()

        logger.info("Starting weather SigLIP training with config: %s", config)

        try:  # noqa: PLR1702
            optimiser = optim.AdamW(
                self.model.parameters(),
                lr=config.learning_rate,
                weight_decay=0.01,
                eps=1e-8,
            )

            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimiser, T_max=config.num_epochs, eta_min=config.learning_rate * 0.1
            )

            log_data, log_file_path = self._setup_training_logs(log_file)
            since = time.time()
            train_set_size = len(train_loader.dataset)

            # Track best model performance
            best_val_loss = float("inf")
            best_epoch = 0

            pbar = tqdm(
                iterable=range(1, config.num_epochs + 1),
                desc="Fine-tuning SigLIP Model",
                total=config.num_epochs,
                unit="epoch",
            )

            for epoch in pbar:
                try:
                    self.model.train()
                    running_loss = 0.0

                    for batch_idx, sample in enumerate(train_loader):
                        try:
                            loss_value, batch_size = self._process_weather_batch(
                                sample, optimiser
                            )
                            running_loss += loss_value * batch_size
                        except RuntimeError as e:
                            if "out of memory" in str(e).lower():
                                logger.warning(
                                    "GPU out of memory at batch %d, skipping", batch_idx
                                )
                                torch.cuda.empty_cache()
                                continue
                            raise

                    train_loss = running_loss / train_set_size

                    val_accuracy, val_loss = self.evaluate_model(val_loader)
                    scheduler.step()

                    epoch_results = TrainingEpochResults(
                        epoch=epoch,
                        train_loss=train_loss,
                        val_loss=val_loss,
                        val_accuracy=val_accuracy,
                    )

                    self._log_epoch_results(
                        epoch_results, log_data=log_data, log_file_path=log_file_path
                    )

                    # Check if this is the best model so far
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_epoch = epoch
                        # Save the best model
                        if save_path:
                            self.save_model(save_path)
                            logger.info(
                                "New best model saved at epoch %d with "
                                "validation loss: %.4f",
                                epoch,
                                val_loss,
                            )

                    pbar.set_description(
                        f"Epoch {epoch}/{config.num_epochs} - "
                        f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                        f"Val Acc: {val_accuracy:.4f}, "
                        f"Best Epoch: {best_epoch}"
                    )

                except Exception:
                    logger.exception("Error in weather training epoch %d", epoch)
                    if epoch == 1:
                        raise
                    continue

            training_time = time.time() - since
            logger.info(
                "Weather SigLIP training complete in %.0fm %.0fs",
                training_time // 60,
                training_time % 60,
            )
            logger.info(
                "Best model achieved at epoch %d with validation loss: %.4f",
                best_epoch,
                best_val_loss,
            )

            # Final save is not needed since we save the best model during training

        except Exception as e:
            if isinstance(e, (ModelInitError, ModelTrainingError)):
                raise
            msg = f"SigLIP fine-tuning failed: {e}"
            logger.exception(msg)
            raise ModelTrainingError(msg) from e

    @staticmethod
    def _setup_training_logs(
        log_file: PathLike[str] | None,
    ) -> tuple[list, Path | None]:
        """Setup training logs for model training.

        Args:
            log_file (`PathLike[str]`, optional): Path to save training logs.

        Returns:
            `tuple[list, Path | None]`: Tuple containing log data list and log
                file path.
        """
        log_data = []
        log_file_path = None
        if log_file:
            log_file_path = Path(log_file)
            log_file_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info("Model training logs will be saved to: %s", log_file_path)
        return log_data, log_file_path

    def _process_weather_batch(
        self, sample: dict[str, Any], optimiser: optim.Optimizer
    ) -> tuple[float, int]:
        """Process a single training batch (SigLIP).

        Args:
            sample (`dict[str, Any]`): A batch sample containing 'input_ids' and
                'pixel_values'.
            optimiser (`optim.Optimizer`): Optimiser for updating model parameters.

        Returns:
            `tuple[float, int]`: Tuple containing the loss value and batch size.

        Raises:
            ModelInitError: If the model is not initialised.
        """
        if self.model is None:
            err_msg = "Model not initialised. Call setup_model() first."
            logger.error(err_msg)
            raise ModelInitError(err_msg)

        input_ids = sample["input_ids"].to(self.device)
        batch_size = input_ids.size(0)

        optimiser.zero_grad()

        logits_per_image, logits_per_text = self._forward_and_similarities(sample)

        # BCE-with-logits targets: identity matrix (positives on the diagonal)
        targets = torch.eye(
            batch_size, device=self.device, dtype=logits_per_image.dtype
        )

        # Rebalance positives vs many negatives (pos_weight = B-1 per class)
        pos_weight = torch.full(
            (batch_size,), fill_value=(batch_size - 1.0), device=self.device
        )

        loss_i = F.binary_cross_entropy_with_logits(
            logits_per_image, targets, pos_weight=pos_weight
        )
        loss_t = F.binary_cross_entropy_with_logits(
            logits_per_text, targets, pos_weight=pos_weight
        )
        loss = 0.5 * (loss_i + loss_t)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        optimiser.step()

        return loss.item(), batch_size

    @staticmethod
    def _log_epoch_results(
        results: TrainingEpochResults,
        *,
        log_data: list[dict[str, Any]],
        log_file_path: PathLike[str] | None = None,
    ) -> None:
        """Log the results of a training epoch.

        Args:
            results (`TrainingEpochResults`): Results from the training epoch.
            log_data (`list[dict[str, Any]]`): List to accumulate log data.
            log_file_path (`Path`, optional): Path to save the epoch results as
                CSV.
        """
        epoch_data = results._asdict()
        log_data.append(epoch_data)
        if log_file_path:
            log_file_path = Path(log_file_path)
            if not log_file_path.exists():
                log_file_path.touch()
                logger.info(
                    "Created new log file for model training",
                    log_file_path=log_file_path,
                )
            logger.info(
                "Logging epoch results",
                epoch=results.epoch,
                log_file_path=log_file_path,
            )
            df = pd.DataFrame([epoch_data])
            df.to_csv(log_file_path, mode="a", header=(results.epoch == 1), index=False)

    def save_model(self, save_path: PathLike[str]) -> None:
        """Save the trained SigLIP model with LoRA adapter.

        Args:
            save_path (`PathLike[str]`): Directory path to save the model.

        Raises:
            ModelInitError: If the model is not initialised or saving fails.
        """
        if self.model is None:
            msg = "Model not initialised. Call setup_model() first."
            logger.error(msg)
            raise ModelInitError(msg)
        try:
            save_path_obj = Path(save_path)
            save_path_obj.mkdir(parents=True, exist_ok=True)
            self.model.save_pretrained(save_path_obj)
            logger.info("Weather SigLIP LoRA adapter saved to: %s", save_path_obj)
        except Exception as e:
            msg = f"Failed to save weather model: {e}"
            logger.exception(msg)
            raise ModelInitError(msg) from e

    def load_model(
        self, model_path: PathLike[str], adapter_name: str = "hrrr_adapter"
    ) -> None:
        """Load a pre-trained SigLIP model with LoRA adapter.

        Args:
            model_path (`PathLike[str]`): Path to the pre-trained LoRA adapter
                directory.
            adapter_name (`str`, optional): Name for the loaded adapter.
                Defaults to "hrrr_adapter".

        Raises:
            ModelInitError: If the model is not initialised or loading fails.
        """
        try:
            if self.model is None:
                self.setup_model()
            model_path_obj = Path(model_path)
            if not model_path_obj.exists():
                msg = f"Weather model path does not exist: {model_path}"
                logger.error(msg)
                raise ModelInitError(msg)  # noqa: TRY301
            self.model.load_adapter(model_path, adapter_name=adapter_name)
            # Set the newly loaded adapter as the active adapter
            self.model.set_adapter(adapter_name)
            logger.info(
                "Weather SigLIP LoRA adapter '%s' loaded from: %s",
                adapter_name,
                model_path,
            )
        except Exception as e:
            if isinstance(e, ModelInitError):
                raise
            msg = f"Failed to load weather model: {e}"
            logger.exception(msg)
            raise ModelInitError(msg) from e

    def test_caption_to_image(
        self,
        image_path: str,
        weather_descriptions: list[str],
        *,
        return_probs: bool = True,
    ) -> dict[str, Any]:
        """Test the model by predicting weather descriptions for a given image.

        This version follows SigLIP's sigmoid objective at inference time:
        probabilities are computed independently per (image, caption) pair via
        a sigmoid over the raw similarity logits (not a softmax over captions).

        Args:
            image_path (`str`): Path to the image to test.
            weather_descriptions (`list[str]`): List of candidate weather descriptions.
            return_probs (`bool`, optional): If True, include per-caption probabilities.

        Returns:
            `dict[str, Any]`: Contains the predicted index, caption, confidence, and
                optionally the full probability list (sigmoid outputs).

        Raises:
            ModelInitError: If the model is not initialised.
            DataError: If the image path is invalid or no descriptions are provided.
        """
        if self.model is None:
            msg = "Model not initialised. Call setup_model() first."
            logger.error(msg)
            raise ModelInitError(msg)

        if not weather_descriptions:
            msg = "No weather descriptions provided"
            logger.error(msg)
            raise DataError(msg)

        try:
            image_path_obj = Path(image_path)
            if not image_path_obj.exists():
                msg = f"Image does not exist: {image_path}"
                logger.error(msg)
                raise DataError(msg)  # noqa: TRY301

            image = Image.open(image_path_obj)
            if image.mode != "RGB":
                image = image.convert("RGB")

            inputs = self.processor(
                text=weather_descriptions,
                images=image,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256,
            )

            for key in inputs:
                inputs[key] = inputs[key].to(self.device)

            self.model.eval()
            with torch.no_grad():
                outputs = self.model(**inputs)
                text_embeds = nn.functional.normalize(outputs.text_embeds, p=2, dim=-1)
                image_embeds = nn.functional.normalize(
                    outputs.image_embeds, p=2, dim=-1
                )

                scale = (
                    self.model.logit_scale.exp()
                    if hasattr(self.model, "logit_scale")
                    else self.logit_scale.exp()
                )
                logits = scale * (image_embeds @ text_embeds.t())  # (1, num_texts)

                # SigLIP independent probabilities per pair
                probs = torch.sigmoid(logits)

            # Top-1 by raw logits (equivalently by probs since sigmoid is monotonic)
            predicted_idx = torch.argmax(logits, dim=1).item()
            confidence = probs[0, predicted_idx].item()

            results = {
                "image_path": str(image_path),
                "weather_descriptions": weather_descriptions,
                "predicted_description_idx": predicted_idx,
                "predicted_description": weather_descriptions[predicted_idx],
                "confidence": confidence,
            }

            if return_probs:
                results["probabilities"] = probs.squeeze(0).cpu().numpy().tolist()

            logger.info(
                "Weather image test complete - Predicted: %s (confidence: %.3f)",
                weather_descriptions[predicted_idx],
                results["confidence"],
            )

        except Exception as e:
            if isinstance(e, (ModelInitError, DataError)):
                raise
            msg = f"Weather image testing failed: {e}"
            logger.exception(msg)
            raise DataError(msg) from e

        else:
            return results

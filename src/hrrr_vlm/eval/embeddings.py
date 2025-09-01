"""Module for extracting and managing embeddings from fine-tuned HRRR-VLM models."""

import json
from os import PathLike
from pathlib import Path
from typing import Any, Literal, NamedTuple

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from hrrr_vlm.eval.retrieval import WeatherReport
from hrrr_vlm.train.train import HRRRLoRASigLIPTrainer
from hrrr_vlm.utils.logger import configure_logger

# Configure logging
logger = configure_logger()


class EmbeddingData(NamedTuple):
    """Container for embedding data with metadata.

    Attributes:
        image_embeddings: Normalised image embeddings [N, embed_dim]
        text_embeddings: Normalised text embeddings [N, embed_dim]
        captions: List of original captions
        metadata: List of metadata dictionaries containing weather info
        sample_ids: List of unique sample identifiers
        image_filenames: List of corresponding image filenames
    """

    image_embeddings: torch.Tensor
    text_embeddings: torch.Tensor
    captions: list[str]
    metadata: list[dict[str, Any]]
    sample_ids: list[str]
    image_filenames: list[str]


class EmbeddingExtractor:
    """Class for extracting embeddings from SigLIP models with metadata preservation.

    Attributes:
        trainer (`HRRRLoRASigLIPTrainer`): The SigLIP model.
        device (`torch.device`): Device to run extraction on.
        processor (`SiglipProcessor`): Processor for tokenisation and image
            processing.
    """

    def __init__(
        self, trainer: HRRRLoRASigLIPTrainer, device: torch.device | None = None
    ) -> None:
        """Initialize the embedding extractor.

        Args:
            trainer: The trained SigLIP model trainer
            device: Device to run extraction on (auto-detected if None)

        Raises:
            ValueError: If trainer model is not initialized
        """
        self.trainer = trainer
        self.device = device or trainer.device
        self.processor = trainer.processor

        if trainer.model is None:
            msg = "Trainer model is not initialized"
            raise ValueError(msg)

    @staticmethod
    def _get_dataset_item(
        dataset: Dataset | Subset, data_idx: int
    ) -> dict[str, Any] | None:
        """Safely get dataset item, handling both Subset and regular datasets.

        Args:
            dataset (`Dataset` or `Subset`): The image-caption dataset/subset.
            data_idx (`int`): Index in the current batch context.

        Returns:
            `dict[str, Any]`: Dataset item dictionary or None if not found.
        """
        try:
            # If it's a Subset, we need to access the underlying dataset
            if isinstance(dataset, Subset):
                # Get the actual index in the original dataset
                original_idx = dataset.indices[data_idx]
                return dataset.dataset.data[original_idx]
            # Regular dataset
            return dataset.data[data_idx]
        except (IndexError, AttributeError):
            logger.warning("Could not access dataset item at index %d", data_idx)
            return None

    def extract_embeddings_with_metadata(
        self,
        dataloader: DataLoader,
        max_samples: int | None = None,
        *,
        normalise: bool = True,
    ) -> EmbeddingData:
        """Extract embeddings along with its metadata.

        Args:
            dataloader (`DataLoader`): DataLoader containing the dataset.
            max_samples (`int`, optional): Maximum number of samples to process.
            normalise (`bool`): Whether to L2 normalise embeddings.

        Returns:
            `EmbeddingData`: Container for embeddings and metadata.
        """
        logger.info("Extracting embeddings with metadata preservation")

        self.trainer.model.eval()
        image_embeddings = []
        text_embeddings = []
        captions = []
        metadata_list = []
        sample_ids = []
        image_filenames = []

        samples_processed = 0
        dataset = dataloader.dataset

        with torch.no_grad():
            for batch_idx, batch in enumerate(
                tqdm(dataloader, desc="Extracting embeddings")
            ):
                if max_samples and samples_processed >= max_samples:
                    break

                # Move tensors to device
                input_ids = batch["input_ids"].to(self.device)
                pixel_values = batch["pixel_values"].to(self.device)
                attention_mask = batch.get(
                    "attention_mask", torch.ones_like(input_ids)
                ).to(self.device)

                # Forward pass through model
                outputs = self.trainer.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                )

                # Extract and optionally normalize embeddings
                img_embeds = outputs.image_embeds
                txt_embeds = outputs.text_embeds

                if normalise:
                    img_embeds = F.normalize(img_embeds, p=2, dim=-1)
                    txt_embeds = F.normalize(txt_embeds, p=2, dim=-1)

                image_embeddings.append(img_embeds.cpu())
                text_embeddings.append(txt_embeds.cpu())

                # Decode captions
                batch_captions = self.processor.batch_decode(
                    input_ids, skip_special_tokens=True
                )
                captions.extend(batch_captions)

                # Extract metadata from dataset
                # Handle both Subset and regular datasets
                start_idx = batch_idx * dataloader.batch_size

                for i in range(len(batch_captions)):
                    data_idx = start_idx + i

                    # Get the actual dataset item
                    item = self._get_dataset_item(dataset, data_idx)

                    if item is not None:
                        # Parse weather information from caption
                        weather_report = WeatherReport(batch_captions[i])

                        # Combine dataset metadata with parsed weather info
                        metadata = {
                            "sample_id": item["sample_id"],
                            "date": item["date"],
                            "variable": item["variable"],
                            "model": item["model"],
                            "region": item.get("region", "Unknown"),
                            "season": item.get("season", "Unknown"),
                            "image_filename": item["image_filename"],
                            # Weather-specific metadata from parsing
                            "avg_temperature": weather_report.avg_temperature,
                            "temperature_range": weather_report.temperature_range,
                            "wind_speed": weather_report.wind_speed,
                            "precipitation": weather_report.precipitation,
                            "humidity": weather_report.humidity,
                            "conditions": weather_report.conditions,
                        }

                        metadata_list.append(metadata)
                        sample_ids.append(item["sample_id"])
                        image_filenames.append(item["image_filename"])
                    else:
                        # Fallback metadata if we can't access the dataset item
                        weather_report = WeatherReport(batch_captions[i])

                        metadata = {
                            "sample_id": f"unknown_{data_idx}",
                            "date": "Unknown",
                            "variable": "temperature",  # Default assumption
                            "model": "hrrr",  # Default assumption
                            "region": weather_report.region or "Unknown",
                            "season": weather_report.season or "Unknown",
                            "image_filename": f"unknown_{data_idx}.png",
                            # Weather-specific metadata from parsing
                            "avg_temperature": weather_report.avg_temperature,
                            "temperature_range": weather_report.temperature_range,
                            "wind_speed": weather_report.wind_speed,
                            "precipitation": weather_report.precipitation,
                            "humidity": weather_report.humidity,
                            "conditions": weather_report.conditions,
                        }

                        metadata_list.append(metadata)
                        sample_ids.append(f"unknown_{data_idx}")
                        image_filenames.append(f"unknown_{data_idx}.png")

                samples_processed += len(batch_captions)

                if max_samples and samples_processed >= max_samples:
                    break

        # Concatenate all embeddings
        image_embeddings = torch.cat(image_embeddings, dim=0)
        text_embeddings = torch.cat(text_embeddings, dim=0)

        logger.info(
            "Embeddings extracted",
            num_samples=len(captions),
            image_shape=image_embeddings.shape,
            text_shape=text_embeddings.shape,
        )

        return EmbeddingData(
            image_embeddings=image_embeddings,
            text_embeddings=text_embeddings,
            captions=captions,
            metadata=metadata_list,
            sample_ids=sample_ids,
            image_filenames=image_filenames,
        )

    @staticmethod
    def save_embeddings(
        embedding_data: EmbeddingData,
        save_path: PathLike[str],
        *,
        save_format: Literal["npz", "pt", "safetensors"] = "safetensors",
    ) -> None:
        """Save embedding data to disk.

        Args:
            embedding_data (`EmbeddingData`): Container for embeddings and metadata.
            save_path (`PathLike[str]`): Path to save the data.
            save_format (`str`): Format to save in. Defaults to 'safetensors'
                for security.

        Raises:
            ValueError: If unsupported format is specified.
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        if save_format == "npz":
            np.savez_compressed(
                save_path,
                image_embeddings=embedding_data.image_embeddings.numpy(),
                text_embeddings=embedding_data.text_embeddings.numpy(),
                captions=np.array(embedding_data.captions),
                sample_ids=np.array(embedding_data.sample_ids),
                image_filenames=np.array(embedding_data.image_filenames),
            )

            # Save metadata separately as JSON
            metadata_path = save_path.with_suffix(".metadata.json")
            with metadata_path.open("w", encoding="utf-8") as f:
                json.dump(embedding_data.metadata, f, indent=2, default=str)
        elif save_format == "safetensors":
            # Safetensors only supports tensors, so we save metadata separately
            tensor_dict = {
                "image_embeddings": embedding_data.image_embeddings,
                "text_embeddings": embedding_data.text_embeddings,
                "sample_ids": torch.tensor(
                    [
                        int(sid.split("_")[-1]) if sid.split("_")[-1].isdigit() else -1
                        for sid in embedding_data.sample_ids
                    ],
                    dtype=torch.int32,
                ),
                "image_filenames": torch.tensor(
                    [
                        int(Path(fname).stem.split("_")[-1])
                        if Path(fname).stem.split("_")[-1].isdigit()
                        else -1
                        for fname in embedding_data.image_filenames
                    ],
                    dtype=torch.int32,
                ),
            }
            save_file(tensor_dict, str(save_path))

            # Save non-tensor data (captions and metadata) as JSON
            metadata_path = save_path.with_suffix(".metadata.json")
            with metadata_path.open("w", encoding="utf-8") as f:
                json.dump(
                    {
                        "captions": embedding_data.captions,
                        "metadata": embedding_data.metadata,
                        "sample_ids": embedding_data.sample_ids,
                        "image_filenames": embedding_data.image_filenames,
                    },
                    f,
                    indent=2,
                    default=str,
                )
        elif save_format == "pt":
            torch.save(
                {
                    "image_embeddings": embedding_data.image_embeddings,
                    "text_embeddings": embedding_data.text_embeddings,
                    "captions": embedding_data.captions,
                    "metadata": embedding_data.metadata,
                    "sample_ids": embedding_data.sample_ids,
                    "image_filenames": embedding_data.image_filenames,
                },
                save_path,
            )
        else:
            msg = f"Unsupported format: {save_format}"
            raise ValueError(msg)

        logger.info("Embeddings saved", save_path=str(save_path), format=save_format)

    @staticmethod
    def load_embeddings(
        load_path: PathLike[str],
        *,
        load_format: Literal["npz", "pt", "safetensors"] = "safetensors",
    ) -> EmbeddingData:
        """Load embedding data from disk.

        Args:
            load_path (`PathLike[str]`): Path to load the data from.
            load_format (`str`): Format to load from. Defaults to 'safetensors'.

        Returns:
            `EmbeddingData`: Container for embeddings and metadata.

        Raises:
            ValueError: If unsupported format is specified.
        """
        load_path = Path(load_path)

        if load_format == "npz":
            data = np.load(load_path)

            # Load metadata separately
            metadata_path = load_path.with_suffix(".metadata.json")
            with metadata_path.open(encoding="utf-8") as f:
                metadata = json.load(f)

            return EmbeddingData(
                image_embeddings=torch.from_numpy(data["image_embeddings"]),
                text_embeddings=torch.from_numpy(data["text_embeddings"]),
                captions=data["captions"].tolist(),
                metadata=metadata,
                sample_ids=data["sample_ids"].tolist(),
                image_filenames=data["image_filenames"].tolist(),
            )
        if load_format == "safetensors":
            data = torch.load(load_path)

            # Load non-tensor data (captions and metadata) from JSON
            metadata_path = load_path.with_suffix(".metadata.json")
            with metadata_path.open(encoding="utf-8") as f:
                meta = json.load(f)

            return EmbeddingData(
                image_embeddings=data["image_embeddings"],
                text_embeddings=data["text_embeddings"],
                captions=meta["captions"],
                metadata=meta["metadata"],
                sample_ids=meta["sample_ids"],
                image_filenames=meta["image_filenames"],
            )
        if load_format == "pt":
            data = torch.load(load_path)

            return EmbeddingData(
                image_embeddings=data["image_embeddings"],
                text_embeddings=data["text_embeddings"],
                captions=data["captions"],
                metadata=data["metadata"],
                sample_ids=data["sample_ids"],
                image_filenames=data["image_filenames"],
            )
        msg = f"Unsupported format: {load_format}"
        raise ValueError(msg)

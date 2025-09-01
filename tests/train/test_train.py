"""Comprehensive unit tests for the train.py module."""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import torch
from PIL import Image
from pydantic import ValidationError
from torch.utils.data import DataLoader

from hrrr_vlm.train.exceptions import DataError, ModelInitError, ModelTrainingError
from hrrr_vlm.train.train import (
    HRRRLoRASigLIPTrainer,
    ModelTrainingConfig,
    TrainingEpochResults,
    training_init,
)


class TestTrainingInit:
    """Test training_init function."""

    @patch("hrrr_vlm.train.train.torch.cuda.is_available")
    @patch("hrrr_vlm.train.train.torch.cuda.manual_seed_all")
    @patch("hrrr_vlm.train.train.torch.cuda.manual_seed")
    @patch("hrrr_vlm.train.train.torch.manual_seed")
    @patch("hrrr_vlm.train.train.random.seed")
    @patch("hrrr_vlm.train.train.np.random.default_rng")
    def test_training_init_with_cuda(
        self,
        mock_np_rng: Mock,
        mock_random_seed: Mock,
        mock_torch_manual_seed: Mock,
        mock_cuda_manual_seed: Mock,
        mock_cuda_manual_seed_all: Mock,
        mock_cuda_available: Mock,
    ) -> None:
        """Test training_init when CUDA is available."""
        mock_cuda_available.return_value = True

        training_init(42)

        mock_np_rng.assert_called_once_with(42)
        mock_torch_manual_seed.assert_called_once_with(42)
        mock_random_seed.assert_called_once_with(42)
        mock_cuda_manual_seed.assert_called_once_with(42)
        mock_cuda_manual_seed_all.assert_called_once_with(42)

    @patch("hrrr_vlm.train.train.torch.cuda.is_available")
    @patch("hrrr_vlm.train.train.torch.manual_seed")
    @patch("hrrr_vlm.train.train.random.seed")
    @patch("hrrr_vlm.train.train.np.random.default_rng")
    def test_training_init_without_cuda(
        self,
        mock_np_rng: Mock,
        mock_random_seed: Mock,
        mock_torch_manual_seed: Mock,
        mock_cuda_available: Mock,
    ) -> None:
        """Test training_init when CUDA is not available."""
        mock_cuda_available.return_value = False

        training_init(123)

        mock_np_rng.assert_called_once_with(123)
        mock_torch_manual_seed.assert_called_once_with(123)
        mock_random_seed.assert_called_once_with(123)

    def test_training_init_default_seed(self) -> None:
        """Test training_init with default seed value."""
        with patch("hrrr_vlm.train.train.torch.manual_seed") as mock_torch:
            training_init()
            mock_torch.assert_called_once_with(42)


class TestModelTrainingConfig:
    """Test ModelTrainingConfig pydantic model."""

    def test_config_default_values(self) -> None:
        """Test config initialization with default values."""
        config = ModelTrainingConfig()

        assert config.num_epochs == 10
        assert config.learning_rate == 5e-5
        assert config.batch_size == 32
        assert config.train_split == 0.8
        assert config.val_split == 0.1
        assert config.test_split == 0.1
        assert config.num_workers == 0
        assert config.max_length == 1024

    def test_config_custom_values(self) -> None:
        """Test config initialization with custom values."""
        config = ModelTrainingConfig(
            num_epochs=20,
            learning_rate=1e-4,
            batch_size=16,
            train_split=0.7,
            val_split=0.2,
            test_split=0.1,
            num_workers=4,
            max_length=512,
        )

        assert config.num_epochs == 20
        assert config.learning_rate == 1e-4
        assert config.batch_size == 16
        assert config.train_split == 0.7
        assert config.val_split == 0.2
        assert config.test_split == 0.1
        assert config.num_workers == 4
        assert config.max_length == 512

    def test_config_validation_positive_epochs(self) -> None:
        """Test config validation for positive epochs."""
        with pytest.raises(ValidationError) as exc_info:
            ModelTrainingConfig(num_epochs=0)

        assert "greater than 0" in str(exc_info.value)

    def test_config_validation_positive_learning_rate(self) -> None:
        """Test config validation for positive learning rate."""
        with pytest.raises(ValidationError) as exc_info:
            ModelTrainingConfig(learning_rate=0.0)

        assert "greater than 0" in str(exc_info.value)

    def test_config_validation_positive_batch_size(self) -> None:
        """Test config validation for positive batch size."""
        with pytest.raises(ValidationError) as exc_info:
            ModelTrainingConfig(batch_size=-1)

        assert "greater than 0" in str(exc_info.value)

    def test_config_validation_split_ratios(self) -> None:
        """Test config validation for split ratios."""
        with pytest.raises(ValidationError) as exc_info:
            ModelTrainingConfig(train_split=1.5)

        assert "less than or equal to 1" in str(exc_info.value)

    def test_config_validation_negative_workers(self) -> None:
        """Test config validation for non-negative workers."""
        with pytest.raises(ValidationError) as exc_info:
            ModelTrainingConfig(num_workers=-1)

        assert "greater than or equal to 0" in str(exc_info.value)

    def test_config_validation_max_length(self) -> None:
        """Test config validation for max_length."""
        with pytest.raises(ValidationError) as exc_info:
            ModelTrainingConfig(max_length=0)

        assert "greater than 0" in str(exc_info.value)

    def test_config_str_strip_whitespace(self) -> None:
        """Test config strips whitespace from string fields."""
        # Note: ModelTrainingConfig doesn't have string fields in this version
        # This test documents the expected behavior
        assert ModelTrainingConfig.model_config["str_strip_whitespace"] is True

    def test_config_extra_forbid(self) -> None:
        """Test config forbids extra fields."""
        with pytest.raises(ValidationError) as exc_info:
            ModelTrainingConfig(extra_field="not_allowed")

        assert "Extra inputs are not permitted" in str(exc_info.value)


class TestTrainingEpochResults:
    """Test TrainingEpochResults NamedTuple."""

    def test_epoch_results_creation(self) -> None:
        """Test creating TrainingEpochResults."""
        results = TrainingEpochResults(
            epoch=5, train_loss=0.25, val_loss=0.30, val_accuracy=0.85
        )

        assert results.epoch == 5
        assert results.train_loss == 0.25
        assert results.val_loss == 0.30
        assert results.val_accuracy == 0.85

    def test_epoch_results_immutable(self) -> None:
        """Test TrainingEpochResults is immutable."""
        results = TrainingEpochResults(1, 0.1, 0.2, 0.9)

        with pytest.raises(AttributeError):
            results.epoch = 2

    def test_epoch_results_asdict(self) -> None:
        """Test TrainingEpochResults _asdict method."""
        results = TrainingEpochResults(3, 0.15, 0.18, 0.92)
        results_dict = results._asdict()

        expected = {
            "epoch": 3,
            "train_loss": 0.15,
            "val_loss": 0.18,
            "val_accuracy": 0.92,
        }
        assert results_dict == expected


class TestHRRRLoRASigLIPTrainerInit:
    """Test HRRRLoRASigLIPTrainer initialization."""

    @patch("hrrr_vlm.train.train.SiglipProcessor.from_pretrained")
    @patch("hrrr_vlm.train.train.torch.cuda.is_available")
    @patch("hrrr_vlm.train.train.torch.backends.mps.is_built")
    def test_init_default_params(
        self, mock_mps_available: Mock, mock_cuda_available: Mock, mock_processor: Mock
    ) -> None:
        """Test trainer initialization with default parameters."""
        mock_cuda_available.return_value = False
        mock_mps_available.return_value = False
        mock_processor.return_value = Mock()

        trainer = HRRRLoRASigLIPTrainer()

        assert trainer.model_name == "google/siglip-base-patch16-224"
        assert trainer.device == torch.device("cpu")
        assert trainer.lora_config["r"] == 32
        assert trainer.lora_config["lora_alpha"] == 32
        assert trainer.model is None
        assert trainer.logit_scale is None
        mock_processor.assert_called_once_with("google/siglip-base-patch16-224")

    @patch("hrrr_vlm.train.train.SiglipProcessor.from_pretrained")
    @patch("hrrr_vlm.train.train.torch.cuda.is_available")
    def test_init_cuda_device(
        self, mock_cuda_available: Mock, mock_processor: Mock
    ) -> None:
        """Test trainer initialization with CUDA device."""
        mock_cuda_available.return_value = True
        mock_processor.return_value = Mock()

        trainer = HRRRLoRASigLIPTrainer()

        assert trainer.device == torch.device("cuda")

    @patch("hrrr_vlm.train.train.SiglipProcessor.from_pretrained")
    @patch("hrrr_vlm.train.train.torch.cuda.is_available")
    @patch("hrrr_vlm.train.train.torch.backends.mps.is_built")
    def test_init_mps_device(
        self, mock_mps_available: Mock, mock_cuda_available: Mock, mock_processor: Mock
    ) -> None:
        """Test trainer initialization with MPS device."""
        mock_cuda_available.return_value = False
        mock_mps_available.return_value = True
        mock_processor.return_value = Mock()

        trainer = HRRRLoRASigLIPTrainer()

        assert trainer.device == torch.device("mps")

    @patch("hrrr_vlm.train.train.SiglipProcessor.from_pretrained")
    def test_init_custom_params(self, mock_processor: Mock) -> None:
        """Test trainer initialization with custom parameters."""
        mock_processor.return_value = Mock()
        custom_lora_config = {"r": 16, "lora_alpha": 16}
        custom_device = torch.device("cpu")

        trainer = HRRRLoRASigLIPTrainer(
            model_name="custom/model",
            lora_config=custom_lora_config,
            device=custom_device,
        )

        assert trainer.model_name == "custom/model"
        assert trainer.device == custom_device
        assert trainer.lora_config["r"] == 16
        mock_processor.assert_called_once_with("custom/model")

    @patch("hrrr_vlm.train.train.SiglipProcessor.from_pretrained")
    def test_init_processor_failure(self, mock_processor: Mock) -> None:
        """Test trainer initialization when processor loading fails."""
        mock_processor.side_effect = Exception("Processor load failed")

        with pytest.raises(ModelInitError) as exc_info:
            HRRRLoRASigLIPTrainer()

        assert "Failed to load SigLIP processor" in str(exc_info.value)


class TestHRRRLoRASigLIPTrainerSetupModel:
    """Test HRRRLoRASigLIPTrainer setup_model method."""

    @pytest.fixture
    def trainer(self) -> HRRRLoRASigLIPTrainer:
        """Create a trainer instance for testing.

        Returns:
            `HRRRLoRASigLIPTrainer`: Test trainer instance.
        """
        with patch("hrrr_vlm.train.train.SiglipProcessor.from_pretrained"):
            return HRRRLoRASigLIPTrainer()

    @patch("hrrr_vlm.train.train.get_peft_model")
    @patch("hrrr_vlm.train.train.LoraConfig")
    @patch("hrrr_vlm.train.train.SiglipModel.from_pretrained")
    def test_setup_model_success(
        self,
        mock_siglip_model: Mock,
        mock_lora_config: Mock,
        mock_get_peft_model: Mock,
        trainer: HRRRLoRASigLIPTrainer,
    ) -> None:
        """Test successful model setup."""
        # Setup mocks
        mock_base_model = Mock()
        mock_base_model.to.return_value = mock_base_model
        mock_siglip_model.return_value = mock_base_model

        mock_peft_model = Mock()
        mock_peft_model.logit_scale = torch.nn.Parameter(torch.tensor(2.0))
        mock_get_peft_model.return_value = mock_peft_model

        with patch.object(trainer, "print_trainable_parameters"):
            trainer.setup_model()

        mock_siglip_model.assert_called_once_with(trainer.model_name)
        mock_lora_config.assert_called_once()
        mock_get_peft_model.assert_called_once()
        assert trainer.model == mock_peft_model
        assert trainer.logit_scale == mock_peft_model.logit_scale

    @patch("hrrr_vlm.train.train.get_peft_model")
    @patch("hrrr_vlm.train.train.LoraConfig")
    @patch("hrrr_vlm.train.train.SiglipModel.from_pretrained")
    def test_setup_model_no_logit_scale(
        self,
        mock_siglip_model: Mock,
        mock_lora_config: Mock,  # noqa: ARG002
        mock_get_peft_model: Mock,
        trainer: HRRRLoRASigLIPTrainer,
    ) -> None:
        """Test model setup when logit_scale doesn't exist."""
        # Setup mocks
        mock_base_model = Mock()
        mock_base_model.to.return_value = mock_base_model
        mock_siglip_model.return_value = mock_base_model

        mock_peft_model = Mock()
        # Mock hasattr to return False for logit_scale
        mock_peft_model.register_parameter = Mock()
        mock_get_peft_model.return_value = mock_peft_model

        with (
            patch.object(trainer, "print_trainable_parameters"),
            patch("builtins.hasattr", return_value=False),
        ):
            trainer.setup_model()

        # Should create new logit_scale parameter
        mock_peft_model.register_parameter.assert_called_once()
        call_args = mock_peft_model.register_parameter.call_args
        assert call_args[0][0] == "logit_scale"
        assert isinstance(call_args[0][1], torch.nn.Parameter)

    @patch("hrrr_vlm.train.train.SiglipModel.from_pretrained")
    def test_setup_model_failure(
        self, mock_siglip_model: Mock, trainer: HRRRLoRASigLIPTrainer
    ) -> None:
        """Test model setup failure."""
        mock_siglip_model.side_effect = Exception("Model load failed")

        with pytest.raises(ModelInitError) as exc_info:
            trainer.setup_model()

        assert "Failed to setup SigLIP weather model" in str(exc_info.value)


class TestHRRRLoRASigLIPTrainerPrintParameters:
    """Test HRRRLoRASigLIPTrainer print_trainable_parameters method."""

    @pytest.fixture
    def trainer(self) -> HRRRLoRASigLIPTrainer:
        """Create a trainer instance for testing.

        Returns:
            `HRRRLoRASigLIPTrainer`: Test trainer instance.
        """
        with patch("hrrr_vlm.train.train.SiglipProcessor.from_pretrained"):
            return HRRRLoRASigLIPTrainer()

    def test_print_parameters_no_model(self, trainer: HRRRLoRASigLIPTrainer) -> None:
        """Test print_trainable_parameters when model is None."""
        trainer.print_trainable_parameters()
        # Should log warning and return without error

    def test_print_parameters_success(self, trainer: HRRRLoRASigLIPTrainer) -> None:
        """Test successful parameter printing."""
        # Create mock model with parameters
        mock_model = Mock()
        param1 = Mock()
        param1.numel.return_value = 100
        param1.requires_grad = True

        param2 = Mock()
        param2.numel.return_value = 50
        param2.requires_grad = False

        mock_model.named_parameters.return_value = [
            ("param1", param1),
            ("param2", param2),
        ]
        trainer.model = mock_model

        trainer.print_trainable_parameters()
        # Should calculate and log parameters without error

    def test_print_parameters_exception(self, trainer: HRRRLoRASigLIPTrainer) -> None:
        """Test parameter printing when exception occurs."""
        mock_model = Mock()
        mock_model.named_parameters.side_effect = Exception("Parameter error")
        trainer.model = mock_model

        trainer.print_trainable_parameters()
        # Should handle exception gracefully


class TestHRRRLoRASigLIPTrainerBatchBuilder:
    """Test HRRRLoRASigLIPTrainer training_batch_builder method."""

    @pytest.fixture
    def trainer(self) -> HRRRLoRASigLIPTrainer:
        """Create a trainer instance for testing.

        Returns:
            `HRRRLoRASigLIPTrainer`: Test trainer instance.
        """
        with patch("hrrr_vlm.train.train.SiglipProcessor.from_pretrained"):
            return HRRRLoRASigLIPTrainer()

    @pytest.fixture
    def sample_images(self) -> list[Image.Image]:
        """Create sample PIL images for testing.

        Returns:
            `list[Image.Image]`: List of sample images.
        """
        images = []
        for _ in range(2):
            img = Image.new("RGB", (64, 64), color="red")
            images.append(img)
        return images

    @patch("hrrr_vlm.train.train.SiglipProcessor.from_pretrained")
    def test_batch_builder_success(
        self,
        mock_processor_class: Mock,
        trainer: HRRRLoRASigLIPTrainer,
        sample_images: list[Image.Image],
    ) -> None:
        """Test successful batch building."""
        mock_processor = Mock()
        mock_processor.return_value = {
            "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
            "pixel_values": torch.randn(2, 3, 224, 224),
            "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 0]]),
        }
        mock_processor_class.return_value = mock_processor

        samples = list(zip(sample_images, ["caption 1", "caption 2"], strict=True))

        result = trainer.training_batch_builder(samples)

        assert "input_ids" in result
        assert "pixel_values" in result
        mock_processor.assert_called_once_with(
            text=["caption 1", "caption 2"],
            images=sample_images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        )

    @patch("hrrr_vlm.train.train.SiglipProcessor.from_pretrained")
    def test_batch_builder_failure(
        self,
        mock_processor_class: Mock,
        trainer: HRRRLoRASigLIPTrainer,
        sample_images: list[Image.Image],
    ) -> None:
        """Test batch builder failure."""
        mock_processor_class.side_effect = Exception("Processing failed")

        samples = list(zip(sample_images, ["caption 1", "caption 2"], strict=True))

        with pytest.raises(DataError) as exc_info:
            trainer.training_batch_builder(samples)

        assert "Failed to process weather batch" in str(exc_info.value)


class TestHRRRLoRASigLIPTrainerCreateDataLoaders:
    """Test HRRRLoRASigLIPTrainer create_data_loaders method."""

    @pytest.fixture
    def trainer(self) -> HRRRLoRASigLIPTrainer:
        """Create a trainer instance for testing.

        Returns:
            `HRRRLoRASigLIPTrainer`: Test trainer instance.
        """
        with patch("hrrr_vlm.train.train.SiglipProcessor.from_pretrained"):
            return HRRRLoRASigLIPTrainer()

    @pytest.fixture
    def temp_dirs(self) -> tuple[Path, Path]:  # type: ignore[no-untyped-def]
        """Create temporary directories for testing.

        Yields:
            `tuple[Path, Path]`: Tuple containing the temporary directory path
                and images directory path.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            images_dir = temp_path / "images"
            images_dir.mkdir()

            # Create sample image files
            for i in range(3):
                img_path = images_dir / f"test_{i}.png"
                img = Image.new("RGB", (64, 64), color="red")
                img.save(img_path)

            yield temp_path, images_dir

    @pytest.fixture
    def sample_jsonl_file(self, temp_dirs: tuple[Path, Path]) -> Path:
        """Create a sample JSONL file for testing.

        Args:
            temp_dirs (`tuple[Path, Path]`): Temporary directories fixture.

        Returns:
            `Path`: Path to the created JSONL file.
        """
        temp_path, _images_dir = temp_dirs
        jsonl_file = temp_path / "test_data.jsonl"

        sample_data = [
            {
                "image": "test_0.png",
                "caption": "Temperature weather pattern over continental US",
                "variable": "temperature",
                "model": "hrrr",
                "region": "continental_us",
                "date": "2019-01-01",
            },
            {
                "image": "test_1.png",
                "caption": "Wind speed patterns across the region",
                "variable": "wind_speed",
                "model": "hrrr",
                "region": "continental_us",
                "date": "2019-01-02",
            },
            {
                "image": "test_2.png",
                "caption": "Precipitation forecast for the area",
                "variable": "precipitation",
                "model": "hrrr",
                "region": "continental_us",
                "date": "2019-01-03",
            },
        ]

        with jsonl_file.open("w") as f:
            f.writelines(json.dumps(item) + "\n" for item in sample_data)

        return jsonl_file

    @patch("hrrr_vlm.train.train.HRRRImageCaptionDataSetup")
    def test_create_data_loaders_dataset_failure(
        self,
        mock_dataset_class: Mock,
        trainer: HRRRLoRASigLIPTrainer,
        sample_jsonl_file: Path,
        temp_dirs: tuple[Path, Path],
    ) -> None:
        """Test data loader creation when dataset creation fails."""
        _, images_dir = temp_dirs
        mock_dataset_class.side_effect = Exception("Dataset creation failed")

        with pytest.raises(DataError) as exc_info:
            trainer.create_data_loaders(sample_jsonl_file, images_dir)

        assert "Failed to create data loaders" in str(exc_info.value)


class TestHRRRLoRASigLIPTrainerForwardAndSimilarities:
    """Test HRRRLoRASigLIPTrainer _forward_and_similarities method."""

    @pytest.fixture
    def trainer(self) -> HRRRLoRASigLIPTrainer:
        """Create a trainer instance for testing.

        Returns:
            `HRRRLoRASigLIPTrainer`: Test trainer instance.
        """
        with patch("hrrr_vlm.train.train.SiglipProcessor.from_pretrained"):
            return HRRRLoRASigLIPTrainer()

    def test_forward_no_model(self, trainer: HRRRLoRASigLIPTrainer) -> None:
        """Test forward pass when model is not initialised."""
        sample = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "pixel_values": torch.randn(1, 3, 224, 224),
        }

        with pytest.raises(ModelInitError) as exc_info:
            trainer._forward_and_similarities(sample)

        assert "Model not initialised" in str(exc_info.value)

    def test_forward_success(self, trainer: HRRRLoRASigLIPTrainer) -> None:
        """Test successful forward pass."""
        # Setup mock model
        mock_model = Mock()
        mock_model.logit_scale = torch.nn.Parameter(torch.tensor(2.0))

        # Mock model outputs
        mock_outputs = Mock()
        mock_outputs.text_embeds = torch.randn(2, 128)
        mock_outputs.image_embeds = torch.randn(2, 128)
        mock_model.return_value = mock_outputs

        trainer.model = mock_model
        trainer.logit_scale = mock_model.logit_scale

        sample = {
            "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
            "pixel_values": torch.randn(2, 3, 224, 224),
            "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 0]]),
        }

        logits_per_image, logits_per_text = trainer._forward_and_similarities(sample)

        assert logits_per_image.shape == (2, 2)
        assert logits_per_text.shape == (2, 2)
        mock_model.assert_called_once()

    def test_forward_no_attention_mask(self, trainer: HRRRLoRASigLIPTrainer) -> None:
        """Test forward pass when attention_mask is not provided."""
        # Setup mock model
        mock_model = Mock()
        mock_model.logit_scale = torch.nn.Parameter(torch.tensor(2.0))

        mock_outputs = Mock()
        mock_outputs.text_embeds = torch.randn(1, 128)
        mock_outputs.image_embeds = torch.randn(1, 128)
        mock_model.return_value = mock_outputs

        trainer.model = mock_model
        trainer.logit_scale = mock_model.logit_scale

        sample = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "pixel_values": torch.randn(1, 3, 224, 224),
        }

        logits_per_image, logits_per_text = trainer._forward_and_similarities(sample)

        assert logits_per_image.shape == (1, 1)
        assert logits_per_text.shape == (1, 1)


class TestHRRRLoRASigLIPTrainerEvaluateModel:
    """Test HRRRLoRASigLIPTrainer evaluate_model method."""

    @pytest.fixture
    def trainer(self) -> HRRRLoRASigLIPTrainer:
        """Create a trainer instance for testing.

        Returns:
            `HRRRLoRASigLIPTrainer`: Test trainer instance.
        """
        with patch("hrrr_vlm.train.train.SiglipProcessor.from_pretrained"):
            return HRRRLoRASigLIPTrainer()

    @pytest.fixture
    def mock_val_loader(self) -> Mock:
        """Create a mock validation data loader.

        Returns:
            `Mock`: Mock DataLoader instance.
        """
        mock_loader = Mock(spec=DataLoader)

        # Create sample batches
        batch1 = {
            "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
            "pixel_values": torch.randn(2, 3, 224, 224),
        }
        batch2 = {
            "input_ids": torch.tensor([[7, 8, 9]]),
            "pixel_values": torch.randn(1, 3, 224, 224),
        }

        mock_loader.__iter__ = Mock(return_value=iter([batch1, batch2]))
        return mock_loader

    def test_evaluate_no_model(
        self, trainer: HRRRLoRASigLIPTrainer, mock_val_loader: Mock
    ) -> None:
        """Test evaluation when model is not initialised."""
        with pytest.raises(ModelInitError) as exc_info:
            trainer.evaluate_model(mock_val_loader)

        assert "Model not initialised" in str(exc_info.value)

    def test_evaluate_exception(
        self, trainer: HRRRLoRASigLIPTrainer, mock_val_loader: DataLoader
    ) -> None:
        """Test evaluation when exception occurs."""
        mock_model = Mock()
        mock_model.eval.side_effect = Exception("Evaluation failed")
        trainer.model = mock_model

        with pytest.raises(ModelTrainingError) as exc_info:
            trainer.evaluate_model(mock_val_loader)

        assert "Model evaluation failed" in str(exc_info.value)


class TestHRRRLoRASigLIPTrainerProcessBatch:
    """Test HRRRLoRASigLIPTrainer _process_weather_batch method."""

    @pytest.fixture
    def trainer(self) -> HRRRLoRASigLIPTrainer:
        """Create a trainer instance for testing.

        Returns:
            `HRRRLoRASigLIPTrainer`: Test trainer instance.
        """
        with patch("hrrr_vlm.train.train.SiglipProcessor.from_pretrained"):
            return HRRRLoRASigLIPTrainer()

    @pytest.fixture
    def mock_optimiser(self) -> Mock:
        """Create a mock optimiser.

        Returns:
            `Mock`: Mock optimiser instance.
        """
        mock_opt = Mock()
        mock_opt.zero_grad = Mock()
        mock_opt.step = Mock()
        return mock_opt

    def test_process_batch_no_model(
        self, trainer: HRRRLoRASigLIPTrainer, mock_optimiser: Mock
    ) -> None:
        """Test batch processing when model is not initialised."""
        sample = {"input_ids": torch.tensor([[1, 2, 3]])}

        with pytest.raises(ModelInitError) as exc_info:
            trainer._process_weather_batch(sample, mock_optimiser)

        assert "Model not initialised" in str(exc_info.value)

    def test_process_batch_success(
        self, trainer: HRRRLoRASigLIPTrainer, mock_optimiser: Mock
    ) -> None:
        """Test successful batch processing."""
        # Setup mock model
        mock_model = Mock()
        mock_model.parameters.return_value = [torch.nn.Parameter(torch.randn(10, 10))]
        trainer.model = mock_model
        trainer.device = torch.device("cpu")  # Force CPU to avoid device mismatch

        sample = {
            "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
            "pixel_values": torch.randn(2, 3, 224, 224),
        }

        # Mock forward_and_similarities to return tensors that can compute loss
        with patch.object(trainer, "_forward_and_similarities") as mock_forward:
            mock_forward.return_value = (
                torch.tensor([[1.0, 0.5], [0.5, 1.0]], requires_grad=True),
                torch.tensor([[1.0, 0.5], [0.5, 1.0]], requires_grad=True),
            )

            with patch(
                "hrrr_vlm.train.train.torch.nn.utils.clip_grad_norm_"
            ) as mock_clip:
                loss, batch_size = trainer._process_weather_batch(
                    sample, mock_optimiser
                )

                assert isinstance(loss, float)
                assert batch_size == 2
                mock_optimiser.zero_grad.assert_called_once()
                mock_optimiser.step.assert_called_once()
                mock_clip.assert_called_once()


class TestHRRRLoRASigLIPTrainerTrainModel:
    """Test HRRRLoRASigLIPTrainer train_model method."""

    @pytest.fixture
    def trainer(self) -> HRRRLoRASigLIPTrainer:
        """Create a trainer instance for testing.

        Returns:
            `HRRRLoRASigLIPTrainer`: Test trainer instance.
        """
        with patch("hrrr_vlm.train.train.SiglipProcessor.from_pretrained"):
            return HRRRLoRASigLIPTrainer()

    @pytest.fixture
    def mock_data_loaders(self) -> tuple[Mock, Mock]:
        """Create mock data loaders.

        Returns:
            `tuple[Mock, Mock]`: Tuple containing mock train and validation
                DataLoaders.
        """
        train_loader = Mock(spec=DataLoader)
        val_loader = Mock(spec=DataLoader)

        # Mock dataset lengths
        train_loader.dataset = Mock()
        train_loader.dataset.__len__ = Mock(return_value=80)
        val_loader.dataset = Mock()
        val_loader.dataset.__len__ = Mock(return_value=20)

        return train_loader, val_loader

    def test_train_no_model(
        self, trainer: HRRRLoRASigLIPTrainer, mock_data_loaders: tuple[Mock, Mock]
    ) -> None:
        """Test training when model is not initialised."""
        train_loader, val_loader = mock_data_loaders

        with pytest.raises(ModelInitError) as exc_info:
            trainer.train_model(train_loader, val_loader)

        assert "Model not initialised" in str(exc_info.value)

    def test_train_exception(
        self, trainer: HRRRLoRASigLIPTrainer, mock_data_loaders: tuple[Mock, Mock]
    ) -> None:
        """Test training when exception occurs."""
        train_loader, val_loader = mock_data_loaders

        mock_model = Mock()
        mock_model.parameters.side_effect = Exception("Training failed")
        trainer.model = mock_model

        with pytest.raises(ModelTrainingError) as exc_info:
            trainer.train_model(train_loader, val_loader)

        assert "SigLIP fine-tuning failed" in str(exc_info.value)


class TestHRRRLoRASigLIPTrainerSaveLoad:
    """Test HRRRLoRASigLIPTrainer save_model and load_model methods."""

    @pytest.fixture
    def trainer(self) -> HRRRLoRASigLIPTrainer:
        """Create a trainer instance for testing.

        Returns:
            `HRRRLoRASigLIPTrainer`: Test trainer instance.
        """
        with patch("hrrr_vlm.train.train.SiglipProcessor.from_pretrained"):
            return HRRRLoRASigLIPTrainer()

    def test_save_no_model(self, trainer: HRRRLoRASigLIPTrainer) -> None:
        """Test saving when model is not initialised."""
        with pytest.raises(ModelInitError) as exc_info:
            trainer.save_model("/tmp/test_model")

        assert "Model not initialised" in str(exc_info.value)

    def test_save_success(self, trainer: HRRRLoRASigLIPTrainer) -> None:
        """Test successful model saving."""
        mock_model = Mock()
        mock_model.save_pretrained = Mock()
        trainer.model = mock_model

        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "test_model"

            trainer.save_model(save_path)

            mock_model.save_pretrained.assert_called_once_with(save_path)

    def test_save_exception(self, trainer: HRRRLoRASigLIPTrainer) -> None:
        """Test saving when exception occurs."""
        mock_model = Mock()
        mock_model.save_pretrained.side_effect = Exception("Save failed")
        trainer.model = mock_model

        with pytest.raises(ModelInitError) as exc_info:
            trainer.save_model("/tmp/test_model")

        assert "Failed to save weather model" in str(exc_info.value)

    def test_load_no_model(self, trainer: HRRRLoRASigLIPTrainer) -> None:
        """Test loading when model is not initialised."""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "test_model"
            model_path.mkdir()

            # Mock setup_model to set a mock model
            with patch.object(trainer, "setup_model") as mock_setup:
                mock_model = Mock()
                mock_model.load_adapter = Mock()
                mock_model.set_adapter = Mock()

                def setup_side_effect() -> None:
                    trainer.model = mock_model

                mock_setup.side_effect = setup_side_effect

                trainer.load_model(model_path)
                mock_setup.assert_called_once()

    def test_load_nonexistent_path(self, trainer: HRRRLoRASigLIPTrainer) -> None:
        """Test loading from non-existent path."""
        with pytest.raises(ModelInitError) as exc_info:
            trainer.load_model("/nonexistent/path")

        assert "Weather model path does not exist" in str(exc_info.value)

    def test_load_success(self, trainer: HRRRLoRASigLIPTrainer) -> None:
        """Test successful model loading."""
        mock_model = Mock()
        mock_model.load_adapter = Mock()
        mock_model.set_adapter = Mock()
        trainer.model = mock_model

        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "test_model"
            model_path.mkdir()

            trainer.load_model(model_path, "custom_adapter")

            mock_model.load_adapter.assert_called_once_with(
                model_path, adapter_name="custom_adapter"
            )
            mock_model.set_adapter.assert_called_once_with("custom_adapter")

    def test_load_exception(self, trainer: HRRRLoRASigLIPTrainer) -> None:
        """Test loading when exception occurs."""
        mock_model = Mock()
        mock_model.load_adapter.side_effect = Exception("Load failed")
        trainer.model = mock_model

        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "test_model"
            model_path.mkdir()

            with pytest.raises(ModelInitError) as exc_info:
                trainer.load_model(model_path)

            assert "Failed to load weather model" in str(exc_info.value)


class TestHRRRLoRASigLIPTrainerTestCaptionToImage:
    """Test HRRRLoRASigLIPTrainer test_caption_to_image method."""

    @pytest.fixture
    def trainer(self) -> HRRRLoRASigLIPTrainer:
        """Create a trainer instance for testing.

        Returns:
            `HRRRLoRASigLIPTrainer`: Test trainer instance.
        """
        with patch("hrrr_vlm.train.train.SiglipProcessor.from_pretrained"):
            return HRRRLoRASigLIPTrainer()

    @pytest.fixture
    def sample_image_path(self) -> Path:  # type: ignore[no-untyped-def]
        """Create a sample image file for testing.

        Yields:
            `Path`: Path to the created image file.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            img_path = Path(temp_dir) / "test_image.png"
            img = Image.new("RGB", (64, 64), color="blue")
            img.save(img_path)
            yield img_path

    def test_test_caption_no_model(
        self, trainer: HRRRLoRASigLIPTrainer, sample_image_path: Path
    ) -> None:
        """Test caption testing when model is not initialised."""
        with pytest.raises(ModelInitError) as exc_info:
            trainer.test_caption_to_image(
                str(sample_image_path), ["caption1", "caption2"]
            )

        assert "Model not initialised" in str(exc_info.value)

    def test_test_caption_no_descriptions(
        self, trainer: HRRRLoRASigLIPTrainer, sample_image_path: Path
    ) -> None:
        """Test caption testing with empty descriptions."""
        trainer.model = Mock()

        with pytest.raises(DataError) as exc_info:
            trainer.test_caption_to_image(str(sample_image_path), [])

        assert "No weather descriptions provided" in str(exc_info.value)

    def test_test_caption_nonexistent_image(
        self, trainer: HRRRLoRASigLIPTrainer
    ) -> None:
        """Test caption testing with non-existent image."""
        trainer.model = Mock()

        with pytest.raises(DataError) as exc_info:
            trainer.test_caption_to_image("/nonexistent/image.png", ["caption1"])

        assert "Image does not exist" in str(exc_info.value)

    def test_test_caption_success(
        self, trainer: HRRRLoRASigLIPTrainer, sample_image_path: Path
    ) -> None:
        """Test successful caption testing."""
        mock_model = Mock()
        mock_model.eval = Mock()
        trainer.model = mock_model
        trainer.processor = Mock()
        trainer.device = torch.device("cpu")  # Force CPU device

        # Mock processor output
        trainer.processor.return_value = {
            "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
            "pixel_values": torch.randn(1, 3, 224, 224),
        }

        # Mock model outputs with proper tensor structure
        mock_outputs = Mock()
        mock_outputs.text_embeds = torch.randn(2, 128)  # 2 captions, 128 dims
        mock_outputs.image_embeds = torch.randn(1, 128)  # 1 image, 128 dims
        mock_model.return_value = mock_outputs

        # Mock logit_scale
        mock_model.logit_scale = torch.nn.Parameter(torch.tensor(4.6052))  # log(1/0.01)

        descriptions = ["sunny weather", "cloudy conditions"]
        results = trainer.test_caption_to_image(str(sample_image_path), descriptions)

        assert "image_path" in results
        assert "weather_descriptions" in results
        assert "predicted_description_idx" in results
        assert "predicted_description" in results
        assert "confidence" in results
        assert results["predicted_description"] in descriptions

    def test_test_caption_exception(
        self, trainer: HRRRLoRASigLIPTrainer, sample_image_path: Path
    ) -> None:
        """Test caption testing when exception occurs."""
        mock_model = Mock()
        trainer.model = mock_model
        trainer.processor = Mock()
        trainer.processor.side_effect = Exception("Processing failed")

        with pytest.raises(DataError) as exc_info:
            trainer.test_caption_to_image(str(sample_image_path), ["caption1"])

        assert "Weather image testing failed" in str(exc_info.value)


class TestHRRRLoRASigLIPTrainerLogEpochResults:
    """Test HRRRLoRASigLIPTrainer _log_epoch_results static method."""

    def test_log_epoch_results_no_file(self) -> None:
        """Test logging epoch results without file."""
        results = TrainingEpochResults(1, 0.5, 0.4, 0.9)
        log_data = []

        HRRRLoRASigLIPTrainer._log_epoch_results(results, log_data=log_data)

        assert len(log_data) == 1
        assert log_data[0]["epoch"] == 1
        assert log_data[0]["train_loss"] == 0.5

    def test_log_epoch_results_with_file(self) -> None:
        """Test logging epoch results with file."""
        results = TrainingEpochResults(1, 0.5, 0.4, 0.9)
        log_data = []

        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "training.csv"

            with patch("hrrr_vlm.train.train.pd.DataFrame") as mock_df_class:
                mock_df = Mock()
                mock_df.to_csv = Mock()
                mock_df_class.return_value = mock_df

                HRRRLoRASigLIPTrainer._log_epoch_results(
                    results, log_data=log_data, log_file_path=log_file
                )

                mock_df.to_csv.assert_called_once()


class TestHRRRLoRASigLIPTrainerSetupTrainingLogs:
    """Test HRRRLoRASigLIPTrainer _setup_training_logs static method."""

    def test_setup_logs_no_file(self) -> None:
        """Test log setup without file."""
        log_data, log_file_path = HRRRLoRASigLIPTrainer._setup_training_logs(None)

        assert log_data == []
        assert log_file_path is None

    def test_setup_logs_with_file(self) -> None:
        """Test log setup with file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "logs" / "training.csv"

            log_data, log_file_path = HRRRLoRASigLIPTrainer._setup_training_logs(
                log_file
            )

            assert log_data == []
            assert log_file_path == log_file
            assert log_file_path.parent.exists()

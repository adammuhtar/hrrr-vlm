"""Unit tests for the data_loader.py module."""

import json
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import pytest
import torch
from PIL import Image
from transformers import SiglipProcessor

from hrrr_vlm.train.data_loader import HRRRImageCaptionDataSetup
from hrrr_vlm.train.exceptions import DataError

# Test constants
TEST_SAMPLE_ID = "test_sample_001"
TEST_VARIABLE = "temperature"
TEST_MODEL = "hrrr"
TEST_DATE = "2019-04-15"
TEST_CAPTION = "Temperature data showing conditions across the region"
TEST_IMAGE_FILENAME = "test_image.png"
TEST_REGION = "continental_us"
TEST_MAX_LENGTH = 256


@pytest.fixture
def mock_processor() -> Mock:
    """Create a mock SigLIP processor.

    Returns:
        Mock: Mock SigLIP processor instance.
    """
    processor = Mock(spec=SiglipProcessor)
    processor.tokenizer = Mock()
    processor.feature_extractor = Mock()
    return processor


@pytest.fixture
def sample_data_item() -> dict[str, Any]:
    """Create a sample data item.

    Returns:
        dict[str, Any]: Sample data item with all required fields.
    """
    return {
        "image_path": f"images/{TEST_IMAGE_FILENAME}",
        "caption": TEST_CAPTION,
        "sample_id": TEST_SAMPLE_ID,
        "variable": TEST_VARIABLE,
        "model": TEST_MODEL,
        "date": TEST_DATE,
        "metadata": {"region": TEST_REGION},
    }


@pytest.fixture
def sample_jsonl_data(sample_data_item: dict[str, Any]) -> str:
    """Create sample JSONL data.

    Args:
        sample_data_item: Sample data item fixture.

    Returns:
        str: JSONL formatted data.
    """
    return json.dumps(sample_data_item) + "\n"


@pytest.fixture
def temp_dirs() -> tuple[Path, Path]:  # type: ignore[misc]
    """Create temporary directories for testing.

    Yields:
        tuple[Path, Path]: Temporary data directory and images directory.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        data_dir = Path(temp_dir) / "data"
        images_dir = Path(temp_dir) / "images"
        data_dir.mkdir()
        images_dir.mkdir()
        yield data_dir, images_dir


@pytest.fixture
def sample_image() -> Image.Image:
    """Create a sample image for testing.

    Returns:
        Image.Image: Sample RGB image.
    """
    return Image.new("RGB", (512, 512), color="blue")


class TestHRRRImageCaptionDataSetupInit:
    """Test HRRRImageCaptionDataSetup initialization."""

    def test_init_success(
        self, temp_dirs: tuple[Path, Path], sample_jsonl_data: str, mock_processor: Mock
    ) -> None:
        """Test successful initialization."""
        data_dir, images_dir = temp_dirs
        data_file = data_dir / "test_data.jsonl"
        data_file.write_text(sample_jsonl_data)

        # Create the image file
        image_path = images_dir / TEST_IMAGE_FILENAME
        with image_path.open("wb") as f:
            f.write(b"fake image data")

        dataset = HRRRImageCaptionDataSetup(
            data_file=data_file,
            images_dir=images_dir,
            processor=mock_processor,
            max_length=TEST_MAX_LENGTH,
        )

        assert dataset.images_dir == images_dir
        assert dataset.processor == mock_processor
        assert dataset.max_length == TEST_MAX_LENGTH
        assert len(dataset.data) == 1

    def test_init_missing_images_dir(
        self, temp_dirs: tuple[Path, Path], sample_jsonl_data: str, mock_processor: Mock
    ) -> None:
        """Test initialization with missing images directory."""
        data_dir, _ = temp_dirs
        data_file = data_dir / "test_data.jsonl"
        data_file.write_text(sample_jsonl_data)
        missing_images_dir = data_dir / "nonexistent"

        with pytest.raises(DataError) as exc_info:
            HRRRImageCaptionDataSetup(
                data_file=data_file,
                images_dir=missing_images_dir,
                processor=mock_processor,
            )

        assert "Images directory does not exist" in str(exc_info.value)

    def test_init_missing_data_file(
        self, temp_dirs: tuple[Path, Path], mock_processor: Mock
    ) -> None:
        """Test initialization with missing data file."""
        _, images_dir = temp_dirs
        missing_data_file = images_dir / "nonexistent.jsonl"

        with pytest.raises(DataError) as exc_info:
            HRRRImageCaptionDataSetup(
                data_file=missing_data_file,
                images_dir=images_dir,
                processor=mock_processor,
            )

        assert "Data file does not exist" in str(exc_info.value)

    def test_init_empty_data_file(
        self, temp_dirs: tuple[Path, Path], mock_processor: Mock
    ) -> None:
        """Test initialization with empty data file."""
        data_dir, images_dir = temp_dirs
        data_file = data_dir / "empty.jsonl"
        data_file.write_text("")

        with pytest.raises(DataError) as exc_info:
            HRRRImageCaptionDataSetup(
                data_file=data_file, images_dir=images_dir, processor=mock_processor
            )

        assert "No valid weather data found" in str(exc_info.value)

    def test_init_invalid_json(
        self, temp_dirs: tuple[Path, Path], mock_processor: Mock
    ) -> None:
        """Test initialization with invalid JSON data."""
        data_dir, images_dir = temp_dirs
        data_file = data_dir / "invalid.jsonl"
        data_file.write_text("invalid json line\n")

        with pytest.raises(DataError) as exc_info:
            HRRRImageCaptionDataSetup(
                data_file=data_file, images_dir=images_dir, processor=mock_processor
            )

        assert "No valid weather data found" in str(exc_info.value)


class TestDataLoading:
    """Test data loading functionality."""

    def test_load_data_missing_required_fields(
        self, temp_dirs: tuple[Path, Path], mock_processor: Mock
    ) -> None:
        """Test loading data with missing required fields."""
        data_dir, images_dir = temp_dirs
        data_file = data_dir / "incomplete.jsonl"

        # Missing 'caption' field
        incomplete_item = {
            "image_path": f"images/{TEST_IMAGE_FILENAME}",
            "sample_id": TEST_SAMPLE_ID,
            "variable": TEST_VARIABLE,
            "model": TEST_MODEL,
            "date": TEST_DATE,
        }
        data_file.write_text(json.dumps(incomplete_item) + "\n")

        with pytest.raises(DataError):
            HRRRImageCaptionDataSetup(
                data_file=data_file, images_dir=images_dir, processor=mock_processor
            )

    def test_load_data_mixed_valid_invalid(
        self,
        temp_dirs: tuple[Path, Path],
        sample_data_item: dict[str, Any],
        mock_processor: Mock,
    ) -> None:
        """Test loading data with mix of valid and invalid items."""
        data_dir, images_dir = temp_dirs
        data_file = data_dir / "mixed.jsonl"

        # Create image file
        image_path = images_dir / TEST_IMAGE_FILENAME
        with image_path.open("wb") as f:
            f.write(b"fake image data")

        mixed_data = [
            json.dumps(sample_data_item),  # Valid
            "invalid json",  # Invalid JSON
            json.dumps({"incomplete": "item"}),  # Missing required fields
            json.dumps(sample_data_item),  # Valid
        ]
        data_file.write_text("\n".join(mixed_data) + "\n")

        dataset = HRRRImageCaptionDataSetup(
            data_file=data_file, images_dir=images_dir, processor=mock_processor
        )

        # Should only have 2 valid items
        assert len(dataset.data) == 2


class TestImagePathProcessing:
    """Test image path processing functionality."""

    def test_process_image_path_relative(
        self,
        temp_dirs: tuple[Path, Path],
        sample_data_item: dict[str, Any],
        mock_processor: Mock,
    ) -> None:
        """Test processing relative image paths."""
        data_dir, images_dir = temp_dirs
        data_file = data_dir / "test.jsonl"

        # Create image in images directory
        image_path = images_dir / TEST_IMAGE_FILENAME
        with image_path.open("wb") as f:
            f.write(b"fake image data")

        data_file.write_text(json.dumps(sample_data_item) + "\n")

        dataset = HRRRImageCaptionDataSetup(
            data_file=data_file, images_dir=images_dir, processor=mock_processor
        )

        item = dataset.data[0]
        assert "image_filename" in item
        assert "resolved_image_path" in item
        assert item["image_filename"] == TEST_IMAGE_FILENAME

    def test_process_image_path_missing_image(
        self,
        temp_dirs: tuple[Path, Path],
        sample_data_item: dict[str, Any],
        mock_processor: Mock,
    ) -> None:
        """Test processing path when image file is missing."""
        data_dir, images_dir = temp_dirs
        data_file = data_dir / "test.jsonl"

        # Don't create the image file
        data_file.write_text(json.dumps(sample_data_item) + "\n")

        # Should raise DataError due to no valid data
        with pytest.raises(DataError):
            HRRRImageCaptionDataSetup(
                data_file=data_file, images_dir=images_dir, processor=mock_processor
            )


class TestSeasonExtraction:
    """Test season extraction from dates."""

    def test_get_season_from_date_winter(self) -> None:
        """Test winter season extraction."""
        assert HRRRImageCaptionDataSetup._get_season_from_date("2019-12-15") == "winter"
        assert HRRRImageCaptionDataSetup._get_season_from_date("2019-01-15") == "winter"
        assert HRRRImageCaptionDataSetup._get_season_from_date("2019-02-15") == "winter"

    def test_get_season_from_date_spring(self) -> None:
        """Test spring season extraction."""
        assert HRRRImageCaptionDataSetup._get_season_from_date("2019-03-15") == "spring"
        assert HRRRImageCaptionDataSetup._get_season_from_date("2019-04-15") == "spring"
        assert HRRRImageCaptionDataSetup._get_season_from_date("2019-05-15") == "spring"

    def test_get_season_from_date_summer(self) -> None:
        """Test summer season extraction."""
        assert HRRRImageCaptionDataSetup._get_season_from_date("2019-06-15") == "summer"
        assert HRRRImageCaptionDataSetup._get_season_from_date("2019-07-15") == "summer"
        assert HRRRImageCaptionDataSetup._get_season_from_date("2019-08-15") == "summer"

    def test_get_season_from_date_fall(self) -> None:
        """Test fall season extraction."""
        assert HRRRImageCaptionDataSetup._get_season_from_date("2019-09-15") == "fall"
        assert HRRRImageCaptionDataSetup._get_season_from_date("2019-10-15") == "fall"
        assert HRRRImageCaptionDataSetup._get_season_from_date("2019-11-15") == "fall"

    def test_get_season_from_date_invalid(self) -> None:
        """Test handling of invalid date strings."""
        assert (
            HRRRImageCaptionDataSetup._get_season_from_date("invalid-date") == "unknown"
        )
        assert HRRRImageCaptionDataSetup._get_season_from_date("2019") == "unknown"
        assert HRRRImageCaptionDataSetup._get_season_from_date("") == "unknown"


class TestDatasetMethods:
    """Test dataset methods (__len__, __getitem__)."""

    def test_len(
        self, temp_dirs: tuple[Path, Path], sample_jsonl_data: str, mock_processor: Mock
    ) -> None:
        """Test __len__ method."""
        data_dir, images_dir = temp_dirs
        data_file = data_dir / "test.jsonl"

        # Create multiple items
        multiple_data = sample_jsonl_data * 3
        data_file.write_text(multiple_data)

        # Create image file
        image_path = images_dir / TEST_IMAGE_FILENAME
        with image_path.open("wb") as f:
            f.write(b"fake image data")

        dataset = HRRRImageCaptionDataSetup(
            data_file=data_file, images_dir=images_dir, processor=mock_processor
        )

        assert len(dataset) == 3

    @patch("PIL.Image.open")
    def test_getitem_success(
        self,
        mock_image_open: Mock,
        temp_dirs: tuple[Path, Path],
        sample_jsonl_data: str,
        mock_processor: Mock,
    ) -> None:
        """Test successful __getitem__ call."""
        data_dir, images_dir = temp_dirs
        data_file = data_dir / "test.jsonl"
        data_file.write_text(sample_jsonl_data)

        # Create image file
        image_path = images_dir / TEST_IMAGE_FILENAME
        with image_path.open("wb") as f:
            f.write(b"fake image data")

        # Mock PIL Image operations
        mock_img = Mock(spec=Image.Image)
        mock_img.mode = "RGB"
        mock_img.verify.return_value = None
        mock_image_open.return_value = mock_img

        dataset = HRRRImageCaptionDataSetup(
            data_file=data_file, images_dir=images_dir, processor=mock_processor
        )

        img, caption = dataset[0]

        assert img == mock_img
        assert isinstance(caption, str)
        assert TEST_CAPTION in caption

    def test_getitem_index_out_of_range(
        self, temp_dirs: tuple[Path, Path], sample_jsonl_data: str, mock_processor: Mock
    ) -> None:
        """Test __getitem__ with index out of range."""
        data_dir, images_dir = temp_dirs
        data_file = data_dir / "test.jsonl"
        data_file.write_text(sample_jsonl_data)

        # Create image file
        image_path = images_dir / TEST_IMAGE_FILENAME
        with image_path.open("wb") as f:
            f.write(b"fake image data")

        dataset = HRRRImageCaptionDataSetup(
            data_file=data_file, images_dir=images_dir, processor=mock_processor
        )

        with pytest.raises(DataError) as exc_info:
            dataset[10]  # Out of range

        assert "Index 10 out of range" in str(exc_info.value)

    def test_getitem_tensor_index(
        self, temp_dirs: tuple[Path, Path], sample_jsonl_data: str, mock_processor: Mock
    ) -> None:
        """Test __getitem__ with tensor index."""
        data_dir, images_dir = temp_dirs
        data_file = data_dir / "test.jsonl"
        data_file.write_text(sample_jsonl_data)

        # Create image file
        image_path = images_dir / TEST_IMAGE_FILENAME
        with image_path.open("wb") as f:
            f.write(b"fake image data")

        with patch("PIL.Image.open") as mock_image_open:
            mock_img = Mock(spec=Image.Image)
            mock_img.mode = "RGB"
            mock_img.verify.return_value = None
            mock_image_open.return_value = mock_img

            dataset = HRRRImageCaptionDataSetup(
                data_file=data_file, images_dir=images_dir, processor=mock_processor
            )

            tensor_idx = torch.tensor(0)
            img, caption = dataset[tensor_idx]

            assert img == mock_img
            assert isinstance(caption, str)

    @patch("PIL.Image.open")
    def test_getitem_image_load_failure(
        self,
        mock_image_open: Mock,
        temp_dirs: tuple[Path, Path],
        sample_jsonl_data: str,
        mock_processor: Mock,
    ) -> None:
        """Test __getitem__ when image loading fails."""
        data_dir, images_dir = temp_dirs
        data_file = data_dir / "test.jsonl"
        data_file.write_text(sample_jsonl_data)

        # Create image file
        image_path = images_dir / TEST_IMAGE_FILENAME
        with image_path.open("wb") as f:
            f.write(b"fake image data")

        # Mock image opening to fail
        mock_image_open.side_effect = FileNotFoundError("Image not found")

        with patch("PIL.Image.new") as mock_image_new:
            mock_blank_img = Mock(spec=Image.Image)
            mock_image_new.return_value = mock_blank_img

            dataset = HRRRImageCaptionDataSetup(
                data_file=data_file, images_dir=images_dir, processor=mock_processor
            )

            img, _caption = dataset[0]

            # Should return blank image
            assert img == mock_blank_img
            mock_image_new.assert_called_with("RGB", (512, 512), color="lightgray")

    @patch("PIL.Image.open")
    def test_getitem_image_mode_conversion(
        self,
        mock_image_open: Mock,
        temp_dirs: tuple[Path, Path],
        sample_jsonl_data: str,
        mock_processor: Mock,
    ) -> None:
        """Test __getitem__ with image mode conversion."""
        data_dir, images_dir = temp_dirs
        data_file = data_dir / "test.jsonl"
        data_file.write_text(sample_jsonl_data)

        # Create image file
        image_path = images_dir / TEST_IMAGE_FILENAME
        with image_path.open("wb") as f:
            f.write(b"fake image data")

        # Mock PIL Image with non-RGB mode
        mock_img = Mock(spec=Image.Image)
        mock_img.mode = "L"  # Grayscale
        mock_img.verify.return_value = None
        mock_converted_img = Mock(spec=Image.Image)
        mock_img.convert.return_value = mock_converted_img
        mock_image_open.return_value = mock_img

        dataset = HRRRImageCaptionDataSetup(
            data_file=data_file, images_dir=images_dir, processor=mock_processor
        )

        img, _caption = dataset[0]

        assert img == mock_converted_img
        mock_img.convert.assert_called_with("RGB")


class TestCaptionEnhancement:
    """Test caption enhancement functionality."""

    def test_enhance_weather_caption_basic(
        self, temp_dirs: tuple[Path, Path], sample_jsonl_data: str, mock_processor: Mock
    ) -> None:
        """Test basic caption enhancement."""
        data_dir, images_dir = temp_dirs
        data_file = data_dir / "test.jsonl"
        data_file.write_text(sample_jsonl_data)

        # Create image file
        image_path = images_dir / TEST_IMAGE_FILENAME
        with image_path.open("wb") as f:
            f.write(b"fake image data")

        dataset = HRRRImageCaptionDataSetup(
            data_file=data_file, images_dir=images_dir, processor=mock_processor
        )

        item = dataset.data[0]
        enhanced = dataset._enhance_weather_caption(TEST_CAPTION, item)

        assert "HRRR model" in enhanced
        assert "spring season" in enhanced
        assert "continental_us region" in enhanced

    def test_enhance_weather_caption_no_region(
        self, temp_dirs: tuple[Path, Path], mock_processor: Mock
    ) -> None:
        """Test caption enhancement without region."""
        data_dir, images_dir = temp_dirs
        data_file = data_dir / "test.jsonl"

        # Create data without region
        data_item = {
            "image_path": f"images/{TEST_IMAGE_FILENAME}",
            "caption": TEST_CAPTION,
            "sample_id": TEST_SAMPLE_ID,
            "variable": TEST_VARIABLE,
            "model": TEST_MODEL,
            "date": TEST_DATE,
            "metadata": {},
        }
        data_file.write_text(json.dumps(data_item) + "\n")

        # Create image file
        image_path = images_dir / TEST_IMAGE_FILENAME
        with image_path.open("wb") as f:
            f.write(b"fake image data")

        dataset = HRRRImageCaptionDataSetup(
            data_file=data_file, images_dir=images_dir, processor=mock_processor
        )

        item = dataset.data[0]
        enhanced = dataset._enhance_weather_caption(TEST_CAPTION, item)

        assert "HRRR model" in enhanced
        assert "spring season" in enhanced
        assert "continental_us region" not in enhanced  # No region info should be added

    def test_enhance_weather_caption_truncation(
        self, temp_dirs: tuple[Path, Path], mock_processor: Mock
    ) -> None:
        """Test caption enhancement with truncation."""
        data_dir, images_dir = temp_dirs
        data_file = data_dir / "test.jsonl"

        # Create very long caption
        long_caption = "A " * 200  # Very long caption
        data_item = {
            "image_path": f"images/{TEST_IMAGE_FILENAME}",
            "caption": long_caption,
            "sample_id": TEST_SAMPLE_ID,
            "variable": TEST_VARIABLE,
            "model": TEST_MODEL,
            "date": TEST_DATE,
            "metadata": {"region": TEST_REGION},
        }
        data_file.write_text(json.dumps(data_item) + "\n")

        # Create image file
        image_path = images_dir / TEST_IMAGE_FILENAME
        with image_path.open("wb") as f:
            f.write(b"fake image data")

        dataset = HRRRImageCaptionDataSetup(
            data_file=data_file,
            images_dir=images_dir,
            processor=mock_processor,
            max_length=50,  # Short max length
        )

        item = dataset.data[0]
        enhanced = dataset._enhance_weather_caption(long_caption, item)

        assert len(enhanced) <= 50
        assert enhanced.endswith("...")


class TestWeatherStatistics:
    """Test weather statistics functionality."""

    def test_get_weather_statistics_single_item(
        self, temp_dirs: tuple[Path, Path], sample_jsonl_data: str, mock_processor: Mock
    ) -> None:
        """Test statistics with single item."""
        data_dir, images_dir = temp_dirs
        data_file = data_dir / "test.jsonl"
        data_file.write_text(sample_jsonl_data)

        # Create image file
        image_path = images_dir / TEST_IMAGE_FILENAME
        with image_path.open("wb") as f:
            f.write(b"fake image data")

        dataset = HRRRImageCaptionDataSetup(
            data_file=data_file, images_dir=images_dir, processor=mock_processor
        )

        stats = dataset.get_weather_statistics()

        assert stats["total_samples"] == 1
        assert stats["variables"][TEST_VARIABLE] == 1
        assert stats["models"][TEST_MODEL] == 1
        assert stats["regions"][TEST_REGION] == 1
        assert stats["seasons"]["spring"] == 1
        assert stats["date_range"]["min"] == TEST_DATE
        assert stats["date_range"]["max"] == TEST_DATE

    def test_get_weather_statistics_multiple_items(
        self, temp_dirs: tuple[Path, Path], mock_processor: Mock
    ) -> None:
        """Test statistics with multiple diverse items."""
        data_dir, images_dir = temp_dirs
        data_file = data_dir / "test.jsonl"

        # Create diverse data items
        data_items = [
            {
                "image_path": f"images/{TEST_IMAGE_FILENAME}",
                "caption": "Temperature data",
                "sample_id": "sample_1",
                "variable": "temperature",
                "model": "hrrr",
                "date": "2019-04-15",
                "metadata": {"region": "continental_us"},
            },
            {
                "image_path": f"images/{TEST_IMAGE_FILENAME}",
                "caption": "Wind data",
                "sample_id": "sample_2",
                "variable": "wind_speed",
                "model": "hrrrak",
                "date": "2019-07-20",
                "metadata": {"region": "alaska"},
            },
            {
                "image_path": f"images/{TEST_IMAGE_FILENAME}",
                "caption": "Precipitation data",
                "sample_id": "sample_3",
                "variable": "precipitation",
                "model": "hrrr",
                "date": "2019-01-10",
                "metadata": {"region": "continental_us"},
            },
        ]

        jsonl_data = "\n".join(json.dumps(item) for item in data_items) + "\n"
        data_file.write_text(jsonl_data)

        # Create image file
        image_path = images_dir / TEST_IMAGE_FILENAME
        with image_path.open("wb") as f:
            f.write(b"fake image data")

        dataset = HRRRImageCaptionDataSetup(
            data_file=data_file, images_dir=images_dir, processor=mock_processor
        )

        stats = dataset.get_weather_statistics()

        assert stats["total_samples"] == 3
        assert stats["variables"]["temperature"] == 1
        assert stats["variables"]["wind_speed"] == 1
        assert stats["variables"]["precipitation"] == 1
        assert stats["models"]["hrrr"] == 2
        assert stats["models"]["hrrrak"] == 1
        assert stats["regions"]["continental_us"] == 2
        assert stats["regions"]["alaska"] == 1
        assert stats["seasons"]["spring"] == 1
        assert stats["seasons"]["summer"] == 1
        assert stats["seasons"]["winter"] == 1
        assert stats["date_range"]["min"] == "2019-01-10"
        assert stats["date_range"]["max"] == "2019-07-20"

    def test_get_weather_statistics_empty_dates(
        self, temp_dirs: tuple[Path, Path], mock_processor: Mock
    ) -> None:
        """Test statistics with empty dates."""
        data_dir, images_dir = temp_dirs
        data_file = data_dir / "test.jsonl"

        # Create data with empty dates
        data_item = {
            "image_path": f"images/{TEST_IMAGE_FILENAME}",
            "caption": TEST_CAPTION,
            "sample_id": TEST_SAMPLE_ID,
            "variable": TEST_VARIABLE,
            "model": TEST_MODEL,
            "date": "",  # Empty date
            "metadata": {"region": TEST_REGION},
        }
        data_file.write_text(json.dumps(data_item) + "\n")

        # Create image file
        image_path = images_dir / TEST_IMAGE_FILENAME
        with image_path.open("wb") as f:
            f.write(b"fake image data")

        dataset = HRRRImageCaptionDataSetup(
            data_file=data_file, images_dir=images_dir, processor=mock_processor
        )

        stats = dataset.get_weather_statistics()

        assert stats["date_range"]["min"] is None
        assert stats["date_range"]["max"] is None


class TestDataSetupIntegration:
    """Integration tests for HRRRImageCaptionDataSetup."""

    def test_full_workflow(
        self, temp_dirs: tuple[Path, Path], mock_processor: Mock
    ) -> None:
        """Test complete workflow with realistic data."""
        data_dir, images_dir = temp_dirs
        data_file = data_dir / "test.jsonl"

        # Create realistic data
        data_items = [
            {
                "image_path": "images/temp_2019_04_15.png",
                "caption": "Temperature analysis showing warm conditions",
                "sample_id": "temp_001",
                "variable": "temperature",
                "model": "hrrr",
                "date": "2019-04-15",
                "metadata": {
                    "region": "continental_us",
                    "generation_time": "2019-04-15T12:00:00",
                },
            },
            {
                "image_path": "images/wind_2019_04_15.png",
                "caption": "Wind speed patterns across the region",
                "sample_id": "wind_001",
                "variable": "wind_speed",
                "model": "hrrr",
                "date": "2019-04-15",
                "metadata": {
                    "region": "continental_us",
                    "generation_time": "2019-04-15T12:00:00",
                },
            },
        ]

        jsonl_data = "\n".join(json.dumps(item) for item in data_items) + "\n"
        data_file.write_text(jsonl_data)

        # Create image files
        for item in data_items:
            image_filename = Path(item["image_path"]).name
            image_path = images_dir / image_filename
            with image_path.open("wb") as f:
                f.write(b"fake image data")

        with patch("PIL.Image.open") as mock_image_open:
            mock_img = Mock(spec=Image.Image)
            mock_img.mode = "RGB"
            mock_img.verify.return_value = None
            mock_image_open.return_value = mock_img

            dataset = HRRRImageCaptionDataSetup(
                data_file=data_file, images_dir=images_dir, processor=mock_processor
            )

            # Test basic functionality
            assert len(dataset) == 2

            # Test data access
            img1, caption1 = dataset[0]
            img2, caption2 = dataset[1]

            assert img1 == mock_img
            assert img2 == mock_img
            assert isinstance(caption1, str)
            assert isinstance(caption2, str)

            # Test statistics
            stats = dataset.get_weather_statistics()
            assert stats["total_samples"] == 2
            assert stats["variables"]["temperature"] == 1
            assert stats["variables"]["wind_speed"] == 1
            assert stats["models"]["hrrr"] == 2
            assert stats["regions"]["continental_us"] == 2

    def test_edge_cases_handling(
        self, temp_dirs: tuple[Path, Path], mock_processor: Mock
    ) -> None:
        """Test handling of various edge cases."""
        data_dir, images_dir = temp_dirs
        data_file = data_dir / "edge_cases.jsonl"

        # Create data with various edge cases
        edge_case_items = [
            # Normal item
            {
                "image_path": f"images/{TEST_IMAGE_FILENAME}",
                "caption": "Normal caption",
                "sample_id": "normal_001",
                "variable": "temperature",
                "model": "hrrr",
                "date": "2019-04-15",
                "metadata": {"region": "continental_us"},
            },
            # Item with unknown region
            {
                "image_path": f"images/{TEST_IMAGE_FILENAME}",
                "caption": "Unknown region caption",
                "sample_id": "unknown_001",
                "variable": "temperature",
                "model": "hrrr",
                "date": "2019-04-15",
                "metadata": {},
            },
            # Item with edge case date
            {
                "image_path": f"images/{TEST_IMAGE_FILENAME}",
                "caption": "Edge case date",
                "sample_id": "edge_001",
                "variable": "temperature",
                "model": "hrrr",
                "date": "invalid-date-format",  # Invalid date format
                "metadata": {"region": "test"},
            },
        ]

        jsonl_data = "\n".join(json.dumps(item) for item in edge_case_items) + "\n"
        data_file.write_text(jsonl_data)

        # Create image file
        image_path = images_dir / TEST_IMAGE_FILENAME
        with image_path.open("wb") as f:
            f.write(b"fake image data")

        with patch("PIL.Image.open") as mock_image_open:
            mock_img = Mock(spec=Image.Image)
            mock_img.mode = "RGB"
            mock_img.verify.return_value = None
            mock_image_open.return_value = mock_img

            dataset = HRRRImageCaptionDataSetup(
                data_file=data_file, images_dir=images_dir, processor=mock_processor
            )

            # Should handle all edge cases gracefully
            assert len(dataset) == 3

            # Check that unknown regions are handled
            stats = dataset.get_weather_statistics()
            assert "Unknown" in stats["regions"]
            assert "unknown" in stats["seasons"]  # From invalid date

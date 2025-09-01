"""Unit tests for the loader.py module."""

import json
from datetime import date
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from PIL import Image

from hrrr_vlm.data.config import DataGeneratorConfig
from hrrr_vlm.data.loader import (
    DEFAULT_VARIABLES,
    REGION_MODEL_PAIRS,
    HRRRImageCaptionDataset,
    get_first_mondays,
)
from hrrr_vlm.data.models import TrainingRecord

# Test constants
TEST_SAMPLE_ID = "test_sample_001"
TEST_IMAGE_PATH = "test_image.png"
TEST_CAPTION = "Temperature map showing warm conditions"
TEST_VARIABLE = "temperature"
TEST_MODEL = "hrrr"
TEST_DATE = "2019-04-15"


class TestGetFirstMondays:
    """Test the get_first_mondays utility function."""

    def test_get_first_mondays_basic(self) -> None:
        """Test basic functionality of get_first_mondays."""
        # Test January 2019 (should get 5 Mondays)
        mondays = get_first_mondays(2019, 1, 2019, 1)

        assert isinstance(mondays, list)
        assert len(mondays) > 0
        assert all(isinstance(date_str, str) for date_str in mondays)

        # Check format YYYY-MM-DD (length 10, 2 dashes)
        for date_str in mondays:
            assert len(date_str) == 10
            assert date_str.count("-") == 2

    def test_get_first_mondays_single_month(self) -> None:
        """Test getting Mondays for a single month."""
        mondays = get_first_mondays(2019, 4, 2019, 4)

        # April 2019 should have 4-5 Mondays (typical month range)
        assert 4 <= len(mondays) <= 5

    def test_get_first_mondays_multiple_months(self) -> None:
        """Test getting Mondays across multiple months."""
        mondays = get_first_mondays(2019, 1, 2019, 3)

        # Should have Mondays from Jan, Feb, and March (12+ total)
        assert len(mondays) >= 12

        # Check that we have dates from all months
        dates_set = set(mondays)
        has_jan = any("2019-01" in d for d in dates_set)
        has_feb = any("2019-02" in d for d in dates_set)
        has_mar = any("2019-03" in d for d in dates_set)

        assert has_jan
        assert has_feb
        assert has_mar

    def test_get_first_mondays_edge_cases(self) -> None:
        """Test edge cases for get_first_mondays."""
        # Test single week
        mondays = get_first_mondays(2019, 4, 2019, 4)
        assert len(mondays) >= 1

        # Test year boundary
        mondays = get_first_mondays(2019, 12, 2020, 1)
        assert len(mondays) >= 8

    def test_get_first_mondays_monday_validation(self) -> None:
        """Test that returned dates are actually Mondays."""
        mondays = get_first_mondays(2019, 4, 2019, 4)

        for date_str in mondays:
            year, month, day = map(int, date_str.split("-"))
            test_date = date(year, month, day)
            # Monday is weekday 0
            assert test_date.weekday() == 0


class TestHRRRImageCaptionDatasetBasic:
    """Test basic functionality of HRRRImageCaptionDataset."""

    def test_constants_exist(self) -> None:
        """Test that required constants are defined."""
        assert isinstance(REGION_MODEL_PAIRS, (list, tuple))
        assert len(REGION_MODEL_PAIRS) > 0
        assert isinstance(DEFAULT_VARIABLES, (list, tuple))
        assert len(DEFAULT_VARIABLES) > 0

    def test_region_model_pairs_structure(self) -> None:
        """Test structure of REGION_MODEL_PAIRS."""
        for pair in REGION_MODEL_PAIRS:
            assert isinstance(pair, tuple)
            assert len(pair) == 2
            region, model = pair
            assert isinstance(region, str)
            assert isinstance(model, str)

    def test_default_variables_structure(self) -> None:
        """Test structure of DEFAULT_VARIABLES."""
        for variable in DEFAULT_VARIABLES:
            assert isinstance(variable, str)
            assert len(variable) > 0


class TestHRRRImageCaptionDatasetInit:
    """Test initialization of HRRRImageCaptionDataset."""

    def _create_test_jsonl(self, tmp_path: Path, num_records: int = 3) -> Path:
        """Create a test JSONL file with sample records.

        Returns:
            Path to the created JSONL file.
        """
        jsonl_file = tmp_path / "test_training.jsonl"

        records = []
        for i in range(num_records):
            # Create test image
            img_path = tmp_path / f"test_image_{i}.png"
            test_image = Image.new("RGB", (64, 64), color="red")
            test_image.save(img_path)

            record = {
                "sample_id": f"test_{i:03d}",
                "image_path": str(img_path),
                "caption": f"Test caption {i}",
                "variable": TEST_VARIABLE,
                "model": TEST_MODEL,
                "date": TEST_DATE,
                "metadata": {"fxx": 0, "region": "Continental US"},
            }
            records.append(record)

        with jsonl_file.open("w") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")

        return jsonl_file

    def test_init_file_not_found(self) -> None:
        """Test initialization with non-existent JSONL file."""
        mock_processor = Mock()

        with pytest.raises(FileNotFoundError, match="JSONL file not found"):
            HRRRImageCaptionDataset(
                jsonl_file="/nonexistent/file.jsonl", processor=mock_processor
            )

    def test_init_success(self, tmp_path: Path) -> None:
        """Test successful initialization."""
        jsonl_file = self._create_test_jsonl(tmp_path)
        mock_processor = Mock()

        dataset = HRRRImageCaptionDataset(
            jsonl_file=jsonl_file, processor=mock_processor
        )

        assert dataset.jsonl_file == jsonl_file
        assert dataset.processor == mock_processor
        assert dataset.max_length == 77
        assert len(dataset.records) == 3

    def test_init_with_parameters(self, tmp_path: Path) -> None:
        """Test initialization with custom parameters."""
        jsonl_file = self._create_test_jsonl(tmp_path)
        mock_processor = Mock()
        images_base_path = tmp_path / "images"

        dataset = HRRRImageCaptionDataset(
            jsonl_file=jsonl_file,
            processor=mock_processor,
            max_length=50,
            transform_images=False,
            images_base_path=images_base_path,
        )

        assert dataset.max_length == 50
        assert dataset.transform_images is False
        assert dataset.images_base_path == images_base_path

    def test_init_max_length_enforcement(self, tmp_path: Path) -> None:
        """Test that max_length is enforced to CLIP's limit."""
        jsonl_file = self._create_test_jsonl(tmp_path)
        mock_processor = Mock()

        dataset = HRRRImageCaptionDataset(
            jsonl_file=jsonl_file,
            processor=mock_processor,
            max_length=100,  # Above CLIP limit
        )

        assert dataset.max_length == 77

    def test_init_empty_jsonl(self, tmp_path: Path) -> None:
        """Test initialization with empty JSONL file."""
        jsonl_file = tmp_path / "empty.jsonl"
        jsonl_file.touch()  # Create empty file
        mock_processor = Mock()

        with pytest.raises(ValueError, match="No valid training records found"):
            HRRRImageCaptionDataset(jsonl_file=jsonl_file, processor=mock_processor)


class TestHRRRImageCaptionDatasetMethods:
    """Test methods of HRRRImageCaptionDataset."""

    @pytest.fixture
    def sample_dataset(self, tmp_path: Path) -> HRRRImageCaptionDataset:
        """Create a sample dataset for testing.

        Returns:
            A sample HRRRImageCaptionDataset instance.
        """
        jsonl_file = tmp_path / "test_training.jsonl"

        # Create test images and records
        records = []
        for i in range(3):
            img_path = tmp_path / f"test_image_{i}.png"
            test_image = Image.new("RGB", (64, 64), color=("red", "green", "blue")[i])
            test_image.save(img_path)

            record = {
                "sample_id": f"test_{i:03d}",
                "image_path": str(img_path),
                "caption": f"Test caption {i} with weather data",
                "variable": ["temperature", "wind_speed", "precipitation"][i],
                "model": TEST_MODEL,
                "date": f"2019-04-{15 + i:02d}",
                "metadata": {"fxx": 0, "region": "Continental US"},
            }
            records.append(record)

        with jsonl_file.open("w") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")

        mock_processor = Mock()
        return HRRRImageCaptionDataset(jsonl_file=jsonl_file, processor=mock_processor)

    def test_len(self, sample_dataset: HRRRImageCaptionDataset) -> None:
        """Test __len__ method."""
        assert len(sample_dataset) == 3

    def test_getitem_success(self, sample_dataset: HRRRImageCaptionDataset) -> None:
        """Test successful __getitem__ method."""
        image, caption = sample_dataset[0]

        assert isinstance(image, Image.Image)
        assert isinstance(caption, str)
        assert image.mode == "RGB"
        assert "Test caption 0" in caption

    def test_getitem_index_error(self, sample_dataset: HRRRImageCaptionDataset) -> None:
        """Test __getitem__ with invalid index."""
        with pytest.raises(IndexError, match=r"Index .* out of range"):
            _ = sample_dataset[10]

    def test_get_record_success(self, sample_dataset: HRRRImageCaptionDataset) -> None:
        """Test get_record method."""
        record = sample_dataset.get_record(0)

        assert isinstance(record, TrainingRecord)
        assert record.sample_id == "test_000"
        assert "Test caption 0" in record.caption

    def test_get_record_index_error(
        self, sample_dataset: HRRRImageCaptionDataset
    ) -> None:
        """Test get_record with invalid index."""
        with pytest.raises(IndexError, match=r"Index .* out of range"):
            sample_dataset.get_record(10)

    def test_filter_by_variable(self, sample_dataset: HRRRImageCaptionDataset) -> None:
        """Test filter_by_variable method."""
        filtered = sample_dataset.filter_by_variable("temperature")

        assert len(filtered) == 1
        assert isinstance(filtered, HRRRImageCaptionDataset)

        record = filtered.get_record(0)
        assert record.variable == "temperature"

    def test_filter_by_variable_no_matches(
        self, sample_dataset: HRRRImageCaptionDataset
    ) -> None:
        """Test filter_by_variable with no matches."""
        filtered = sample_dataset.filter_by_variable("nonexistent")

        assert len(filtered) == 0
        assert isinstance(filtered, HRRRImageCaptionDataset)

    def test_get_statistics(self, sample_dataset: HRRRImageCaptionDataset) -> None:
        """Test get_statistics method."""
        stats = sample_dataset.get_statistics()

        assert isinstance(stats, dict)
        assert stats["total_samples"] == 3
        assert "variables" in stats
        assert "models" in stats
        assert "dates" in stats

        # Check variables count
        assert stats["variables"]["temperature"] == 1
        assert stats["variables"]["wind_speed"] == 1
        assert stats["variables"]["precipitation"] == 1

        # Check models count
        assert stats["models"][TEST_MODEL] == 3

        # Check date range
        assert stats["dates"]["min"] == "2019-04-15"
        assert stats["dates"]["max"] == "2019-04-17"


class TestHRRRImageCaptionDatasetErrorHandling:
    """Test error handling in HRRRImageCaptionDataset."""

    def test_getitem_missing_image(self, tmp_path: Path) -> None:
        """Test __getitem__ with missing image file."""
        jsonl_file = tmp_path / "test.jsonl"

        # Create a real image file first, then delete it after creating the dataset
        img_path = tmp_path / "test.png"
        test_image = Image.new("RGB", (64, 64), color="red")
        test_image.save(img_path)

        record = {
            "sample_id": "test_001",
            "image_path": str(img_path),  # Use absolute path initially
            "caption": "Test caption",
            "variable": TEST_VARIABLE,
            "model": TEST_MODEL,
            "date": TEST_DATE,
            "metadata": {"fxx": 0, "region": "Continental US"},
        }

        with jsonl_file.open("w") as f:
            f.write(json.dumps(record) + "\n")

        mock_processor = Mock()
        dataset = HRRRImageCaptionDataset(
            jsonl_file=jsonl_file, processor=mock_processor
        )

        # Now delete the image file to test the missing image scenario
        img_path.unlink()

        # Should create fallback image
        with patch("hrrr_vlm.data.loader.logger") as mock_logger:
            image, _caption = dataset[0]

            assert isinstance(image, Image.Image)
            assert image.size == (512, 512)
            mock_logger.warning.assert_called()

    def test_getitem_long_caption(self, tmp_path: Path) -> None:
        """Test __getitem__ with very long caption."""
        jsonl_file = tmp_path / "test.jsonl"

        # Create test image
        img_path = tmp_path / "test.png"
        test_image = Image.new("RGB", (64, 64), color="red")
        test_image.save(img_path)

        long_caption = "Very long caption " * 100  # Very long caption

        record = {
            "sample_id": "test_001",
            "image_path": str(img_path),
            "caption": long_caption,
            "variable": TEST_VARIABLE,
            "model": TEST_MODEL,
            "date": TEST_DATE,
            "metadata": {"fxx": 0, "region": "Continental US"},
        }

        with jsonl_file.open("w") as f:
            f.write(json.dumps(record) + "\n")

        mock_processor = Mock()
        dataset = HRRRImageCaptionDataset(
            jsonl_file=jsonl_file, processor=mock_processor, max_length=50
        )

        _, caption = dataset[0]

        # Caption should be truncated
        assert len(caption) <= 50 * 4  # Rough character limit
        assert caption.endswith("...")


class TestHRRRImageCaptionDatasetClassMethods:
    """Test class methods of HRRRImageCaptionDataset."""

    @patch("hrrr_vlm.data.loader.snapshot_download")
    def test_from_huggingface(self, mock_snapshot: Mock, tmp_path: Path) -> None:
        """Test from_huggingface class method."""
        # Setup mock
        mock_snapshot.return_value = str(tmp_path)

        # Create test JSONL file
        jsonl_file = tmp_path / "training_dataset.jsonl"
        img_path = tmp_path / "test.png"
        test_image = Image.new("RGB", (64, 64), color="red")
        test_image.save(img_path)

        record = {
            "sample_id": "test_001",
            "image_path": str(img_path),
            "caption": "Test caption",
            "variable": TEST_VARIABLE,
            "model": TEST_MODEL,
            "date": TEST_DATE,
            "metadata": {"fxx": 0, "region": "Continental US"},
        }

        with jsonl_file.open("w") as f:
            f.write(json.dumps(record) + "\n")

        mock_processor = Mock()

        dataset = HRRRImageCaptionDataset.from_huggingface(
            repo_id="test/repo",
            processor=mock_processor,
            cache_dir=str(tmp_path / "cache"),
        )

        assert isinstance(dataset, HRRRImageCaptionDataset)
        assert len(dataset) == 1
        mock_snapshot.assert_called_once_with(
            repo_id="test/repo", repo_type="dataset", cache_dir=str(tmp_path / "cache")
        )

    def test_create_from_existing_jsonl(self, tmp_path: Path) -> None:
        """Test create_from_existing_jsonl static method."""
        # Create test JSONL file
        jsonl_file = tmp_path / "existing.jsonl"
        img_path = tmp_path / "test.png"
        test_image = Image.new("RGB", (64, 64), color="red")
        test_image.save(img_path)

        record = {
            "sample_id": "test_001",
            "image_path": str(img_path),
            "caption": "Test caption",
            "variable": TEST_VARIABLE,
            "model": TEST_MODEL,
            "date": TEST_DATE,
            "metadata": {"fxx": 0, "region": "Continental US"},
        }

        with jsonl_file.open("w") as f:
            f.write(json.dumps(record) + "\n")

        mock_processor = Mock()

        dataset = HRRRImageCaptionDataset.create_from_existing_jsonl(
            jsonl_file=jsonl_file, processor=mock_processor
        )

        assert isinstance(dataset, HRRRImageCaptionDataset)
        assert len(dataset) == 1

    @patch("hrrr_vlm.data.loader.WeatherDataGenerator")
    def test_create_dataset(self, mock_generator_class: Mock, tmp_path: Path) -> None:
        """Test create_dataset class method."""
        # Setup mock generator
        mock_generator = Mock()
        mock_generator_class.return_value = mock_generator
        mock_generator.generate_samples.return_value = {"successful": 5}

        # Create mock JSONL file
        jsonl_file = tmp_path / "training_dataset.jsonl"
        img_path = tmp_path / "test.png"
        test_image = Image.new("RGB", (64, 64), color="red")
        test_image.save(img_path)

        record = {
            "sample_id": "test_001",
            "image_path": str(img_path),
            "caption": "Test caption",
            "variable": TEST_VARIABLE,
            "model": TEST_MODEL,
            "date": TEST_DATE,
            "metadata": {"fxx": 0, "region": "Continental US"},
        }

        with jsonl_file.open("w") as f:
            f.write(json.dumps(record) + "\n")

        mock_generator.create_training_records.return_value = jsonl_file

        # Test
        mock_config = Mock(spec=DataGeneratorConfig)
        mock_processor = Mock()
        dates = ["2019-04-15", "2019-04-16"]

        dataset = HRRRImageCaptionDataset.create_dataset(
            config=mock_config,
            processor=mock_processor,
            dates=dates,
            variables=["temperature", "wind_speed"],
            regions=["Continental US"],
        )

        assert isinstance(dataset, HRRRImageCaptionDataset)
        assert len(dataset) == 1

        # Verify generator was called correctly
        mock_generator_class.assert_called_once_with(mock_config)
        mock_generator.generate_samples.assert_called()
        mock_generator.create_training_records.assert_called_once_with(
            "training_dataset.jsonl"
        )


class TestHRRRImageCaptionDatasetIntegration:
    """Integration tests for HRRRImageCaptionDataset."""

    def test_full_workflow(self, tmp_path: Path) -> None:
        """Test complete workflow from JSONL creation to data loading."""
        # Create test data
        jsonl_file = tmp_path / "full_test.jsonl"
        images_dir = tmp_path / "images"
        images_dir.mkdir()

        records = []
        for i in range(5):
            img_path = images_dir / f"weather_{i}.png"
            # Create different colored images
            color = ("red", "green", "blue", "yellow", "purple")[i]
            test_image = Image.new("RGB", (128, 128), color=color)
            test_image.save(img_path)

            record = {
                "sample_id": f"weather_{i:03d}",
                "image_path": str(img_path),  # Use absolute path instead of relative
                "caption": f"Weather conditions {i} showing {color} patterns",
                "variable": [
                    "temperature",
                    "wind_speed",
                    "precipitation",
                    "humidity",
                    "temperature",
                ][i],
                "model": TEST_MODEL,
                "date": f"2019-04-{15 + i:02d}",
                "metadata": {"fxx": 0, "region": "Continental US"},
            }
            records.append(record)

        with jsonl_file.open("w") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")

        # Create dataset
        mock_processor = Mock()
        dataset = HRRRImageCaptionDataset(
            jsonl_file=jsonl_file, processor=mock_processor
        )

        # Test all operations
        assert len(dataset) == 5

        # Test data access
        for i in range(5):
            image, caption = dataset[i]
            assert isinstance(image, Image.Image)
            assert isinstance(caption, str)
            assert image.size == (128, 128)
            assert f"Weather conditions {i}" in caption

        # Test filtering
        temp_dataset = dataset.filter_by_variable("temperature")
        assert len(temp_dataset) == 2

        # Test statistics
        stats = dataset.get_statistics()
        assert stats["total_samples"] == 5
        assert stats["variables"]["temperature"] == 2
        assert stats["variables"]["wind_speed"] == 1
        assert stats["dates"]["min"] == "2019-04-15"
        assert stats["dates"]["max"] == "2019-04-19"

        # Test record access
        record = dataset.get_record(0)
        assert record.sample_id == "weather_000"
        assert record.variable == "temperature"

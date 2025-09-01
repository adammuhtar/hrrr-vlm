"""Module for retrieval evaluation of fine-tuned SigLIP models on NOAA HRRR
image-caption dataset.

This module provides evaluation metrics for meteorological image-text retrieval:
- Recall@K: Measures retrieval coverage at different cutoffs
- MRR (Mean Reciprocal Rank): Measures ranking quality
- nDCG@K: Measures normalised discounted cumulative gain
"""

import json
import re
from collections.abc import Sequence
from os import PathLike
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from hrrr_vlm.data.constants import REGIONS
from hrrr_vlm.eval.exceptions import RetrievalEvaluationError
from hrrr_vlm.train.exceptions import ModelInitError
from hrrr_vlm.train.train import HRRRLoRASigLIPTrainer
from hrrr_vlm.utils.logger import configure_logger

# Configure logging
logger = configure_logger()


class RetrievalResults(NamedTuple):
    """Results from retrieval evaluation."""

    # Recall@K metrics
    i2t_recall_at_1: float
    i2t_recall_at_5: float
    i2t_recall_at_10: float
    t2i_recall_at_1: float
    t2i_recall_at_5: float
    t2i_recall_at_10: float
    mean_recall_at_1: float
    mean_recall_at_5: float
    mean_recall_at_10: float

    # MRR metrics
    i2t_mrr: float
    t2i_mrr: float
    mean_mrr: float

    # nDCG metrics
    i2t_ndcg_at_5: float
    i2t_ndcg_at_10: float
    t2i_ndcg_at_5: float
    t2i_ndcg_at_10: float
    mean_ndcg_at_5: float
    mean_ndcg_at_10: float


class QueryTestResults(NamedTuple):
    """Results from individual query testing."""

    total_queries: int
    correct_predictions: int
    accuracy: float
    mean_confidence: float
    results_by_region: dict[str, dict[str, Any]]
    results_by_date: dict[str, dict[str, Any]]
    detailed_results: list[dict[str, Any]]


class WeatherReport:
    """Structured representation of HRRR captions.

    Attributes:
        raw_caption (`str`): Original caption text.
        region (`str`): Geographical region.
        date (`str`): Date of the weather report.
        season (`str`): Season information.
        temperature_range (`tuple[float, float]`): Min and max temperatures in
            Celsius.
        avg_temperature (`float`): Average temperature in Celsius.
        wind_speed (`float`): Average wind speed in km/h.
        precipitation (`str`): Precipitation level description.
        humidity (`float`): Humidity percentage.
        conditions (`str`): General weather conditions description.
    """

    def __init__(self, caption: str) -> None:
        """Parse a weather caption into structured components.

        Args:
            caption (`str`): Raw weather caption string.
        """
        self.raw_caption = caption
        self.region = self.extract_region(caption)
        self.date = self.extract_date(caption)
        self.season = self.extract_season(caption)
        self.temperature_range = self.extract_temperature_range(caption)
        self.avg_temperature = self.extract_avg_temperature(caption)
        self.wind_speed = self.extract_wind_speed(caption)
        self.precipitation = self.extract_precipitation(caption)
        self.humidity = self.extract_humidity(caption)
        self.conditions = self.extract_conditions(caption)

    @staticmethod
    def extract_region(caption: str) -> str:
        """Extract geographical region from caption.

        Args:
            caption (`str`): Raw weather caption string.

        Returns:
            `str`: Extracted region or "Unknown" if not found.
        """
        region_patterns = ["Continental US", *list(REGIONS.keys())]

        for pattern in region_patterns:
            if pattern.lower() in caption.lower():
                return pattern
        return "Unknown"

    @staticmethod
    def extract_date(caption: str) -> str:
        """Extract date from caption.

        Args:
            caption (`str`): Raw weather caption string.

        Returns:
            `str`: Extracted date in "DD Mon YYYY" format or "Unknown" if not found.
        """
        # Extract date patterns e.g. "06 Mar 2023", "29 May 2023"
        date_pattern = r"(\d{1,2} \w{3} \d{4})"
        match = re.search(date_pattern, caption)
        return match.group(1) if match else "Unknown"

    @staticmethod
    def extract_season(caption: str) -> str:
        """Extract season information.

        Args:
            caption (`str`): Raw weather caption string.

        Returns:
            `str`: Extracted season or "Unknown" if not found.
        """
        # Look for season patterns like "early spring", "late spring", "mid summer"
        season_pattern = r"(early|mid|late)\s+(spring|summer|fall|autumn|winter)"
        match = re.search(season_pattern, caption.lower())
        if match:
            return f"{match.group(1)} {match.group(2)}"

        # Fallback to just season names
        seasons = ["spring", "summer", "fall", "autumn", "winter"]
        for season in seasons:
            if season in caption.lower():
                return season
        return "Unknown"

    @staticmethod
    def extract_temperature_range(caption: str) -> tuple[float, float] | None:
        """Extract temperature range in Celsius.

        Args:
            caption (`str`): Raw weather caption string.

        Returns:
            `tuple[float, float]`: Tuple of (min_temp, max_temp) or None if not
                found.
        """
        temp_range_patterns = [
            r"from ([-\d.]+)°C to ([-\d.]+)°C",
            r"ranging from ([-\d.]+)°C to ([-\d.]+)°C",
            r"lows of ([-\d.]+)°C and highs of ([-\d.]+)°C",
        ]

        for pattern in temp_range_patterns:
            match = re.search(pattern, caption)
            if match:
                return (float(match.group(1)), float(match.group(2)))

        return None

    @staticmethod
    def extract_avg_temperature(caption: str) -> float | None:
        """Extract average temperature.

        Args:
            caption (`str`): Raw weather caption string.

        Returns:
            `float`: Average temperature or None if not found.
        """
        avg_pattern = r"averaging ([-\d.]+)°C"
        match = re.search(avg_pattern, caption)
        return float(match.group(1)) if match else None

    @staticmethod
    def extract_wind_speed(caption: str) -> float | None:
        """Extract wind speed.

        Args:
            caption (`str`): Raw weather caption string.

        Returns:
            `float`: Wind speed or None if not found.
        """
        wind_pattern = r"winds? averaging ([\d.]+) km/h"
        match = re.search(wind_pattern, caption, re.IGNORECASE)
        return float(match.group(1)) if match else None

    @staticmethod
    def extract_precipitation(caption: str) -> str:
        """Extract precipitation level.

        Args:
            caption (`str`): Raw weather caption string.

        Returns:
            `str`: Precipitation level description or "Unknown" if not found.
        """
        precip_terms = [
            "low precipitation",
            "moderate precipitation",
            "high precipitation",
            "no precipitation",
            "heavy precipitation",
            "light precipitation",
        ]

        for term in precip_terms:
            if term in caption.lower():
                return term
        return "Unknown"

    @staticmethod
    def extract_humidity(caption: str) -> float | None:
        """Extract humidity percentage.

        Args:
            caption (`str`): Raw weather caption string.

        Returns:
            `float`: Humidity percentage or None if not found.
        """
        humidity_pattern = r"humidity \(([\d.]+)%\)"
        match = re.search(humidity_pattern, caption)
        return float(match.group(1)) if match else None

    @staticmethod
    def extract_conditions(caption: str) -> str:
        """Extract general weather conditions.

        Args:
            caption (`str`): Raw weather caption string.

        Returns:
            `str`: Weather condition description or "Unknown" if not found.
        """
        conditions = [
            "cold conditions",
            "warm conditions",
            "hot conditions",
            "cool conditions",
            "mild conditions",
            "extreme conditions",
        ]

        for condition in conditions:
            if condition in caption.lower():
                return condition
        return "Unknown"


class WeatherRelevanceScorer:
    """Scorer for relevance of candidate image/caption relative to query.

    Attributes:
        temp_tolerance (`float`): Temperature tolerance in Celsius.
        wind_tolerance (`float`): Wind speed tolerance in km/h.
        humidity_tolerance (`float`): Humidity tolerance in percentage.
        weights (`dict[str, float]`): Weights for different similarity components.
        active_keys (`list[str]`): List of active similarity components.
        max_total_weight (`float`): Maximum possible total weight for normalisation.
    """

    def __init__(
        self,
        temp_tolerance: float = 5.0,
        wind_tolerance: float = 10.0,
        humidity_tolerance: float = 15.0,
        exact_match_weight: float = 1.0,
        region_weight: float = 0.8,
        season_weight: float = 0.6,
        temp_weight: float = 0.7,
        conditions_weight: float = 0.5,
    ) -> None:
        """Initialise weather relevance scorer with tolerance parameters.

        Args:
            temp_tolerance (`float`, optional): Temperature tolerance in Celsius.
            wind_tolerance (`float`, optional): Wind speed tolerance in km/h.
            humidity_tolerance (`float`, optional): Humidity tolerance in percentage.
            exact_match_weight (`float`, optional): Weight for exact caption matches.
            region_weight (`float`, optional): Weight for region similarity.
            season_weight (`float`, optional): Weight for season similarity.
            temp_weight (`float`, optional): Weight for temperature similarity.
            conditions_weight (`float`, optional): Weight for weather conditions
                similarity.
        """
        self.temp_tolerance = temp_tolerance
        self.wind_tolerance = wind_tolerance
        self.humidity_tolerance = humidity_tolerance

        # Weights for different similarity components
        self.weights = {
            "exact_match": exact_match_weight,
            "region": region_weight,
            "season": season_weight,
            "temperature": temp_weight,
            "conditions": conditions_weight,
        }
        self.active_keys = ["region", "season", "temperature", "conditions"]
        self.max_total_weight = sum(self.weights[k] for k in self.active_keys)

    def compute_relevance_matrix(
        self,
        image_indices: Sequence[int],
        text_indices: Sequence[int],
        dataset_captions: Sequence[str],
    ) -> np.ndarray:
        """Compute relevance matrix for HRRR weather data.

        Args:
            image_indices (`Sequence[int]`): Indices of images in the dataset.
            text_indices (`Sequence[int]`): Indices of text captions in the dataset.
            dataset_captions (`Sequence[str]`): All captions in the dataset.

        Returns:
            `np.ndarray`: Relevance matrix of shape (num_images, num_texts) with
                relevance scores in [0, 1].
        """
        logger.info("Computing weather relevance matrix with meteorological similarity")

        n_images = len(image_indices)
        n_texts = len(text_indices)
        relevance = np.zeros((n_images, n_texts))

        # Parse all captions into structured weather reports
        weather_reports = [
            WeatherReport(dataset_captions[i]) for i in range(len(dataset_captions))
        ]

        for i, img_idx in enumerate(image_indices):
            for j, txt_idx in enumerate(text_indices):
                if img_idx == txt_idx:
                    # Exact match - perfect relevance
                    relevance[i][j] = 1.0
                else:
                    # Compute meteorological similarity
                    relevance[i][j] = self.compute_weather_similarity(
                        weather_reports[img_idx], weather_reports[txt_idx]
                    )

        return relevance

    def compute_weather_similarity(
        self, report1: WeatherReport, report2: WeatherReport
    ) -> float:
        """Compute similarity between two weather reports.

        Args:
            report1 (`WeatherReport`): First weather report.
            report2 (`WeatherReport`): Second weather report.

        Returns:
            `float`: Similarity score in [0, 1].
        """
        if report1.raw_caption == report2.raw_caption:
            return 1.0

        similarity_score = 0.0

        # Region
        if report1.region != "Unknown" and report2.region != "Unknown":
            region_sim = 1.0 if report1.region == report2.region else 0.0
            similarity_score += region_sim * self.weights["region"]

        # Season
        if report1.season != "Unknown" and report2.season != "Unknown":
            season_sim = self.compute_season_similarity(report1.season, report2.season)
            similarity_score += season_sim * self.weights["season"]

        # Temperature
        if report1.avg_temperature is not None and report2.avg_temperature is not None:
            temp_sim = self.compute_temperature_similarity(
                report1.avg_temperature, report2.avg_temperature
            )
            similarity_score += temp_sim * self.weights["temperature"]

        # Conditions
        if report1.conditions != "Unknown" and report2.conditions != "Unknown":
            conditions_sim = 1.0 if report1.conditions == report2.conditions else 0.0
            similarity_score += conditions_sim * self.weights["conditions"]

        # Normalise by the maximum possible weight
        return (
            similarity_score / self.max_total_weight
            if self.max_total_weight > 0
            else 0.0
        )

    @staticmethod
    def compute_season_similarity(season1: str, season2: str) -> float:
        """Compute similarity between seasons.

        Args:
            season1 (`str`): First season string.
            season2 (`str`): Second season string.

        Returns:
            `float`: Similarity score in [0, 1].
        """
        if season1 == season2:
            return 1.0

        # Adjacent seasons get partial credit
        season_order = ["winter", "spring", "summer", "fall", "autumn"]

        # Handle autumn/fall equivalence
        s1 = season1.replace("autumn", "fall")
        s2 = season2.replace("autumn", "fall")

        try:
            idx1 = season_order.index(
                s1.split()[-1]
            )  # Get last word (handles "mid spring")
            idx2 = season_order.index(s2.split()[-1])

            # Circular distance
            diff = min(abs(idx1 - idx2), 4 - abs(idx1 - idx2))

            if diff == 0:
                return 1.0
            if diff == 1:
                return 0.5  # Adjacent seasons
        except (ValueError, IndexError):
            return 0.0
        else:
            return 0.0

    def compute_temperature_similarity(self, temp1: float, temp2: float) -> float:
        """Compute similarity between temperatures.

        Args:
            temp1 (`float`): First temperature in Celsius.
            temp2 (`float`): Second temperature in Celsius.

        Returns:
            `float`: Similarity score in [0, 1].
        """
        diff = abs(temp1 - temp2)

        if diff <= self.temp_tolerance:
            return 1.0 - (diff / self.temp_tolerance)
        return 0.0


class RetrievalMetrics:
    """Utility class for computing retrieval evaluation metrics."""

    @staticmethod
    def recall_at_k(
        rankings: np.ndarray,
        relevance: np.ndarray,
        k: int,
        threshold: float | None = None,
    ) -> float:
        """Recall@K when there is 1 relevant item within the top-k results.

        If `threshold` is None, treat `relevance` as binary (nonzero is relevant).
        Otherwise, mark items with relevance >= threshold as relevant.

        Args:
            rankings (`np.ndarray`): Array of shape (num_queries, num_items) with
                ranked item indices for each query.
            relevance (`np.ndarray`): Binary or graded relevance matrix of shape
                (num_queries, num_items) with relevance scores.
            k (`int`): Cutoff rank for computing recall.
            threshold (`float`, optional): Minimum relevance score to consider an
                item as relevant.

        Returns:
            `float`: Mean Recall@K score across all queries.
        """
        k = min(k, rankings.shape[1])
        hits = []
        for i in range(rankings.shape[0]):
            topk = rankings[i, :k]
            if threshold is None:
                is_hit = np.any(relevance[i, topk] > 0.0)
            else:
                is_hit = np.any(relevance[i, topk] >= threshold)
            hits.append(float(is_hit))
        return float(np.mean(hits))

    @staticmethod
    def mean_reciprocal_rank(
        rankings: np.ndarray, relevance: np.ndarray, relevance_threshold: float = 0.8
    ) -> float:
        """Compute Mean Reciprocal Rank (MRR) with relevance threshold.

        Args:
            rankings (`np.ndarray`): Array of shape (num_queries, num_items) with
                ranked item indices for each query.
            relevance (`np.ndarray`): Binary or graded relevance matrix of shape
                (num_queries, num_items) with relevance scores.
            relevance_threshold (`float`, optional): Minimum relevance score to
                consider an item as relevant.

        Returns:
            `float`: Mean Reciprocal Rank score across all queries.
        """
        reciprocal_ranks = []

        for i in range(rankings.shape[0]):
            # Find first relevant item in ranking
            for rank, item_idx in enumerate(rankings[i]):
                if relevance[i, item_idx] >= relevance_threshold:
                    reciprocal_ranks.append(1.0 / (rank + 1))
                    break
            else:
                # No relevant item found
                reciprocal_ranks.append(0.0)

        return float(np.mean(reciprocal_ranks))

    @staticmethod
    def ndcg_at_k(rankings: np.ndarray, relevance: np.ndarray, k: int) -> float:
        """Compute Normalised Discounted Cumulative Gain at K.

        Args:
            rankings (`np.ndarray`): Array of shape (num_queries, num_items) with
                ranked item indices for each query.
            relevance (`np.ndarray`): Binary or graded relevance matrix of shape
                (num_queries, num_items) with relevance scores.
            k (`int`): Cutoff rank for computing nDCG.

        Returns:
            `float`: Mean nDCG@K score across all queries.
        """
        k = min(k, rankings.shape[1])

        ndcg_scores = []

        for i in range(rankings.shape[0]):
            # DCG@K
            dcg = 0.0
            for rank in range(min(k, len(rankings[i]))):
                item_idx = rankings[i, rank]
                relevance_score = relevance[i, item_idx]
                dcg += relevance_score / np.log2(
                    rank + 2
                )  # rank + 2 because log2(1) = 0

            # IDCG@K (Ideal DCG)
            ideal_relevance = np.sort(relevance[i])[::-1]  # Sort in descending order
            idcg = 0.0
            for rank in range(min(k, len(ideal_relevance))):
                idcg += ideal_relevance[rank] / np.log2(rank + 2)

            # nDCG@K
            if idcg > 0:
                ndcg_scores.append(dcg / idcg)
            else:
                ndcg_scores.append(0.0)

        return float(np.mean(ndcg_scores))


class RetrievalEvaluator:
    """Main class for evaluating retrieval performance of SigLIP models fine-tuned
    on NOAA HRRR image-caption datasets.

    Attributes:
        trainer (`HRRRLoRASigLIPTrainer`): Fine-tuned SigLIP model loaded in
            HRRRLoRASigLIPTrainer instance.
        relevance_scorer (`WeatherRelevanceScorer`): Custom relevance scorer
            for weather data.
        metrics (`RetrievalMetrics`): Utility class for computing retrieval metrics.
    """

    def __init__(
        self,
        trainer: HRRRLoRASigLIPTrainer,
        relevance_scorer: WeatherRelevanceScorer | None = None,
    ) -> None:
        """Initialise the retrieval evaluator.

        Args:
            trainer (`HRRRLoRASigLIPTrainer`): Fine-tuned SigLIP model loaded in
                HRRRLoRASigLIPTrainer instance.
            relevance_scorer (`WeatherRelevanceScorer`): Custom relevance scorer
                for weather data.

        Raises:
            ModelInitError: If the trainer's model is not initialised.
        """
        self.trainer = trainer
        self.relevance_scorer = relevance_scorer or WeatherRelevanceScorer()
        self.metrics = RetrievalMetrics()

        if self.trainer.model is None:
            msg = "Trainer model must be initialised before evaluation"
            raise ModelInitError(msg)

    @staticmethod
    def load_jsonl_data(jsonl_path: PathLike[str]) -> list[dict[str, Any]]:
        """Load data from JSONL file.

        Args:
            jsonl_path (`PathLike[str]`): Path to the JSONL file.

        Returns:
            List of data samples
        """
        data: list[dict[str, Any]] = []
        path_obj = Path(jsonl_path)
        with path_obj.open(encoding="utf-8") as f:
            data.extend(json.loads(line.strip()) for line in f)

        logger.info("Loaded %d samples from %s", len(data), jsonl_path)
        return data

    @staticmethod
    def create_distractor_captions(
        correct_caption: str,
        all_captions: list[str],
        num_distractors: int = 4,
        rng: np.random.Generator | None = None,
    ) -> list[str]:
        """Create distractor captions for testing.

        Args:
            correct_caption (`str`): Correct caption for the given image.
            all_captions (`list[str]`): All available captions to sample from.
            num_distractors (`int`): Number of distractor captions to create.
            rng (`np.random.Generator`, optional): Random number generator
                instance.

        Returns:
            `list[str]`: List containing correct caption first, followed by
                distractors.
        """
        if rng is None:
            rng = np.random.default_rng(42)

        # Filter out the correct caption from potential distractors
        potential_distractors = [cap for cap in all_captions if cap != correct_caption]

        # Select random distractors
        if len(potential_distractors) >= num_distractors:
            distractors = rng.choice(
                potential_distractors, num_distractors, replace=False
            ).tolist()
        else:
            # If not enough unique captions, repeat some
            distractors = (
                potential_distractors
                * ((num_distractors // len(potential_distractors)) + 1)
            )[:num_distractors]

        # Return correct caption first, then distractors
        return [correct_caption, *distractors]

    def test_individual_queries(
        self,
        jsonl_path: PathLike[str],
        num_samples: int | None = None,
        num_distractors: int = 9,
        random_seed: int = 42,
    ) -> QueryTestResults:
        """Test individual image queries using trainer's test_caption_to_image method.

        Args:
            jsonl_path (`PathLike[str]`): Path to JSONL file with test data.
            num_samples (`int`, optional): Number of samples to test (None for all).
            num_distractors (`int`): Number of distractor captions per query.
            random_seed (`int`): Random seed for reproducibility.

        Returns:
            `QueryTestResults`: QueryTestResults containing detailed test results.

        Raises:
            RetrievalEvaluationError: If individual query testing fails
        """
        rng = np.random.default_rng(random_seed)

        try:
            # Load test data
            test_data = self.load_jsonl_data(jsonl_path)

            if num_samples is not None:
                test_data = test_data[:num_samples]

            # Extract all captions for creating distractors
            all_captions = [item["caption"] for item in test_data]

            detailed_results = []
            region_stats = {}
            date_stats = {}

            logger.debug(
                "Testing individual queries with distractors",
                total_queries=len(test_data),
                num_distractors=num_distractors,
            )

            for _i, item in enumerate(tqdm(test_data, desc="Testing queries")):
                try:
                    # Create test captions (correct + distractors)
                    test_captions = self.create_distractor_captions(
                        item["caption"], all_captions, num_distractors, rng
                    )

                    # Test using trainer's method
                    result = self.trainer.test_caption_to_image(
                        image_path=item["image_path"],
                        weather_descriptions=test_captions,
                        return_probs=True,
                    )

                    # Check if correct (index 0 should be predicted)
                    is_correct = result["predicted_description_idx"] == 0
                    confidence = result["confidence"]

                    # Extract region and date for analysis
                    weather_report = WeatherReport(item["caption"])
                    region = weather_report.region
                    date = weather_report.date

                    # Store detailed result
                    detailed_result = {
                        "sample_id": item["sample_id"],
                        "image_path": item["image_path"],
                        "correct_caption": item["caption"],
                        "predicted_idx": result["predicted_description_idx"],
                        "predicted_caption": result["predicted_description"],
                        "is_correct": is_correct,
                        "confidence": confidence,
                        "region": region,
                        "date": date,
                        "all_probabilities": result["probabilities"],
                    }
                    detailed_results.append(detailed_result)

                    # Update region statistics
                    if region not in region_stats:
                        region_stats[region] = {
                            "correct": 0,
                            "total": 0,
                            "confidences": [],
                        }
                    region_stats[region]["total"] += 1
                    region_stats[region]["confidences"].append(confidence)
                    if is_correct:
                        region_stats[region]["correct"] += 1

                    # Update date statistics
                    if date not in date_stats:
                        date_stats[date] = {"correct": 0, "total": 0, "confidences": []}
                    date_stats[date]["total"] += 1
                    date_stats[date]["confidences"].append(confidence)
                    if is_correct:
                        date_stats[date]["correct"] += 1

                except Exception as e:
                    logger.warning(
                        "Error during testing",
                        sample_id=item["sample_id"],
                        error=str(e),
                    )
                    continue

            # Calculate summary statistics
            total_queries = len(detailed_results)
            correct_predictions = sum(1 for r in detailed_results if r["is_correct"])
            accuracy = correct_predictions / total_queries if total_queries > 0 else 0.0
            mean_confidence = (
                sum(r["confidence"] for r in detailed_results) / total_queries
                if total_queries > 0
                else 0.0
            )

            # Process region and date statistics
            for stats in region_stats.values():
                stats["accuracy"] = (
                    stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
                )
                stats["mean_confidence"] = (
                    sum(stats["confidences"]) / len(stats["confidences"])
                    if stats["confidences"]
                    else 0.0
                )
                del stats["confidences"]  # Remove raw data to save memory

            for stats in date_stats.values():
                stats["accuracy"] = (
                    stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
                )
                stats["mean_confidence"] = (
                    sum(stats["confidences"]) / len(stats["confidences"])
                    if stats["confidences"]
                    else 0.0
                )
                del stats["confidences"]  # Remove raw data to save memory

            logger.info(
                "Individual query testing completed",
                accuracy=accuracy,
                score=(correct_predictions / total_queries),
            )

            return QueryTestResults(
                total_queries=total_queries,
                correct_predictions=correct_predictions,
                accuracy=accuracy,
                mean_confidence=mean_confidence,
                results_by_region=region_stats,
                results_by_date=date_stats,
                detailed_results=detailed_results,
            )

        except Exception as e:
            error_msg = f"Individual query testing failed: {e}"
            logger.exception(error_msg)
            raise RetrievalEvaluationError(error_msg) from e

    def extract_embeddings(
        self, dataloader: DataLoader, max_samples: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
        """Extract image and text embeddings from the dataset.

        Args:
            dataloader (`DataLoader`): DataLoader for the dataset.
            max_samples (`int`, optional): Maximum number of samples to process.

        Returns:
            `tuple[torch.Tensor, torch.Tensor, list[str]]`: Tuple containing:
                - Image embeddings tensor of shape (num_samples, embed_dim)
                - Text embeddings tensor of shape (num_samples, embed_dim)
                - List of captions corresponding to each sample

        Raises:
            ModelInitError: If the trainer's model is not initialised.
        """
        if self.trainer.model is None:
            msg = "Model not initialised"
            raise ModelInitError(msg)

        logger.info("Extracting embeddings for retrieval evaluation")

        self.trainer.model.eval()
        image_embeddings = []
        text_embeddings = []
        captions = []

        samples_processed = 0

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Extracting embeddings"):
                if max_samples and samples_processed >= max_samples:
                    break

                # Move to device
                input_ids = batch["input_ids"].to(self.trainer.device)
                pixel_values = batch["pixel_values"].to(self.trainer.device)

                # Get attention mask or create one
                if "attention_mask" in batch:
                    attention_mask = batch["attention_mask"].to(self.trainer.device)
                else:
                    attention_mask = torch.ones_like(input_ids).to(self.trainer.device)

                # Forward pass
                outputs = self.trainer.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                )

                # Normalise embeddings
                img_embeds = F.normalize(outputs.image_embeds, p=2, dim=-1)
                txt_embeds = F.normalize(outputs.text_embeds, p=2, dim=-1)

                image_embeddings.append(img_embeds.cpu())
                text_embeddings.append(txt_embeds.cpu())

                # Decode captions for relevance computation
                batch_captions = self.trainer.processor.batch_decode(
                    input_ids, skip_special_tokens=True
                )
                captions.extend(batch_captions)

                samples_processed += len(batch_captions)

                if max_samples and samples_processed >= max_samples:
                    break

        # Concatenate all embeddings
        image_embeddings = torch.cat(image_embeddings, dim=0)
        text_embeddings = torch.cat(text_embeddings, dim=0)

        logger.info(
            "Extracted embeddings",
            total_samples=len(captions),
            image_shape=image_embeddings.shape,
            text_shape=text_embeddings.shape,
        )

        return image_embeddings, text_embeddings, captions

    @staticmethod
    def compute_similarity_rankings(
        query_embeddings: torch.Tensor, candidate_embeddings: torch.Tensor
    ) -> np.ndarray:
        """Compute similarity-based rankings between queries and candidates.

        Args:
            query_embeddings (`torch.Tensor`): Tensor of shape (num_queries,
                embed_dim) containing query embeddings.
            candidate_embeddings (`torch.Tensor`): Tensor of shape (num_candidates,
                embed_dim) containing candidate embeddings.

        Returns:
            `np.ndarray`: Array of shape (num_queries, num_candidates) with ranked
                candidate indices for each query.
        """
        # Compute similarity matrix
        similarities = torch.mm(query_embeddings, candidate_embeddings.t())

        # Get rankings (indices sorted by descending similarity)
        rankings = torch.argsort(similarities, dim=1, descending=True)

        return rankings.numpy()

    def evaluate_retrieval(
        self,
        dataloader: DataLoader,
        max_samples: int | None = None,
        k_values: list[int] | None = None,
    ) -> RetrievalResults:
        """Evaluate retrieval performance on HRRR image-caption dataset.

        Args:
            dataloader (`DataLoader`): DataLoader for the dataset.
            max_samples (`int`, optional): Maximum number of samples to process.
            k_values (`list[int]`, optional): List of K values for Recall@K.

        Returns:
            `RetrievalResults`: RetrievalResults containing various evaluation
                metrics.

        Raises:
            RetrievalEvaluationError: If retrieval evaluation fails.
        """
        if k_values is None:
            k_values = [1, 5, 10]

        try:
            # Extract embeddings
            image_embeddings, text_embeddings, captions = self.extract_embeddings(
                dataloader, max_samples
            )
            n_samples = len(captions)
            sample_indices = list(range(n_samples))

            # Compute meteorology-aware relevance matrix
            logger.info("Computing HRRR weather relevance matrix")
            relevance_matrix = self.relevance_scorer.compute_relevance_matrix(
                sample_indices, sample_indices, captions
            )
            relevance_matrix_t = relevance_matrix.T

            # Log relevance stats
            non_zero_relevance = relevance_matrix[relevance_matrix > 0]
            logger.info(
                "Relevance statistics",
                non_zero_entries=len(non_zero_relevance),
                mean_relevance=float(non_zero_relevance.mean())
                if len(non_zero_relevance) > 0
                else 0.0,
                max_relevance=float(non_zero_relevance.max())
                if len(non_zero_relevance) > 0
                else 0.0,
            )

            # Build rankings
            logger.info("Evaluating image-to-text retrieval")
            i2t_rankings = self.compute_similarity_rankings(
                image_embeddings, text_embeddings
            )

            logger.info("Evaluating text-to-image retrieval")
            t2i_rankings = self.compute_similarity_rankings(
                text_embeddings, image_embeddings
            )

            # Initialise results container
            results: dict[str, float] = {}

            # Recall@K metrics
            exact_r = np.eye(n_samples, dtype=float)
            i2t_recall_at_1 = self.metrics.recall_at_k(i2t_rankings, exact_r, 1)
            i2t_recall_at_5 = self.metrics.recall_at_k(i2t_rankings, exact_r, 5)
            i2t_recall_at_10 = self.metrics.recall_at_k(i2t_rankings, exact_r, 10)
            t2i_recall_at_1 = self.metrics.recall_at_k(t2i_rankings, exact_r, 1)
            t2i_recall_at_5 = self.metrics.recall_at_k(t2i_rankings, exact_r, 5)
            t2i_recall_at_10 = self.metrics.recall_at_k(t2i_rankings, exact_r, 10)
            results["i2t_recall_at_1"] = i2t_recall_at_1
            results["i2t_recall_at_5"] = i2t_recall_at_5
            results["i2t_recall_at_10"] = i2t_recall_at_10
            results["t2i_recall_at_1"] = t2i_recall_at_1
            results["t2i_recall_at_5"] = t2i_recall_at_5
            results["t2i_recall_at_10"] = t2i_recall_at_10
            results["mean_recall_at_1"] = (i2t_recall_at_1 + t2i_recall_at_1) / 2
            results["mean_recall_at_5"] = (i2t_recall_at_5 + t2i_recall_at_5) / 2
            results["mean_recall_at_10"] = (i2t_recall_at_10 + t2i_recall_at_10) / 2

            # MRR metrics
            results["i2t_mrr"] = self.metrics.mean_reciprocal_rank(
                i2t_rankings, relevance_matrix
            )
            results["t2i_mrr"] = self.metrics.mean_reciprocal_rank(
                t2i_rankings, relevance_matrix_t
            )
            results["mean_mrr"] = (results["i2t_mrr"] + results["t2i_mrr"]) / 2

            # nDCG metrics
            results["i2t_ndcg_at_5"] = self.metrics.ndcg_at_k(
                i2t_rankings, relevance_matrix, 5
            )
            results["i2t_ndcg_at_10"] = self.metrics.ndcg_at_k(
                i2t_rankings, relevance_matrix, 10
            )
            results["t2i_ndcg_at_5"] = self.metrics.ndcg_at_k(
                t2i_rankings, relevance_matrix_t, 5
            )
            results["t2i_ndcg_at_10"] = self.metrics.ndcg_at_k(
                t2i_rankings, relevance_matrix_t, 10
            )
            results["mean_ndcg_at_5"] = (
                results["i2t_ndcg_at_5"] + results["t2i_ndcg_at_5"]
            ) / 2
            results["mean_ndcg_at_10"] = (
                results["i2t_ndcg_at_10"] + results["t2i_ndcg_at_10"]
            ) / 2

            logger.info("Retrieval evaluation completed", num_samples=n_samples)

            return RetrievalResults(**results)

        except Exception as e:
            error_msg = f"Retrieval evaluation failed: {e}"
            logger.exception(error_msg)
            raise RetrievalEvaluationError(error_msg) from e

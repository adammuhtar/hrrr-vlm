"""Predict hurricanes using weather embeddings with Logistic Regression (L2)."""

import json
from datetime import UTC, datetime, timedelta
from os import PathLike
from pathlib import Path
from typing import Any, ClassVar, NamedTuple

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

from hrrr_vlm.eval.embeddings import EmbeddingData
from hrrr_vlm.utils.logger import configure_logger

# Configure logger
logger = configure_logger()

PREDICTION_THRESHOLD = 0.5


class LogisticRegressionConfig(NamedTuple):
    """Parameters for L2-regularised logistic regression.

    Attributes:
        regularisation_strength: Inverse of regularisation strength; must be a
            positive float. Smaller values specify stronger regularisation.
        max_iter: Maximum number of iterations taken for the solvers to converge.
        solver: Algorithm to use in the optimisation process.
        class_weight: Weights associated with classes in the form {class_label:
            weight}. If "balanced", the class weights will be adjusted inversely
            proportional to class frequencies in the input data as n_samples /
            (n_classes * np.bincount(y)).
    """

    regularisation_strength: float = 1.0  # sklearn's C
    max_iter: int = 1000
    solver: str = "lbfgs"
    class_weight: str | dict | None = "balanced"  # handle class imbalance


class HurricanePredictionResults(NamedTuple):
    """Evaluation bundle for the fitted model.

    Attributes:
        predictions (`np.ndarray`): Predicted probabilities for the positive class.
        labels (`np.ndarray`): True binary labels.
        accuracy (`float`): Accuracy score.
        precision (`float`): Precision score.
        recall (`float`): Recall score.
        f1 (`float`): F1 score.
        auc_roc (`float`): Area Under the Receiver Operating Characteristic Curve.
        feature_importance (`np.ndarray`): Absolute coefficients of the model.
        confusion_matrix (`np.ndarray`): Confusion matrix.
        classification_report (`str`): Text summary of precision, recall, F1-score.
        cross_val_scores (`np.ndarray`): Cross-validation F1 scores.
        model_type (`str`): Type of model used (e.g., 'logistic_l2').
        fitted_model (`Any`): The trained model instance.
        test_features (`np.ndarray | None`): Features used in the test set.
        scaler (`Any`): The fitted scaler instance.
    """

    predictions: np.ndarray
    labels: np.ndarray
    accuracy: float
    precision: float
    recall: float
    f1: float
    auc_roc: float
    feature_importance: np.ndarray | None
    confusion_matrix: np.ndarray
    classification_report: str
    cross_val_scores: np.ndarray
    model_type: str
    fitted_model: Any | None = None
    test_features: np.ndarray | None = None
    scaler: Any | None = None


class RegionMapper:
    """Maps between different regional naming conventions."""

    REGION_MAPPING: ClassVar[dict[str, list[str]]] = {
        "Northeast": ["Northeast"],
        "Ohio Valley": ["Ohio Valley"],
        "South": ["South"],
        "Southeast": ["Southeast"],
        "Continental US": ["Continental US"],
        "Alaska": ["Alaska"],
        "Northern Rockies and Plains": ["Northern Rockies and Plains"],
        "Northwest": ["Northwest"],
        "Southwest": ["Southwest"],
        "Upper Midwest": ["Upper Midwest"],
        "West": ["West"],
    }

    REVERSE_MAPPING: ClassVar[dict[str, str]] = {
        jsonl: h for h, arr in REGION_MAPPING.items() for jsonl in arr
    }

    @classmethod
    def map_jsonl_to_hurricane_region(cls, jsonl_region: str) -> str:
        """Map JSONL region name to hurricane dataset region name, defaulting to
            'Continental US'.

        Args:
            jsonl_region (`str`): Region name from JSONL metadata.

        Returns:
            `str`: Corresponding hurricane dataset region name.
        """
        return cls.REVERSE_MAPPING.get(jsonl_region, "Continental US")


class HurricaneDataProcessor:
    """Loads hurricane CSV and provides date/region matching.

    Attributes:
        hurricane_df (`pl.DataFrame`): Original hurricane data.
        expanded_hurricane_df (`pl.DataFrame`): Expanded data with one row per region.
    """

    def __init__(self, hurricane_csv_path: PathLike[str]) -> None:
        """Initialise and preprocess hurricane data.

        Args:
            hurricane_csv_path (`PathLike[str]`): Path to the hurricane CSV file.
        """
        self.hurricane_df = pl.read_csv(hurricane_csv_path)
        self.preprocess_hurricane_data()

    def preprocess_hurricane_data(self) -> None:
        """Preprocess the hurricane DataFrame for easier querying."""
        self.hurricane_df = self.hurricane_df.with_columns(
            [
                pl.col("Start Date")
                .str.to_datetime(format="%b %d, %Y")
                .dt.replace_time_zone("UTC")
                .alias("start_date"),
                pl.col("End Date")
                .str.to_datetime(format="%b %d, %Y")
                .dt.replace_time_zone("UTC")
                .alias("end_date"),
            ]
        )

        expanded = []
        for row in self.hurricane_df.iter_rows(named=True):
            for region in [r.strip() for r in row["Regions"].split(",")]:
                new_row = row.copy()
                new_row["region"] = region
                expanded.append(new_row)
        self.expanded_hurricane_df = pl.DataFrame(expanded)
        logger.info(
            "Processed hurricane records",
            csv_samples=len(self.hurricane_df),
            region_samples=len(self.expanded_hurricane_df),
        )

    def is_hurricane_period(
        self, date: str, region: str, lookahead_days: int = 3
    ) -> tuple[bool, str | None]:
        """Check if a given date and region falls within a hurricane period.

        Args:
            date (`str`): Date string in 'YYYY-MM-DD' format.
            region (`str`): Region name from JSONL metadata.
            lookahead_days (`int`): Number of days to look ahead for hurricane
                start.

        Returns:
            `tuple[bool, str | None]`: Tuple indicating if within hurricane period
                and the hurricane name.
        """
        try:
            target = datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=UTC)
            hurricane_region = RegionMapper.map_jsonl_to_hurricane_region(region)
            start_window, end_window = target, target + timedelta(days=lookahead_days)
            for h in self.expanded_hurricane_df.iter_rows(named=True):
                if (
                    h["region"] == hurricane_region
                    and start_window < h["start_date"]
                    and end_window >= h["start_date"]
                ):
                    return True, h["Hurricane"]
        except (ValueError, KeyError) as e:
            logger.warning("Error processing date %s, region %s: %s", date, region, e)
        return False, None


class WeatherFeatureExtractor:
    """Extracts lightweight one-hot encoding features from metadata."""

    @staticmethod
    def extract_weather_features(metadata: dict[str, Any]) -> dict[str, float]:
        """Extract weather-related features from metadata.

        Args:
            metadata (`dict[str, Any]`): Metadata dictionary.

        Returns:
            `dict[str, float]`: Extracted feature dictionary.
        """
        features: dict[str, float] = {}

        # Seasonal one-hot
        season = metadata.get("season", "Unknown").lower()
        features["is_winter"] = 1.0 if "winter" in season else 0.0
        features["is_spring"] = 1.0 if "spring" in season else 0.0
        features["is_summer"] = 1.0 if "summer" in season else 0.0
        features["is_autumn"] = 1.0 if "autumn" in season or "fall" in season else 0.0

        # Region one-hot (explicit regions + fallback to Continental US)
        region = metadata.get("region", "Unknown")
        all_regions = [
            "Alaska",
            "Northeast",
            "Northern Rockies and Plains",
            "Northwest",
            "Ohio Valley",
            "South",
            "Southeast",
            "Southwest",
            "Upper Midwest",
            "West",
        ]
        for r in all_regions:
            features[f"is_region_{r.lower().replace(' ', '_')}"] = (
                1.0 if region == r else 0.0
            )
        features["is_region_continental_us"] = 1.0 if region not in all_regions else 0.0

        return features


class HurricanePredictor:
    """Prepares data and trains/evaluates L2 Logistic Regression.

    Attributes:
        hurricane_processor (`HurricaneDataProcessor`): Processor for hurricane
            data.
        feature_extractor (`WeatherFeatureExtractor`): Extractor for weather
            features.
        scaler (`StandardScaler`): Scaler for feature normalisation.
    """

    def __init__(self, hurricane_csv_path: PathLike[str]) -> None:
        """Initialise with hurricane data processor and feature extractor.

        Args:
            hurricane_csv_path (`PathLike[str]`): Path to the hurricane CSV file.
        """
        self.hurricane_processor = HurricaneDataProcessor(hurricane_csv_path)
        self.feature_extractor = WeatherFeatureExtractor()
        self.scaler = StandardScaler()

    def prepare_prediction_dataset(
        self,
        embedding_data: EmbeddingData,
        *,
        lookahead_days: int = 7,
        use_embeddings: bool = True,
        use_weather_features: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
        """Prepare features and labels for hurricane prediction.

        Args:
            embedding_data (`EmbeddingData`): Embedding data with metadata.
            lookahead_days (`int`): Days to look ahead for hurricane start.
            use_embeddings (`bool`): Whether to include image embeddings.
            use_weather_features (`bool`): Whether to include weather features.

        Returns:
            `tuple[np.ndarray, np.ndarray, list[str], list[str]]`: Features array,
                labels array, feature names, and sample info.
        """
        features_list: list[list[float]] = []
        labels: list[int] = []
        feature_names: list[str] = []
        sample_info: list[str] = []

        logger.info("Preparing dataset for %d samples", len(embedding_data.metadata))

        for i, metadata in enumerate(embedding_data.metadata):
            row_feats: list[float] = []
            names_this_row: list[str] = []

            # Embedding features
            if use_embeddings and i < len(embedding_data.image_embeddings):
                img_emb = embedding_data.image_embeddings[i].numpy()
                row_feats.extend(img_emb)
                if not feature_names:
                    names_this_row.extend([f"img_emb_{j}" for j in range(len(img_emb))])

            # Weather metadata features
            if use_weather_features:
                w = self.feature_extractor.extract_weather_features(metadata)
                row_feats.extend(w.values())
                if not feature_names:
                    names_this_row.extend(list(w.keys()))

            if not feature_names:
                feature_names = names_this_row

            features_list.append(row_feats)

            # Label from hurricane proximity
            date = metadata.get("date", "")
            region = metadata.get("region", "")
            is_hurr, name = self.hurricane_processor.is_hurricane_period(
                date, region, lookahead_days
            )
            labels.append(1 if is_hurr else 0)
            sample_info.append(f"{date}_{region}_{name or 'no_hurricane'}")

        X = np.array(features_list)  # noqa: N806
        y = np.array(labels)

        logger.info(
            "Prepared: %d samples, %d features, positives=%d (%.2f%%)",
            len(X),
            X.shape[1] if len(X) > 0 else 0,
            y.sum(),
            (y.mean() * 100) if len(y) else 0,
        )
        return X, y, feature_names, sample_info

    def train_and_evaluate(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        *,
        test_size: float = 0.2,
        cv_folds: int = 5,
        random_state: int = 42,
        config: LogisticRegressionConfig | None = None,
    ) -> HurricanePredictionResults:
        """Train and evaluate L2-regularised logistic regression.

        Args:
            features (`np.ndarray`): Feature array.
            labels (`np.ndarray`): Binary label array.
            test_size (`float`): Proportion of data for testing.
            cv_folds (`int`): Number of cross-validation folds.
            random_state (`int`): Random seed for reproducibility.
            config (`LogisticRegressionConfig`, optional): Configuration for
                logistic regression.

        Returns:
            `HurricanePredictionResults`: Results of the evaluation.

        Raises:
            ValueError: If no features are provided.
        """
        if config is None:
            config = LogisticRegressionConfig()
        if len(features) == 0:
            err_msg = "No features provided for training."
            logger.error(err_msg)
            raise ValueError(err_msg)

        X_train, X_test, y_train, y_test = train_test_split(  # noqa: N806
            features,
            labels,
            test_size=test_size,
            random_state=random_state,
            stratify=labels if len(set(labels)) > 1 else None,
        )

        X_train_s = self.scaler.fit_transform(X_train)  # noqa: N806
        X_test_s = self.scaler.transform(X_test)  # noqa: N806

        model = LogisticRegression(
            random_state=random_state,
            C=config.regularisation_strength,
            max_iter=config.max_iter,
            solver=config.solver,
            class_weight=config.class_weight,
            penalty="l2",
        )

        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)
        y_proba = model.predict_proba(X_test_s)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        try:
            auc_roc = roc_auc_score(y_test, y_proba)
        except ValueError:
            auc_roc = 0.0
            logger.warning("AUC-ROC undefined: only one class present in test set")

        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        # Cross-validate inside training split (scaled)
        cv_scores = cross_val_score(
            model, X_train_s, y_train, cv=cv_folds, scoring="f1"
        )

        feature_importance = np.abs(model.coef_[0]) if hasattr(model, "coef_") else None

        return HurricanePredictionResults(
            predictions=y_proba,
            labels=y_test,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1=f1,
            auc_roc=auc_roc,
            feature_importance=feature_importance,
            confusion_matrix=cm,
            classification_report=report,
            cross_val_scores=cv_scores,
            model_type="logistic_l2",
            fitted_model=model,
            test_features=X_test_s,
            scaler=self.scaler,
        )

    @staticmethod
    def save_results(
        results: HurricanePredictionResults,
        feature_names: list[str],
        sample_info: list[str],
        save_path: PathLike[str],
    ) -> None:
        """Save results and predictions to JSON and CSV files.

        Args:
            results (`HurricanePredictionResults`): Results to save.
            feature_names (`list[str]`): Names of the features used.
            sample_info (`list[str]`): Sample identifiers.
            save_path (`PathLike[str]`): Path to save the results JSON file.
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        results_dict = {
            "model_type": results.model_type,
            "metrics": {
                "accuracy": results.accuracy,
                "precision": results.precision,
                "recall": results.recall,
                "f1": results.f1,
                "auc_roc": results.auc_roc,
            },
            "cross_validation": {
                "scores": results.cross_val_scores.tolist(),
                "mean": float(results.cross_val_scores.mean()),
                "std": float(results.cross_val_scores.std()),
            },
            "confusion_matrix": results.confusion_matrix.tolist(),
            "classification_report": results.classification_report,
            "feature_importance": (
                dict(
                    zip(feature_names, results.feature_importance.tolist(), strict=True)
                )
                if results.feature_importance is not None
                else None
            ),
        }

        with save_path.open("w", encoding="utf-8") as f:
            json.dump(results_dict, f, indent=2)

        preds_path = save_path.with_suffix(".predictions.csv")
        preds_df = pl.DataFrame(
            {
                "sample_info": sample_info[: len(results.predictions)],
                "true_label": results.labels,
                "predicted_probability": results.predictions,
                "predicted_label": (results.predictions > PREDICTION_THRESHOLD).astype(
                    int
                ),
            }
        )
        preds_df.write_csv(preds_path)

        # Save ROC curve data
        HurricanePredictor.save_roc_curve_data(results, save_path)

        logger.info("Results saved to %s", save_path)

    @staticmethod
    def save_roc_curve_data(
        results: HurricanePredictionResults, save_path: PathLike[str]
    ) -> None:
        """Save ROC curve data (FPR, TPR, thresholds) to CSV.

        Args:
            results (`HurricanePredictionResults`): Results containing predictions
                and labels.
            save_path (`PathLike[str]`): Base path to save the ROC curve CSV file.
        """
        save_path = Path(save_path)

        try:
            # Calculate ROC curve
            fpr, tpr, thresholds = roc_curve(results.labels, results.predictions)

            # Create ROC curve DataFrame
            roc_df = pl.DataFrame(
                {
                    "false_positive_rate": fpr,
                    "true_positive_rate": tpr,
                    "threshold": thresholds,
                    "auc_roc": results.auc_roc,
                }
            )

            # Save ROC curve data
            roc_path = save_path.with_suffix(".roc_curve.csv")
            roc_df.write_csv(roc_path)
            logger.info("ROC curve data saved to %s", roc_path)

        except ValueError as e:
            logger.warning("Could not save ROC curve data: %s", e)


def run_linear_probe_analysis(
    embedding_data: EmbeddingData,
    hurricane_csv_path: PathLike[str],
    output_dir: PathLike[str],
    *,
    lookahead_days: int = 3,
) -> dict[str, HurricanePredictionResults]:
    """Run linear probe analysis using only embeddings generated by frozen vision
    encoder.

    Args:
        embedding_data (`EmbeddingData`): Embedding data with metadata.
        hurricane_csv_path (`PathLike[str]`): Path to the hurricane CSV file.
        output_dir (`PathLike[str]`): Directory to save results.
        lookahead_days (`int`): Days to look ahead for hurricane start.

    Returns:
        `dict[str, HurricanePredictionResults]`: Mapping of feature combination names
            to their respective results.
    """
    # Standard linear probe configuration
    linear_probe_config = LogisticRegressionConfig(
        regularisation_strength=1.0,
        max_iter=1000,
        solver="lbfgs",
        class_weight="balanced",
    )

    # Only use image embeddings
    feature_combinations = [("linear_probe", True, False)]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    predictor = HurricanePredictor(hurricane_csv_path)
    all_results = {}

    for combo_name, use_embeddings, use_weather in feature_combinations:
        logger.info("Running linear probe: %s", combo_name)

        # Prepare dataset with only image embeddings
        features, labels, feature_names, sample_info = (
            predictor.prepare_prediction_dataset(
                embedding_data,
                lookahead_days=lookahead_days,
                use_embeddings=use_embeddings,
                use_weather_features=use_weather,  # This should be False
            )
        )

        if len(features) == 0:
            logger.warning("No features for %s, skipping", combo_name)
            continue

        # Train with standard linear probe config
        results = predictor.train_and_evaluate(
            features, labels, config=linear_probe_config
        )

        all_results[combo_name] = results

        # Save results
        save_path = output_dir / f"{combo_name}_results.json"
        predictor.save_results(results, feature_names, sample_info, save_path)

    return all_results


def filter_embeddings_by_date(
    embedding_data: EmbeddingData, cutoff_date: str = "2025-05-31"
) -> EmbeddingData:
    """Return a copy of EmbeddingData filtered to samples with date before the
    cutoff date.

    Args:
        embedding_data (`EmbeddingData`): Original embedding data.
        cutoff_date (`str`): Cutoff date in 'YYYY-MM-DD' format.

    Returns:
        `EmbeddingData`: Filtered embedding data.
    """
    cutoff = datetime.strptime(cutoff_date, "%Y-%m-%d").replace(tzinfo=UTC)
    valid_idx: list[int] = []
    for i, md in enumerate(embedding_data.metadata):
        ds = md.get("date", "")
        if not ds:
            continue
        try:
            if datetime.strptime(ds, "%Y-%m-%d").replace(tzinfo=UTC) <= cutoff:
                valid_idx.append(i)
        except ValueError:
            continue

    logger.info(
        "Filtering samples",
        original_count=len(embedding_data.metadata),
        filtered_count=len(valid_idx),
        cutoff_date=cutoff_date,
    )

    # Rebuild EmbeddingData with the filtered indices
    return EmbeddingData(
        image_embeddings=embedding_data.image_embeddings[valid_idx],
        text_embeddings=embedding_data.text_embeddings[valid_idx],
        captions=[embedding_data.captions[i] for i in valid_idx],
        metadata=[embedding_data.metadata[i] for i in valid_idx],
        sample_ids=(
            [embedding_data.sample_ids[i] for i in valid_idx]
            if hasattr(embedding_data, "sample_ids")
            else [f"sample_{i}" for i in valid_idx]
        ),
        image_filenames=(
            [embedding_data.image_filenames[i] for i in valid_idx]
            if hasattr(embedding_data, "image_filenames")
            else [f"image_{i}.png" for i in valid_idx]
        ),
    )

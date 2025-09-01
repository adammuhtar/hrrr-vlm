"""Module for hierarchical clustering and dendrogram visualisation of embeddings."""

import textwrap
from datetime import UTC, datetime
from os import PathLike
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from scipy.cluster.hierarchy import cophenet, dendrogram, linkage
from scipy.spatial.distance import pdist
from scipy.stats import pearsonr

from hrrr_vlm.eval.embeddings import EmbeddingData
from hrrr_vlm.utils.logger import configure_logger

# Configure logging
logger = configure_logger()


class DendrogramGenerator:
    """Generator to create dendrograms of embeddings grouped by metadata."""

    def __init__(self, *, figsize: tuple[int, int] = (15, 10), dpi: int = 100) -> None:
        """Initialise the dendrogram generator.

        Args:
            figsize (`tuple[int, int]`): Figure size for plots.
            dpi (`int`): DPI for plot resolution.
        """
        self.figsize = figsize
        self.dpi = dpi

        # Set plotting style
        plt.style.use("seaborn-v0_8")
        sns.set_palette("husl")

    def create_regional_dendrogram(
        self,
        embedding_data: EmbeddingData,
        *,
        embedding_type: Literal["image", "text", "combined"] = "combined",
        aggregation_method: str = "mean",
        linkage_method: str = "ward",
        distance_metric: str = "euclidean",
        save_path: PathLike[str] | None = None,
    ) -> dict[str, Any]:
        """Create dendrogram showing hierarchical relationships between regions.

        Args:
            embedding_data (`EmbeddingData`): Embedding data with regional metadata.
            embedding_type (`str`): Type of embeddings to use ('image', 'text',
                'combined').
            aggregation_method (`str`): How to aggregate embeddings per region
                ('mean', 'median').
            linkage_method (`str`): Linkage method for hierarchical clustering.
            distance_metric (`str`): Distance metric for clustering.
            save_path (`PathLike[str]`, optional): Path to save the plot.

        Returns:
            `dict[str, Any]`: Dictionary containing dendrogram analysis results.
        """
        # Prepare embeddings and extract regional information
        embeddings = self.prepare_embeddings(embedding_data, embedding_type)
        regions = [meta["region"] for meta in embedding_data.metadata]

        # Aggregate embeddings by region
        region_embeddings, region_names = self.aggregate_by_category(
            embeddings, regions, aggregation_method
        )

        # Perform hierarchical clustering
        linkage_matrix = linkage(
            region_embeddings, method=linkage_method, metric=distance_metric
        )

        # Create dendrogram plot
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        dendrogram_result = dendrogram(
            linkage_matrix,
            labels=region_names,
            ax=ax,
            orientation="top",
            distance_sort="descending",
            show_leaf_counts=True,
        )

        ax.set_xlabel("Regions", fontsize=12)
        ax.set_ylabel("Distance", fontsize=12)

        # Set smaller figure size (approximately 1/6 of A4)
        fig.set_size_inches(5.5, 4.5)

        # Wrap long region names and rotate labels for better readability
        wrapped_labels = [
            "\n".join(textwrap.wrap(label, width=16)) for label in region_names
        ]
        ax.set_xticklabels(wrapped_labels, rotation=45, ha="right", fontsize=10)
        ax.tick_params(axis="y", labelsize=10)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            logger.info("Saved regional dendrogram to %s", save_path)

        plt.show()

        # Analyse clustering results
        analysis = self.analyse_dendrogram_clusters(
            linkage_matrix, region_names, "regions", region_embeddings
        )
        analysis.update(
            {
                "embedding_type": embedding_type,
                "aggregation_method": aggregation_method,
                "linkage_method": linkage_method,
                "distance_metric": distance_metric,
                "dendrogram_data": dendrogram_result,
            }
        )

        return analysis

    def create_seasonal_dendrogram(
        self,
        embedding_data: EmbeddingData,
        *,
        embedding_type: Literal["image", "text", "combined"] = "combined",
        aggregation_method: str = "mean",
        linkage_method: str = "ward",
        distance_metric: str = "euclidean",
        save_path: PathLike[str] | None = None,
    ) -> dict[str, Any]:
        """Create dendrogram showing hierarchical relationships between seasons.

        Args:
            embedding_data (`EmbeddingData`): Embedding data with regional metadata.
            embedding_type (`str`): Type of embeddings to use ('image', 'text',
                'combined').
            aggregation_method (`str`): How to aggregate embeddings per region
                ('mean', 'median').
            linkage_method (`str`): Linkage method for hierarchical clustering.
            distance_metric (`str`): Distance metric for clustering.
            save_path (`PathLike[str]`, optional): Path to save the plot.

        Returns:
            `dict[str, Any]`: Dictionary containing dendrogram analysis results
        """
        # Extract seasonal information from captions (parsing seasons from text)
        seasons = self.extract_seasons_from_data(embedding_data)

        # Prepare embeddings
        embeddings = self.prepare_embeddings(embedding_data, embedding_type)

        # Aggregate embeddings by season
        season_embeddings, season_names = self.aggregate_by_category(
            embeddings, seasons, aggregation_method
        )

        # Perform hierarchical clustering
        linkage_matrix = linkage(
            season_embeddings, method=linkage_method, metric=distance_metric
        )

        # Create dendrogram plot
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        dendrogram_result = dendrogram(
            linkage_matrix,
            labels=season_names,
            ax=ax,
            orientation="top",
            distance_sort="descending",
            show_leaf_counts=True,
        )

        ax.set_xlabel("Seasons", fontsize=12)
        ax.set_ylabel("Distance", fontsize=12)

        # Set smaller figure size (approximately 1/6 of A4)
        fig.set_size_inches(5.5, 4.5)

        # Rotate labels for better readability
        ax.set_xticklabels(season_names, rotation=45, ha="right", fontsize=10)
        ax.tick_params(axis="y", labelsize=10)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            logger.info("Saved seasonal dendrogram to %s", save_path)

        plt.show()

        # Analyse clustering results
        analysis = self.analyse_dendrogram_clusters(
            linkage_matrix, season_names, "seasons", season_embeddings
        )
        analysis.update(
            {
                "embedding_type": embedding_type,
                "aggregation_method": aggregation_method,
                "linkage_method": linkage_method,
                "distance_metric": distance_metric,
                "dendrogram_data": dendrogram_result,
            }
        )

        return analysis

    @staticmethod
    def prepare_embeddings(
        embedding_data: EmbeddingData,
        embedding_type: Literal["image", "text", "combined"] = "combined",
    ) -> np.ndarray:
        """Prepare embeddings for clustering based on type.

        Args:
            embedding_data (`EmbeddingData`): Embedding data.
            embedding_type (`str`): Type of embeddings to use ('image', 'text',
                'combined').

        Returns:
            `np.ndarray`: Prepared embeddings array.

        Raises:
            ValueError: If an unsupported embedding type is provided.
        """
        if embedding_type == "image":
            return embedding_data.image_embeddings.numpy()
        if embedding_type == "text":
            return embedding_data.text_embeddings.numpy()
        if embedding_type == "combined":
            # Concatenate image and text embeddings
            img_emb = embedding_data.image_embeddings.numpy()
            txt_emb = embedding_data.text_embeddings.numpy()
            return np.concatenate([img_emb, txt_emb], axis=1)
        msg = f"Unsupported embedding type: {embedding_type}"
        raise ValueError(msg)

    @staticmethod
    def extract_seasons_from_data(embedding_data: EmbeddingData) -> list[str]:
        """Extract seasonal information from captions and metadata.

        Args:
            embedding_data (`EmbeddingData`): Embedding data with metadata.

        Returns:
            `list[str]`: List of seasons corresponding to each embedding.
        """
        seasons = []
        for i, meta in enumerate(embedding_data.metadata):  # noqa: PLR1702
            # First try to get season from metadata
            if "season" in meta and meta["season"] != "Unknown":
                seasons.append(meta["season"])
            else:
                # Extract from caption text
                caption = embedding_data.captions[i].lower()
                if any(term in caption for term in ["winter", "cold", "snow"]):
                    seasons.append("winter")
                elif any(term in caption for term in ["spring", "mild"]):
                    seasons.append("spring")
                elif any(term in caption for term in ["summer", "hot"]):
                    seasons.append("summer")
                elif any(term in caption for term in ["autumn", "fall"]):
                    seasons.append("autumn")
                else:
                    # Parse from date if available
                    date_str = meta.get("date", "")
                    if date_str:
                        try:
                            date = datetime.strptime(date_str, "%Y-%m-%d").replace(
                                tzinfo=UTC
                            )
                            month = date.month
                            if month in {12, 1, 2}:
                                seasons.append("winter")
                            elif month in {3, 4, 5}:
                                seasons.append("spring")
                            elif month in {6, 7, 8}:
                                seasons.append("summer")
                            elif month in {9, 10, 11}:
                                seasons.append("autumn")
                        except ValueError:
                            seasons.append("unknown")
                    else:
                        seasons.append("unknown")
        return seasons

    @staticmethod
    def aggregate_by_category(
        embeddings: np.ndarray, categories: list[str], method: str
    ) -> tuple[np.ndarray, list[str]]:
        """Aggregate embeddings by category using specified method.

        Args:
            embeddings (`np.ndarray`): Embeddings array.
            categories (`list[str]`): List of categories corresponding to embeddings.
            method (`str`): Aggregation method ('mean', 'median').

        Returns:
            `tuple[np.ndarray, list[str]]`: Tuple of aggregated embeddings and
                category names.

        Raises:
            ValueError: If an unsupported aggregation method is provided.
        """
        df = pl.DataFrame(embeddings)
        df = df.with_columns(pl.Series("category", categories))

        if method == "mean":
            aggregated = df.group_by("category").mean()
        elif method == "median":
            aggregated = df.group_by("category").median()
        else:
            msg = f"Unsupported aggregation method: {method}"
            raise ValueError(msg)

        category_names = aggregated["category"].to_list()
        category_embeddings = aggregated.drop("category").to_numpy()

        return category_embeddings, category_names

    @staticmethod
    def analyse_dendrogram_clusters(
        linkage_matrix: np.ndarray,
        labels: list[str],
        category_type: str,
        original_data: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Analyse the hierarchical clustering results.

        Args:
            linkage_matrix (`np.ndarray`): Linkage matrix from clustering.
            labels (`list[str]`): Labels for each cluster.
            category_type (`str`): Type of categories (e.g., 'regions', 'seasons').
            original_data (`np.ndarray`, optional): Original data used for
                clustering, for cophenetic correlation calculation.

        Returns:
            `dict[str, Any]`: Dictionary containing analysis results.
        """
        n_clusters = len(labels)

        # Calculate cophenetic correlation if original data is provided
        cophenetic_corr = None
        if original_data is not None:
            try:
                # Calculate original pairwise distances
                original_distances = pdist(original_data)

                # Calculate cophenetic distances from linkage matrix
                cophenetic_distances, _ = cophenet(linkage_matrix, original_distances)

                # Calculate correlation
                correlation, _ = pearsonr(original_distances, cophenetic_distances)
                cophenetic_corr = correlation
            except Exception:
                # If cophenetic calculation fails, skip it
                cophenetic_corr = None

        return {
            "n_categories": n_clusters,
            "category_type": category_type,
            "categories": labels,
            "cophenetic_correlation": cophenetic_corr,
            "max_distance": linkage_matrix[:, 2].max(),
            "min_distance": linkage_matrix[:, 2].min(),
            "distance_range": linkage_matrix[:, 2].max() - linkage_matrix[:, 2].min(),
        }

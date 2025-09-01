"""Module to evaluate SigLIP VLMs on NOAA HRRR image-text tasks."""

from .dendrograms import DendrogramGenerator
from .embeddings import EmbeddingData, EmbeddingExtractor
from .exceptions import RetrievalEvaluationError
from .linear_probe import (
    PREDICTION_THRESHOLD,
    HurricaneDataProcessor,
    HurricanePredictionResults,
    HurricanePredictor,
    filter_embeddings_by_date,
    run_linear_probe_analysis,
)
from .retrieval import RetrievalEvaluator
from .visualisation import ClusteringResults, EmbeddingClusterer

__all__ = [
    "PREDICTION_THRESHOLD",
    "ClusteringResults",
    "DendrogramGenerator",
    "EmbeddingClusterer",
    "EmbeddingData",
    "EmbeddingExtractor",
    "HurricaneDataProcessor",
    "HurricanePredictionResults",
    "HurricanePredictor",
    "RetrievalEvaluationError",
    "RetrievalEvaluator",
    "filter_embeddings_by_date",
    "run_linear_probe_analysis",
]

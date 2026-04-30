"""Prediction service — orchestrates pre-processing, inference, and post-processing."""

from __future__ import annotations

import numpy as np

from app.core.config import settings
from app.models.mood_model import MoodModel
from app.schemas.prediction import MoodScore, PredictionResponse


class PredictionService:
    """Thin service layer that turns raw text into a PredictionResponse."""

    def __init__(self, model: MoodModel | None = None) -> None:
        self._model = model or MoodModel.get_instance()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def predict(self, text: str) -> PredictionResponse:
        inputs = self._preprocess(text)
        raw_scores: np.ndarray = self._model.predict(inputs)
        return self._postprocess(raw_scores)

    # ------------------------------------------------------------------
    # Internal helpers (override in subclasses for custom pipelines)
    # ------------------------------------------------------------------
    def _preprocess(self, text: str) -> np.ndarray:
        """Convert *text* to the numpy array expected by the model.

        Replace this stub with the real tokenisation / feature extraction
        logic once the model artefact is available.
        """
        # Placeholder: return a zero vector of shape (1, 128)
        return np.zeros((1, 128), dtype=np.float32)

    def _postprocess(self, raw_scores: np.ndarray) -> PredictionResponse:
        labels = settings.MOOD_LABELS
        probs: list[float] = raw_scores[0].tolist()

        scores = [MoodScore(label=lbl, score=float(p)) for lbl, p in zip(labels, probs)]
        best_idx: int = int(np.argmax(probs))

        return PredictionResponse(
            predicted_mood=labels[best_idx],
            confidence=float(probs[best_idx]),
            scores=scores,
        )

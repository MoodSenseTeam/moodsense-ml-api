"""Model wrapper and prediction orchestration."""

from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Any

import numpy as np

from app.config import settings


def _clean_text(text: str) -> str:
    """Replicate training-time text cleaning."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+|#\w+", "", text)
    text = re.sub(r"\[username\]|\[url\]|\[sensitive-no\]|\bUSER\b|\bURL\b", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
from app.schemas import MoodScore, PredictionResponse


class MoodModel:
    """Lazy-loaded singleton around a saved TensorFlow/Keras model."""

    _instance: MoodModel | None = None

    def __init__(self) -> None:
        self._model: Any = None
        self._tokenizer: Any = None

    @classmethod
    def get_instance(cls) -> MoodModel:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load(
        self,
        model_path: str | None = None,
        tokenizer_path: str | None = None,
    ) -> None:
        import tensorflow as tf

        path = model_path or settings.MODEL_PATH
        self._model = tf.keras.models.load_model(path)

        tok_path = tokenizer_path or settings.TOKENIZER_PATH
        with open(tok_path, "rb") as f:
            self._tokenizer = pickle.load(f)  # noqa: S301

    @property
    def is_loaded(self) -> bool:
        return self._model is not None and self._tokenizer is not None

    @property
    def tokenizer(self) -> Any:
        return self._tokenizer

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        if not self.is_loaded:
            raise RuntimeError("Model is not loaded. Call MoodModel.load() first.")
        return self._model.predict(inputs)


class PredictionService:
    """Orchestrates preprocess → model inference → postprocess."""

    def __init__(self, model: MoodModel | None = None) -> None:
        self._model: MoodModel = model or MoodModel.get_instance()

    def predict(self, text: str) -> PredictionResponse:
        inputs = self._preprocess(text)
        raw_scores: np.ndarray = self._model.predict(inputs)
        return self._postprocess(raw_scores)

    def _preprocess(self, text: str) -> np.ndarray:
        from tensorflow.keras.preprocessing.sequence import pad_sequences

        cleaned = _clean_text(text)
        sequences = self._model.tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(sequences, maxlen=settings.MAX_LEN, padding="post")
        return np.array(padded, dtype=np.int32)

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

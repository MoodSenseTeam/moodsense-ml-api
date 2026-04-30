"""TensorFlow/Keras mood classification model wrapper."""

from __future__ import annotations

import numpy as np

from app.core.config import settings


class MoodModel:
    """Lazy-loaded wrapper around a saved TensorFlow/Keras model."""

    _instance: MoodModel | None = None

    def __init__(self) -> None:
        self._model = None

    # ------------------------------------------------------------------
    # Singleton helper (optional — services can also instantiate directly)
    # ------------------------------------------------------------------
    @classmethod
    def get_instance(cls) -> MoodModel:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def load(self, model_path: str | None = None) -> None:
        """Load the saved Keras model from *model_path* (or settings)."""
        import tensorflow as tf  # noqa: PLC0415

        path = model_path or settings.MODEL_PATH
        self._model = tf.keras.models.load_model(path)

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """Return raw softmax probabilities for *inputs*."""
        if not self.is_loaded:
            raise RuntimeError("Model is not loaded. Call MoodModel.load() first.")
        return self._model.predict(inputs)

"""Model wrapper and prediction orchestration."""

from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import joblib

from app.config import settings
from app.schemas import MoodScore, PredictionResponse


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


class MoodModel:
    """Lazy-loaded singleton around a saved TensorFlow/Keras model and survey-based models."""

    _instance: MoodModel | None = None

    def __init__(self) -> None:
        self._model: Any = None
        self._tokenizer: Any = None
        self._survey_model: Any = None
        self._scaler: Any = None
        self._imputer: Any = None

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

        # Load Text Model
        path = model_path or settings.MODEL_PATH
        self._model = tf.keras.models.load_model(path)

        # Load Tokenizer
        tok_path = tokenizer_path or settings.TOKENIZER_PATH
        with open(tok_path, "rb") as f:
            self._tokenizer = pickle.load(f)  # noqa: S301

        # Load Survey Model Preprocessors
        self._scaler = joblib.load(settings.SCALER_PATH)
        self._imputer = joblib.load(settings.IMPUTER_PATH)

        # Load XGBoost or RandomForest Survey Model
        survey_path = (
            settings.XGB_MODEL_PATH
            if settings.SURVEY_MODEL_TYPE == "xgb"
            else settings.RF_MODEL_PATH
        )
        self._survey_model = joblib.load(survey_path)

    @property
    def is_loaded(self) -> bool:
        return (
            self._model is not None
            and self._tokenizer is not None
            and self._survey_model is not None
            and self._scaler is not None
            and self._imputer is not None
        )

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

    def predict(
        self,
        text: str,
        sleep_hours: float | None = None,
        activity_level: str | None = None,
        how_you_feeling: str | None = None,
    ) -> PredictionResponse:
        if not self._model.is_loaded:
            raise RuntimeError("Model is not loaded. Call MoodModel.load() first.")

        # Run text-based model
        text_inputs = self._preprocess(text)
        probs_text: np.ndarray = self._model.predict(text_inputs)[0]

        # Check if survey features are present
        has_survey = (
            sleep_hours is not None
            or activity_level is not None
            or how_you_feeling is not None
        )

        if has_survey:
            # Run survey-based model
            survey_inputs = self._preprocess_survey(sleep_hours, activity_level, how_you_feeling)
            probs_survey: np.ndarray = self._model._survey_model.predict_proba(survey_inputs)[0]

            # Align survey output labels (0: Low/Happy, 1: Moderate/Normal, 2: High/Stress)
            # to settings.MOOD_LABELS which is ["stress", "happy", "normal"]
            # Class 0 (Low) maps to index 1 ("happy")
            # Class 1 (Moderate) maps to index 2 ("normal")
            # Class 2 (High) maps to index 0 ("stress")
            probs_survey_mapped = np.array([
                probs_survey[2],  # stress
                probs_survey[0],  # happy
                probs_survey[1],  # normal
            ])

            # Late Fusion weighting
            w = settings.LATE_FUSION_WEIGHT
            probs_combined = w * probs_text + (1 - w) * probs_survey_mapped
        else:
            probs_combined = probs_text

        labels = settings.MOOD_LABELS
        scores = [MoodScore(label=lbl, score=float(p)) for lbl, p in zip(labels, probs_combined)]
        best_idx: int = int(np.argmax(probs_combined))

        return PredictionResponse(
            predicted_mood=labels[best_idx],
            confidence=float(probs_combined[best_idx]),
            scores=scores,
        )

    def _preprocess(self, text: str) -> np.ndarray:
        from tensorflow.keras.preprocessing.sequence import pad_sequences

        cleaned = _clean_text(text)
        sequences = self._model.tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(sequences, maxlen=settings.MAX_LEN, padding="post")
        return np.array(padded, dtype=np.int32)

    def _preprocess_survey(
        self,
        sleep_hours: float | None,
        activity_level: str | None,
        how_you_feeling: str | None,
    ) -> np.ndarray:
        # 1. Map sleep hours
        sleep_cat = "unknown"
        if sleep_hours is not None:
            if sleep_hours < 3.0:
                sleep_cat = "less than 3 hours"
            elif sleep_hours < 5.0:
                sleep_cat = "3-5 hours"
            elif sleep_hours < 7.0:
                sleep_cat = "5-7 hours"
            elif sleep_hours < 9.0:
                sleep_cat = "7-9 hours"
            else:
                sleep_cat = "more than 9 hours"

        # 2. Map health (using how_you_feeling as overall health mapping)
        health_map = {
            "very_happy": 4.0,  # "very high" -> 4
            "happy": 3.0,      # "high" -> 3
            "normal": 2.0,     # "moderate" -> 2
            "stress": 1.0,     # "low" -> 1
            "very_stress": 0.0 # "very low" -> 0
        }
        health_val = 2.0  # default to "moderate"
        if how_you_feeling is not None:
            feeling_lower = how_you_feeling.lower().strip()
            health_val = health_map.get(feeling_lower, 2.0)

        # 3. Map stress level for training leakage compatibility
        stress_lvl_cat = "moderate"
        if how_you_feeling is not None:
            feeling_lower = how_you_feeling.lower().strip()
            if feeling_lower in ["very_happy", "happy"]:
                stress_lvl_cat = "very low"
            elif feeling_lower == "normal":
                stress_lvl_cat = "moderate"
            elif feeling_lower in ["stress", "very_stress"]:
                stress_lvl_cat = "very high"

        # 4. Map extracurriculars
        active_cat = "no"
        if activity_level is not None:
            active_lower = activity_level.lower().strip()
            if active_lower in ["low", "moderate", "high"]:
                active_cat = "yes"

        # 5. Populate dummy inputs matching imputer's features
        input_data = {
            'How Would You Describe The Quality Of Your Sleep On Most Nights': np.nan, # skipped during fit
            'How Would You Rate Your Overall Health Currently': health_val,
            'Smartphone': 1.0, # default to "yes" -> 1.0
            'Laptop': 1.0,     # default to "yes" -> 1.0
            'Desktop': 0.0,    # default to "no" -> 0.0

            # One-hot encoded Sleep Hours
            'On Average How Many Hours Of Sleep Do You Get On A Typical Night During The School Week_3-5 hours, less than 3 hours': 1.0 if sleep_cat == "3-5 hours" else 0.0,
            'On Average How Many Hours Of Sleep Do You Get On A Typical Night During The School Week_5-7 hours': 1.0 if sleep_cat == "5-7 hours" else 0.0,
            'On Average How Many Hours Of Sleep Do You Get On A Typical Night During The School Week_7-9 hours': 1.0 if sleep_cat == "7-9 hours" else 0.0,
            'On Average How Many Hours Of Sleep Do You Get On A Typical Night During The School Week_less than 3 hours': 1.0 if sleep_cat == "less than 3 hours" else 0.0,
            "On Average How Many Hours Of Sleep Do You Get On A Typical Night During The School Week_moderately - there's some impact, but it's not dramatic.": 0.0,
            'On Average How Many Hours Of Sleep Do You Get On A Typical Night During The School Week_more than 9 hours': 1.0 if sleep_cat == "more than 9 hours" else 0.0,
            'On Average How Many Hours Of Sleep Do You Get On A Typical Night During The School Week_not at all - my diet does not affect my academics.': 0.0,
            'On Average How Many Hours Of Sleep Do You Get On A Typical Night During The School Week_significantly - a good diet boosts my academic performance.': 0.0,
            "On Average How Many Hours Of Sleep Do You Get On A Typical Night During The School Week_slightly - there's a slight impact, but it's hardly noticeable.": 0.0,
            'On Average How Many Hours Of Sleep Do You Get On A Typical Night During The School Week_unknown': 1.0 if sleep_cat == "unknown" else 0.0,

            # One-hot encoded Extracurriculars
            'Are You Involved In Any Extracurricular Activities_music or arts': 0.0,
            'Are You Involved In Any Extracurricular Activities_no (go to 3)': 1.0 if active_cat == "no" else 0.0,
            'Are You Involved In Any Extracurricular Activities_other (please specify)': 0.0,
            'Are You Involved In Any Extracurricular Activities_sports': 0.0,
            'Are You Involved In Any Extracurricular Activities_student government': 0.0,
            'Are You Involved In Any Extracurricular Activities_unknown': 0.0,
            'Are You Involved In Any Extracurricular Activities_volunteer work': 0.0,
            'Are You Involved In Any Extracurricular Activities_yes (go to 2.10)': 1.0 if active_cat == "yes" else 0.0,
            'Are You Involved In Any Extracurricular Activities_yes (go to 2.10), no (go to 3)': 0.0,

            # One-hot encoded Stress Level (target leakage compatibility)
            'How Would You Rate Your Stress Level During This Academic Term_low': 1.0 if stress_lvl_cat == "low" else 0.0,
            'How Would You Rate Your Stress Level During This Academic Term_moderate': 1.0 if stress_lvl_cat == "moderate" else 0.0,
            'How Would You Rate Your Stress Level During This Academic Term_unknown': 1.0 if stress_lvl_cat == "unknown" else 0.0,
            'How Would You Rate Your Stress Level During This Academic Term_very high': 1.0 if stress_lvl_cat == "very high" else 0.0,
            'How Would You Rate Your Stress Level During This Academic Term_very low': 1.0 if stress_lvl_cat == "very low" else 0.0,
        }

        # Align with the 29 features of the imputer
        df_in = pd.DataFrame([input_data])
        X_imp = self._model._imputer.transform(df_in)
        X_scaled = self._model._scaler.transform(X_imp)
        return X_scaled

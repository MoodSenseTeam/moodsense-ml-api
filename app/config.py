"""Application settings loaded from environment variables."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

_MODELS_DIR = Path(__file__).resolve().parent / "ml-models" / "sentiment-analysis"
_SURVEY_DIR = Path(__file__).resolve().parent / "ml-models" / "mood-prediction"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    PROJECT_NAME: str = "MoodSense ML API"
    VERSION: str = "0.1.0"

    # Sentiment Model settings
    MODEL_PATH: str = str(_MODELS_DIR / "bilstm_mhsa.keras")
    TOKENIZER_PATH: str = str(_MODELS_DIR / "tokenizer.pkl")
    METADATA_PATH: str = str(_MODELS_DIR / "bilstm_mhsa_metadata.json")
    MAX_LEN: int = 64

    # Survey Model settings
    SURVEY_MODEL_TYPE: str = "xgb"  # "xgb" or "rf"
    XGB_MODEL_PATH: str = str(_SURVEY_DIR / "xgb_focus_model.joblib")
    RF_MODEL_PATH: str = str(_SURVEY_DIR / "rf_focus_model.joblib")
    SCALER_PATH: str = str(_SURVEY_DIR / "scaler.joblib")
    IMPUTER_PATH: str = str(_SURVEY_DIR / "imputer.joblib")
    LATE_FUSION_WEIGHT: float = 0.5  # Weight of the text-based model prediction (0.0 to 1.0)

    MOOD_LABELS: list[str] = [
        "stress",
        "happy",
        "normal",
    ]


settings = Settings()

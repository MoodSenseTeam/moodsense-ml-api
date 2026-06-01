"""Application settings loaded from environment variables."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

_MODELS_DIR = Path(__file__).resolve().parent / "ml-models" / "sentiment-analysis"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    PROJECT_NAME: str = "MoodSense ML API"
    VERSION: str = "0.1.0"

    MODEL_PATH: str = str(_MODELS_DIR / "bilstm_mhsa.keras")
    TOKENIZER_PATH: str = str(_MODELS_DIR / "tokenizer.pkl")
    METADATA_PATH: str = str(_MODELS_DIR / "bilstm_mhsa_metadata.json")
    MAX_LEN: int = 64

    MOOD_LABELS: list[str] = [
        "stress",
        "happy",
        "normal",
    ]


settings = Settings()

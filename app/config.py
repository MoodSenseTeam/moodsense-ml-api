"""Application settings loaded from environment variables."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    PROJECT_NAME: str = "MoodSense ML API"
    VERSION: str = "0.1.0"

    MODEL_PATH: str = "app/models/mood_model"

    MOOD_LABELS: list[str] = [
        "angry",
        "disgust",
        "fear",
        "happy",
        "neutral",
        "sad",
        "surprise",
    ]


settings = Settings()

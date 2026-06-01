"""Pydantic schemas for the prediction endpoint."""

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Input text to classify.")
    sleep_hours: float | None = Field(None, ge=0.0, le=24.0, description="Sleep duration in hours.")
    activity_level: str | None = Field(None, description="Activity level: NONE, LOW, MODERATE, HIGH")
    how_you_feeling: str | None = Field(None, description="Current feeling state: VERY_HAPPY, HAPPY, NORMAL, STRESS, VERY_STRESS")


class MoodScore(BaseModel):
    label: str
    score: float = Field(..., ge=0.0, le=1.0)


class PredictionResponse(BaseModel):
    predicted_mood: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    scores: list[MoodScore]

"""Pydantic schemas for the prediction endpoint."""

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Input text to classify.")


class MoodScore(BaseModel):
    label: str
    score: float = Field(..., ge=0.0, le=1.0)


class PredictionResponse(BaseModel):
    predicted_mood: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    scores: list[MoodScore]

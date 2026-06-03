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

class InsightRequest(BaseModel):
    sleep_hours: float = Field(..., ge=0.0, le=24.0, description="Sleep duration in hours.")
    activity_level: str = Field(..., description="Activity level: NONE, LOW, MODERATE, HIGH")
    study_hours: float = Field(..., ge=0.0, le=24.0, description="Study duration in hours.")
    social_score: int = Field(..., ge=0, le=10, description="Social interaction score out of 10.")
    how_you_feeling: str = Field(..., description="Current feeling state: VERY_HAPPY, HAPPY, NORMAL, STRESS, VERY_STRESS")
    notes: str | None = Field(None, description="Additional notes from the user.")

class Recommendation(BaseModel):
    name: str = Field(..., description="Name of the recommendation.")
    description: str = Field(..., description="Description of the recommendation.")
    duration: str = Field(..., description="Duration it takes (e.g., '10 minutes', '30 minutes').")


class InsightResponse(BaseModel):
    insight: str = Field(..., description="Insight about the user's well-being.")
    recommendations: list[Recommendation] = Field(..., description="Empathetic, actionable recommendations for the user.")



class FactorsRequest(BaseModel):
    sleep_hours: float = Field(..., ge=0.0, le=24.0, description="Sleep duration in hours.")
    activity_level: str = Field(..., description="Activity level: NONE, LOW, MODERATE, HIGH")
    study_hours: float = Field(..., ge=0.0, le=24.0, description="Study duration in hours.")
    social_score: int = Field(..., ge=0, le=10, description="Social interaction score out of 10.")
    how_you_feeling: str = Field(..., description="Current feeling state: VERY_HAPPY, HAPPY, NORMAL, STRESS, VERY_STRESS")
    notes: str | None = Field(None, description="Additional notes from the user.")


class Factor(BaseModel):
    name: str = Field(..., description="Nama faktor (contoh: 'Tidur', 'Belajar', 'Sosialisasi', 'Aktivitas Fisik', 'Catatan').")
    value: str = Field(..., description="Nilai atau deskripsi nilai faktor tersebut (contoh: '4.0 jam', 'Rendah', '2.0 jam', '8/10', dll).")
    description: str = Field(..., description="Penjelasan mengapa faktor ini menjadi pemicu stres (stressor) atau pendukung kebahagiaan (booster) dalam Bahasa Indonesia.")


class FactorsResponse(BaseModel):
    stressors: list[Factor] = Field(..., description="Faktor-faktor yang memicu stres atau kelelahan.")
    boosters: list[Factor] = Field(..., description="Faktor-faktor yang meningkatkan kebahagiaan atau menjaga keseimbangan.")
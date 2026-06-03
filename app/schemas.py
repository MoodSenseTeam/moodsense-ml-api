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


class TrendPoint(BaseModel):
    date: str = Field(..., description="Tanggal dalam format YYYY-MM-DD.")
    average_mood: float = Field(..., ge=0.0, le=10.0, description="Rata-rata mood di hari tersebut (0-10).")


class ForecastRequest(BaseModel):
    weekly_trend: list[TrendPoint] = Field(..., min_length=1, description="Data tren mood 7 hari terakhir.")
    average_mood: float = Field(..., ge=0.0, le=10.0, description="Rata-rata mood keseluruhan (0-10).")
    sleep_quality: float = Field(..., ge=0.0, le=10.0, description="Skor kualitas tidur (0-10).")
    check_in_streak: int = Field(..., ge=0, description="Streak check-in berapa hari berturut-turut.")
    latest_mood: str | None = Field(None, description="Perasaan terakhir: VERY_HAPPY, HAPPY, NORMAL, STRESS, VERY_STRESS.")


class DailyForecast(BaseModel):
    day: str = Field(..., description="Label hari (contoh: 'Besok', 'Lusa', '3 Hari Lagi').")
    date: str = Field(..., description="Tanggal dalam format YYYY-MM-DD.")
    predicted_mood: float = Field(..., ge=0.0, le=10.0, description="Prediksi skor mood (0-10).")
    label: str = Field(..., description="Label mood (contoh: 'Baik', 'Biasa', 'Kurang').")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Tingkat keyakinan prediksi (0-1).")


class ForecastResponse(BaseModel):
    forecasts: list[DailyForecast] = Field(..., description="Proyeksi mood 3-5 hari ke depan.")
    trend_direction: str = Field(..., description="Arah tren: 'meningkat', 'menurun', atau 'stabil'.")
    trend_analysis: str = Field(..., description="Analisis singkat (1-2 kalimat) tentang tren mood.")
    prevention_tips: list[str] = Field(default_factory=list, description="Tips pencegahan jika tren menurun.")
    boost_tips: list[str] = Field(default_factory=list, description="Tips mempertahankan jika tren baik.")
"""Prediction API route."""

from fastapi import APIRouter, Depends, HTTPException
from app.genai.inference import get_ai_insight, get_mood_forecast, get_stress_happiness_factors
from app.inference import PredictionService
from app.schemas import (
    PredictionRequest,
    PredictionResponse,
    ForecastRequest,
    ForecastResponse,
    InsightRequest,
    InsightResponse,
    FactorsRequest,
    FactorsResponse,
)

router = APIRouter(tags=["prediction"])


def get_prediction_service() -> PredictionService:
    return PredictionService()


@router.post("/predict", response_model=PredictionResponse, tags=["prediction"])
def predict(
    request: PredictionRequest,
    service: PredictionService = Depends(get_prediction_service),
) -> PredictionResponse:
    """Classify the mood expressed in *request.text* using optional late fusion of survey features."""
    try:
        return service.predict(
            text=request.text,
            sleep_hours=request.sleep_hours,
            activity_level=request.activity_level,
            how_you_feeling=request.how_you_feeling,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

@router.post("/insight", response_model=InsightResponse, tags=["prediction"])
def get_insight(
    request: InsightRequest,
) -> InsightResponse:
    """Get insight about the user's well-being based on their daily check-in data."""
    try:
        return get_ai_insight(
            sleep_hours=request.sleep_hours,
            activity_level=request.activity_level,
            study_hours=request.study_hours,
            social_score=request.social_score,
            how_you_feeling=request.how_you_feeling,
            notes=request.notes,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/factors", response_model=FactorsResponse, tags=["prediction"])
def extract_factors(
    request: FactorsRequest,
) -> FactorsResponse:
    """Extract and analyze stress/happiness factors from the user's daily check-in data."""
    try:
        return get_stress_happiness_factors(
            sleep_hours=request.sleep_hours,
            activity_level=request.activity_level,
            study_hours=request.study_hours,
            social_score=request.social_score,
            how_you_feeling=request.how_you_feeling,
            notes=request.notes,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/forecast", response_model=ForecastResponse, tags=["prediction"])
def forecast_mood(
    request: ForecastRequest,
) -> ForecastResponse:
    """Proyeksikan mood 5 hari ke depan berdasarkan data tren historis menggunakan Gemini."""
    try:
        return get_mood_forecast(request)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
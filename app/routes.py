"""Prediction API route."""

from fastapi import APIRouter, Depends, HTTPException
from app.genai.inference import get_ai_insight
from app.inference import PredictionService
from app.schemas import PredictionRequest, PredictionResponse, InsightRequest, InsightResponse

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
        insight_text = get_ai_insight(
            sleep_hours=request.sleep_hours,
            activity_level=request.activity_level,
            study_hours=request.study_hours,
            social_score=request.social_score,
            how_you_feeling=request.how_you_feeling,
            notes=request.notes,
        )
        return InsightResponse(insight=insight_text)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
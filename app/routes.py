"""Prediction API route."""

from fastapi import APIRouter, Depends, HTTPException

from app.inference import PredictionService
from app.schemas import PredictionRequest, PredictionResponse

router = APIRouter(tags=["prediction"])


def get_prediction_service() -> PredictionService:
    return PredictionService()


@router.post("/predict", response_model=PredictionResponse)
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

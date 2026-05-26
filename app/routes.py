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
    """Classify the mood expressed in *request.text*."""
    try:
        return service.predict(request.text)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

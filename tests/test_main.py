"""Tests for the health-check and prediction endpoints."""

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.routes import get_prediction_service
from app.schemas import MoodScore, PredictionResponse

client = TestClient(app)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# Prediction endpoint (model not loaded — expects 503)
# ---------------------------------------------------------------------------

def test_predict_without_model_returns_503():
    response = client.post("/api/v1/predict", json={"text": "I feel great today!"})
    assert response.status_code == 503


# ---------------------------------------------------------------------------
# Prediction endpoint (service stubbed via dependency override)
# ---------------------------------------------------------------------------

class _StubPredictionService:
    """Stub that bypasses TensorFlow and returns a fixed response."""

    def predict(
        self,
        text: str,
        sleep_hours: float | None = None,
        activity_level: str | None = None,
        how_you_feeling: str | None = None,
    ) -> PredictionResponse:
        from app.config import settings

        labels = settings.MOOD_LABELS
        uniform = 1.0 / len(labels)
        scores = [MoodScore(label=lbl, score=uniform) for lbl in labels]
        return PredictionResponse(
            predicted_mood=labels[0],
            confidence=uniform,
            scores=scores,
        )


@pytest.fixture(autouse=False)
def override_service():
    app.dependency_overrides[get_prediction_service] = lambda: _StubPredictionService()
    yield
    app.dependency_overrides.clear()


def test_predict_with_stubbed_service(override_service):
    response = client.post("/api/v1/predict", json={"text": "Hello world"})
    assert response.status_code == 200
    data = response.json()
    assert "predicted_mood" in data
    assert "confidence" in data
    assert len(data["scores"]) == 3


# ---------------------------------------------------------------------------
# Insight endpoint (mocked get_ai_insight)
# ---------------------------------------------------------------------------

from unittest.mock import patch

def test_get_insight_success():
    with patch("app.routes.get_ai_insight") as mock_get_insight:
        mock_get_insight.return_value = "Ini adalah insight uji coba."
        response = client.post(
            "/api/v1/insight",
            json={
                "sleep_hours": 7.0,
                "activity_level": "moderate",
                "study_hours": 5.0,
                "social_score": 8,
                "how_you_feeling": "normal",
                "notes": "Belajar cukup melelahkan hari ini."
            }
        )
        assert response.status_code == 200
        assert response.json() == {"insight": "Ini adalah insight uji coba."}
        mock_get_insight.assert_called_once_with(
            sleep_hours=7.0,
            activity_level="moderate",
            study_hours=5.0,
            social_score=8,
            how_you_feeling="normal",
            notes="Belajar cukup melelahkan hari ini."
        )


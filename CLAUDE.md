# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```
# Install deps & venv
uv sync

# Dev server (port 8000)
uv run uvicorn app.main:app --reload

# Tests
uv run pytest                           # all tests
uv run pytest tests/test_main.py -v     # single file, verbose

# Docker
docker build -t moodsense-ml-api .
docker run -p 8000:8000 moodsense-ml-api
```

## Architecture

This is an **inference-only** FastAPI + TensorFlow API that classifies text into 7 moods (angry, disgust, fear, happy, neutral, sad, surprise). No training happens here — a pre-trained `SavedModel` is loaded and served.

**Request flow:** `POST /api/v1/predict` → `PredictionRequest` (Pydantic) → `PredictionService.predict()` → `_preprocess` → `MoodModel.predict()` → `_postprocess` → `PredictionResponse`

**Key files:**

| File | Role |
|------|------|
| `app/main.py` | FastAPI app factory, router registration, `/health` |
| `app/routes.py` | `POST /api/v1/predict` route, 503 on model-not-loaded |
| `app/inference.py` | `MoodModel` (lazy singleton wrapping `tf.keras.models.load_model()`) + `PredictionService` (preprocess → inference → postprocess) |
| `app/schemas.py` | `PredictionRequest` (text), `PredictionResponse` (predicted_mood, confidence, scores) |
| `app/config.py` | `pydantic-settings` — `MODEL_PATH`, `MOOD_LABELS` from env/`.env` |

**Important details:**
- `PredictionService._preprocess()` is a **stub** returning `np.zeros((1, 128))`. Replace with real tokenization matching the model.
- `MoodModel` is a singleton; `PredictionService` accepts an optional model instance for testability.
- Tests use FastAPI `dependency_overrides` to swap in a stub service — no TensorFlow needed in tests.
- Python 3.12 required (`.python-version`).
- Model artifact goes at `app/models/mood_model/` (TensorFlow `SavedModel` directory).

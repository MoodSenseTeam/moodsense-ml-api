# MoodSense ML API — Release Plan

Inference-only FastAPI + TensorFlow API for 7-mood text classification: *angry, disgust, fear, happy, neutral, sad, surprise*.

---

## 1. Model & Preprocessing

- [ ] Place the trained `SavedModel` at `app/models/mood_model/`.
- [ ] Replace `PredictionService._preprocess()` stub with real tokenization matching the model.
- [ ] Ship tokenizer artifact alongside the model (`tokenizer.json` or `tokenizer.pkl`).
- [ ] Verify model input shape matches `_preprocess` output and output order matches `MOOD_LABELS`.

## 2. Code Cleanup

- [ ] Populate `pyproject.toml` dependencies (fastapi, uvicorn, tensorflow, pydantic-settings, numpy).
- [ ] Remove or fix root `main.py` — `python main.py` should start the server.
- [ ] Add `.env.example` with documented defaults (`MODEL_PATH`, `MOOD_LABELS`).
- [ ] Add `.dockerignore` (`.venv/`, `__pycache__/`, `.git/`, `tests/`, `.env`).

## 3. API Hardening

- [ ] Add `max_length` to `PredictionRequest.text` (e.g. 512 chars) to prevent OOM.
- [ ] Eager-load model at startup so the pod fails fast if the artifact is missing.
- [ ] Add request logging middleware (duration, status code).
- [ ] Consider single worker or inference lock — TF models aren't thread-safe for GPU.

## 4. Testing

- [ ] Unit test `_preprocess` with known input → expected output shape.
- [ ] Unit test `_postprocess` with mock softmax → correct top label.
- [ ] Integration test: load real model, `POST /predict`, verify 200 + valid response.
- [ ] Edge cases: empty text (422), very long text, non-ASCII/emoji input.

## 5. Containerization

- [ ] Dockerfile: `python:3.12-slim`, `PYTHONUNBUFFERED=1`, non-root user.
- [ ] Small model (<100 MB): bake into image. Large model: download from GCS/S3 at startup.

## 6. Deploy

- [ ] Health check (`/health`) wired into orchestrator liveness/readiness probe.
- [ ] Resource limits (CPU, RAM). TF inference on CPU is fine for low traffic.
- [ ] Smoke test against live endpoint post-deploy.
- [ ] Monitor: request count, latency (p50/p95/p99), error rate.

---

## Priority

| Priority | Task                                | Why                                     |
| -------- | ----------------------------------- | --------------------------------------- |
| P0       | Model artifact + real preprocessing | Core inference path                     |
| P1       | Dependencies + startup model load   | Blocking a clean build; fail fast       |
| P1       | Integration test with real model    | Catch contract mismatches before deploy |
| P2       | Input validation (max_length)       | Production safety                       |
| P2       | Container hardening                 | Security baseline                       |
| P3       | Request logging                     | Observability                           |

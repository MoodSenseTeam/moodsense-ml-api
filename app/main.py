"""MoodSense ML API — entry point."""

from fastapi import FastAPI

from app.api.routes import predict
from app.core.config import settings

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Mood/sentiment analysis API powered by TensorFlow.",
)

app.include_router(predict.router, prefix="/api/v1")


@app.get("/health", tags=["health"])
def health_check():
    return {"status": "ok"}

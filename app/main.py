"""MoodSense ML API — entry point."""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app import routes
from app.config import settings
from app.inference import MoodModel


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Eagerly load the model so the server fails fast if files are missing."""
    model = MoodModel.get_instance()
    model.load()
    yield


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Mood/sentiment analysis API powered by TensorFlow.",
    lifespan=lifespan,
)

app.include_router(routes.router, prefix="/api/v1")


@app.get("/health", tags=["health"])
def health_check():
    return {"status": "ok"}

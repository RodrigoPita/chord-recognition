"""FastAPI application for chord recognition.

Run with:
    uv run uvicorn chord_rec.api:app --reload
"""

from fastapi import FastAPI

from chord_rec.api.routes import router

app = FastAPI(title="Chord Recognition API", version="0.1.0")
app.include_router(router)

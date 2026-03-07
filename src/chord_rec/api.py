"""FastAPI application for chord recognition.

POST /recognize  — upload a WAV file, get back a list of chord segments.

Run with:
    uv run uvicorn chord_rec.api:app --reload
"""

import tempfile
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from chord_rec.chromagram import ChromagramConfig, compute_chromagram
from chord_rec.recognition import ChordRecognizer, ChordSegment, RecognizerConfig, decode_chord_sequence

app = FastAPI(title="Chord Recognition API", version="0.1.0")


@app.post("/recognize", response_model=list[ChordSegment])
async def recognize(
    file: UploadFile = File(..., description="Audio file (.wav)"),
    version: Literal["STFT", "CQT", "IIR"] = Form("CQT"),
    chord_set: Literal["basic", "extended"] = Form("basic"),
    p: float = Form(0.15, ge=0.0, le=1.0),
) -> JSONResponse:
    """Recognize chords in an uploaded audio file."""
    if not file.filename or not file.filename.lower().endswith(".wav"):
        raise HTTPException(status_code=422, detail="Only .wav files are supported.")

    audio_bytes = await file.read()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = Path(tmp.name)

    try:
        config = ChromagramConfig(version=version)
        X, Fs_X, _, _, _ = compute_chromagram(tmp_path, config)

        recognizer = ChordRecognizer(config=RecognizerConfig(chord_set=chord_set, p=p))
        chord_hmm, _, _ = recognizer.recognize(X)

        segments = decode_chord_sequence(chord_hmm, recognizer.chord_labels, Fs_X)
    finally:
        tmp_path.unlink(missing_ok=True)

    return JSONResponse(content=[s.model_dump() for s in segments])

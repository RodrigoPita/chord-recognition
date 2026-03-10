import tempfile
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

router = APIRouter()


@router.get("/")
def root() -> dict:
    """API information."""
    return {
        "name": "Chord Recognition API",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/health",
        "endpoints": [
            {"method": "POST", "path": "/recognize", "description": "Recognize chords in a WAV file"},
        ],
    }


@router.get("/health")
def health() -> dict:
    """Health check."""
    return {"status": "ok"}


_MPB_MATRICES = Path(__file__).parent.parent / "data" / "matrices"


@router.post("/recognize")
async def recognize(
    file: UploadFile = File(..., description="Audio file (.wav)"),
    version: Literal["STFT", "CQT", "IIR"] = Form("CQT"),
    chord_set: Literal["basic", "extended"] = Form("basic"),
    matrix: Literal["uniform", "mpb"] = Form(
        "uniform",
        description=(
            "Transition matrix for the HMM. 'uniform' gives equal probability to all "
            "chord changes. 'mpb' uses a matrix derived from a Brazilian Popular Music "
            "(MPB) corpus, biasing the model toward genre-typical progressions."
        ),
    ),
    p: float = Form(
        0.15,
        ge=0.0,
        le=1.0,
        description=(
            "HMM self-transition probability. Controls how likely the model is to stay "
            "on the same chord versus switching to a new one. Higher values produce "
            "smoother, longer segments; lower values allow more frequent changes."
        ),
    ),
) -> JSONResponse:
    """Recognize chords in an uploaded audio file."""
    # Heavy imports (librosa, numba, libfmp) are deferred to here so uvicorn
    # can bind the port immediately at startup without waiting for them.
    import pandas as pd
    from chord_rec.chromagram import ChromagramConfig, compute_chromagram
    from chord_rec.recognition import ChordRecognizer, RecognizerConfig, decode_chord_sequence, get_chord_labels

    if not file.filename or not file.filename.lower().endswith(".wav"):
        raise HTTPException(status_code=422, detail="Only .wav files are supported.")

    transition_matrix = None
    if matrix == "mpb":
        df = pd.read_csv(_MPB_MATRICES / f"mpb_{chord_set}.csv", index_col=0)
        labels = get_chord_labels(chord_set)
        transition_matrix = df.reindex(index=labels, columns=labels)

    audio_bytes = await file.read()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = Path(tmp.name)

    try:
        config = ChromagramConfig(version=version)
        X, Fs_X, _, _, _ = compute_chromagram(tmp_path, config)

        recognizer = ChordRecognizer(
            config=RecognizerConfig(chord_set=chord_set, p=p),
            transition_matrix=transition_matrix,
        )
        chord_hmm, _, _ = recognizer.recognize(X)

        segments = decode_chord_sequence(chord_hmm, recognizer.chord_labels, Fs_X)
    finally:
        tmp_path.unlink(missing_ok=True)

    return JSONResponse(content=[s.model_dump() for s in segments])

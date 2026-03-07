import io
import wave

import numpy as np
import pytest
from fastapi.testclient import TestClient

from chord_rec.api import app

client = TestClient(app)


def _make_wav(duration_s: float = 1.0, sample_rate: int = 22050) -> bytes:
    """Generate a minimal sine-wave WAV in memory."""
    n = int(duration_s * sample_rate)
    t = np.linspace(0, duration_s, n, endpoint=False)
    samples = (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples.tobytes())
    return buf.getvalue()


WAV_BYTES = _make_wav()


class TestRecognizeEndpoint:
    def test_returns_200(self):
        resp = client.post(
            "/recognize",
            files={"file": ("test.wav", WAV_BYTES, "audio/wav")},
        )
        assert resp.status_code == 200

    def test_response_is_list(self):
        resp = client.post(
            "/recognize",
            files={"file": ("test.wav", WAV_BYTES, "audio/wav")},
        )
        assert isinstance(resp.json(), list)

    def test_segments_have_expected_fields(self):
        resp = client.post(
            "/recognize",
            files={"file": ("test.wav", WAV_BYTES, "audio/wav")},
        )
        for seg in resp.json():
            assert "start" in seg
            assert "end" in seg
            assert "label" in seg
            assert "duration" in seg

    def test_segments_are_contiguous(self):
        resp = client.post(
            "/recognize",
            files={"file": ("test.wav", WAV_BYTES, "audio/wav")},
        )
        segs = resp.json()
        for i in range(1, len(segs)):
            assert segs[i]["start"] == pytest.approx(segs[i - 1]["end"], abs=1e-4)

    def test_custom_version_and_chord_set(self):
        resp = client.post(
            "/recognize",
            files={"file": ("test.wav", WAV_BYTES, "audio/wav")},
            data={"version": "STFT", "chord_set": "extended"},
        )
        assert resp.status_code == 200

    def test_rejects_non_wav(self):
        resp = client.post(
            "/recognize",
            files={"file": ("track.mp3", b"not a wav", "audio/mpeg")},
        )
        assert resp.status_code == 422

    def test_invalid_p_rejected(self):
        resp = client.post(
            "/recognize",
            files={"file": ("test.wav", WAV_BYTES, "audio/wav")},
            data={"p": "1.5"},
        )
        assert resp.status_code == 422

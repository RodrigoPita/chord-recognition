---
title: Chord Recognition
emoji: 🎸
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

<h1 align="center">Chord Recognition 🎸</h1>

<div align="center">

[![Python](https://img.shields.io/badge/python-3.12-gray?style=for-the-badge&colorA=3776AB&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-gray?style=for-the-badge&colorA=009688&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![uv](https://img.shields.io/badge/uv-gray?style=for-the-badge&colorA=DE5FE9&logo=uv&logoColor=white)](https://github.com/astral-sh/uv)
[![HuggingFace](https://img.shields.io/badge/🤗%20Live%20API-gray?style=for-the-badge&colorA=FFD21E)](https://rodrigopita-chord-recognition.hf.space/docs)

</div>

A chord recognition library and REST API focused on complex harmonies found in MPB, Bossa Nova, and Jazz. Built on top of the [Fundamentals of Music Processing (FMP)](https://www.audiolabs-erlangen.de/resources/MIR/FMP/C5/C5.html) modules by Meinard Müller.

Originally developed as a Computer Science graduation thesis (TCC) at UFRJ.

---

## ✨ Features

* **🎼 Three chromagram front-ends:** STFT, CQT, and IIR
* **🎹 Two chord vocabularies:** basic (24 chords: major + minor triads) and extended (120 chords: sevenths, half-diminished, diminished, augmented, and more)
* **🤖 HMM-based recognition:** Viterbi algorithm over chord similarity for smooth, musically coherent segmentation
* **📊 Corpus transition matrices:** plug in a transition matrix derived from a real corpus (e.g. MPB) to bias the model toward genre-specific progressions
* **📐 Evaluation metrics:** precision, recall, and F-measure against ground-truth annotations
* **🌐 REST API:** deploy locally or use the [hosted API on HuggingFace Spaces](https://rodrigopita-chord-recognition.hf.space)

---

## 🚀 Live API

The API is deployed at **https://rodrigopita-chord-recognition.hf.space** — interactive docs at [/docs](https://rodrigopita-chord-recognition.hf.space/docs).

```bash
curl -X POST https://rodrigopita-chord-recognition.hf.space/recognize \
  -F "file=@audio.wav" \
  -F "version=CQT" \
  -F "chord_set=basic"
```

Returns a JSON array of timed chord segments:

```json
[
  {"start": 0.0, "end": 2.32, "label": "D", "duration": 2.32},
  {"start": 2.32, "end": 6.31, "label": "C#m", "duration": 3.99}
]
```

> **Note:** The first request after a period of inactivity may take ~20 seconds as the free-tier container wakes up and numba JIT-compiles on first use.

---

## 💻 Running Locally

### Prerequisites

* Python 3.10+
* [uv](https://github.com/astral-sh/uv)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/RodrigoPita/chord-recognition.git
   cd chord-recognition
   ```

2. **Install dependencies:**
   ```bash
   uv sync
   ```

   To include development dependencies (pytest):
   ```bash
   uv sync --extra dev
   ```

### REST API

```bash
uv run uvicorn chord_rec.api:app --reload
```

Then `POST /recognize` at `http://localhost:8000`:

```bash
curl -X POST http://localhost:8000/recognize \
  -F "file=@audio.wav" \
  -F "version=CQT" \
  -F "chord_set=basic"
```

Interactive docs at `http://localhost:8000/docs`.

### CLI

```bash
uv run python scripts/recognize.py audio.wav
uv run python scripts/recognize.py audio.wav --version CQT --chord-set extended
uv run python scripts/recognize.py audio.wav --annotation labels.csv   # evaluate
uv run python scripts/recognize.py audio.wav --output chords.csv       # save results
uv run python scripts/recognize.py audio.wav --plot                    # chord timeline
```

---

## 🐍 Python Library

### Basic chord recognition

```python
from chord_rec.chromagram import ChromagramConfig, compute_chromagram
from chord_rec.recognition import ChordRecognizer

config = ChromagramConfig(version="CQT")
X, Fs_X, _, _, _ = compute_chromagram("audio.wav", config)

recognizer = ChordRecognizer()
chord_hmm, chord_template, chord_sim = recognizer.recognize(X)
```

### Evaluating against an annotation

```python
from chord_rec.data_utils import Song
from chord_rec.evaluation import convert_chord_ann_matrix, compute_eval_measures

song = Song(audio_path="audio.wav", annotation_path="labels.csv", name="My Song")
ann_matrix, *_ = convert_chord_ann_matrix(song.annotation_path, recognizer.chord_labels, Fs=Fs_X, N=X.shape[1])

result = compute_eval_measures(ann_matrix, chord_hmm)
print(result)  # P=0.812, R=0.794, F=0.803
```

### Using a corpus-derived transition matrix

```python
import pandas as pd
from chord_rec.recognition import ChordRecognizer, RecognizerConfig

df = pd.read_csv("data/matrices/mpb_transition_matrix.csv", index_col=0)
recognizer = ChordRecognizer(
    config=RecognizerConfig(chord_set="basic", p=0.5),
    transition_matrix=df,
)
chord_hmm, _, _ = recognizer.recognize(X)
```

### Finding the optimal self-transition probability

```python
best_p = recognizer.find_best_p(X, ann_matrix)
print(f"Best p: {best_p:.2f}")
```

---

## 🐳 Deploying to HuggingFace Spaces

The API is packaged as a Docker Space. To deploy your own copy:

1. **Create a new Space** at [huggingface.co/new-space](https://huggingface.co/new-space), selecting **Docker** as the SDK.

2. **Add the HF remote:**
   ```bash
   git remote add hf https://huggingface.co/spaces/<your-username>/<your-space-name>
   ```

3. **Push:**
   ```bash
   git push hf main
   ```

HF builds the Docker image and starts the container automatically on every push.

**Key Dockerfile requirements:**

* Create a user with uid 1000 and switch to it before `CMD` — HF Spaces enforces this and will silently refuse to schedule the container otherwise:
  ```dockerfile
  RUN useradd -m -u 1000 user
  # ... install deps ...
  USER user
  ```
* Bind uvicorn to `0.0.0.0`, not `localhost`.
* Set `HOME` to the user's home directory so numba and other libraries can write their caches.

---

## 📋 Annotation Format

Chord annotation files are semicolon-separated CSVs:

```
"Start";"End";"Label"
0.000000;2.511111;"Am"
2.511111;4.930000;"C"
```

If your source annotations are in Audacity-style tab-separated TXT files:

```python
from chord_rec.data_utils import txt_to_csv, txt_to_csv_aligned

txt_to_csv("input.txt", "output.csv")          # direct conversion
txt_to_csv_aligned("input.txt", "output.csv")  # aligns segment boundaries
```

---

## 📚 References

* Müller, M. (2015). *Fundamentals of Music Processing*. Springer. [FMP Notebooks C5](https://www.audiolabs-erlangen.de/resources/MIR/FMP/C5/C5.html)
* Caetano, M. et al. *Projeto MPB* — corpus used for the MPB transition matrix.
* Original TCC repository: [RodrigoPita/TCC](https://github.com/RodrigoPita/TCC)

---

<div align="center">

Made with ❤️ by [Rodrigo Pita](https://github.com/RodrigoPita)

**[⬆ back to top](#chord-recognition-)**

</div>

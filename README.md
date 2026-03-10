---
title: Chord Recognition
emoji: 🎸
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# chord-recognition

A chord recognition library focused on complex harmonies found in MPB, Bossa Nova, and Jazz. Built on top of the [Fundamentals of Music Processing (FMP)](https://www.audiolabs-erlangen.de/resources/MIR/FMP/C5/C5.html) modules by Meinard Müller.

Originally developed as a Computer Science graduation thesis (TCC) at UFRJ.

**Live API:** [https://rodrigopita-chord-recognition.hf.space](https://rodrigopita-chord-recognition.hf.space) — interactive docs at [/docs](https://rodrigopita-chord-recognition.hf.space/docs)

## Features

- Three chromagram front-ends: **STFT**, **CQT**, and **IIR**
- Two chord vocabularies: **basic** (24 chords: major + minor triads) and **extended** (120 chords: includes seventh, half-diminished, diminished, augmented, and more)
- Template-based and **HMM-based** chord recognition via the Viterbi algorithm
- Support for custom transition matrices derived from corpus analysis (e.g. the MPB corpus)
- Evaluation metrics: precision, recall, and F-measure
- Utilities for loading annotation files and finding the optimal self-transition probability

## Installation

Requires Python 3.10+. Dependencies are managed with [uv](https://github.com/astral-sh/uv).

```bash
git clone https://github.com/your-username/chord-recognition.git
cd chord-recognition
uv sync
```

To include development dependencies (pytest):

```bash
uv sync --extra dev
```

## Usage

### Basic chord recognition

```python
from chord_rec.data_utils import Song
from chord_rec.chromagram import ChromagramConfig, compute_chromagram
from chord_rec.recognition import ChordRecognizer
from chord_rec.evaluation import convert_chord_ann_matrix, compute_eval_measures

song = Song(
    audio_path="data/audios/my_song.wav",
    annotation_path="data/labels/my_song.csv",
    name="My Song",
)

config = ChromagramConfig(version="CQT")
X, Fs_X, _, _, _ = compute_chromagram(song.audio_path, config)

recognizer = ChordRecognizer()
chord_hmm, chord_template, chord_sim = recognizer.recognize(X)
```

### Evaluating against an annotation

```python
chord_labels = recognizer.chord_labels
ann_matrix, *_ = convert_chord_ann_matrix(song.annotation_path, chord_labels, Fs=Fs_X, N=X.shape[1])

result = compute_eval_measures(ann_matrix, chord_hmm)
print(result)  # P=0.812, R=0.794, F=0.803 (TP=..., FP=..., FN=...)
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

### CLI

```bash
uv run python scripts/recognize.py audio.wav
uv run python scripts/recognize.py audio.wav --version CQT --chord-set extended
uv run python scripts/recognize.py audio.wav --annotation labels.csv   # evaluate
uv run python scripts/recognize.py audio.wav --output chords.csv       # save results
uv run python scripts/recognize.py audio.wav --plot                    # chord timeline
```

### REST API

#### Hosted (HuggingFace Spaces)

The API is deployed at **https://rodrigopita-chord-recognition.hf.space**. Interactive docs: [/docs](https://rodrigopita-chord-recognition.hf.space/docs).

```bash
curl -X POST https://rodrigopita-chord-recognition.hf.space/recognize \
  -F "file=@audio.wav" \
  -F "version=CQT" \
  -F "chord_set=basic"
```

Note: the first request after a period of inactivity may take ~20 seconds as the free-tier container wakes up and numba JIT-compiles on first use.

#### Local

```bash
uv run uvicorn chord_rec.api:app --reload
```

Then `POST /recognize` with a multipart form upload:

```bash
curl -X POST http://localhost:8000/recognize \
  -F "file=@audio.wav" \
  -F "version=CQT" \
  -F "chord_set=basic"
```

Returns a JSON array of chord segments:

```json
[
  {"start": 0.0, "end": 2.32, "label": "D", "duration": 2.32},
  {"start": 2.32, "end": 6.31, "label": "C#m", "duration": 3.99}
]
```

Interactive API docs are available at `http://localhost:8000/docs`.

## Deploying to HuggingFace Spaces

The API is packaged as a Docker Space. To deploy your own copy:

1. Create a new Space at [huggingface.co/new-space](https://huggingface.co/new-space), selecting **Docker** as the SDK and setting `app_port: 7860` in the README front-matter.

2. Add the HF Space as a git remote:
   ```bash
   git remote add hf https://huggingface.co/spaces/<your-username>/<your-space-name>
   ```

3. Push:
   ```bash
   git push hf main
   ```

HF will build the Docker image and start the container automatically on every push.

**Key requirements for the Dockerfile:**
- Create a user with uid 1000 and switch to it before the `CMD` — HF Spaces enforces this and will silently refuse to schedule the container otherwise:
  ```dockerfile
  RUN useradd -m -u 1000 user
  # ... install deps ...
  USER user
  ```
- Bind uvicorn to `0.0.0.0`, not `localhost`.
- Set `HOME` to the user's home directory so numba and other libraries can write their caches.

## Project structure

```
chord-recognition/
├── src/
│   └── chord_rec/
│       ├── constants.py     # Chord templates, note names, chordal types
│       ├── data_utils.py    # Song model, file loading, annotation conversion
│       ├── chromagram.py    # Chromagram computation (STFT, CQT, IIR)
│       ├── recognition.py   # Template recognition, Viterbi, ChordRecognizer
│       ├── evaluation.py    # Precision/recall/F-measure, annotation parsing
│       ├── plotting.py      # Visualisation helpers
│       └── api.py           # FastAPI application (POST /recognize)
├── scripts/
│   └── recognize.py         # CLI for running recognition on a local file
├── data/
│   ├── audios/              # WAV files (gitignored)
│   ├── labels/              # Chord annotation CSV files
│   └── matrices/            # Transition matrix CSV files
└── tests/
```

## Annotation format

Chord annotation files are semicolon-separated CSVs with the following structure:

```
"Start";"End";"Label"
0.000000;2.511111;"Am"
2.511111;4.930000;"C"
```

If your source annotations are in Audacity-style tab-separated TXT files, use:

```python
from chord_rec.data_utils import txt_to_csv, txt_to_csv_aligned

txt_to_csv("input.txt", "output.csv")          # direct conversion
txt_to_csv_aligned("input.txt", "output.csv")  # aligns segment boundaries
```

## Original repository

This project is a refactored version of the original TCC repository: [RodrigoPita/TCC](https://github.com/RodrigoPita/TCC)

## References

- Müller, M. (2015). *Fundamentals of Music Processing*. Springer. [FMP Notebooks C5](https://www.audiolabs-erlangen.de/resources/MIR/FMP/C5/C5.html)
- Caetano, M. et al. *Projeto MPB* — corpus used for the MPB transition matrix.

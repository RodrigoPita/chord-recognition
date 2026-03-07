# chord-recognition

A chord recognition library focused on complex harmonies found in MPB, Bossa Nova, and Jazz. Built on top of the [Fundamentals of Music Processing (FMP)](https://www.audiolabs-erlangen.de/resources/MIR/FMP/C5/C5.html) modules by Meinard Müller.

Originally developed as a Computer Science graduation thesis (TCC) at UFRJ.

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
│       └── plotting.py      # Visualisation helpers
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

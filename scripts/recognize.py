"""Chord recognition script.

Runs the HMM-based chord recognizer on an audio file and prints the
detected chord sequence. Optionally evaluates against a reference
annotation and saves or plots the results.

Usage:
    uv run python scripts/recognize.py audio.wav
    uv run python scripts/recognize.py audio.wav --version CQT
    uv run python scripts/recognize.py audio.wav --chord-set extended
    uv run python scripts/recognize.py audio.wav --annotation labels.csv
    uv run python scripts/recognize.py audio.wav --output chords.csv
    uv run python scripts/recognize.py audio.wav --plot
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from chord_rec.chromagram import ChromagramConfig, compute_chromagram
from chord_rec.evaluation import compute_eval_measures, convert_chord_ann_matrix
from chord_rec.recognition import ChordRecognizer, RecognizerConfig, decode_chord_sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Recognize chords in an audio file.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('audio', type=Path, help='Path to the audio file (.wav)')
    parser.add_argument(
        '--version', choices=['STFT', 'CQT', 'IIR'], default='CQT',
        help='Chromagram front-end',
    )
    parser.add_argument(
        '--chord-set', choices=['basic', 'extended'], default='basic',
        dest='chord_set', help='Chord vocabulary',
    )
    parser.add_argument(
        '--p', type=float, default=0.15,
        help='Self-transition probability for the HMM',
    )
    parser.add_argument(
        '--matrix', type=Path, default=None,
        help='Path to a transition matrix CSV (e.g. MPB corpus matrix)',
    )
    parser.add_argument(
        '--annotation', type=Path, default=None,
        help='Path to a reference annotation CSV for evaluation',
    )
    parser.add_argument(
        '--output', type=Path, default=None,
        help='Save chord segments to this CSV file',
    )
    parser.add_argument(
        '--plot', action='store_true',
        help='Show a chord timeline plot',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.audio.exists():
        print(f'Error: audio file not found: {args.audio}', file=sys.stderr)
        sys.exit(1)

    # --- Chromagram ---
    print(f'Computing {args.version} chromagram for: {args.audio.name}')
    config = ChromagramConfig(version=args.version)
    X, Fs_X, _, _, x_dur = compute_chromagram(args.audio, config)
    print(f'  Duration: {x_dur:.1f}s  |  Frames: {X.shape[1]}  |  Feature rate: {Fs_X:.1f} Hz')

    # --- Recognition ---
    import pandas as pd
    transition_matrix = None
    if args.matrix:
        print(f'Loading transition matrix: {args.matrix.name}')
        transition_matrix = pd.read_csv(args.matrix, index_col=0)

    recognizer = ChordRecognizer(
        config=RecognizerConfig(chord_set=args.chord_set, p=args.p),
        transition_matrix=transition_matrix,
    )
    chord_hmm, _, _ = recognizer.recognize(X)

    # --- Decode ---
    segments = decode_chord_sequence(chord_hmm, recognizer.chord_labels, Fs_X)

    # --- Print ---
    print(f'\nChord sequence ({len(segments)} segments):\n')
    for seg in segments:
        print(f'  {seg.start:7.2f}s → {seg.end:7.2f}s   {seg.label:<8}  ({seg.duration:.2f}s)')

    # --- Evaluation (optional) ---
    if args.annotation:
        if not args.annotation.exists():
            print(f'\nWarning: annotation file not found: {args.annotation}', file=sys.stderr)
        else:
            print(f'\nEvaluating against: {args.annotation.name}')
            ann_matrix, *_ = convert_chord_ann_matrix(
                args.annotation, recognizer.chord_labels, Fs=Fs_X, N=X.shape[1],
            )
            result = compute_eval_measures(ann_matrix, chord_hmm)
            print(f'  {result}')

    # --- Save (optional) ---
    if args.output:
        with open(args.output, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['start', 'end', 'duration', 'label'])
            for seg in segments:
                writer.writerow([seg.start, seg.end, seg.duration, seg.label])
        print(f'\nSaved to: {args.output}')

    # --- Plot (optional) ---
    if args.plot:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import numpy as np

        unique_labels = list(dict.fromkeys(seg.label for seg in segments))
        cmap = plt.get_cmap('tab20', len(unique_labels))
        color_map = {label: cmap(i) for i, label in enumerate(unique_labels)}

        fig, ax = plt.subplots(figsize=(14, 2))
        for seg in segments:
            ax.barh(
                0, seg.duration, left=seg.start, height=0.5,
                color=color_map[seg.label], edgecolor='white', linewidth=0.5,
            )
            if seg.duration > x_dur / 40:
                ax.text(
                    seg.start + seg.duration / 2, 0, seg.label,
                    ha='center', va='center', fontsize=7, fontweight='bold',
                )

        legend = [mpatches.Patch(color=color_map[l], label=l) for l in unique_labels]
        ax.legend(handles=legend, loc='upper right', fontsize=7, ncol=4)
        ax.set_xlim(0, x_dur)
        ax.set_xlabel('Time (seconds)')
        ax.set_yticks([])
        ax.set_title(f'Chord recognition — {args.audio.name}')
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    main()

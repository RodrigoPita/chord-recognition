import copy
from pathlib import Path

import numpy as np
import libfmp.c4
from pydantic import BaseModel


class EvaluationResult(BaseModel):
    precision: float
    recall: float
    f_measure: float
    true_positives: int
    false_positives: int
    false_negatives: int

    def __str__(self) -> str:
        return (
            f'P={self.precision:.3f}, R={self.recall:.3f}, F={self.f_measure:.3f} '
            f'(TP={self.true_positives}, FP={self.false_positives}, FN={self.false_negatives})'
        )


_FLAT_TO_SHARP: dict[str, str] = {
    'Db': 'C#',
    'Eb': 'D#',
    'Gb': 'F#',
    'Ab': 'G#',
    'Bb': 'A#',
}


def convert_chord_label(ann: list) -> list:
    """Normalise chord labels in a segment annotation.

    Replaces ':min' suffix with 'm' and converts flat spellings to
    their sharp enharmonic equivalents (e.g. Bb → A#).
    """
    ann_conv = copy.deepcopy(ann)
    for segment in ann_conv:
        segment[2] = segment[2].replace(':min', 'm')
        for flat, sharp in _FLAT_TO_SHARP.items():
            segment[2] = segment[2].replace(flat, sharp)
    return ann_conv


def convert_sequence_ann(seq: list[str], Fs: float = 1) -> list:
    """Convert a frame-level label sequence to a segment-based annotation."""
    return [[(m - 0.5) / Fs, (m + 0.5) / Fs, seq[m]] for m in range(len(seq))]


def convert_chord_ann_matrix(
    annotation_path: Path | str,
    chord_labels: list[str],
    Fs: float = 1,
    N: int | None = None,
    last: bool = False,
) -> tuple[np.ndarray, list, list, list, list]:
    """Convert a segment-based chord annotation file into multiple representations.

    Args:
        annotation_path: Path to the CSV annotation file.
        chord_labels: Ordered list of chord label strings.
        Fs: Feature rate in Hz (Default: 1).
        N: Target number of frames. Pads or trims ann_matrix to match.
        last: If True, pads with the last chord label; otherwise uses 'N'.

    Returns:
        ann_matrix: Binary time-chord matrix (num_chords x N).
        ann_frame: Frame-level label sequence.
        ann_seg_frame: Segment annotation in frame indices (continuous).
        ann_seg_ind: Segment annotation in frame indices.
        ann_seg_sec: Segment annotation in seconds.
    """
    path = str(annotation_path)
    ann_seg_sec, _ = libfmp.c4.read_structure_annotation(path)
    ann_seg_sec = convert_chord_label(ann_seg_sec)
    ann_seg_ind, _ = libfmp.c4.read_structure_annotation(path, Fs=Fs, index=True)
    ann_seg_ind = convert_chord_label(ann_seg_ind)

    ann_frame: list[str] = libfmp.c4.convert_ann_to_seq_label(ann_seg_ind)

    if N is None:
        N = len(ann_frame)
    if N < len(ann_frame):
        ann_frame = ann_frame[:N]
    if N > len(ann_frame):
        pad = ann_frame[-1] if last else 'N'
        ann_frame = ann_frame + [pad] * (N - len(ann_frame))

    ann_seg_frame = convert_sequence_ann(ann_frame, Fs=1)

    ann_matrix = np.zeros((len(chord_labels), N))
    for n, label in enumerate(ann_frame):
        if label in chord_labels:
            ann_matrix[chord_labels.index(label), n] = 1

    return ann_matrix, ann_frame, ann_seg_frame, ann_seg_ind, ann_seg_sec


def compute_eval_measures(I_ref: np.ndarray, I_est: np.ndarray) -> EvaluationResult:
    """Compute precision, recall, and F-measure between reference and estimate.

    Args:
        I_ref: Binary reference matrix.
        I_est: Binary estimate matrix, same shape as I_ref.

    Returns:
        EvaluationResult with precision, recall, f_measure, TP, FP, FN.
    """
    assert I_ref.shape == I_est.shape, "Input matrices must have the same shape"
    TP = int(np.sum(np.logical_and(I_ref, I_est)))
    FP = int(np.sum(I_est > 0)) - TP
    FN = int(np.sum(I_ref > 0)) - TP
    if TP > 0:
        P = TP / (TP + FP)
        R = TP / (TP + FN)
        F = 2 * P * R / (P + R)
    else:
        P = R = F = 0.0
    return EvaluationResult(
        precision=P,
        recall=R,
        f_measure=F,
        true_positives=TP,
        false_positives=FP,
        false_negatives=FN,
    )


def compute_mean_f_measure(
    result_dict: dict[int, np.ndarray],
    indices: list[int],
) -> np.ndarray:
    """Compute the element-wise mean of F-measure arrays across songs."""
    result_mean = np.copy(result_dict[indices[0]])
    for i in indices[1:]:
        result_mean += result_dict[i]
    return result_mean / len(indices)

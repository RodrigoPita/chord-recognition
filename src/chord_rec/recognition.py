from typing import Literal

import numpy as np
import pandas as pd
from numba import jit
from scipy.linalg import circulant
from pydantic import BaseModel, Field, computed_field
import libfmp.c3

from chord_rec.constants import FUNDAMENTALS, CHORDAL_TYPES, TEMPLATES_IN_C


ChordSet = Literal['basic', 'extended']


class ChordSegment(BaseModel):
    start: float
    end: float
    label: str

    @computed_field
    @property
    def duration(self) -> float:
        return round(self.end - self.start, 6)


class RecognizerConfig(BaseModel):
    chord_set: ChordSet = 'basic'
    p: float = Field(
        0.15,
        ge=0.0,
        le=1.0,
        description=(
            "HMM self-transition probability. Controls how likely the model is to stay "
            "on the same chord versus switching to a new one. Higher values produce "
            "smoother, longer segments; lower values allow more frequent changes."
        ),
    )


def get_chord_labels(
    chord_set: ChordSet = 'basic',
    ext_minor: str = 'm',
    nonchord: bool = False,
) -> list[str]:
    """Generate the ordered list of chord labels for a given chord set.

    Args:
        chord_set: 'basic' for 24 chords (12 major + 12 minor triads),
                   'extended' for 120 chords (all types in TEMPLATES_IN_C).
        ext_minor: Suffix appended to minor chord names (Default: 'm').
        nonchord: If True, appends a 'N' (no-chord) label at the end.

    Returns:
        List of chord label strings.
    """
    notes = list(FUNDAMENTALS)
    labels = notes + [n + ext_minor for n in notes]

    if chord_set == 'extended':
        ext_types = list(TEMPLATES_IN_C.keys())
        # Skip '' (major triad) and 'm' (minor triad) — already included above
        for chord_type in ext_types:
            if chord_type not in ('', 'm'):
                labels += [n + chord_type for n in notes]

    if nonchord:
        labels.append('N')

    return labels


def generate_chord_templates(
    chord_set: ChordSet = 'basic',
    nonchord: bool = False,
) -> np.ndarray:
    """Generate binary chord template matrix.

    Args:
        chord_set: 'basic' produces a 12 x 24 matrix (or 25 with nonchord),
                   'extended' produces a 12 x 120 matrix.
        nonchord: If True, appends an all-zeros nonchord column (basic only).

    Returns:
        Template matrix of shape (12, num_chords).
    """
    if chord_set == 'basic':
        template_cmaj = np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0])
        template_cmin = np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0])
        num_chords = 25 if nonchord else 24
        templates = np.zeros((12, num_chords))
        for shift in range(12):
            templates[:, shift] = np.roll(template_cmaj, shift)
            templates[:, shift + 12] = np.roll(template_cmin, shift)
    else:
        chord_names = [
            f + CHORDAL_TYPES[t]
            for f in FUNDAMENTALS
            for t in CHORDAL_TYPES
        ]
        df = pd.DataFrame(0, index=range(12), columns=chord_names)
        for fundamental in FUNDAMENTALS:
            root = FUNDAMENTALS.index(fundamental)
            for chord_type, intervals in TEMPLATES_IN_C.items():
                col = fundamental + chord_type
                if col in df.columns:
                    for interval in intervals:
                        df.at[(root + interval) % 12, col] = 1
        templates = df.to_numpy()

    return templates


def chord_recognition_template(
    X: np.ndarray,
    norm_sim: str = '1',
    chord_set: ChordSet = 'basic',
    nonchord: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Template-based chord recognition via cosine similarity.

    Args:
        X: Chromagram matrix (12 x N_frames).
        norm_sim: Norm used to normalise the similarity matrix ('1', '2', 'max').
        chord_set: Chord vocabulary to use.
        nonchord: Whether to include a no-chord template.

    Returns:
        chord_sim: Chord similarity matrix (num_chords x N_frames).
        chord_max: Binarised matrix with 1 at the argmax chord per frame.
    """
    templates = generate_chord_templates(chord_set, nonchord)
    X_norm = libfmp.c3.normalize_feature_sequence(X, norm='2')
    templates_norm = libfmp.c3.normalize_feature_sequence(templates, norm='2')
    chord_sim = np.matmul(templates_norm.T, X_norm)
    if norm_sim is not None:
        chord_sim = libfmp.c3.normalize_feature_sequence(chord_sim, norm=norm_sim)
    chord_max_index = np.argmax(chord_sim, axis=0)
    chord_max = np.zeros(chord_sim.shape, dtype=np.int32)
    for n in range(chord_sim.shape[1]):
        chord_max[chord_max_index[n], n] = 1
    return chord_sim, chord_max


@jit(nopython=True)
def viterbi(
    A: np.ndarray,
    C: np.ndarray,
    B: np.ndarray,
    O: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Viterbi algorithm for HMM decoding (probability domain).

    Args:
        A: State transition matrix (I x I).
        C: Initial state distribution (I,).
        B: Emission probability matrix (I x K).
        O: Observation index sequence (N,).

    Returns:
        S_opt: Optimal state sequence (N,).
        D: Accumulated probability matrix (I x N).
        E: Backtracking matrix (I x N-1).
    """
    I = A.shape[0]
    N = len(O)
    D = np.zeros((I, N))
    E = np.zeros((I, N - 1), dtype=np.int32)
    D[:, 0] = np.multiply(C, B[:, O[0]])
    for n in range(1, N):
        for i in range(I):
            temp = np.multiply(A[:, i], D[:, n - 1])
            D[i, n] = np.max(temp) * B[i, O[n]]
            E[i, n - 1] = np.argmax(temp)
    S_opt = np.zeros(N, dtype=np.int32)
    S_opt[-1] = np.argmax(D[:, -1])
    for n in range(N - 2, -1, -1):
        S_opt[n] = E[int(S_opt[n + 1]), n]
    return S_opt, D, E


@jit(nopython=True)
def viterbi_log(
    A: np.ndarray,
    C: np.ndarray,
    B: np.ndarray,
    O: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Viterbi algorithm for HMM decoding (log domain).

    Args:
        A: State transition matrix (I x I).
        C: Initial state distribution (I,).
        B: Emission probability matrix (I x K).
        O: Observation index sequence (N,).

    Returns:
        S_opt: Optimal state sequence (N,).
        D_log: Accumulated log-probability matrix (I x N).
        E: Backtracking matrix (I x N-1).
    """
    I = A.shape[0]
    N = len(O)
    tiny = np.finfo(0.).tiny
    A_log = np.log(A + tiny)
    C_log = np.log(C + tiny)
    B_log = np.log(B + tiny)
    D_log = np.zeros((I, N))
    E = np.zeros((I, N - 1), dtype=np.int32)
    D_log[:, 0] = C_log + B_log[:, O[0]]
    for n in range(1, N):
        for i in range(I):
            temp = A_log[:, i] + D_log[:, n - 1]
            D_log[i, n] = np.max(temp) + B_log[i, O[n]]
            E[i, n - 1] = np.argmax(temp)
    S_opt = np.zeros(N, dtype=np.int32)
    S_opt[-1] = np.argmax(D_log[:, -1])
    for n in range(N - 2, -1, -1):
        S_opt[n] = E[int(S_opt[n + 1]), n]
    return S_opt, D_log, E


@jit(nopython=True)
def viterbi_log_likelihood(
    A: np.ndarray,
    C: np.ndarray,
    B_O: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Viterbi algorithm operating directly on a pre-computed likelihood matrix.

    Args:
        A: State transition matrix (I x I).
        C: Initial state distribution (1 x I).
        B_O: Likelihood matrix (I x N).

    Returns:
        S_mat: Binary matrix of the optimal state sequence (I x N).
        S_opt: Optimal state sequence (N,).
        D_log: Accumulated log-probability matrix (I x N).
        E: Backtracking matrix (I x N-1).
    """
    I = A.shape[0]
    N = B_O.shape[1]
    tiny = np.finfo(0.).tiny
    A_log = np.log(A + tiny)
    C_log = np.log(C + tiny)
    B_O_log = np.log(B_O + tiny)
    D_log = np.zeros((I, N))
    E = np.zeros((I, N - 1), dtype=np.int32)
    D_log[:, 0] = C_log + B_O_log[:, 0]
    for n in range(1, N):
        for i in range(I):
            temp = A_log[:, i] + D_log[:, n - 1]
            D_log[i, n] = np.max(temp) + B_O_log[i, n]
            E[i, n - 1] = np.argmax(temp)
    S_opt = np.zeros(N, dtype=np.int32)
    S_opt[-1] = np.argmax(D_log[:, -1])
    for n in range(N - 2, -1, -1):
        S_opt[n] = E[int(S_opt[n + 1]), n]
    S_mat = np.zeros((I, N), dtype=np.int32)
    for n in range(N):
        S_mat[S_opt[n], n] = 1
    return S_mat, S_opt, D_log, E


def uniform_transition_matrix(p: float = 0.01, N: int = 24) -> np.ndarray:
    """Build a uniform transition matrix with self-transition probability p.

    Args:
        p: Self-transition probability (diagonal entries).
        N: Number of states.

    Returns:
        Transition matrix of shape (N, N) with rows summing to 1.
    """
    off_diag = (1 - p) / (N - 1)
    A = off_diag * np.ones((N, N))
    np.fill_diagonal(A, p)
    return A


def edit_diagonal(df: pd.DataFrame, p: float) -> pd.DataFrame:
    """Set the diagonal of a transition matrix to p, rescaling off-diagonal
    entries proportionally so each row still sums to 1.

    Args:
        df: Square transition matrix as a DataFrame.
        p: New self-transition probability.

    Returns:
        Updated DataFrame with the same shape.
    """
    n = df.shape[0]
    result = df.copy().astype(float)
    for i in range(n):
        result.iat[i, i] = p
        off_diag_target = 1 - p
        current_off_diag = result.iloc[i].sum() - p
        if current_off_diag == 0:
            for j in range(n):
                if i != j:
                    result.iat[i, j] = off_diag_target / (n - 1)
        else:
            scale = off_diag_target / current_off_diag
            for j in range(n):
                if i != j:
                    result.iat[i, j] *= scale
    return result


def matrix_circular_mean(A: np.ndarray) -> np.ndarray:
    """Compute a circulant matrix from the mean of sheared diagonals of A."""
    N = A.shape[0]
    A_shear = np.zeros((N, N))
    for n in range(N):
        A_shear[:, n] = np.roll(A[:, n], -n)
    return circulant(np.sum(A_shear, axis=1)) / N


def matrix_chord24_trans_inv(A: np.ndarray) -> np.ndarray:
    """Make a 24-state chord transition matrix transposition-invariant.

    Applies circular mean independently to each of the four quadrants
    (major→major, major→minor, minor→major, minor→minor).
    """
    A_ti = np.zeros(A.shape)
    A_ti[0:12, 0:12] = matrix_circular_mean(A[0:12, 0:12])
    A_ti[0:12, 12:24] = matrix_circular_mean(A[0:12, 12:24])
    A_ti[12:24, 0:12] = matrix_circular_mean(A[12:24, 0:12])
    A_ti[12:24, 12:24] = matrix_circular_mean(A[12:24, 12:24])
    return A_ti


def decode_chord_sequence(
    chord_matrix: np.ndarray,
    chord_labels: list[str],
    Fs_X: float,
) -> list[ChordSegment]:
    """Convert a binary chord matrix into a list of timed chord segments.

    Consecutive frames with the same chord are merged into a single segment.

    Args:
        chord_matrix: Binary matrix of shape (num_chords x N_frames), e.g.
            the HMM output from ChordRecognizer.recognize().
        chord_labels: Ordered list of chord label strings matching the matrix rows.
        Fs_X: Feature rate in Hz used to convert frame indices to seconds.

    Returns:
        List of ChordSegment, each with start/end times in seconds and a label.
    """
    frame_indices = np.argmax(chord_matrix, axis=0)
    segments: list[ChordSegment] = []
    if len(frame_indices) == 0:
        return segments

    current_label = chord_labels[frame_indices[0]]
    start_frame = 0

    for i in range(1, len(frame_indices)):
        label = chord_labels[frame_indices[i]]
        if label != current_label:
            segments.append(ChordSegment(
                start=round(start_frame / Fs_X, 6),
                end=round(i / Fs_X, 6),
                label=current_label,
            ))
            current_label = label
            start_frame = i

    segments.append(ChordSegment(
        start=round(start_frame / Fs_X, 6),
        end=round(len(frame_indices) / Fs_X, 6),
        label=current_label,
    ))
    return segments


class ChordRecognizer:
    """HMM-based chord recognizer.

    Encapsulates a transition matrix and chord vocabulary, exposing a
    clean recognize() interface over the underlying Viterbi decoder.

    Args:
        config: RecognizerConfig with chord_set and self-transition probability p.
        transition_matrix: Optional externally-derived transition matrix (e.g.
            from corpus analysis). If None, a uniform matrix is used.
    """

    def __init__(
        self,
        config: RecognizerConfig | None = None,
        transition_matrix: pd.DataFrame | None = None,
    ) -> None:
        self.config = config or RecognizerConfig()
        self._external_matrix = transition_matrix
        self.chord_labels = get_chord_labels(self.config.chord_set)
        self._A = self._build_transition_matrix()

    def _build_transition_matrix(self) -> np.ndarray:
        n = len(self.chord_labels)
        if self._external_matrix is None:
            return uniform_transition_matrix(p=self.config.p, N=n)
        return edit_diagonal(self._external_matrix, self.config.p).to_numpy()

    def recognize(
        self,
        chromagram: np.ndarray,
        filt_len: int | None = None,
        filt_type: Literal['mean', 'median'] = 'mean',
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run chord recognition on a chromagram.

        Args:
            chromagram: Chromagram matrix (12 x N_frames).
            filt_len: Pre-filter length. No filtering applied if None.
            filt_type: Pre-filter type ('mean' or 'median').

        Returns:
            chord_hmm: HMM-based result as binary matrix (num_chords x N_frames).
            chord_template: Template-based result as binary matrix.
            chord_sim: Chord similarity matrix.
        """
        X = chromagram
        if filt_len is not None:
            if filt_type == 'mean':
                X, _ = libfmp.c3.smooth_downsample_feature_sequence(
                    X, Fs=1, filt_len=filt_len, down_sampling=1,
                )
            else:
                X, _ = libfmp.c3.median_downsample_feature_sequence(
                    X, Fs=1, filt_len=filt_len, down_sampling=1,
                )

        chord_sim, chord_template = chord_recognition_template(
            X, norm_sim='1', chord_set=self.config.chord_set,
        )

        n = len(self._A)
        C = (1 / n) * np.ones((1, n))
        chord_hmm, _, _, _ = viterbi_log_likelihood(self._A, C, chord_sim)

        return chord_hmm, chord_template, chord_sim

    def find_best_p(
        self,
        chromagram: np.ndarray,
        ann_matrix: np.ndarray,
        p_min: float = 0.0,
        p_max: float = 1.0,
        steps: int = 50,
    ) -> float:
        """Find the self-transition probability p that maximises F-measure.

        Searches linearly between p_min and p_max. Does not mutate the
        recognizer's current configuration.

        Args:
            chromagram: Chromagram matrix (12 x N_frames).
            ann_matrix: Binary reference annotation matrix (num_chords x N_frames).
            p_min: Lower bound for p search.
            p_max: Upper bound for p search.
            steps: Number of candidate p values to evaluate.

        Returns:
            The p value that achieved the highest F-measure.
        """
        from chord_rec.evaluation import compute_eval_measures

        p_values = [p_min + (p_max - p_min) * i / steps for i in range(steps + 1)]
        chord_sim, _ = chord_recognition_template(
            chromagram, norm_sim='1', chord_set=self.config.chord_set,
        )
        n = len(self.chord_labels)
        C = (1 / n) * np.ones((1, n))

        best_p, best_f = self.config.p, -1.0
        for p in p_values:
            A = (
                uniform_transition_matrix(p=p, N=n)
                if self._external_matrix is None
                else edit_diagonal(self._external_matrix, p).to_numpy()
            )
            chord_hmm, _, _, _ = viterbi_log_likelihood(A, C, chord_sim)
            result = compute_eval_measures(ann_matrix, chord_hmm)
            if result.f_measure > best_f:
                best_f = result.f_measure
                best_p = p

        return best_p

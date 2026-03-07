import numpy as np
import pandas as pd
import pytest

from chord_rec.recognition import (
    ChordRecognizer,
    ChordSegment,
    RecognizerConfig,
    chord_recognition_template,
    decode_chord_sequence,
    edit_diagonal,
    generate_chord_templates,
    get_chord_labels,
    matrix_chord24_trans_inv,
    matrix_circular_mean,
    uniform_transition_matrix,
)


class TestGetChordLabels:
    def test_basic_returns_24_labels(self):
        labels = get_chord_labels("basic")
        assert len(labels) == 24

    def test_extended_returns_120_labels(self):
        labels = get_chord_labels("extended")
        assert len(labels) == 120

    def test_nonchord_appends_n(self):
        labels = get_chord_labels("basic", nonchord=True)
        assert labels[-1] == "N"
        assert len(labels) == 25

    def test_basic_starts_with_c_major(self):
        labels = get_chord_labels("basic")
        assert labels[0] == "C"

    def test_basic_contains_c_minor(self):
        labels = get_chord_labels("basic", ext_minor="m")
        assert "Cm" in labels

    def test_extended_contains_major_seventh(self):
        labels = get_chord_labels("extended")
        assert "CM7" in labels

    def test_extended_does_not_duplicate_basic_chords(self):
        labels = get_chord_labels("extended")
        assert labels.count("C") == 1
        assert labels.count("Cm") == 1


class TestGenerateChordTemplates:
    def test_basic_shape(self):
        templates = generate_chord_templates("basic")
        assert templates.shape == (12, 24)

    def test_basic_shape_with_nonchord(self):
        templates = generate_chord_templates("basic", nonchord=True)
        assert templates.shape == (12, 25)

    def test_extended_shape(self):
        templates = generate_chord_templates("extended")
        assert templates.shape == (12, 120)

    def test_values_are_binary(self):
        for chord_set in ("basic", "extended"):
            templates = generate_chord_templates(chord_set)
            unique = np.unique(templates)
            assert set(unique).issubset({0, 1})

    def test_c_major_template(self):
        templates = generate_chord_templates("basic")
        # C major: C(0), E(4), G(7)
        c_major = templates[:, 0]
        assert c_major[0] == 1  # C
        assert c_major[4] == 1  # E
        assert c_major[7] == 1  # G
        assert c_major.sum() == 3

    def test_c_minor_template(self):
        templates = generate_chord_templates("basic")
        # C minor: C(0), Eb(3), G(7)
        c_minor = templates[:, 12]
        assert c_minor[0] == 1  # C
        assert c_minor[3] == 1  # Eb
        assert c_minor[7] == 1  # G
        assert c_minor.sum() == 3


class TestUniformTransitionMatrix:
    def test_shape(self):
        A = uniform_transition_matrix(p=0.5, N=24)
        assert A.shape == (24, 24)

    def test_rows_sum_to_one(self):
        A = uniform_transition_matrix(p=0.3, N=24)
        np.testing.assert_allclose(A.sum(axis=1), np.ones(24))

    def test_diagonal_equals_p(self):
        p = 0.7
        A = uniform_transition_matrix(p=p, N=24)
        np.testing.assert_allclose(np.diag(A), p)

    def test_off_diagonal_uniform(self):
        p = 0.4
        N = 24
        A = uniform_transition_matrix(p=p, N=N)
        expected_off = (1 - p) / (N - 1)
        for i in range(N):
            for j in range(N):
                if i != j:
                    assert A[i, j] == pytest.approx(expected_off)


class TestEditDiagonal:
    def _uniform_df(self, N=4, p=0.1):
        A = uniform_transition_matrix(p=p, N=N)
        return pd.DataFrame(A)

    def test_rows_still_sum_to_one(self):
        df = self._uniform_df()
        result = edit_diagonal(df, 0.6)
        np.testing.assert_allclose(result.values.sum(axis=1), np.ones(4), atol=1e-10)

    def test_diagonal_set_to_new_p(self):
        df = self._uniform_df()
        new_p = 0.6
        result = edit_diagonal(df, new_p)
        np.testing.assert_allclose(np.diag(result.values), new_p)

    def test_does_not_mutate_original(self):
        df = self._uniform_df(p=0.1)
        original_diag = np.diag(df.values).copy()
        edit_diagonal(df, 0.9)
        np.testing.assert_allclose(np.diag(df.values), original_diag)


class TestMatrixCircularMean:
    def test_output_shape(self):
        A = np.random.rand(12, 12)
        result = matrix_circular_mean(A)
        assert result.shape == (12, 12)

    def test_rows_sum_preserved(self):
        A = uniform_transition_matrix(p=0.5, N=12)
        result = matrix_circular_mean(A)
        np.testing.assert_allclose(result.sum(axis=1), A.sum(axis=1), atol=1e-10)


class TestMatrixChord24TransInv:
    def test_output_shape(self):
        A = uniform_transition_matrix(p=0.5, N=24)
        result = matrix_chord24_trans_inv(A)
        assert result.shape == (24, 24)

    def test_uniform_matrix_is_unchanged(self):
        A = uniform_transition_matrix(p=0.5, N=24)
        result = matrix_chord24_trans_inv(A)
        np.testing.assert_allclose(result, A, atol=1e-10)


class TestChordRecognitionTemplate:
    def _synthetic_chromagram(self, dominant_pitch: int, n_frames: int = 10) -> np.ndarray:
        X = np.zeros((12, n_frames))
        X[dominant_pitch, :] = 1.0
        return X

    def test_output_shapes(self):
        X = self._synthetic_chromagram(0)
        chord_sim, chord_max = chord_recognition_template(X)
        assert chord_sim.shape[1] == X.shape[1]
        assert chord_max.shape == chord_sim.shape

    def test_chord_max_is_binary(self):
        X = self._synthetic_chromagram(0)
        _, chord_max = chord_recognition_template(X)
        assert set(np.unique(chord_max)).issubset({0, 1})

    def test_chord_max_has_one_per_frame(self):
        X = self._synthetic_chromagram(0)
        _, chord_max = chord_recognition_template(X)
        np.testing.assert_array_equal(chord_max.sum(axis=0), np.ones(X.shape[1]))


class TestChordRecognizer:
    def _synthetic_chromagram(self, n_frames: int = 20) -> np.ndarray:
        rng = np.random.default_rng(42)
        return rng.random((12, n_frames))

    def test_default_config(self):
        recognizer = ChordRecognizer()
        assert recognizer.config.chord_set == "basic"
        assert recognizer.config.p == 0.15

    def test_basic_chord_labels_count(self):
        recognizer = ChordRecognizer()
        assert len(recognizer.chord_labels) == 24

    def test_extended_chord_labels_count(self):
        recognizer = ChordRecognizer(config=RecognizerConfig(chord_set="extended"))
        assert len(recognizer.chord_labels) == 120

    def test_recognize_output_shapes(self):
        X = self._synthetic_chromagram()
        recognizer = ChordRecognizer()
        chord_hmm, chord_template, chord_sim = recognizer.recognize(X)
        assert chord_hmm.shape == (24, 20)
        assert chord_template.shape == (24, 20)
        assert chord_sim.shape[1] == 20

    def test_recognize_hmm_is_binary(self):
        X = self._synthetic_chromagram()
        recognizer = ChordRecognizer()
        chord_hmm, _, _ = recognizer.recognize(X)
        assert set(np.unique(chord_hmm)).issubset({0, 1})

    def test_recognizer_with_external_transition_matrix(self):
        A = uniform_transition_matrix(p=0.5, N=24)
        df = pd.DataFrame(A)
        recognizer = ChordRecognizer(
            config=RecognizerConfig(p=0.5),
            transition_matrix=df,
        )
        X = self._synthetic_chromagram()
        chord_hmm, _, _ = recognizer.recognize(X)
        assert chord_hmm.shape == (24, 20)

    def test_config_immutability(self):
        recognizer = ChordRecognizer(config=RecognizerConfig(p=0.3))
        original_p = recognizer.config.p
        recognizer.find_best_p(
            self._synthetic_chromagram(),
            np.eye(24, 20, dtype=int),
            steps=5,
        )
        assert recognizer.config.p == original_p


class TestDecodeChordSequence:
    def _matrix_from_sequence(self, sequence: list[str], labels: list[str]) -> np.ndarray:
        matrix = np.zeros((len(labels), len(sequence)), dtype=int)
        for frame, label in enumerate(sequence):
            matrix[labels.index(label), frame] = 1
        return matrix

    def test_single_chord_throughout(self):
        labels = ['C', 'Am', 'F', 'G']
        matrix = self._matrix_from_sequence(['C'] * 10, labels)
        segments = decode_chord_sequence(matrix, labels, Fs_X=10.0)
        assert len(segments) == 1
        assert segments[0].label == 'C'
        assert segments[0].start == 0.0
        assert segments[0].end == pytest.approx(1.0)

    def test_two_chords(self):
        labels = ['C', 'Am', 'F', 'G']
        matrix = self._matrix_from_sequence(['C'] * 5 + ['Am'] * 5, labels)
        segments = decode_chord_sequence(matrix, labels, Fs_X=10.0)
        assert len(segments) == 2
        assert segments[0].label == 'C'
        assert segments[1].label == 'Am'
        assert segments[0].end == pytest.approx(segments[1].start)

    def test_segments_cover_full_duration(self):
        labels = ['C', 'Am', 'F', 'G']
        sequence = ['C', 'C', 'Am', 'Am', 'F', 'G']
        matrix = self._matrix_from_sequence(sequence, labels)
        Fs_X = 2.0
        segments = decode_chord_sequence(matrix, labels, Fs_X=Fs_X)
        assert segments[0].start == 0.0
        assert segments[-1].end == pytest.approx(len(sequence) / Fs_X)

    def test_duration_computed_correctly(self):
        labels = ['C', 'Am']
        matrix = self._matrix_from_sequence(['C'] * 4, labels)
        segments = decode_chord_sequence(matrix, labels, Fs_X=2.0)
        assert segments[0].duration == pytest.approx(2.0)

    def test_returns_chord_segment_instances(self):
        labels = ['C', 'Am']
        matrix = self._matrix_from_sequence(['C', 'Am'], labels)
        segments = decode_chord_sequence(matrix, labels, Fs_X=1.0)
        assert all(isinstance(s, ChordSegment) for s in segments)

    def test_empty_matrix_returns_empty_list(self):
        labels = ['C', 'Am']
        matrix = np.zeros((2, 0), dtype=int)
        segments = decode_chord_sequence(matrix, labels, Fs_X=10.0)
        assert segments == []

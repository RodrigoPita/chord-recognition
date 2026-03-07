import numpy as np
import pytest

from chord_rec.evaluation import (
    EvaluationResult,
    compute_eval_measures,
    compute_mean_f_measure,
    convert_chord_label,
    convert_sequence_ann,
)


class TestConvertChordLabel:
    def test_converts_min_suffix(self):
        ann = [[0.0, 1.0, "A:min"]]
        result = convert_chord_label(ann)
        assert result[0][2] == "Am"

    def test_converts_flat_to_sharp(self):
        cases = [
            ("Db", "C#"), ("Eb", "D#"), ("Gb", "F#"), ("Ab", "G#"), ("Bb", "A#"),
        ]
        for flat, sharp in cases:
            ann = [[0.0, 1.0, flat]]
            result = convert_chord_label(ann)
            assert result[0][2] == sharp

    def test_converts_flat_minor(self):
        ann = [[0.0, 1.0, "Bb:min"]]
        result = convert_chord_label(ann)
        assert result[0][2] == "A#m"

    def test_does_not_mutate_original(self):
        ann = [[0.0, 1.0, "Bb"]]
        convert_chord_label(ann)
        assert ann[0][2] == "Bb"

    def test_leaves_sharp_labels_unchanged(self):
        ann = [[0.0, 1.0, "C#"], [1.0, 2.0, "Am"]]
        result = convert_chord_label(ann)
        assert result[0][2] == "C#"
        assert result[1][2] == "Am"


class TestConvertSequenceAnn:
    def test_length_matches_sequence(self):
        seq = ["C", "Am", "F", "G"]
        result = convert_sequence_ann(seq)
        assert len(result) == 4

    def test_labels_are_preserved(self):
        seq = ["C", "Am"]
        result = convert_sequence_ann(seq)
        assert result[0][2] == "C"
        assert result[1][2] == "Am"

    def test_segments_are_centred_on_frame(self):
        seq = ["C"]
        result = convert_sequence_ann(seq, Fs=1)
        assert result[0][0] == -0.5
        assert result[0][1] == 0.5


class TestComputeEvalMeasures:
    def _make_matrix(self, rows, cols, ones: list[tuple[int, int]]):
        m = np.zeros((rows, cols), dtype=int)
        for r, c in ones:
            m[r, c] = 1
        return m

    def test_perfect_match(self):
        m = self._make_matrix(4, 4, [(0, 0), (1, 1), (2, 2), (3, 3)])
        result = compute_eval_measures(m, m.copy())
        assert result.precision == pytest.approx(1.0)
        assert result.recall == pytest.approx(1.0)
        assert result.f_measure == pytest.approx(1.0)
        assert result.true_positives == 4
        assert result.false_positives == 0
        assert result.false_negatives == 0

    def test_no_match(self):
        ref = self._make_matrix(4, 4, [(0, 0)])
        est = self._make_matrix(4, 4, [(1, 0)])
        result = compute_eval_measures(ref, est)
        assert result.precision == 0.0
        assert result.recall == 0.0
        assert result.f_measure == 0.0
        assert result.true_positives == 0

    def test_partial_match(self):
        ref = self._make_matrix(4, 4, [(0, 0), (1, 1), (2, 2), (3, 3)])
        est = self._make_matrix(4, 4, [(0, 0), (1, 1)])
        result = compute_eval_measures(ref, est)
        assert result.true_positives == 2
        assert result.false_negatives == 2
        assert result.false_positives == 0
        assert result.recall == pytest.approx(0.5)
        assert result.precision == pytest.approx(1.0)

    def test_shape_mismatch_raises(self):
        ref = np.zeros((4, 4))
        est = np.zeros((4, 5))
        with pytest.raises(AssertionError):
            compute_eval_measures(ref, est)

    def test_returns_evaluation_result_model(self):
        m = np.eye(3, dtype=int)
        result = compute_eval_measures(m, m)
        assert isinstance(result, EvaluationResult)

    def test_str_representation(self):
        result = EvaluationResult(
            precision=0.8, recall=0.6, f_measure=0.686,
            true_positives=6, false_positives=2, false_negatives=4,
        )
        s = str(result)
        assert "P=0.800" in s
        assert "R=0.600" in s
        assert "F=0.686" in s


class TestComputeMeanFMeasure:
    def test_mean_of_identical_arrays(self):
        arr = np.array([1.0, 0.5, 0.8])
        result = compute_mean_f_measure({0: arr, 1: arr}, [0, 1])
        np.testing.assert_allclose(result, arr)

    def test_mean_of_two_arrays(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        result = compute_mean_f_measure({0: a, 1: b}, [0, 1])
        np.testing.assert_allclose(result, [0.5, 0.5])

# tests/test_metrics.py
"""
Unit tests for evaluation metrics.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from src.evaluation.metrics import (
    acc1, acc2, p_score, mean_absolute_error,
    compute_tempo_metrics, compute_beat_metrics, aggregate_results
)


class TestTempoMetrics:

    def test_acc1_exact(self):
        assert acc1(120.0, 120.0) == 1.0

    def test_acc1_within_tolerance(self):
        assert acc1(122.0, 120.0) == 1.0   # 1.67% error < 4%

    def test_acc1_outside_tolerance(self):
        assert acc1(130.0, 120.0) == 0.0   # 8.3% error > 4%

    def test_acc1_zero_reference(self):
        assert acc1(120.0, 0.0) == 0.0

    def test_acc2_octave_double(self):
        # 240 is 2x of 120 — ACC2 should accept this
        assert acc2(240.0, 120.0) == 1.0

    def test_acc2_octave_half(self):
        # 60 is 0.5x of 120 — ACC2 should accept this
        assert acc2(60.0, 120.0) == 1.0

    def test_acc2_wrong_octave(self):
        assert acc2(150.0, 120.0) == 0.0

    def test_p_score_exact(self):
        assert p_score(120.0, 120.0) == 1.0

    def test_p_score_range(self):
        score = p_score(118.0, 120.0)
        assert 0.0 <= score <= 1.0

    def test_mae_basic(self):
        est = np.array([120.0, 130.0, 90.0])
        ref = np.array([120.0, 120.0, 100.0])
        mae = mean_absolute_error(est, ref)
        assert abs(mae - 10.0/3.0 * 2) < 1e-5   # (0+10+10)/3

    def test_mae_ignores_zero_reference(self):
        est = np.array([120.0, 0.0])
        ref = np.array([120.0, 0.0])
        mae = mean_absolute_error(est, ref)
        assert mae == 0.0   # Only first entry counts

    def test_compute_tempo_metrics_keys(self):
        result = compute_tempo_metrics(120.0, 120.0)
        for key in ["acc1", "acc2", "p_score", "abs_error"]:
            assert key in result

    def test_compute_tempo_metrics_perfect(self):
        result = compute_tempo_metrics(120.0, 120.0)
        assert result["acc1"]     == 1.0
        assert result["acc2"]     == 1.0
        assert result["abs_error"] == 0.0


class TestBeatMetrics:

    def test_perfect_beats(self):
        ref = np.arange(0.5, 10.0, 0.5)   # 120 BPM
        est = ref.copy()
        result = compute_beat_metrics(est, ref)
        assert result["f_measure"] > 0.95
        assert result["precision"] > 0.95
        assert result["recall"]    > 0.95

    def test_empty_estimated(self):
        ref    = np.arange(0.5, 5.0, 0.5)
        result = compute_beat_metrics(np.array([]), ref)
        assert result["f_measure"] == 0.0

    def test_empty_reference(self):
        est    = np.arange(0.5, 5.0, 0.5)
        result = compute_beat_metrics(est, np.array([]))
        assert result["f_measure"] == 0.0

    def test_result_keys(self):
        ref    = np.arange(0.5, 5.0, 0.5)
        result = compute_beat_metrics(ref, ref)
        for key in ["f_measure", "precision", "recall", "cemgil", "information_gain"]:
            assert key in result

    def test_f_measure_range(self):
        ref    = np.arange(0.5, 10.0, 0.5)
        est    = ref + 0.03   # 30ms offset — within 70ms window
        result = compute_beat_metrics(est, ref)
        assert 0.0 <= result["f_measure"] <= 1.0


class TestAggregateResults:

    def test_empty_returns_empty(self):
        assert aggregate_results([]) == {}

    def test_basic_aggregate(self):
        results = [{"acc1": 1.0, "acc2": 1.0}, {"acc1": 0.5, "acc2": 0.8}]
        agg     = aggregate_results(results)
        assert abs(agg["acc1_mean"] - 0.75) < 1e-9
        assert "acc1_std" in agg
        assert agg["acc1_n"] == 2

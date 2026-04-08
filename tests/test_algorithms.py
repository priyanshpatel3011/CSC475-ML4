# tests/test_algorithms.py
"""
Unit tests for all three beat tracking algorithms.
Tests use synthetic signals (pure sine beat grid) to verify correctness.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
import tempfile
import soundfile as sf

from src.algorithms.autocorrelation import AutocorrelationTracker
from src.algorithms.state_space     import StateSpaceTracker


# ── Fixtures ──────────────────────────────────────────────────────────────────

SR        = 22050
BPM_TRUE  = 120.0
DURATION  = 10.0   # seconds


def make_click_track(bpm: float, sr: int = SR, duration: float = DURATION) -> np.ndarray:
    """
    Generate a synthetic click track at a fixed BPM.
    Returns an audio array with impulses at every beat position.
    """
    n_samples    = int(sr * duration)
    y            = np.zeros(n_samples)
    beat_period  = sr * 60.0 / bpm
    beat_samples = np.arange(0, n_samples, beat_period).astype(int)
    beat_samples = beat_samples[beat_samples < n_samples]

    # Gaussian click
    click_width = int(sr * 0.005)  # 5ms
    t_click     = np.exp(-0.5 * (np.arange(-click_width, click_width) / (click_width * 0.3)) ** 2)

    for s in beat_samples:
        lo = max(0, s - click_width)
        hi = min(n_samples, s + click_width)
        y[lo:hi] += t_click[:hi - lo]

    return y / (np.max(np.abs(y)) + 1e-8)


@pytest.fixture
def click_wav(tmp_path):
    """Write synthetic click track to a temp WAV file and return the path."""
    y    = make_click_track(BPM_TRUE, sr=SR)
    path = str(tmp_path / "click.wav")
    sf.write(path, y, SR)
    return path, BPM_TRUE


# ── Autocorrelation Tests ─────────────────────────────────────────────────────

class TestAutocorrelationTracker:

    def test_returns_tuple(self, click_wav):
        path, _ = click_wav
        tracker = AutocorrelationTracker(sr=SR)
        result  = tracker.predict(path)
        assert isinstance(result, tuple) and len(result) == 2

    def test_tempo_within_tolerance(self, click_wav):
        """Estimated tempo should be within 10% of true BPM on a click track."""
        path, true_bpm = click_wav
        tracker = AutocorrelationTracker(sr=SR)
        est_tempo, _ = tracker.predict(path)
        err1 = abs(est_tempo - true_bpm) / true_bpm < 0.10
        err2 = abs(est_tempo - true_bpm/2) / (true_bpm/2) < 0.10
        err3 = abs(est_tempo - true_bpm*2) / (true_bpm*2) < 0.10
        assert err1 or err2 or err3, (
            f"Expected ≈{true_bpm} (or octave) BPM, got {est_tempo:.2f} BPM"
        )

    def test_beat_times_are_sorted(self, click_wav):
        path, _ = click_wav
        tracker = AutocorrelationTracker(sr=SR)
        _, beats = tracker.predict(path)
        assert np.all(np.diff(beats) > 0), "Beat times must be strictly increasing"

    def test_beat_times_are_positive(self, click_wav):
        path, _ = click_wav
        tracker = AutocorrelationTracker(sr=SR)
        _, beats = tracker.predict(path)
        assert np.all(beats >= 0), "Beat times must be non-negative"

    def test_predict_from_audio(self):
        """Test predict_from_audio() with a numpy array input."""
        y       = make_click_track(BPM_TRUE, sr=SR)
        tracker = AutocorrelationTracker(sr=SR)
        tempo, beats = tracker.predict_from_audio(y, SR)
        assert isinstance(tempo, float)
        assert isinstance(beats, np.ndarray)

    def test_handles_silent_audio(self, tmp_path):
        """Silent audio should not raise, returning 0 beats."""
        y    = np.zeros(SR * 5)
        path = str(tmp_path / "silence.wav")
        sf.write(path, y, SR)
        tracker = AutocorrelationTracker(sr=SR)
        tempo, beats = tracker.predict(path)
        assert isinstance(tempo, float)

    def test_invalid_path_raises(self):
        tracker = AutocorrelationTracker(sr=SR)
        with pytest.raises((FileNotFoundError, RuntimeError)):
            tracker.predict("/nonexistent/path/audio.wav")


# ── State-Space Tests ─────────────────────────────────────────────────────────

class TestStateSpaceTracker:

    def test_returns_tuple(self, click_wav):
        path, _ = click_wav
        tracker = StateSpaceTracker(sr=SR)
        result  = tracker.predict(path)
        assert isinstance(result, tuple) and len(result) == 2

    def test_tempo_is_positive(self, click_wav):
        path, _ = click_wav
        tracker = StateSpaceTracker(sr=SR)
        tempo, _ = tracker.predict(path)
        assert tempo > 0, "Tempo should be positive"

    def test_beat_times_sorted(self, click_wav):
        path, _ = click_wav
        tracker = StateSpaceTracker(sr=SR)
        _, beats = tracker.predict(path)
        if len(beats) > 1:
            assert np.all(np.diff(beats) > 0)

    def test_predict_from_audio(self):
        y       = make_click_track(BPM_TRUE, sr=SR)
        tracker = StateSpaceTracker(sr=SR)
        tempo, beats = tracker.predict_from_audio(y, SR)
        assert isinstance(tempo, float)
        assert isinstance(beats, np.ndarray)

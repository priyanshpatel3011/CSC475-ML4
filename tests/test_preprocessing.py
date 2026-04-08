# tests/test_preprocessing.py
"""
Unit tests for audio utilities and dataset loading utilities.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
import soundfile as sf

from src.utils.audio import load_audio, compute_onset_envelope, compute_mel_spectrogram
from src.utils.dataset import split_tracks, Track


SR = 22050


@pytest.fixture
def wav_file(tmp_path):
    """Create a short test WAV file."""
    y    = np.random.randn(SR * 3).astype(np.float32)
    path = str(tmp_path / "test.wav")
    sf.write(path, y, SR)
    return path


class TestAudioUtils:

    def test_load_returns_correct_shape(self, wav_file):
        y, sr = load_audio(wav_file, sr=SR, mono=True)
        assert sr == SR
        assert y.ndim == 1
        assert len(y) == SR * 3

    def test_load_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_audio("/nonexistent/file.wav")

    def test_onset_envelope_shape(self, wav_file):
        y, sr = load_audio(wav_file, sr=SR)
        env   = compute_onset_envelope(y, sr)
        assert env.ndim == 1
        assert len(env) > 0

    def test_mel_spectrogram_shape(self, wav_file):
        y, sr = load_audio(wav_file, sr=SR)
        mel   = compute_mel_spectrogram(y, sr, n_mels=64)
        assert mel.ndim == 2
        assert mel.shape[0] == 64   # n_mels


class TestDatasetSplit:

    def _make_tracks(self, n: int, genres=("rock", "jazz", "blues")):
        tracks = []
        for i in range(n):
            genre = genres[i % len(genres)]
            tracks.append(Track(
                track_id=f"track_{i:03d}",
                audio_path=f"/fake/path/{i}.wav",
                tempo=120.0,
                genre=genre,
                dataset="gtzan"
            ))
        return tracks

    def test_split_ratios_approximate(self):
        tracks = self._make_tracks(100)
        train, val, test = split_tracks(tracks, train_ratio=0.7,
                                         val_ratio=0.15, test_ratio=0.15)
        assert abs(len(train) - 70) <= 3
        assert abs(len(val)   - 15) <= 3
        assert abs(len(test)  - 15) <= 3

    def test_split_no_overlap(self):
        tracks = self._make_tracks(90)
        train, val, test = split_tracks(tracks)
        train_ids = {t.track_id for t in train}
        val_ids   = {t.track_id for t in val}
        test_ids  = {t.track_id for t in test}
        assert len(train_ids & val_ids)  == 0
        assert len(train_ids & test_ids) == 0
        assert len(val_ids   & test_ids) == 0

    def test_split_covers_all_tracks(self):
        tracks = self._make_tracks(90)
        train, val, test = split_tracks(tracks)
        total = len(train) + len(val) + len(test)
        assert total == 90

    def test_split_reproducible(self):
        tracks = self._make_tracks(60)
        _, _, test1 = split_tracks(tracks, seed=42)
        _, _, test2 = split_tracks(tracks, seed=42)
        ids1 = [t.track_id for t in test1]
        ids2 = [t.track_id for t in test2]
        assert ids1 == ids2

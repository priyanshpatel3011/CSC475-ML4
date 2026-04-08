# src/algorithms/autocorrelation.py
"""
Algorithm 1: Autocorrelation-Based Tempo Estimation and Beat Tracking
=====================================================================
Classical signal-processing baseline following Percival & Tzanetakis (2014).

Pipeline:
    1. Compute onset strength envelope
    2. Autocorrelation of the envelope
    3. Generate pulse trains at candidate tempos
    4. Cross-correlation between envelope and each pulse train
    5. Select highest-scoring tempo
    6. Derive beat positions by peak-picking on the onset envelope
"""

import logging
import time
from typing import Tuple

import numpy as np
import scipy.signal

try:
    import librosa
    _HAS_LIBROSA = True
except ImportError:
    librosa = None
    _HAS_LIBROSA = False

from src.utils.config import (
    SAMPLE_RATE, HOP_LENGTH, TEMPO_MIN, TEMPO_MAX
)

logger = logging.getLogger(__name__)


class AutocorrelationTracker:
    """
    Tempo estimator and beat tracker based on onset autocorrelation
    and pulse cross-correlation.

    Usage:
        tracker = AutocorrelationTracker()
        tempo, beat_times = tracker.predict(audio_path)
    """

    def __init__(self,
                 sr:          int   = SAMPLE_RATE,
                 hop_length:  int   = HOP_LENGTH,
                 tempo_min:   float = TEMPO_MIN,
                 tempo_max:   float = TEMPO_MAX,
                 n_candidates: int  = 200):
        """
        Args:
            sr:           Target sample rate.
            hop_length:   Hop size for onset envelope.
            tempo_min:    Minimum BPM to consider.
            tempo_max:    Maximum BPM to consider.
            n_candidates: Number of BPM candidates to evaluate.
        """
        self.sr           = sr
        self.hop_length   = hop_length
        self.tempo_min    = tempo_min
        self.tempo_max    = tempo_max
        self.n_candidates = n_candidates

        # Pre-compute candidate BPM grid
        self.bpm_candidates = np.linspace(tempo_min, tempo_max, n_candidates)

    # ── Public Interface ──────────────────────────────────────────────────────

    def predict(self, audio_path: str) -> Tuple[float, np.ndarray]:
        """
        Estimate global tempo and beat positions for an audio file.

        Args:
            audio_path: Path to audio file.

        Returns:
            (tempo, beat_times):
                tempo      — estimated BPM (float)
                beat_times — beat positions in seconds (1-D numpy array)
        """
        if not _HAS_LIBROSA:
            raise ImportError("librosa is required. pip install librosa")
        t_start = time.perf_counter()

        y, sr = librosa.load(audio_path, sr=self.sr, mono=True)
        tempo, beat_times = self._process(y, sr)

        elapsed = time.perf_counter() - t_start
        logger.debug("AutocorrTracker: %.2f BPM | %d beats | %.3fs",
                     tempo, len(beat_times), elapsed)
        return tempo, beat_times

    def predict_from_audio(self, y: np.ndarray, sr: int) -> Tuple[float, np.ndarray]:
        """
        Estimate tempo and beats directly from a pre-loaded audio array.

        Args:
            y:  Audio time series.
            sr: Sample rate.

        Returns:
            (tempo, beat_times)
        """
        return self._process(y, sr)

    # ── Core Processing ───────────────────────────────────────────────────────

    def _process(self, y: np.ndarray, sr: int) -> Tuple[float, np.ndarray]:
        # Step 1 — Onset strength envelope
        envelope = self._compute_onset_envelope(y, sr)

        # Step 2 — Autocorrelation
        autocorr = self._autocorrelate(envelope)

        # Step 3 & 4 — Pulse cross-correlation
        tempo = self._estimate_tempo_via_pulse_xcorr(autocorr, sr)

        # Step 5 — Beat positions via peak-picking
        beat_times = self._estimate_beats(y, sr, tempo)

        return tempo, beat_times

    def _compute_onset_envelope(self, y: np.ndarray, sr: int) -> np.ndarray:
        """
        Compute onset strength envelope.
        Uses librosa when available; falls back to an energy-flux envelope
        so the module runs in minimal environments.
        """
        if _HAS_LIBROSA:
            envelope = librosa.onset.onset_strength(
                y=y, sr=sr, hop_length=self.hop_length
            )
        else:
            # Energy-flux fallback: RMS energy per hop, positive differences
            hop = self.hop_length
            n_frames = len(y) // hop
            rms = np.array([
                float(np.sqrt(np.mean(y[i*hop:(i+1)*hop] ** 2)))
                for i in range(n_frames)
            ])
            flux = np.diff(rms, prepend=rms[0])
            envelope = np.maximum(flux, 0.0)

        if envelope.max() > 0:
            envelope = envelope / envelope.max()
        return envelope

    def _autocorrelate(self, envelope: np.ndarray) -> np.ndarray:
        """
        Full autocorrelation of the onset envelope.
        Returns the positive-lag half only.
        """
        n = len(envelope)
        # Zero-pad to avoid circular correlation
        padded = np.zeros(2 * n)
        padded[:n] = envelope
        fft_env = np.fft.rfft(padded)
        autocorr = np.fft.irfft(fft_env * np.conj(fft_env))
        autocorr = autocorr[:n]
        # Normalise
        if autocorr[0] > 0:
            autocorr = autocorr / autocorr[0]
        return autocorr

    def _estimate_tempo_via_pulse_xcorr(self,
                                         autocorr: np.ndarray,
                                         sr: int) -> float:
        """
        For each candidate BPM, build a pulse train and compute its
        cross-correlation with the autocorrelation signal.
        Select the BPM with the highest score.
        """
        frames_per_second = sr / self.hop_length
        scores = np.zeros(self.n_candidates)

        for i, bpm in enumerate(self.bpm_candidates):
            period_frames = int(round(frames_per_second * 60.0 / bpm))
            if period_frames <= 0 or period_frames >= len(autocorr):
                continue
            # Build impulse train
            pulse = np.zeros(len(autocorr))
            for k in range(0, len(autocorr), period_frames):
                pulse[k] = 1.0
            # Score = dot product (fast xcorr at lag 0)
            scores[i] = float(np.dot(autocorr, pulse))

        best_idx = int(np.argmax(scores))
        return float(self.bpm_candidates[best_idx])

    def _estimate_beats(self, y: np.ndarray, sr: int, tempo: float) -> np.ndarray:
        """
        Estimate beat times by picking onset envelope peaks spaced
        approximately one beat apart.
        Uses librosa.beat.beat_track when available; falls back to
        evenly-spaced beat positions derived from the estimated tempo.
        """
        if _HAS_LIBROSA:
            _, beat_frames = librosa.beat.beat_track(
                y=y, sr=sr, hop_length=self.hop_length,
                start_bpm=tempo, units="frames"
            )
            beat_times = librosa.frames_to_time(
                beat_frames, sr=sr, hop_length=self.hop_length
            )
        else:
            # Fallback: evenly-spaced beats at estimated tempo
            duration        = len(y) / sr
            beat_interval   = 60.0 / tempo
            beat_times      = np.arange(0.0, duration, beat_interval)
        return beat_times

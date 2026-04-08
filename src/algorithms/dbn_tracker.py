# src/algorithms/dbn_tracker.py
"""
Algorithm 2: Dynamic Bayesian Network (DBN) Beat Tracker
=========================================================
Uses madmom's RNN beat activation function with a DBN and particle
filtering for probabilistic beat state estimation.

Fixes vs original:
    - predict_from_audio added (was missing, breaking unified interface)
"""

import logging
import time
from typing import Tuple

import numpy as np

from src.utils.config import SAMPLE_RATE, TEMPO_MIN, TEMPO_MAX

logger = logging.getLogger(__name__)


class DBNBeatTracker:
    """
    Beat tracker and tempo estimator using madmom's DBN pipeline.

    Usage:
        tracker = DBNBeatTracker()
        tempo, beat_times = tracker.predict(audio_path)
    """

    def __init__(self,
                 sr:                int   = SAMPLE_RATE,
                 tempo_min:         float = TEMPO_MIN,
                 tempo_max:         float = TEMPO_MAX,
                 fps:               int   = 100,
                 transition_lambda: int   = 100):
        self.sr                = sr
        self.tempo_min         = tempo_min
        self.tempo_max         = tempo_max
        self.fps               = fps
        self.transition_lambda = transition_lambda
        self._processor        = None   # Lazy-loaded on first use

    def _load_processor(self):
        """Lazy-load madmom processor to avoid slow import at module level."""
        if self._processor is not None:
            return
        try:
            from madmom.features.beats import (
                RNNBeatProcessor,
                DBNBeatTrackingProcessor,
            )
            from madmom.processors import SequentialProcessor
            rnn = RNNBeatProcessor()
            dbn = DBNBeatTrackingProcessor(
                min_bpm=self.tempo_min,
                max_bpm=self.tempo_max,
                fps=self.fps,
                transition_lambda=self.transition_lambda,
            )
            self._processor = SequentialProcessor([rnn, dbn])
            logger.info("madmom DBN processor loaded successfully")
        except ImportError as exc:
            raise ImportError(
                "madmom is required for DBNBeatTracker.\n"
                "Install with: pip install madmom"
            ) from exc

    # ── Public Interface ──────────────────────────────────────────────────────

    def predict(self, audio_path: str) -> Tuple[float, np.ndarray]:
        """
        Estimate global tempo and beat positions from an audio file.

        Args:
            audio_path: Path to audio file (madmom handles loading).

        Returns:
            (tempo, beat_times): BPM float, beat positions in seconds.
        """
        self._load_processor()
        t_start    = time.perf_counter()
        beat_times = np.array(self._processor(audio_path), dtype=float)
        tempo      = self._tempo_from_beats(beat_times)
        logger.debug("DBNTracker: %.2f BPM | %d beats | %.3fs",
                     tempo, len(beat_times), time.perf_counter() - t_start)
        return tempo, beat_times

    def predict_from_audio(self, y: np.ndarray, sr: int) -> Tuple[float, np.ndarray]:
        """
        Estimate tempo and beats from a pre-loaded audio array.
        FIX: This method was missing in the original, breaking the
        unified algorithm interface used in Evaluator.

        Args:
            y:  Audio time series (mono float32/float64).
            sr: Sample rate.

        Returns:
            (tempo, beat_times)
        """
        import tempfile, soundfile as sf, os
        self._load_processor()
        # madmom requires a file path, so we write a temp wav
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            sf.write(tmp_path, y, sr)
            return self.predict(tmp_path)
        finally:
            os.unlink(tmp_path)

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _tempo_from_beats(beat_times: np.ndarray) -> float:
        """Derive global tempo from median inter-beat interval (IBI)."""
        if len(beat_times) < 2:
            return 0.0
        ibis       = np.diff(beat_times)
        median_ibi = float(np.median(ibis))
        if median_ibi <= 0:
            return 0.0
        return 60.0 / median_ibi

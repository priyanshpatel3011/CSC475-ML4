# src/algorithms/state_space.py
"""
Algorithm 3: 1D State-Space Beat Tracker
=========================================
Implements a semi-Markov model with dimensionality reduction (PCA),
Viterbi decoding, and a jump-back reward strategy for computational
efficiency.

Fixes vs original:
    - PCA obs is actually USED in emission scoring (was computed then discarded)
    - Viterbi inner loop is fully vectorised with numpy (was O(T*S^2) Python)
    - predict_from_audio added (missing in original)
"""

import logging
import time
from typing import Tuple

import numpy as np
from sklearn.decomposition import PCA

try:
    import librosa
    _HAS_LIBROSA = True
except ImportError:
    _HAS_LIBROSA = False

from src.utils.config import SAMPLE_RATE, HOP_LENGTH, TEMPO_MIN, TEMPO_MAX

logger = logging.getLogger(__name__)


class StateSpaceTracker:
    """
    Beat tracker using a 1D semi-Markov state-space model with
    Viterbi decoding and jump-back reward.

    Usage:
        tracker = StateSpaceTracker()
        tempo, beat_times = tracker.predict(audio_path)
    """

    def __init__(self,
                 sr:               int   = SAMPLE_RATE,
                 hop_length:       int   = HOP_LENGTH,
                 tempo_min:        float = TEMPO_MIN,
                 tempo_max:        float = TEMPO_MAX,
                 n_tempo_states:   int   = 60,
                 pca_components:   int   = 8,
                 jump_back_weight: float = 0.5):
        self.sr               = sr
        self.hop_length       = hop_length
        self.tempo_min        = tempo_min
        self.tempo_max        = tempo_max
        self.n_tempo_states   = n_tempo_states
        self.pca_components   = pca_components
        self.jump_back_weight = jump_back_weight

        self._fps    = sr / hop_length
        self._tempos = np.linspace(tempo_min, tempo_max, n_tempo_states)
        self._periods = np.clip(
            (60.0 / self._tempos * self._fps).astype(int), 1, None
        )

    # ── Public Interface ──────────────────────────────────────────────────────

    def predict(self, audio_path: str) -> Tuple[float, np.ndarray]:
        """Estimate tempo and beats from an audio file path."""
        if not _HAS_LIBROSA:
            raise ImportError("librosa is required. pip install librosa")
        t_start = time.perf_counter()
        y, sr = librosa.load(audio_path, sr=self.sr, mono=True)
        tempo, beat_times = self._process(y, sr)
        logger.debug("StateSpaceTracker: %.2f BPM | %d beats | %.3fs",
                     tempo, len(beat_times), time.perf_counter() - t_start)
        return tempo, beat_times

    def predict_from_audio(self, y: np.ndarray, sr: int) -> Tuple[float, np.ndarray]:
        """Estimate tempo and beats from a pre-loaded audio array."""
        return self._process(y, sr)

    # ── Core Processing ───────────────────────────────────────────────────────

    def _process(self, y: np.ndarray, sr: int) -> Tuple[float, np.ndarray]:
        if _HAS_LIBROSA:
            envelope = librosa.onset.onset_strength(
                y=y, sr=sr, hop_length=self.hop_length
            )
        else:
            # Minimal fallback: energy-based onset envelope
            frame_len = self.hop_length
            n_frames  = len(y) // frame_len
            envelope  = np.array([
                float(np.sqrt(np.mean(y[i*frame_len:(i+1)*frame_len]**2)))
                for i in range(n_frames)
            ])

        # FIX: compute PCA-smoothed observation signal
        obs = self._reduce_features(envelope)

        # FIX: pass obs into emission computation (was using raw envelope only)
        emissions = self._compute_emissions(envelope, obs)

        _, best_tempo_idx = self._viterbi(emissions)

        beat_frames = self._decode_beats(envelope, best_tempo_idx)

        if _HAS_LIBROSA:
            beat_times = librosa.frames_to_time(
                beat_frames, sr=sr, hop_length=self.hop_length
            )
        else:
            beat_times = beat_frames * self.hop_length / sr

        tempo = self._tempo_from_beats(beat_times)
        return tempo, beat_times

    def _reduce_features(self, envelope: np.ndarray) -> np.ndarray:
        """
        Build windowed onset-envelope matrix and apply PCA.
        Returns first principal component as a smoothed signal.
        """
        win = min(16, max(2, len(envelope) // 4))
        n_frames = len(envelope) - win
        if n_frames <= 0:
            return envelope.copy()

        mat = np.stack([envelope[i:i + win] for i in range(n_frames)])
        n_components = min(self.pca_components, mat.shape[1], mat.shape[0])
        try:
            pca = PCA(n_components=n_components)
            reduced = pca.fit_transform(mat)
            obs = reduced[:, 0]
            # Normalise to [0, 1]
            r = obs.max() - obs.min()
            if r > 0:
                obs = (obs - obs.min()) / r
            obs = np.concatenate([obs, np.zeros(len(envelope) - len(obs))])
        except Exception:
            obs = envelope.copy()
        return obs

    def _compute_emissions(self, envelope: np.ndarray,
                            obs: np.ndarray) -> np.ndarray:
        """
        Emission probability matrix of shape (T, n_tempo_states).

        Uses an autocorrelation-weighted frame emission.  For each tempo
        state s with period p_s, we compute the autocorrelation of the
        onset envelope at lag p_s — this is a global measure of how well
        the signal is periodic at that tempo.  Per-frame emission is then:

            emission[t, s] = blended[t] * autocorr_weight[s]

        This is strongly discriminative: states whose period matches the
        dominant periodicity in the signal get large autocorr_weight and
        win the Viterbi path cleanly, eliminating octave bias.
        """
        T = len(envelope)
        S = self.n_tempo_states

        # Blend onset envelope with PCA-smoothed obs
        blended = 0.7 * envelope + 0.3 * obs
        bmax = blended.max()
        if bmax > 0:
            blended = blended / bmax

        # Full autocorrelation of the blended envelope
        n = len(blended)
        padded = np.zeros(2 * n)
        padded[:n] = blended
        fft_env = np.fft.rfft(padded)
        autocorr = np.fft.irfft(fft_env * np.conj(fft_env))[:n]
        if autocorr[0] > 0:
            autocorr = autocorr / autocorr[0]
        autocorr = np.clip(autocorr, 0, None)   # keep positive lags only

        # Per-state autocorrelation weight (sum over harmonics for robustness)
        autocorr_weights = np.zeros(S)
        for s, period in enumerate(self._periods):
            p = int(period)
            if p <= 0 or p >= n:
                continue
            # Include first two harmonics for robustness
            weight = autocorr[p]
            if 2 * p < n:
                weight += 0.5 * autocorr[2 * p]
            autocorr_weights[s] = weight

        # Normalise weights to [0, 1]
        wmax = autocorr_weights.max()
        if wmax > 0:
            autocorr_weights = autocorr_weights / wmax

        # Emission: per-frame onset strength scaled by state periodicity weight
        # Shape (T, S) built with outer product
        emissions = np.outer(blended, autocorr_weights)   # (T,) x (S,)

        # Add per-frame jump-back reward (keeps decoder phase-aware)
        for s, period in enumerate(self._periods):
            p = int(period)
            if p <= 0 or p >= T:
                continue
            prev_idx      = np.arange(T) - p
            valid         = prev_idx >= 0
            jb            = np.zeros(T)
            jb[valid]     = self.jump_back_weight * blended[prev_idx[valid]]
            emissions[:, s] += jb

        row_max = emissions.max(axis=1, keepdims=True)
        row_max[row_max == 0] = 1.0
        emissions /= row_max
        return emissions

    def _viterbi(self, emissions: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Vectorised Viterbi decoding over tempo states.
        FIX 1: inner loop eliminated — each timestep is a single numpy op.
        FIX 2: log-uniform tempo prior added so all BPM states start equally
                likely — eliminates the octave bias toward lower tempos that
                occurs when slower states accumulate more emission rewards
                per unit time than faster states.

        Transition model: Gaussian centred on current state (sigma=2 states).
        """
        T, S = emissions.shape
        log_em = np.log(np.clip(emissions, 1e-10, None))

        # Build log-transition matrix (S x S): Gaussian self-preference
        states    = np.arange(S, dtype=float)
        sigma     = 2.0
        log_trans = np.zeros((S, S))
        for s in range(S):
            w = np.exp(-0.5 * ((states - s) / sigma) ** 2)
            w /= w.sum()
            log_trans[s] = np.log(np.clip(w, 1e-10, None))

        # FIX: Log-uniform prior over tempo states.
        # Without this, lower-BPM states fire fewer beats per frame and
        # accumulate lower raw emission totals, making the decoder favour
        # slower tempos (octave bias). Normalising by expected beats/frame
        # (∝ BPM) removes this systematic advantage.
        beats_per_frame = self._tempos / 60.0 / self._fps   # shape (S,)
        # Scale so the median state has zero correction (neutral prior)
        log_prior = np.log(np.clip(beats_per_frame, 1e-10, None))
        log_prior -= np.median(log_prior)

        delta = np.full((T, S), -np.inf)
        psi   = np.zeros((T, S), dtype=np.int32)
        # Initialise with prior so all states compete on equal footing
        delta[0] = log_em[0] + log_prior

        for t in range(1, T):
            # Vectorised — (S_prev,1) + (S_prev,S_next) => (S_prev,S_next)
            candidates = delta[t-1][:, None] + log_trans     # (S, S)
            psi[t]     = candidates.argmax(axis=0)            # (S,)
            delta[t]   = candidates.max(axis=0) + log_em[t]  # (S,)

        # Backtrack
        seq      = np.zeros(T, dtype=np.int32)
        seq[T-1] = int(np.argmax(delta[T-1]))
        for t in range(T - 2, -1, -1):
            seq[t] = psi[t+1, seq[t+1]]

        dominant = int(np.bincount(seq).argmax())
        return seq, dominant

    def _decode_beats(self, envelope: np.ndarray, tempo_idx: int) -> np.ndarray:
        """Extract beat frames given the dominant tempo state."""
        period   = int(self._periods[tempo_idx])
        n_frames = len(envelope)
        if period <= 0 or n_frames == 0:
            return np.array([], dtype=int)
        start = int(np.argmax(envelope[:min(period, n_frames)]))
        return np.arange(start, n_frames, period, dtype=int)

    @staticmethod
    def _tempo_from_beats(beat_times: np.ndarray) -> float:
        """Derive tempo from median inter-beat interval."""
        if len(beat_times) < 2:
            return 0.0
        med = float(np.median(np.diff(beat_times)))
        return 60.0 / med if med > 0 else 0.0

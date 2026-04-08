# src/utils/audio.py
"""
Audio loading and preprocessing utilities.
All audio is standardised to 44.1kHz mono before any processing.
"""

import os
import logging
import numpy as np
import librosa
import soundfile as sf
from typing import Tuple, Optional

from src.utils.config import SAMPLE_RATE, MONO

logger = logging.getLogger(__name__)


def load_audio(path: str, sr: int = SAMPLE_RATE, mono: bool = MONO,
               duration: Optional[float] = None) -> Tuple[np.ndarray, int]:
    """
    Load an audio file and resample to target sample rate.

    Args:
        path:     Path to audio file.
        sr:       Target sample rate (default 44100 Hz).
        mono:     Convert to mono if True.
        duration: Clip duration in seconds (None = full file).

    Returns:
        (y, sr): Audio time series and sample rate.

    Raises:
        FileNotFoundError: If the audio file does not exist.
        RuntimeError:      If librosa cannot decode the file.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Audio file not found: {path}")

    try:
        y, loaded_sr = librosa.load(path, sr=sr, mono=mono, duration=duration)
        logger.debug("Loaded %s  shape=%s  sr=%d", path, y.shape, loaded_sr)
        return y, loaded_sr
    except Exception as exc:
        raise RuntimeError(f"Failed to load {path}: {exc}") from exc


def save_audio(y: np.ndarray, sr: int, path: str) -> None:
    """Save a numpy array as a WAV file using soundfile."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sf.write(path, y, sr)
    logger.debug("Saved audio to %s", path)


def compute_onset_envelope(y: np.ndarray, sr: int,
                            hop_length: int = 512) -> np.ndarray:
    """
    Compute the onset strength envelope.

    Args:
        y:          Audio time series.
        sr:         Sample rate.
        hop_length: Number of samples between successive frames.

    Returns:
        1-D numpy array of onset strength values.
    """
    envelope = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    return envelope


def compute_mel_spectrogram(y: np.ndarray, sr: int,
                             n_fft: int = 2048,
                             hop_length: int = 512,
                             n_mels: int = 128) -> np.ndarray:
    """
    Compute a log-scaled mel spectrogram.

    Returns:
        2-D array of shape (n_mels, time_frames).
    """
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )
    log_S = librosa.power_to_db(S, ref=np.max)
    return log_S


def frames_to_times(frames: np.ndarray, sr: int, hop_length: int = 512) -> np.ndarray:
    """Convert frame indices to time in seconds."""
    return librosa.frames_to_time(frames, sr=sr, hop_length=hop_length)

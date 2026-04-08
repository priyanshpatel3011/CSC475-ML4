#!/usr/bin/env python3
# scripts/extract_features.py
"""
Feature Extraction Pipeline
=============================
Computes and saves mel-spectrograms and onset strength envelopes
for all processed audio files.

Saves features as .npy files mirroring the processed directory structure.

Usage:
    python scripts/extract_features.py --input data/processed --output data/features
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from tqdm import tqdm

from src.utils.audio import (
    load_audio, compute_mel_spectrogram, compute_onset_envelope
)
from src.utils.config import SAMPLE_RATE, HOP_LENGTH, N_FFT, N_MELS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def extract_and_save(audio_path: str,
                     mel_path: str,
                     onset_path: str,
                     sr: int = SAMPLE_RATE) -> bool:
    """
    Extract mel-spectrogram and onset envelope from audio_path.
    Save to mel_path and onset_path as .npy files.

    Returns:
        True on success.
    """
    try:
        y, sr_ = load_audio(audio_path, sr=sr, mono=True)

        mel   = compute_mel_spectrogram(y, sr_, hop_length=HOP_LENGTH,
                                         n_fft=N_FFT, n_mels=N_MELS)
        onset = compute_onset_envelope(y, sr_, hop_length=HOP_LENGTH)

        os.makedirs(os.path.dirname(mel_path),   exist_ok=True)
        os.makedirs(os.path.dirname(onset_path), exist_ok=True)

        np.save(mel_path,   mel)
        np.save(onset_path, onset)
        return True
    except Exception as exc:
        logger.warning("Feature extraction failed for %s: %s", audio_path, exc)
        return False


def main():
    parser = argparse.ArgumentParser(description="Extract audio features")
    parser.add_argument("--input",  required=True, help="Processed audio root")
    parser.add_argument("--output", required=True, help="Feature output root")
    parser.add_argument("--sr",     type=int, default=SAMPLE_RATE)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    wav_files = []
    for dirpath, _, fnames in os.walk(args.input):
        for f in fnames:
            if f.endswith(".wav"):
                wav_files.append(os.path.join(dirpath, f))

    logger.info("Found %d WAV files", len(wav_files))
    n_ok = n_skip = n_fail = 0

    for wav in tqdm(wav_files, desc="Extracting features"):
        rel   = os.path.relpath(wav, args.input)
        base  = os.path.splitext(rel)[0]

        mel_path   = os.path.join(args.output, "mel",   base + ".npy")
        onset_path = os.path.join(args.output, "onset", base + ".npy")

        if os.path.isfile(mel_path) and os.path.isfile(onset_path) and not args.overwrite:
            n_skip += 1
            continue

        if extract_and_save(wav, mel_path, onset_path, sr=args.sr):
            n_ok += 1
        else:
            n_fail += 1

    logger.info("Done. OK: %d | Skipped: %d | Failed: %d", n_ok, n_skip, n_fail)


if __name__ == "__main__":
    main()

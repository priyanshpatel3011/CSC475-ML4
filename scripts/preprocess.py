#!/usr/bin/env python3
# scripts/preprocess.py
"""
Data Preprocessing Pipeline
============================
Converts all raw audio files to 44.1kHz mono WAV format and
organises them into the processed data directory.

Usage:
    python scripts/preprocess.py --input data/raw/gtzan --output data/processed/gtzan
    python scripts/preprocess.py --input data/raw/giantsteps --output data/processed/giantsteps
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tqdm import tqdm
import soundfile as sf

from src.utils.audio import load_audio, save_audio
from src.utils.config import SAMPLE_RATE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".aiff", ".aif"}


def find_audio_files(root: str):
    """Recursively find all supported audio files under root."""
    found = []
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            if os.path.splitext(fname)[1].lower() in SUPPORTED_EXTENSIONS:
                found.append(os.path.join(dirpath, fname))
    return sorted(found)


def preprocess_file(src_path: str, dst_path: str, sr: int = SAMPLE_RATE) -> bool:
    """
    Load src_path, resample to sr Hz mono, and save as WAV to dst_path.

    Returns:
        True on success, False on error.
    """
    try:
        y, loaded_sr = load_audio(src_path, sr=sr, mono=True)
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        save_audio(y, sr, dst_path)
        return True
    except Exception as exc:
        logger.warning("Failed to preprocess %s: %s", src_path, exc)
        return False


def main():
    parser = argparse.ArgumentParser(description="Preprocess audio dataset")
    parser.add_argument("--input",  required=True, help="Root of raw audio directory")
    parser.add_argument("--output", required=True, help="Root of processed output directory")
    parser.add_argument("--sr",     type=int, default=SAMPLE_RATE, help="Target sample rate")
    parser.add_argument("--overwrite", action="store_true", help="Re-process existing files")
    args = parser.parse_args()

    audio_files = find_audio_files(args.input)
    logger.info("Found %d audio files in %s", len(audio_files), args.input)

    n_ok = n_skip = n_fail = 0

    for src in tqdm(audio_files, desc="Preprocessing"):
        # Mirror directory structure under output root
        rel_path = os.path.relpath(src, args.input)
        base, _  = os.path.splitext(rel_path)
        dst      = os.path.join(args.output, base + ".wav")

        if os.path.isfile(dst) and not args.overwrite:
            n_skip += 1
            continue

        if preprocess_file(src, dst, sr=args.sr):
            n_ok += 1
        else:
            n_fail += 1

    logger.info(
        "Done. Processed: %d | Skipped (exists): %d | Failed: %d",
        n_ok, n_skip, n_fail
    )
    if n_fail > 0:
        logger.warning("%d files failed — check logs above.", n_fail)


if __name__ == "__main__":
    main()

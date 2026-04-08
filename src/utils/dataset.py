# src/utils/dataset.py
"""
Dataset discovery, annotation loading, and train/val/test splitting.

Supports GTZAN Tempo-Beat and GiantSteps layouts.

Fixes vs original:
    - load_giantsteps now loads beat annotations from .beats files
      (was hardcoded to None, so beat metrics could never run on GiantSteps)
    - Graceful warning when annotation files are missing instead of silent None
"""

import os
import json
import random
import logging
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import numpy as np

from src.utils.config import (
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO, RANDOM_SEED, GTZAN_GENRES
)

logger = logging.getLogger(__name__)


# ── Data Structures ───────────────────────────────────────────────────────────

class Track:
    """Represents a single annotated audio track."""

    def __init__(self, track_id: str, audio_path: str,
                 tempo: Optional[float] = None,
                 beat_times: Optional[np.ndarray] = None,
                 genre: Optional[str] = None,
                 dataset: str = "unknown"):
        self.track_id   = track_id
        self.audio_path = audio_path
        self.tempo      = tempo
        self.beat_times = beat_times
        self.genre      = genre
        self.dataset    = dataset

    def __repr__(self):
        tempo_str = f"{self.tempo:.1f} BPM" if self.tempo else "?"
        return f"Track(id={self.track_id!r}, genre={self.genre!r}, tempo={tempo_str})"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_tempo_file(path: str) -> Optional[float]:
    """Read a single-float .bpm file. Returns None if missing or invalid."""
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            return float(f.read().strip())
    except (ValueError, OSError):
        logger.warning("Could not parse tempo file: %s", path)
        return None


def _load_beats_file(path: str) -> Optional[np.ndarray]:
    """
    Read a .beats file (one beat time per line, optional beat number after space).
    Returns numpy array or None if file is missing.
    """
    if not os.path.isfile(path):
        return None
    times = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    times.append(float(line.split()[0]))
    except (ValueError, OSError):
        logger.warning("Could not parse beats file: %s", path)
        return None
    return np.array(times, dtype=float) if times else None


# ── GTZAN Loader ──────────────────────────────────────────────────────────────

def load_gtzan(root: str) -> List[Track]:
    """
    Load GTZAN Tempo-Beat dataset.

    Expected layout (matches TempoBeatDownbeat/gtzan_tempo_beat repo):
        root/
          audio/
            <genre>/
              <trackname>.wav
          annotations/
            tempo/   <trackname>.bpm
            beats/   <trackname>.beats

    Download instructions:
        Audio   → https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification
        Annotations → git clone https://github.com/TempoBeatDownbeat/gtzan_tempo_beat

    Returns:
        List of Track objects.
    """
    tracks     = []
    audio_root = os.path.join(root, "audio")
    tempo_root = os.path.join(root, "annotations", "tempo")
    beats_root = os.path.join(root, "annotations", "beats")

    if not os.path.isdir(audio_root):
        logger.error("GTZAN audio directory not found: %s", audio_root)
        logger.error("See README.md → Dataset Setup for download instructions.")
        return tracks

    for genre in GTZAN_GENRES:
        genre_dir = os.path.join(audio_root, genre)
        if not os.path.isdir(genre_dir):
            logger.warning("Genre directory not found (skipping): %s", genre_dir)
            continue

        for fname in sorted(os.listdir(genre_dir)):
            if not fname.endswith(".wav") and not fname.endswith(".au"):
                continue
            track_id   = os.path.splitext(fname)[0]
            audio_path = os.path.join(genre_dir, fname)

            # Map "blues.00000" to "gtzan_blues_00000" for annotations
            mapped_id = track_id
            if "." in track_id and not track_id.startswith("gtzan"):
                parts = track_id.split(".")
                if len(parts) == 2:
                    mapped_id = f"gtzan_{parts[0]}_{parts[1]}"

            tempo      = _load_tempo_file(os.path.join(tempo_root, mapped_id + ".bpm"))
            beat_times = _load_beats_file(os.path.join(beats_root, mapped_id + ".beats"))

            if tempo is None:
                logger.debug("No tempo annotation for %s", track_id)
            if beat_times is None:
                logger.debug("No beat annotation for %s", track_id)

            tracks.append(Track(
                track_id=track_id,
                audio_path=audio_path,
                tempo=tempo,
                beat_times=beat_times,
                genre=genre,
                dataset="gtzan",
            ))

    logger.info("Loaded %d GTZAN tracks", len(tracks))
    return tracks


# ── GiantSteps Loader ─────────────────────────────────────────────────────────

def load_giantsteps(root: str) -> List[Track]:
    """
    Load GiantSteps Tempo dataset.

    Expected layout (matches GiantSteps/giantsteps-tempo-dataset repo):
        root/
          audio/
            <trackname>.mp3   (or .wav after conversion)
          annotations/
            tempo/    <trackname>.bpm
            beats/    <trackname>.beats   ← FIX: now loaded (was None in original)

    Download instructions:
        Annotations → git clone https://github.com/GiantSteps/giantsteps-tempo-dataset
        Audio       → cd giantsteps-tempo-dataset && bash audio_dl.sh
                      (downloads 664 MP3 previews from Beatport CDN)
        Convert wav → bash convert_audio.sh   (requires sox)

    Note: GiantSteps beat annotations are provided in the repo under
          annotations/beats/. If only doing tempo evaluation, beats can
          be absent — tempo metrics will still compute correctly.

    Returns:
        List of Track objects.
    """
    tracks     = []
    audio_root = os.path.join(root, "audio")
    tempo_root = os.path.join(root, "annotations", "tempo")
    beats_root = os.path.join(root, "annotations", "beats")

    if not os.path.isdir(audio_root):
        logger.error("GiantSteps audio directory not found: %s", audio_root)
        logger.error("See README.md → Dataset Setup for download instructions.")
        return tracks

    for fname in sorted(os.listdir(audio_root)):
        if not (fname.endswith(".mp3") or fname.endswith(".wav")):
            continue
        track_id   = os.path.splitext(fname)[0]
        audio_path = os.path.join(audio_root, fname)

        tempo = _load_tempo_file(os.path.join(tempo_root, track_id + ".bpm"))

        # FIX: actually load beat annotations (original hardcoded None here)
        beat_times = _load_beats_file(os.path.join(beats_root, track_id + ".beats"))

        tracks.append(Track(
            track_id=track_id,
            audio_path=audio_path,
            tempo=tempo,
            beat_times=beat_times,
            genre="edm",
            dataset="giantsteps",
        ))

    logger.info("Loaded %d GiantSteps tracks", len(tracks))
    return tracks


# ── Splitting ─────────────────────────────────────────────────────────────────

def split_tracks(tracks: List[Track],
                 train_ratio: float = TRAIN_RATIO,
                 val_ratio:   float = VAL_RATIO,
                 test_ratio:  float = TEST_RATIO,
                 seed:        int   = RANDOM_SEED,
                 stratify:    bool  = True
                 ) -> Tuple[List[Track], List[Track], List[Track]]:
    """
    Stratified train/val/test split preserving genre balance.

    Returns:
        (train_tracks, val_tracks, test_tracks)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"

    rng = random.Random(seed)
    train, val, test = [], [], []

    if stratify:
        buckets: Dict[str, List[Track]] = defaultdict(list)
        for t in tracks:
            buckets[t.genre or "unknown"].append(t)

        for genre, bucket in buckets.items():
            shuffled = bucket[:]
            rng.shuffle(shuffled)
            n       = len(shuffled)
            n_train = int(round(n * train_ratio))
            n_val   = int(round(n * val_ratio))
            train.extend(shuffled[:n_train])
            val.extend(shuffled[n_train:n_train + n_val])
            test.extend(shuffled[n_train + n_val:])
    else:
        shuffled = tracks[:]
        rng.shuffle(shuffled)
        n       = len(shuffled)
        n_train = int(round(n * train_ratio))
        n_val   = int(round(n * val_ratio))
        train   = shuffled[:n_train]
        val     = shuffled[n_train:n_train + n_val]
        test    = shuffled[n_train + n_val:]

    logger.info("Split: %d train / %d val / %d test",
                len(train), len(val), len(test))
    return train, val, test


def save_split(split: Dict[str, List[Track]], path: str) -> None:
    """Serialize split track IDs to JSON for reproducibility."""
    serialized = {
        name: [t.track_id for t in track_list]
        for name, track_list in split.items()
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(serialized, f, indent=2)
    logger.info("Saved split manifest to %s", path)

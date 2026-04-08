#!/usr/bin/env python3
"""
scripts/download_data.py
========================
Downloads and organises all required datasets for the CSC475 project.

Datasets handled:
  1. GTZAN audio          — from Kaggle (requires kaggle CLI)
  2. GTZAN annotations    — from GitHub (TempoBeatDownbeat/gtzan_tempo_beat)
  3. GiantSteps tempo     — from GitHub (GiantSteps/giantsteps-tempo-dataset)
  4. GiantSteps audio     — via bash script bundled in the repo

Usage:
    python scripts/download_data.py [--data-dir data/raw] [--skip-audio]

Requirements:
    git, bash, sox (for wav conversion), kaggle CLI (for GTZAN audio)

Dataset licences:
    GTZAN audio       : Research / non-commercial use
    GTZAN annotations : CC BY 4.0
    GiantSteps        : Creative Commons (see repo)
"""

import argparse
import logging
import os
import subprocess
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ── Links & repos ─────────────────────────────────────────────────────────────

GTZAN_ANNOTATIONS_REPO   = "https://github.com/TempoBeatDownbeat/gtzan_tempo_beat.git"
GIANTSTEPS_TEMPO_REPO    = "https://github.com/GiantSteps/giantsteps-tempo-dataset.git"

GTZAN_KAGGLE_DATASET     = "andradaolteanu/gtzan-dataset-music-genre-classification"

GTZAN_GENRES = [
    "blues", "classical", "country", "disco",
    "hiphop", "jazz", "metal", "pop", "reggae", "rock",
]


def run(cmd: str, cwd: str = None, check: bool = True) -> int:
    """Run a shell command, stream output, return exit code."""
    logger.info("$ %s", cmd)
    result = subprocess.run(cmd, shell=True, cwd=cwd)
    if check and result.returncode != 0:
        logger.error("Command failed (exit %d): %s", result.returncode, cmd)
        sys.exit(result.returncode)
    return result.returncode


def git_clone_or_pull(repo_url: str, dest: str):
    """Clone a repo or pull if it already exists."""
    if os.path.isdir(os.path.join(dest, ".git")):
        logger.info("Repo exists, pulling: %s", dest)
        run(f"git -C {dest} pull --ff-only", check=False)
    else:
        run(f"git clone {repo_url} {dest}")


# ── GTZAN ─────────────────────────────────────────────────────────────────────

def setup_gtzan(data_dir: str, skip_audio: bool):
    """
    Prepare GTZAN dataset.

    Audio:       data/raw/gtzan/audio/<genre>/*.wav
    Annotations: data/raw/gtzan/annotations/{tempo,beats}/
    """
    gtzan_root   = os.path.join(data_dir, "gtzan")
    audio_root   = os.path.join(gtzan_root, "audio")
    annot_root   = os.path.join(gtzan_root, "annotations")
    tempo_root   = os.path.join(annot_root, "tempo")
    beats_root   = os.path.join(annot_root, "beats")

    os.makedirs(gtzan_root,  exist_ok=True)
    os.makedirs(audio_root,  exist_ok=True)
    os.makedirs(tempo_root,  exist_ok=True)
    os.makedirs(beats_root,  exist_ok=True)

    # ── Annotations (GitHub) ──────────────────────────────────────────────────
    annot_repo = os.path.join(data_dir, "_gtzan_tempo_beat_repo")
    logger.info("=== Downloading GTZAN annotations ===")
    git_clone_or_pull(GTZAN_ANNOTATIONS_REPO, annot_repo)

    # Copy .bpm files
    for root, _, files in os.walk(os.path.join(annot_repo, "tempo")):
        for f in files:
            if f.endswith(".bpm"):
                src = os.path.join(root, f)
                dst = os.path.join(tempo_root, f)
                if not os.path.exists(dst):
                    import shutil
                    shutil.copy2(src, dst)

    # Copy .beats files
    for root, _, files in os.walk(os.path.join(annot_repo, "beats")):
        for f in files:
            if f.endswith(".beats"):
                src = os.path.join(root, f)
                dst = os.path.join(beats_root, f)
                if not os.path.exists(dst):
                    import shutil
                    shutil.copy2(src, dst)

    logger.info("GTZAN annotations ready: %s .bpm  |  %s .beats",
                len(os.listdir(tempo_root)), len(os.listdir(beats_root)))

    # ── Audio (Kaggle) ────────────────────────────────────────────────────────
    if skip_audio:
        logger.info("Skipping GTZAN audio download (--skip-audio)")
        return

    logger.info("=== Downloading GTZAN audio via Kaggle ===")
    logger.info("Requires: pip install kaggle && kaggle API key configured")
    logger.info("See: https://www.kaggle.com/docs/api")

    kaggle_dl = os.path.join(data_dir, "_kaggle_gtzan")
    os.makedirs(kaggle_dl, exist_ok=True)
    rc = run(
        f"kaggle datasets download -d {GTZAN_KAGGLE_DATASET} "
        f"--unzip -p {kaggle_dl}",
        check=False
    )
    if rc != 0:
        logger.warning("Kaggle download failed. Manual steps:")
        logger.warning("  1. Go to https://www.kaggle.com/datasets/%s", GTZAN_KAGGLE_DATASET)
        logger.warning("  2. Download and unzip to: %s", kaggle_dl)
        logger.warning("  3. Re-run this script")
        return

    # Kaggle zip unpacks as genres/<genre>/*.wav — reorganise to audio/<genre>/
    import shutil
    kaggle_genres = os.path.join(kaggle_dl, "genres_original")
    if not os.path.isdir(kaggle_genres):
        kaggle_genres = os.path.join(kaggle_dl, "genres")

    for genre in GTZAN_GENRES:
        src_genre = os.path.join(kaggle_genres, genre)
        dst_genre = os.path.join(audio_root, genre)
        if os.path.isdir(src_genre):
            os.makedirs(dst_genre, exist_ok=True)
            for f in os.listdir(src_genre):
                src = os.path.join(src_genre, f)
                dst = os.path.join(dst_genre, f)
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)

    logger.info("GTZAN audio ready: %s", audio_root)


# ── GiantSteps ────────────────────────────────────────────────────────────────

def setup_giantsteps(data_dir: str, skip_audio: bool):
    """
    Prepare GiantSteps Tempo dataset.

    Annotations: data/raw/giantsteps/annotations/{tempo,beats}/
    Audio:       data/raw/giantsteps/audio/*.mp3 (or .wav after conversion)
    """
    gs_root    = os.path.join(data_dir, "giantsteps")
    audio_root = os.path.join(gs_root,  "audio")
    annot_root = os.path.join(gs_root,  "annotations")
    tempo_root = os.path.join(annot_root, "tempo")
    beats_root = os.path.join(annot_root, "beats")

    os.makedirs(gs_root,    exist_ok=True)
    os.makedirs(audio_root, exist_ok=True)
    os.makedirs(tempo_root, exist_ok=True)
    os.makedirs(beats_root, exist_ok=True)

    # ── Clone GiantSteps repo (contains annotations + download script) ────────
    gs_repo = os.path.join(data_dir, "_giantsteps_tempo_repo")
    logger.info("=== Downloading GiantSteps annotations ===")
    git_clone_or_pull(GIANTSTEPS_TEMPO_REPO, gs_repo)

    import shutil

    # Copy .bpm annotation files
    annot_src_tempo = os.path.join(gs_repo, "annotations", "tempo")
    if os.path.isdir(annot_src_tempo):
        for f in os.listdir(annot_src_tempo):
            if f.endswith(".bpm") or f.endswith(".LOFI.bpm"):
                src = os.path.join(annot_src_tempo, f)
                dst = os.path.join(tempo_root, f)
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)

    # Copy .beats annotation files (if present in repo)
    annot_src_beats = os.path.join(gs_repo, "annotations", "beats")
    if os.path.isdir(annot_src_beats):
        for f in os.listdir(annot_src_beats):
            if f.endswith(".beats"):
                src = os.path.join(annot_src_beats, f)
                dst = os.path.join(beats_root, f)
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)

    logger.info("GiantSteps annotations ready: %s .bpm files",
                len(os.listdir(tempo_root)))

    # ── Audio ─────────────────────────────────────────────────────────────────
    if skip_audio:
        logger.info("Skipping GiantSteps audio download (--skip-audio)")
        return

    logger.info("=== Downloading GiantSteps audio ===")
    logger.info("This downloads 664 MP3 previews (~2 min each) from Beatport CDN.")
    logger.info("Requires: bash, curl or wget")

    dl_script = os.path.join(gs_repo, "audio_dl.sh")
    if not os.path.isfile(dl_script):
        logger.warning("audio_dl.sh not found in repo — skipping audio download")
        return

    run(f"bash {dl_script}", cwd=audio_root, check=False)

    # Convert MP3 → WAV (requires sox)
    convert_script = os.path.join(gs_repo, "convert_audio.sh")
    if os.path.isfile(convert_script):
        logger.info("Converting MP3 → WAV (requires sox)")
        run(f"bash {convert_script}", cwd=audio_root, check=False)

    logger.info("GiantSteps audio ready: %s", audio_root)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Download and organise CSC475 datasets"
    )
    parser.add_argument("--data-dir",    default="data/raw",
                        help="Root directory for raw data (default: data/raw)")
    parser.add_argument("--skip-audio",  action="store_true",
                        help="Only download annotations, skip large audio files")
    parser.add_argument("--dataset",     choices=["gtzan", "giantsteps", "all"],
                        default="all",
                        help="Which dataset to download (default: all)")
    args = parser.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)

    if args.dataset in ("gtzan", "all"):
        setup_gtzan(args.data_dir, skip_audio=args.skip_audio)

    if args.dataset in ("giantsteps", "all"):
        setup_giantsteps(args.data_dir, skip_audio=args.skip_audio)

    logger.info("=== Dataset setup complete ===")
    logger.info("Next step: python scripts/preprocess.py --input data/raw/gtzan --output data/processed/gtzan")


if __name__ == "__main__":
    main()

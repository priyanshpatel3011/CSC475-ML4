# src/utils/config.py
"""
Global configuration constants for the CSC475 beat tracking project.
"""

import os

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_RAW       = os.path.join(PROJECT_ROOT, "data", "raw")
DATA_PROCESSED = os.path.join(PROJECT_ROOT, "data", "processed")
DATA_FEATURES  = os.path.join(PROJECT_ROOT, "data", "features")
RESULTS_DIR    = os.path.join(PROJECT_ROOT, "results")
FIGURES_DIR    = os.path.join(RESULTS_DIR, "figures")
METRICS_DIR    = os.path.join(RESULTS_DIR, "metrics")

# ── Audio ────────────────────────────────────────────────────────────────────
SAMPLE_RATE    = 44100      # Hz — all audio resampled to this
MONO           = True
CLIP_DURATION  = 30.0       # seconds (GTZAN clips)

# ── Feature Extraction ───────────────────────────────────────────────────────
HOP_LENGTH     = 512
N_FFT          = 2048
N_MELS         = 128
FMIN           = 27.5
FMAX           = 16000.0

# ── Tempo / Beat ─────────────────────────────────────────────────────────────
TEMPO_MIN      = 40         # BPM lower bound
TEMPO_MAX      = 240        # BPM upper bound
ACC1_TOLERANCE = 0.04       # 4% for ACC1
ACC2_OCTAVE    = True       # Allow 2x/0.5x for ACC2
BEAT_TOLERANCE = 0.07       # 70ms window for F-measure

# ── Dataset Splits ───────────────────────────────────────────────────────────
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15
RANDOM_SEED = 42

# ── GTZAN Genres ─────────────────────────────────────────────────────────────
GTZAN_GENRES = [
    "blues", "classical", "country", "disco",
    "hiphop", "jazz", "metal", "pop", "reggae", "rock"
]

# ── Algorithm Names ───────────────────────────────────────────────────────────
ALGO_AUTOCORR    = "autocorrelation"
ALGO_DBN         = "dbn"
ALGO_STATE_SPACE = "state_space"
ALL_ALGORITHMS   = [ALGO_AUTOCORR, ALGO_DBN, ALGO_STATE_SPACE]

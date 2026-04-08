# src/visualization/plots.py
"""
Visualization module for beat tracking project.

Generates:
    1. Per-genre bar chart (ACC1, F-measure)
    2. Tempo range vs. accuracy scatter plot
    3. Box plots of F-measure by algorithm and genre
    4. Runtime comparison bar chart
    5. Failure case waveform with beat annotations
"""

import os
import logging
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pandas as pd
import librosa
import librosa.display

from src.utils.config import FIGURES_DIR, GTZAN_GENRES, ALL_ALGORITHMS

logger = logging.getLogger(__name__)

# ── Style ─────────────────────────────────────────────────────────────────────
ALGO_COLORS = {
    "autocorrelation": "#2E75B6",
    "dbn":             "#ED7D31",
    "state_space":     "#70AD47",
}
ALGO_LABELS = {
    "autocorrelation": "Autocorrelation",
    "dbn":             "DBN (madmom)",
    "state_space":     "State-Space",
}


def _save(fig, name: str, out_dir: str = FIGURES_DIR, dpi: int = 150) -> str:
    """Save figure to PNG and PDF."""
    os.makedirs(out_dir, exist_ok=True)
    path_png = os.path.join(out_dir, name + ".png")
    path_pdf = os.path.join(out_dir, name + ".pdf")
    fig.savefig(path_png, dpi=dpi, bbox_inches="tight")
    fig.savefig(path_pdf, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved figure: %s", path_png)
    return path_png


# ── 1. Per-Genre Bar Chart ────────────────────────────────────────────────────

def plot_genre_comparison(summary: Dict,
                           metric: str = "acc1_mean",
                           metric_label: str = "ACC1",
                           out_dir: str = FIGURES_DIR) -> str:
    """
    Bar chart comparing algorithm performance per genre.

    Args:
        summary:      Aggregated summary dict from Evaluator.aggregate().
        metric:       Key to extract from by_genre results.
        metric_label: Y-axis label.
        out_dir:      Output directory.

    Returns:
        Path to saved PNG.
    """
    genres = GTZAN_GENRES
    algos  = [a for a in ALL_ALGORITHMS if a in summary]
    x      = np.arange(len(genres))
    width  = 0.25

    fig, ax = plt.subplots(figsize=(14, 5))

    for i, algo in enumerate(algos):
        vals = []
        for genre in genres:
            genre_data = summary[algo]["by_genre"].get(genre, {})
            group      = "tempo" if "acc" in metric or "p_score" in metric else "beat"
            val        = genre_data.get(group, {}).get(metric, 0.0)
            vals.append(val if val is not None else 0.0)

        bars = ax.bar(x + i * width, vals, width,
                      label=ALGO_LABELS.get(algo, algo),
                      color=ALGO_COLORS.get(algo, "#888888"),
                      edgecolor="white", linewidth=0.5)

    ax.set_xlabel("Genre", fontsize=12)
    ax.set_ylabel(metric_label, fontsize=12)
    ax.set_title(f"{metric_label} by Genre", fontsize=14, fontweight="bold")
    ax.set_xticks(x + width)
    ax.set_xticklabels([g.capitalize() for g in genres], rotation=30, ha="right")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    sns.despine(ax=ax)

    return _save(fig, f"genre_comparison_{metric}", out_dir)


# ── 2. Tempo Range vs. Accuracy Scatter ──────────────────────────────────────

def plot_tempo_scatter(results: Dict,
                        metric: str = "acc1",
                        out_dir: str = FIGURES_DIR) -> str:
    """
    Scatter plot: reference tempo (x) vs. per-track ACC1 score (y).
    """
    algos = [a for a in ALL_ALGORITHMS if a in results]
    fig, axes = plt.subplots(1, len(algos), figsize=(5 * len(algos), 4),
                              sharey=True)
    if len(algos) == 1:
        axes = [axes]

    for ax, algo in zip(axes, algos):
        tempos, scores = [], []
        for res in results[algo].values():
            t = res.get("reference_tempo")
            s = res.get("tempo_metrics", {}).get(metric)
            if t is not None and s is not None:
                tempos.append(t)
                scores.append(s)

        ax.scatter(tempos, scores, alpha=0.4, s=18,
                   color=ALGO_COLORS.get(algo, "#888"),
                   edgecolors="none")
        ax.set_title(ALGO_LABELS.get(algo, algo), fontsize=11)
        ax.set_xlabel("Reference Tempo (BPM)", fontsize=10)
        ax.set_ylim(-0.05, 1.05)
        ax.yaxis.grid(True, linestyle="--", alpha=0.4)
        sns.despine(ax=ax)

    axes[0].set_ylabel(metric.upper(), fontsize=10)
    fig.suptitle("Tempo Range vs. Accuracy", fontsize=13, fontweight="bold")
    plt.tight_layout()
    return _save(fig, f"tempo_scatter_{metric}", out_dir)


# ── 3. Box Plots of F-measure ─────────────────────────────────────────────────

def plot_fmeasure_boxplot(results: Dict,
                           out_dir: str = FIGURES_DIR) -> str:
    """
    Box plots of F-measure distribution, one box per algorithm.
    """
    records = []
    for algo in ALL_ALGORITHMS:
        if algo not in results:
            continue
        for res in results[algo].values():
            f = res.get("beat_metrics", {}).get("f_measure")
            if f is not None:
                records.append({
                    "Algorithm": ALGO_LABELS.get(algo, algo),
                    "F-measure": f,
                    "Genre":     res.get("genre", "unknown"),
                })

    if not records:
        logger.warning("No beat metrics to plot in box plot")
        return ""

    df  = pd.DataFrame(records)
    fig, ax = plt.subplots(figsize=(8, 5))

    palette = {ALGO_LABELS.get(a, a): ALGO_COLORS.get(a, "#888")
               for a in ALL_ALGORITHMS}

    sns.boxplot(data=df, x="Algorithm", y="F-measure",
                palette=palette, width=0.5, linewidth=1.2, ax=ax)
    sns.stripplot(data=df, x="Algorithm", y="F-measure",
                  color="black", alpha=0.2, size=2, jitter=True, ax=ax)

    ax.set_ylim(0, 1.05)
    ax.set_title("F-measure Distribution by Algorithm", fontsize=13, fontweight="bold")
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    sns.despine(ax=ax)
    plt.tight_layout()
    return _save(fig, "fmeasure_boxplot", out_dir)


# ── 4. Runtime Comparison ─────────────────────────────────────────────────────

def plot_runtime_comparison(summary: Dict,
                             out_dir: str = FIGURES_DIR) -> str:
    """
    Horizontal bar chart of mean runtime per algorithm.
    """
    algos    = [a for a in ALL_ALGORITHMS if a in summary]
    means    = [summary[a]["overall"].get("runtime_mean_sec", 0.0) for a in algos]
    stds     = [summary[a]["overall"].get("runtime_std_sec",  0.0) for a in algos]
    labels   = [ALGO_LABELS.get(a, a) for a in algos]
    colors   = [ALGO_COLORS.get(a, "#888") for a in algos]

    fig, ax = plt.subplots(figsize=(7, 3))
    y = np.arange(len(algos))
    ax.barh(y, means, xerr=stds, color=colors, height=0.5,
            edgecolor="white", capsize=4)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel("Mean Runtime per Track (seconds)", fontsize=11)
    ax.set_title("Computational Runtime Comparison", fontsize=13, fontweight="bold")
    ax.xaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    sns.despine(ax=ax)
    plt.tight_layout()
    return _save(fig, "runtime_comparison", out_dir)


# ── 5. Failure Case Waveform ──────────────────────────────────────────────────

def plot_failure_case(audio_path: str,
                       estimated_beats: np.ndarray,
                       reference_beats: np.ndarray,
                       track_id: str,
                       algo_name: str,
                       out_dir: str = FIGURES_DIR) -> str:
    """
    Plot waveform with estimated vs. reference beat markers.
    Useful for visualising failure cases.

    Args:
        audio_path:      Path to the audio file.
        estimated_beats: Estimated beat times in seconds.
        reference_beats: Reference beat times in seconds.
        track_id:        Track identifier for title/filename.
        algo_name:       Algorithm name.
        out_dir:         Output directory.

    Returns:
        Path to saved PNG.
    """
    import librosa
    y, sr = librosa.load(audio_path, sr=22050, duration=15.0)
    duration = librosa.get_duration(y=y, sr=sr)

    fig, ax = plt.subplots(figsize=(14, 3))
    librosa.display.waveshow(y, sr=sr, ax=ax, alpha=0.5, color="#AAAAAA")

    # Reference beats — green vertical lines
    ref_in_range = reference_beats[reference_beats <= duration]
    for t in ref_in_range:
        ax.axvline(t, color="green", alpha=0.8, linewidth=1.2, linestyle="-")

    # Estimated beats — red dashed lines
    est_in_range = estimated_beats[estimated_beats <= duration]
    for t in est_in_range:
        ax.axvline(t, color="red", alpha=0.7, linewidth=1.0, linestyle="--")

    green_patch = mpatches.Patch(color="green", label="Reference beats")
    red_patch   = mpatches.Patch(color="red",   label="Estimated beats")
    ax.legend(handles=[green_patch, red_patch], fontsize=9, loc="upper right")

    ax.set_xlabel("Time (s)", fontsize=10)
    ax.set_title(f"Failure Case: {track_id}  [{ALGO_LABELS.get(algo_name, algo_name)}]",
                 fontsize=11)
    sns.despine(ax=ax)
    plt.tight_layout()

    fname = f"failure_{algo_name}_{track_id}"
    return _save(fig, fname, out_dir)

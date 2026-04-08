# src/evaluation/evaluator.py
"""
Evaluator: runs all three algorithms on a dataset split and collects results.

Usage:
    evaluator = Evaluator(algorithms, output_dir="results/metrics")
    results   = evaluator.run(tracks, split_name="test")
    evaluator.save(results, "results/metrics/test_results.json")
"""

import json
import logging
import os
import time
from typing import Dict, List, Optional

import numpy as np
from tqdm import tqdm

from src.evaluation.metrics import compute_tempo_metrics, compute_beat_metrics, aggregate_results
from src.utils.config import ALL_ALGORITHMS

logger = logging.getLogger(__name__)


class Evaluator:
    """
    Runs registered algorithms on a list of Track objects and
    computes all metrics.
    """

    def __init__(self, algorithms: dict, output_dir: str = "results/metrics"):
        """
        Args:
            algorithms: Dict mapping algorithm name -> tracker instance.
                        Each tracker must implement .predict(audio_path)
                        returning (tempo: float, beat_times: np.ndarray).
            output_dir: Directory to save JSON results.
        """
        self.algorithms = algorithms
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def run(self, tracks: list, split_name: str = "test") -> Dict:
        """
        Evaluate all algorithms on the given tracks.

        Args:
            tracks:     List of Track objects with ground-truth annotations.
            split_name: Label for this split (used in logging and filenames).

        Returns:
            Nested dict:
                results[algo_name][track_id] = {
                    "tempo_metrics": {...},
                    "beat_metrics":  {...},
                    "runtime_sec":   float,
                    "estimated_tempo": float,
                    "genre": str,
                }
        """
        results = {algo: {} for algo in self.algorithms}

        for algo_name, tracker in self.algorithms.items():
            logger.info("Evaluating: %s on %s (%d tracks)",
                        algo_name, split_name, len(tracks))

            for track in tqdm(tracks, desc=f"{algo_name}/{split_name}"):
                if not os.path.isfile(track.audio_path):
                    logger.warning("Audio not found, skipping: %s", track.audio_path)
                    continue

                # ── Run algorithm ────────────────────────────────────────────
                t0 = time.perf_counter()
                try:
                    est_tempo, est_beats = tracker.predict(track.audio_path)
                except Exception as exc:
                    logger.error("%s failed on %s: %s", algo_name, track.track_id, exc)
                    est_tempo, est_beats = 0.0, np.array([])
                runtime = time.perf_counter() - t0

                # ── Tempo metrics ────────────────────────────────────────────
                tempo_m = {}
                if track.tempo is not None:
                    tempo_m = compute_tempo_metrics(est_tempo, track.tempo)

                # ── Beat metrics ─────────────────────────────────────────────
                beat_m = {}
                if track.beat_times is not None and len(track.beat_times) > 0:
                    beat_m = compute_beat_metrics(est_beats, track.beat_times)

                results[algo_name][track.track_id] = {
                    "tempo_metrics":    tempo_m,
                    "beat_metrics":     beat_m,
                    "runtime_sec":      runtime,
                    "estimated_tempo":  est_tempo,
                    "reference_tempo":  track.tempo,
                    "genre":            track.genre,
                    "dataset":          track.dataset,
                }

        return results

    def aggregate(self, results: Dict) -> Dict:
        """
        Aggregate per-track results into per-algorithm summary statistics.

        Returns:
            summary[algo_name] = {
                "overall": {metric_mean, metric_std, ...},
                "by_genre": { genre: {metric_mean, ...} }
            }
        """
        summary = {}

        for algo_name, track_results in results.items():
            all_tempo  = []
            all_beat   = []
            by_genre   = {}

            for track_id, res in track_results.items():
                genre = res.get("genre", "unknown")
                if genre not in by_genre:
                    by_genre[genre] = {"tempo": [], "beat": []}

                if res.get("tempo_metrics"):
                    all_tempo.append(res["tempo_metrics"])
                    by_genre[genre]["tempo"].append(res["tempo_metrics"])

                if res.get("beat_metrics"):
                    all_beat.append(res["beat_metrics"])
                    by_genre[genre]["beat"].append(res["beat_metrics"])

            # Runtime stats
            runtimes = [r["runtime_sec"] for r in track_results.values()]

            summary[algo_name] = {
                "overall": {
                    "tempo": aggregate_results(all_tempo),
                    "beat":  aggregate_results(all_beat),
                    "runtime_mean_sec": float(np.mean(runtimes)) if runtimes else 0.0,
                    "runtime_std_sec":  float(np.std(runtimes))  if runtimes else 0.0,
                    "n_tracks": len(track_results),
                },
                "by_genre": {
                    genre: {
                        "tempo": aggregate_results(data["tempo"]),
                        "beat":  aggregate_results(data["beat"]),
                    }
                    for genre, data in by_genre.items()
                }
            }

        return summary

    def save(self, data: dict, filename: str) -> None:
        """Save results dict to a JSON file."""
        path = os.path.join(self.output_dir, filename)
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=_json_serializer)
        logger.info("Saved results to %s", path)

    def load(self, filename: str) -> dict:
        """Load results from a JSON file."""
        path = os.path.join(self.output_dir, filename)
        with open(path) as f:
            return json.load(f)


def _json_serializer(obj):
    """Handle numpy types for JSON serialization."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

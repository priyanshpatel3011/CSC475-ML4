# src/evaluation/metrics.py
"""
Evaluation metrics for tempo estimation and beat tracking.

Tempo Metrics:
    - ACC1  : Accuracy within 4% tolerance
    - ACC2  : ACC1 allowing 2x/0.5x octave errors only (standard definition)
    - P-Score: Continuous score based on tempo ratio
    - MAE   : Mean Absolute Error in BPM

Beat Tracking Metrics:
    - F-measure : Precision/recall with 70ms tolerance window
    - Cemgil    : Gaussian-weighted beat accuracy
    - InfoGain  : Information gain over uniform beat distribution
"""

import numpy as np
from typing import Optional

try:
    import mir_eval
    _HAS_MIR_EVAL = True
except ImportError:
    _HAS_MIR_EVAL = False


# ── Tempo Metrics ─────────────────────────────────────────────────────────────

def acc1(estimated: float, reference: float, tolerance: float = 0.04) -> float:
    """ACC1: 1 if |estimated/reference - 1| <= tolerance, else 0."""
    if reference <= 0 or estimated <= 0:
        return 0.0
    return 1.0 if abs(estimated / reference - 1.0) <= tolerance else 0.0


def acc2(estimated: float, reference: float, tolerance: float = 0.04) -> float:
    """
    ACC2: Standard MIREX definition — ACC1 extended to allow 2x and 0.5x
    octave errors ONLY.  3x/1/3x are NOT part of the standard definition.
    """
    if reference <= 0 or estimated <= 0:
        return 0.0
    for factor in [1.0, 2.0, 0.5]:
        if abs(estimated / (reference * factor) - 1.0) <= tolerance:
            return 1.0
    return 0.0


def p_score(estimated: float, reference: float, tolerance: float = 0.08) -> float:
    """P-Score: Continuous tempo accuracy in [0, 1]. Checks 1x, 2x, 0.5x."""
    if reference <= 0 or estimated <= 0:
        return 0.0
    ratio = estimated / reference
    for factor in [1.0, 2.0, 0.5]:
        err = abs(ratio / factor - 1.0)
        if err <= tolerance:
            return max(0.0, 1.0 - err / tolerance)
    return 0.0


def mean_absolute_error(estimated: np.ndarray, reference: np.ndarray) -> float:
    """Mean Absolute Error in BPM over a set of tracks."""
    mask = reference > 0
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs(estimated[mask] - reference[mask])))


def compute_tempo_metrics(estimated: float, reference: float) -> dict:
    """Compute all tempo metrics for a single track."""
    return {
        "acc1":      acc1(estimated, reference),
        "acc2":      acc2(estimated, reference),
        "p_score":   p_score(estimated, reference),
        "abs_error": abs(estimated - reference) if reference > 0 else float("nan"),
    }


# ── Beat Tracking Metrics ─────────────────────────────────────────────────────

def _f_measure_numpy(ref: np.ndarray, est: np.ndarray, tolerance: float = 0.07):
    """Pure-numpy F-measure fallback when mir_eval is not installed."""
    if len(ref) == 0 or len(est) == 0:
        return 0.0, 0.0, 0.0
    matched_ref = np.zeros(len(ref), dtype=bool)
    tp = 0
    for e in est:
        diffs = np.abs(ref - e)
        idx = int(np.argmin(diffs))
        if diffs[idx] <= tolerance and not matched_ref[idx]:
            matched_ref[idx] = True
            tp += 1
    precision = tp / len(est) if len(est) > 0 else 0.0
    recall    = tp / len(ref) if len(ref) > 0 else 0.0
    f = (2 * precision * recall / (precision + recall)
         if precision + recall > 0 else 0.0)
    return float(f), float(precision), float(recall)


def compute_beat_metrics(estimated_beats: np.ndarray,
                         reference_beats: np.ndarray,
                         tolerance: float = 0.07) -> dict:
    """
    Compute all beat tracking metrics for a single track.
    Uses mir_eval when available, falls back to pure-numpy F-measure.
    """
    empty = {"f_measure": 0.0, "precision": 0.0, "recall": 0.0,
             "cemgil": 0.0, "information_gain": 0.0}
    if len(estimated_beats) == 0 or len(reference_beats) == 0:
        return empty

    est = np.sort(estimated_beats[estimated_beats >= 0])
    ref = np.sort(reference_beats[reference_beats >= 0])

    if _HAS_MIR_EVAL:
        try:
            f = mir_eval.beat.f_measure(ref, est, f_measure_threshold=tolerance)
            _, p, r = _f_measure_numpy(ref, est, tolerance)
            cemgil  = mir_eval.beat.cemgil(ref, est)[0]
            ig      = mir_eval.beat.information_gain(ref, est)
        except Exception:
            f = p = r = cemgil = ig = 0.0
    else:
        f, p, r = _f_measure_numpy(ref, est, tolerance)
        cemgil = ig = 0.0

    return {"f_measure": float(f), "precision": float(p), "recall": float(r),
            "cemgil": float(cemgil), "information_gain": float(ig)}


# ── Aggregate ─────────────────────────────────────────────────────────────────

def aggregate_results(results: list) -> dict:
    """Aggregate per-track metric dicts into means and std."""
    if not results:
        return {}
    keys = results[0].keys()
    agg  = {}
    for key in keys:
        vals = [r[key] for r in results
                if r.get(key) is not None
                and not (isinstance(r[key], float) and np.isnan(r[key]))]
        if vals:
            agg[f"{key}_mean"] = float(np.mean(vals))
            agg[f"{key}_std"]  = float(np.std(vals))
            agg[f"{key}_n"]    = len(vals)
    return agg

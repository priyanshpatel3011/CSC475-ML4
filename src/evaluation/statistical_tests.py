# src/evaluation/statistical_tests.py
"""
Statistical significance testing for pairwise algorithm comparison.

Tests:
    - Wilcoxon signed-rank test (non-parametric, paired)
    - Effect size: Cohen's d
    - Generates a formatted comparison table
"""

import logging
from itertools import combinations
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

logger = logging.getLogger(__name__)

SIGNIFICANCE_THRESHOLD = 0.05


def extract_metric_arrays(results: Dict,
                           algo_names: List[str],
                           metric_key: str,
                           metric_group: str = "tempo_metrics"
                           ) -> Dict[str, np.ndarray]:
    """
    Extract per-track metric values for each algorithm into numpy arrays.
    Only tracks present in ALL algorithms are included (paired test requirement).

    Args:
        results:      Nested results dict from Evaluator.run().
        algo_names:   Algorithm names to compare.
        metric_key:   Metric to extract, e.g. "acc1" or "f_measure".
        metric_group: "tempo_metrics" or "beat_metrics".

    Returns:
        Dict mapping algo_name -> 1-D numpy array of per-track metric values.
    """
    # Find common track IDs
    track_sets = [set(results[a].keys()) for a in algo_names]
    common_ids = sorted(set.intersection(*track_sets))

    arrays = {}
    for algo in algo_names:
        vals = []
        for tid in common_ids:
            m = results[algo][tid].get(metric_group, {})
            val = m.get(metric_key)
            vals.append(val if val is not None else np.nan)
        arrays[algo] = np.array(vals, dtype=float)

    logger.info("Paired comparison on %d common tracks", len(common_ids))
    return arrays


def pairwise_wilcoxon(metric_arrays: Dict[str, np.ndarray],
                       alpha: float = SIGNIFICANCE_THRESHOLD
                       ) -> pd.DataFrame:
    """
    Perform pairwise Wilcoxon signed-rank tests.

    Args:
        metric_arrays: Dict algo_name -> array of per-track values.
        alpha:         Significance threshold.

    Returns:
        DataFrame with columns:
            algo_a, algo_b, mean_a, mean_b, statistic, p_value,
            significant, effect_size (Cohen's d), winner
    """
    algo_names = list(metric_arrays.keys())
    rows = []

    for a, b in combinations(algo_names, 2):
        arr_a = metric_arrays[a]
        arr_b = metric_arrays[b]

        # Drop NaN pairs
        valid = ~(np.isnan(arr_a) | np.isnan(arr_b))
        if valid.sum() < 5:
            logger.warning("Too few valid samples for %s vs %s", a, b)
            continue

        x = arr_a[valid]
        y = arr_b[valid]

        # Wilcoxon signed-rank test
        try:
            stat, p = wilcoxon(x, y, alternative="two-sided", zero_method="wilcox")
        except ValueError:
            stat, p = np.nan, 1.0

        # Cohen's d (standardised mean difference)
        d_val = cohens_d(x, y)

        rows.append({
            "algo_a":      a,
            "algo_b":      b,
            "mean_a":      float(np.mean(x)),
            "mean_b":      float(np.mean(y)),
            "statistic":   float(stat) if not np.isnan(stat) else None,
            "p_value":     float(p),
            "significant": bool(p < alpha),
            "effect_size": float(d_val),
            "winner":      a if np.mean(x) > np.mean(y) else b,
        })

    df = pd.DataFrame(rows)
    return df


def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute Cohen's d effect size between two paired arrays.

    Returns:
        Cohen's d value (positive means x > y).
    """
    diff = x - y
    if len(diff) < 2:
        return 0.0
    return float(np.mean(diff) / (np.std(diff, ddof=1) + 1e-10))


def print_comparison_table(df: pd.DataFrame, metric_name: str) -> None:
    """Pretty-print the pairwise comparison table."""
    print(f"\n{'='*65}")
    print(f"  Pairwise Statistical Comparison — {metric_name.upper()}")
    print(f"{'='*65}")
    print(f"{'Algo A':<20} {'Algo B':<20} {'Mean A':>7} {'Mean B':>7} "
          f"{'p':>7} {'Sig':>5} {'d':>6} {'Winner'}")
    print(f"{'-'*65}")
    for _, row in df.iterrows():
        sig = "YES" if row["significant"] else "no"
        print(f"{row['algo_a']:<20} {row['algo_b']:<20} "
              f"{row['mean_a']:>7.3f} {row['mean_b']:>7.3f} "
              f"{row['p_value']:>7.4f} {sig:>5} "
              f"{row['effect_size']:>6.3f} {row['winner']}")
    print(f"{'='*65}\n")

#!/usr/bin/env python3
# scripts/run_evaluation.py
"""
Run Full Evaluation
====================
Loads a dataset, runs all three algorithms on the test split,
computes metrics, runs statistical tests, and saves results.

Usage:
    python scripts/run_evaluation.py --dataset gtzan --split test
    python scripts/run_evaluation.py --dataset giantsteps --split test
"""

import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.algorithms.autocorrelation import AutocorrelationTracker
from src.algorithms.dbn_tracker      import DBNBeatTracker
from src.algorithms.state_space      import StateSpaceTracker
from src.evaluation.evaluator        import Evaluator
from src.evaluation.statistical_tests import (
    extract_metric_arrays, pairwise_wilcoxon, print_comparison_table
)
from src.visualization.plots import (
    plot_genre_comparison, plot_tempo_scatter,
    plot_fmeasure_boxplot, plot_runtime_comparison
)
from src.utils.dataset import load_gtzan, load_giantsteps, split_tracks
from src.utils.config   import (
    DATA_PROCESSED, RESULTS_DIR, FIGURES_DIR, ALL_ALGORITHMS
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


DATASETS = {
    "gtzan":      lambda: load_gtzan(os.path.join(DATA_PROCESSED, "gtzan")),
    "giantsteps": lambda: load_giantsteps(os.path.join(DATA_PROCESSED, "giantsteps")),
}


def main():
    parser = argparse.ArgumentParser(description="Run beat tracking evaluation")
    parser.add_argument("--dataset", choices=list(DATASETS.keys()),
                        default="gtzan", help="Dataset to evaluate")
    parser.add_argument("--split",   choices=["train", "val", "test"],
                        default="test", help="Which split to evaluate on")
    parser.add_argument("--skip-dbn", action="store_true",
                        help="Skip DBN if madmom not available")
    parser.add_argument("--out-dir", default=RESULTS_DIR)
    args = parser.parse_args()

    # ── Load dataset ──────────────────────────────────────────────────────────
    logger.info("Loading dataset: %s", args.dataset)
    tracks = DATASETS[args.dataset]()
    if not tracks:
        logger.error("No tracks loaded. Check data directory: %s",
                     os.path.join(DATA_PROCESSED, args.dataset))
        sys.exit(1)

    train_tracks, val_tracks, test_tracks = split_tracks(tracks)
    split_map = {"train": train_tracks, "val": val_tracks, "test": test_tracks}
    eval_tracks = split_map[args.split]
    logger.info("Evaluating on %s split: %d tracks", args.split, len(eval_tracks))

    # ── Initialise algorithms ─────────────────────────────────────────────────
    algorithms = {
        "autocorrelation": AutocorrelationTracker(),
        "state_space":     StateSpaceTracker(),
    }
    if not args.skip_dbn:
        try:
            algorithms["dbn"] = DBNBeatTracker()
            logger.info("DBN tracker loaded")
        except ImportError:
            logger.warning("madmom not available — skipping DBN tracker")

    # ── Run evaluation ────────────────────────────────────────────────────────
    metrics_dir = os.path.join(args.out_dir, "metrics")
    evaluator   = Evaluator(algorithms, output_dir=metrics_dir)
    results     = evaluator.run(eval_tracks, split_name=args.split)

    # Save raw per-track results
    fname = f"{args.dataset}_{args.split}_results.json"
    evaluator.save(results, fname)

    # ── Aggregate ─────────────────────────────────────────────────────────────
    summary = evaluator.aggregate(results)
    evaluator.save(summary, f"{args.dataset}_{args.split}_summary.json")

    # Print overall summary
    print(f"\n{'='*55}")
    print(f"  Results: {args.dataset.upper()} — {args.split.upper()} split")
    print(f"{'='*55}")
    for algo in list(algorithms.keys()):
        if algo not in summary:
            continue
        ov = summary[algo]["overall"]
        t  = ov.get("tempo", {})
        b  = ov.get("beat",  {})
        print(f"\n  {algo.upper()}")
        print(f"    ACC1:        {t.get('acc1_mean', 0):.3f} "
              f"± {t.get('acc1_std', 0):.3f}")
        print(f"    ACC2:        {t.get('acc2_mean', 0):.3f} "
              f"± {t.get('acc2_std', 0):.3f}")
        print(f"    F-measure:   {b.get('f_measure_mean', 0):.3f} "
              f"± {b.get('f_measure_std', 0):.3f}")
        print(f"    Cemgil:      {b.get('cemgil_mean', 0):.3f} "
              f"± {b.get('cemgil_std', 0):.3f}")
        print(f"    Runtime/trk: {ov.get('runtime_mean_sec', 0):.3f}s")
    print()

    # ── Statistical Tests ─────────────────────────────────────────────────────
    algo_names = list(algorithms.keys())
    if len(algo_names) >= 2:
        for metric_key, metric_group, label in [
            ("acc1",      "tempo_metrics", "ACC1"),
            ("f_measure", "beat_metrics",  "F-measure"),
        ]:
            try:
                arrays = extract_metric_arrays(results, algo_names,
                                               metric_key, metric_group)
                df     = pairwise_wilcoxon(arrays)
                print_comparison_table(df, label)
                evaluator.save(df.to_dict(orient="records"),
                                f"{args.dataset}_{args.split}_stats_{metric_key}.json")
            except Exception as exc:
                logger.warning("Statistical test failed for %s: %s", metric_key, exc)

    # ── Visualisations ────────────────────────────────────────────────────────
    fig_dir = os.path.join(args.out_dir, "figures", args.dataset)
    try:
        plot_genre_comparison(summary, metric="acc1_mean",
                               metric_label="ACC1", out_dir=fig_dir)
        plot_genre_comparison(summary, metric="f_measure_mean",
                               metric_label="F-measure", out_dir=fig_dir)
        plot_tempo_scatter(results, metric="acc1", out_dir=fig_dir)
        plot_fmeasure_boxplot(results, out_dir=fig_dir)
        plot_runtime_comparison(summary, out_dir=fig_dir)
        logger.info("All figures saved to %s", fig_dir)
    except Exception as exc:
        logger.warning("Some plots failed: %s", exc)

    logger.info("Evaluation complete.")


if __name__ == "__main__":
    main()

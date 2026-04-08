#!/usr/bin/env python3
# scripts/run_all.py
"""
Master Pipeline Script
=======================
Runs the entire pipeline end-to-end:
    1. Preprocess raw audio (GTZAN + GiantSteps)
    2. Extract features
    3. Evaluate all algorithms on both datasets (test split)
    4. Generate all figures and statistical reports

Usage:
    python scripts/run_all.py
    python scripts/run_all.py --skip-preprocess   # if data already preprocessed
    python scripts/run_all.py --skip-dbn          # if madmom not installed
"""

import argparse
import logging
import os
import subprocess
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s"
)
logger = logging.getLogger("run_all")

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))


def run(cmd: list, step: str) -> int:
    """Run a subprocess command and log the result."""
    logger.info("STEP: %s", step)
    logger.info("CMD : %s", " ".join(cmd))
    ret = subprocess.run(cmd)
    if ret.returncode != 0:
        logger.error("FAILED at step: %s (exit code %d)", step, ret.returncode)
    else:
        logger.info("OK: %s", step)
    return ret.returncode


def main():
    parser = argparse.ArgumentParser(description="Run full CSC475 pipeline")
    parser.add_argument("--skip-preprocess",  action="store_true")
    parser.add_argument("--skip-features",    action="store_true")
    parser.add_argument("--skip-dbn",         action="store_true")
    args = parser.parse_args()

    py = sys.executable
    failures = []

    # ── 1. Preprocess ─────────────────────────────────────────────────────────
    if not args.skip_preprocess:
        for dataset in ["gtzan", "giantsteps"]:
            rc = run(
                [py, os.path.join(SCRIPTS_DIR, "preprocess.py"),
                 "--input",  f"data/raw/{dataset}",
                 "--output", f"data/processed/{dataset}"],
                f"Preprocess {dataset}"
            )
            if rc != 0:
                failures.append(f"preprocess/{dataset}")

    # ── 2. Feature Extraction ─────────────────────────────────────────────────
    if not args.skip_features:
        rc = run(
            [py, os.path.join(SCRIPTS_DIR, "extract_features.py"),
             "--input",  "data/processed",
             "--output", "data/features"],
            "Feature extraction"
        )
        if rc != 0:
            failures.append("extract_features")

    # ── 3. Evaluation ─────────────────────────────────────────────────────────
    for dataset in ["gtzan", "giantsteps"]:
        cmd = [py, os.path.join(SCRIPTS_DIR, "run_evaluation.py"),
               "--dataset", dataset, "--split", "test"]
        if args.skip_dbn:
            cmd.append("--skip-dbn")
        rc = run(cmd, f"Evaluate {dataset}")
        if rc != 0:
            failures.append(f"evaluate/{dataset}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "="*50)
    if failures:
        print(f"  Pipeline completed with {len(failures)} failure(s):")
        for f in failures:
            print(f"    - {f}")
    else:
        print("  Pipeline completed successfully!")
    print("  Results: results/metrics/")
    print("  Figures: results/figures/")
    print("="*50)


if __name__ == "__main__":
    main()

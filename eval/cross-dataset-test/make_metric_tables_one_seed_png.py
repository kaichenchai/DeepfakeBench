#!/usr/bin/env python3
"""
Render cross-dataset metric tables for one random seed.

The input files are the per-model JSON results written under:
    logs/cross_dataset_test/multiple_seeds/<seed>/

The output is one PNG per metric in this directory, named
``<metric>_seed_<seed>.png``.  Cells use the same layout, ranking, and
overall-average column as make_metric_tables_png.py.  Since there is only one
measurement per cell, the displayed standard deviation is zero.

Example:
    python eval/cross-dataset-test/make_metric_tables_one_seed_png.py \
        --seed 1024
"""

import argparse
import json
import os

from make_metric_tables_png import (
    DATASETS,
    METRICS,
    MODELS,
    render_metric,
)


HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RESULTS_ROOT = os.path.abspath(
    os.path.join(HERE, "..", "..", "logs", "cross_dataset_test", "multiple_seeds")
)


def load_seed_results(seed, results_root, datasets, metric):
    """Return {model: {display_dataset: (value, 0.0)}} for one seed."""
    seed_dir = os.path.join(results_root, str(seed))
    data = {model: {} for model in MODELS}
    missing = []

    for model in MODELS:
        filename = f"cross_dataset_results_{model}_seed_{seed}.json"
        path = os.path.join(seed_dir, filename)
        if not os.path.isfile(path):
            missing.append(path)
            continue

        with open(path, encoding="utf-8") as fh:
            results = json.load(fh)

        for display_name, raw_name in datasets:
            metrics = results.get(raw_name)
            if not isinstance(metrics, dict):
                continue
            value = metrics.get(metric)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                data[model][display_name] = (float(value), 0.0)

    if missing:
        missing_paths = "\n".join(f"  {path}" for path in missing)
        raise FileNotFoundError(
            f"Missing result files for seed {seed}:\n{missing_paths}"
        )
    return data


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=1024,
                        help="Random seed to render (default: 1024).")
    parser.add_argument("--results-root", default=DEFAULT_RESULTS_ROOT,
                        help="Root containing <seed>/ result directories.")
    parser.add_argument("--exclude", nargs="+", default=[],
                        help="Display dataset labels to drop, e.g. UADFV.")
    parser.add_argument("--no-avg", action="store_true",
                        help="Omit the overall-average (Avg.) column.")
    return parser.parse_args()


def main():
    args = parse_args()
    datasets = [dataset for dataset in DATASETS
                if dataset[0] not in args.exclude]

    for metric in METRICS:
        data = load_seed_results(args.seed, args.results_root, datasets, metric)
        out_path = os.path.join(HERE, f"{metric}_seed_{args.seed}.png")
        render_metric(
            metric,
            data,
            datasets,
            out_path,
            show_avg=not args.no_avg,
            title_suffix=f"(seed {args.seed})",
        )


if __name__ == "__main__":
    main()

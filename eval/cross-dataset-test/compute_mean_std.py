#!/usr/bin/env python3
"""
Aggregate cross-dataset test results across the 5 random seeds (256/1024/1234/2048/9999)
and produce, for each metric, a CSV table of model (rows) vs dataset (columns),
reporting mean and std deviation over seeds.

Inputs (default):
    logs/cross_dataset_test/multiple_seeds/<seed>/cross_dataset_results_<model>_seed_<seed>.json

Outputs (default: eval/):
    acc.csv, ap.csv, auc.csv, eer.csv, video_auc.csv
Each CSV has columns "<dataset>_mean", "<dataset>_std" for every evaluated dataset.
Row names are the model labels parsed from the result filenames.

std is the SAMPLE standard deviation (ddof=1, n-1), the usual choice when reporting
mean +/- std over independent random-seed runs.
"""

import argparse
import csv
import glob
import json
import os
import re

import numpy as np

METRICS = ["acc", "ap", "auc", "eer", "video_auc"]

# Filename -> (model, seed), e.g.
#   cross_dataset_results_ce_hsic_seed_1024.json  -> ("ce_hsic", "1024")
FILENAME_RE = re.compile(r"^cross_dataset_results_(?P<model>.+)_seed_(?P<seed>\d+)\.json$")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results_dir", type=str,
                   default="./logs/cross_dataset_test/multiple_seeds",
                   help="Root folder containing <seed>/ subfolders of result JSONs.")
    p.add_argument("--out_dir", type=str, default="./eval",
                   help="Folder to write the per-metric CSVs.")
    p.add_argument("--metrics", type=str, nargs="+", default=METRICS,
                   help=f"Metrics to aggregate (default: {METRICS}).")
    return p.parse_args()


def main():
    args = parse_args()

    files = sorted(glob.glob(os.path.join(args.results_dir, "*", "cross_dataset_results_*.json")))
    if not files:
        raise SystemExit(f"No result files found under {args.results_dir}")

    # data[model][metric][dataset][seed] = value
    data = {}
    seeds_per_model = {}
    skipped = []

    for f in files:
        m = FILENAME_RE.match(os.path.basename(f))
        if not m:
            skipped.append(f)
            continue
        model, seed = m.group("model"), m.group("seed")
        seeds_per_model.setdefault(model, set()).add(seed)

        with open(f) as fh:
            results = json.load(fh)

        for dataset, metrics in results.items():
            if not isinstance(metrics, dict):
                continue
            for metric in args.metrics:
                if metric in metrics and isinstance(metrics[metric], (int, float, np.floating)):
                    data.setdefault(model, {}).setdefault(metric, {}).setdefault(dataset, {})[seed] = \
                        float(metrics[metric])

    # Deterministic ordering of models and datasets
    models = sorted(data.keys())
    # Datasets: union of all datasets seen (sorted for a stable column layout)
    datasets = sorted({ds for model in data.values()
                       for metric in model.values()
                       for ds in metric.keys()})

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Parsed {len(files)} files -> {len(models)} models x {len(datasets)} datasets")
    print("Seeds found per model:")
    for model in models:
        print(f"  {model:55s} seeds={sorted(seeds_per_model[model])}")

    if skipped:
        print(f"\n[WARN] Skipped {len(skipped)} unparseable files:")
        for s in skipped:
            print("  ", s)

    for metric in args.metrics:
        out_path = os.path.join(args.out_dir, f"{metric}.csv")
        with open(out_path, "w", newline="") as fh:
            writer = csv.writer(fh)
            header = ["model"]
            for ds in datasets:
                header.append(f"{ds}_mean")
                header.append(f"{ds}_std")
            writer.writerow(header)

            for model in models:
                row = [model]
                for ds in datasets:
                    vals = data.get(model, {}).get(metric, {}).get(ds, {})
                    if not vals:
                        row += ["", ""]
                        continue
                    seeds = sorted(vals, key=int)
                    arr = np.array([vals[s] for s in seeds], dtype=float)
                    row.append(f"{arr.mean():.6f}")
                    std = arr.std(ddof=1) if len(arr) > 1 else 0.0
                    row.append(f"{std:.6f}")
                writer.writerow(row)

        n_cols = 1 + 2 * len(datasets)
        print(f"Wrote {out_path}  ({len(models)} models x {n_cols} columns)")


if __name__ == "__main__":
    main()

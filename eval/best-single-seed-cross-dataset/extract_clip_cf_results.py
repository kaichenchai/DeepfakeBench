#!/usr/bin/env python3
"""Collect per-seed cross-dataset metrics from clip counterfactual JSON files.

The output is one row per seed/model/dataset.  The first three columns are
``seed``, ``model``, and ``dataset``; all remaining columns are metric names
found in the input JSON files.

By default, JSON files are searched below:
    logs/cross_dataset_test/multiple_seeds

Only files whose filename contains ``clip_cf`` are included.  The model
column is taken from the filename, with the standard result prefix and
``_seed_<seed>.json`` suffix removed.

The loader accepts either the usual cross-dataset result mapping::

    {"FaceForensics++": {"acc": 0.9, "auc": 0.95}}

or a JSON object containing a ``clip_df`` value.  ``clip_df`` may be a list
of records, a dataframe-style ``{"columns": ..., "data": ...}`` object, or
the same dataset-to-metrics mapping.

Example:
    python eval/best-single-seed-cross-dataset/extract_clip_df_results.py
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
DEFAULT_RESULTS_ROOT = HERE.parents[1] / "logs" / "cross_dataset_test" / "multiple_seeds"
DEFAULT_OUTPUT = HERE / "clip_df_results.csv"
FILE_RE = re.compile(r"^cross_dataset_results_(?P<model>.+)_seed_(?P<seed>\d+)\.json$")
SEED_RE = re.compile(r"(?:^|[_/-])seed[_-]?(?P<seed>\d+)(?:$|[_/-])", re.IGNORECASE)


def _is_metric_value(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _records_from_payload(payload: Any) -> list[dict[str, Any]]:
    """Convert supported clip_df/result representations into flat records."""
    if isinstance(payload, dict) and "clip_df" in payload:
        payload = payload["clip_df"]

    if isinstance(payload, dict) and isinstance(payload.get("columns"), list):
        columns = payload["columns"]
        return [
            dict(zip(columns, row))
            for row in payload.get("data", [])
            if isinstance(row, list)
        ]

    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]

    if isinstance(payload, dict):
        records = []
        for dataset, metrics in payload.items():
            if not isinstance(metrics, dict):
                continue
            row = {"dataset": dataset}
            row.update({key: value for key, value in metrics.items() if _is_metric_value(value)})
            if len(row) > 1:
                records.append(row)
        return records

    return []


def _find_clip_df_payload(payload: Any) -> Any | None:
    """Return the first nested clip_df value, if present."""
    if isinstance(payload, dict):
        if "clip_df" in payload:
            return payload
        for value in payload.values():
            found = _find_clip_df_payload(value)
            if found is not None:
                return found
    elif isinstance(payload, list):
        for value in payload:
            found = _find_clip_df_payload(value)
            if found is not None:
                return found
    return None


def _metadata(path: Path, root: Path) -> tuple[str, str]:
    match = FILE_RE.match(path.name)
    relative_parts = path.relative_to(root).parts if path.is_relative_to(root) else path.parts
    seed_match = SEED_RE.search("/".join(relative_parts))
    seed = (match.group("seed") if match else None) or (
        seed_match.group("seed") if seed_match else ""
    )
    model = match.group("model") if match else ""
    if not model:
        model = next((part for part in reversed(relative_parts[:-1]) if part), path.stem)
    return seed, model


def collect(results_root: Path) -> tuple[list[dict[str, Any]], list[Path]]:
    rows: list[dict[str, Any]] = []
    skipped: list[Path] = []
    for path in sorted(results_root.rglob("*.json")):
        if "clip_cf" not in path.name:
            continue
        try:
            with path.open(encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError):
            skipped.append(path)
            continue

        clip_payload = _find_clip_df_payload(payload)
        records = _records_from_payload(clip_payload) if clip_payload is not None else _records_from_payload(payload)
        seed, model = _metadata(path, results_root)
        for record in records:
            dataset = record.get("dataset") or record.get("name") or record.get("name dataset")
            if dataset is None:
                continue
            row = {"seed": seed, "model": model, "dataset": dataset}
            row.update(
                {key: value for key, value in record.items()
                 if key not in {"seed", "model", "dataset", "name", "name dataset"}
                 and _is_metric_value(value)}
            )
            rows.append(row)
    rows.sort(key=lambda row: (int(row["seed"]) if str(row["seed"]).isdigit() else str(row["seed"]),
                                   row["model"], row["dataset"]))
    return rows, skipped


def write_csv(rows: list[dict[str, Any]], output: Path) -> None:
    metrics = sorted({key for row in rows for key in row if key not in {"seed", "model", "dataset"}})
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["seed", "model", "dataset", *metrics])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if not args.results_root.is_dir():
        raise SystemExit(f"Results directory does not exist: {args.results_root}")
    rows, skipped = collect(args.results_root)
    if not rows:
        raise SystemExit(f"No usable metric records found under {args.results_root}")
    write_csv(rows, args.output)
    print(f"Wrote {len(rows)} rows to {args.output}")
    if skipped:
        print(f"Skipped {len(skipped)} unreadable JSON files")


if __name__ == "__main__":
    main()

"""Convert probe cosine-similarity arrays to one PGFPlots-ready CSV file."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert cos_sim_reals.npy and cos_sim_fakes.npy to one CSV."
    )
    parser.add_argument("--real", required=True, type=Path, help="Path to cos_sim_reals.npy")
    parser.add_argument("--fake", required=True, type=Path, help="Path to cos_sim_fakes.npy")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="CSV output path (default: cos_sim_histogram.csv beside the input arrays)",
    )
    parser.add_argument("--bins", type=int, default=60, help="Number of shared histogram bins")
    return parser.parse_args()


def load_values(path: Path, label: str) -> np.ndarray:
    values = np.asarray(np.load(path), dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError(f"{label} input contains no finite values: {path}")
    return values


def write_histogram(
    path: Path,
    edges: np.ndarray,
    real_counts: np.ndarray,
    fake_counts: np.ndarray,
) -> None:
    centers = (edges[:-1] + edges[1:]) / 2.0
    widths = np.diff(edges)
    real_density = real_counts / (real_counts.sum() * widths)
    fake_density = fake_counts / (fake_counts.sum() * widths)
    with path.open("w", newline="", encoding="ascii") as output:
        writer = csv.writer(output)
        writer.writerow(
            [
                "bin_left",
                "bin_right",
                "bin_center",
                "real_count",
                "fake_count",
                "real_density",
                "fake_density",
            ]
        )
        for index, (left, right, center) in enumerate(
            zip(edges[:-1], edges[1:], centers)
        ):
            writer.writerow(
                [
                    f"{left:.12g}",
                    f"{right:.12g}",
                    f"{center:.12g}",
                    int(real_counts[index]),
                    int(fake_counts[index]),
                    f"{real_density[index]:.12g}",
                    f"{fake_density[index]:.12g}",
                ]
            )


def main() -> None:
    args = parse_args()
    if args.bins < 1:
        raise ValueError("--bins must be positive")

    real = load_values(args.real, "real")
    fake = load_values(args.fake, "fake")
    combined = np.concatenate((real, fake))
    lower, upper = float(combined.min()), float(combined.max())
    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        lower, upper = -0.1, 1.0
    edges = np.linspace(lower, upper, args.bins + 1)
    real_counts, _ = np.histogram(real, bins=edges)
    fake_counts, _ = np.histogram(fake, bins=edges)

    output = args.output or args.real.parent / "cos_sim_histogram.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    write_histogram(output, edges, real_counts, fake_counts)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()

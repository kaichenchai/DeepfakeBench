"""Extract detector features and compute a 2-D t-SNE embedding for pgfplots.

The feature extraction is shared with ``eval/pca/extract_features.py`` so both
evaluation scripts use the same dataset, preprocessing, and checkpoint logic.
The real/fake ground-truth label is kept for every point.

The pipeline can run in one pass or in two cached stages, so the expensive GPU
forward passes run once and t-SNE can be re-run cheaply:

    # stage 1 (GPU): extract and cache features + labels
    python eval/tsne/extract_tsne.py \
        --config training/config/detector/effort_ce_orthogonal_weight.yaml \
        --weights /path/to/ckpt_final.pth \
        --dataset FaceForensics++ \
        --save-features eval/tsne/features/effort \
        --extract-only

    # stage 2 (CPU only): t-SNE from the cache
    python eval/tsne/extract_tsne.py \
        --features eval/tsne/features/effort \
        --output-dir eval/tsne/data/effort

Outputs inside ``--save-features``:
    features.npy        -- ``(samples, dimensions)`` feature matrix
    labels.npy          -- ``(samples,)`` ground truth, 0=real and 1=fake
    extraction_meta.json -- config, checkpoint, and dataset provenance

Outputs inside ``--output-dir``:
    tsne.csv  -- columns ``x,y,label,class`` (combined data)
    real.csv  -- same columns, label 0 only
    fake.csv  -- same columns, label 1 only
    meta.json -- extraction and t-SNE parameters

t-SNE is computed on every extracted sample; ``--max-points`` then
downsamples the 2-D coordinates that are written for plotting.

Example:
    python eval/tsne/extract_tsne.py \
        --model-type vanilla-clip \
        --config training/config/detector/effort.yaml \
        --dataset FaceForensics++ \
        --output-dir eval/tsne/data/vanilla_clip
"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
TRAINING_DIR = REPO_ROOT / "training"
for path in (REPO_ROOT, TRAINING_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

LABEL_NAMES = {0: "real", 1: "fake"}
FEATURES_FILE = "features.npy"
LABELS_FILE = "labels.npy"
EXTRACTION_META_FILE = "extraction_meta.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract detector features and write pgfplots-ready t-SNE data."
    )
    parser.add_argument(
        "--model-type",
        choices=("detector", "vanilla-clip"),
        default="detector",
        help="Extract from a trained detector or local vanilla CLIP (default: detector).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Detector YAML config, also used for dataset settings in vanilla-clip mode.",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        help="Detector checkpoint. Not used by vanilla-clip mode.",
    )
    parser.add_argument(
        "--clip-path",
        type=Path,
        default=REPO_ROOT / "models--openai--clip-vit-large-patch14",
        help="Local Hugging Face CLIP directory for vanilla-clip mode.",
    )
    parser.add_argument(
        "--dataset",
        help="Dataset name to process. Required unless --features is used.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for tsne.csv, real.csv, fake.csv, and meta.json.",
    )
    parser.add_argument(
        "--save-features",
        type=Path,
        help=(
            "Cache the extracted features and ground-truth labels in this "
            "directory (features.npy, labels.npy, extraction_meta.json)."
        ),
    )
    parser.add_argument(
        "--extract-only",
        action="store_true",
        help="Stop after --save-features; do not run t-SNE.",
    )
    parser.add_argument(
        "--features",
        type=Path,
        help=(
            "Reuse cached features from this directory instead of running a "
            "model. Makes --config, --weights, and --dataset unnecessary."
        ),
    )
    parser.add_argument(
        "--batch-size", type=int, default=None, help="Override test_batchSize."
    )
    parser.add_argument(
        "--workers", type=int, default=None, help="Override the config worker count."
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Inference device (default: auto).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional limit on extracted samples.",
    )
    parser.add_argument(
        "--perplexity",
        type=float,
        default=30.0,
        help="t-SNE perplexity (default: 30).",
    )
    parser.add_argument(
        "--pca-components",
        type=int,
        default=50,
        help="PCA dimensions before t-SNE; 0 disables PCA (default: 50).",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=1000,
        help="t-SNE optimisation iterations (default: 1000).",
    )
    parser.add_argument(
        "--angle",
        type=float,
        default=0.5,
        help="Barnes-Hut accuracy/speed trade-off in [0, 1] (default: 0.5).",
    )
    parser.add_argument(
        "--seed", type=int, default=1024, help="Random seed (default: 1024)."
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="t-SNE worker count; -1 uses all cores (default: -1).",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=100,
        help="Maximum points written for plotting; 0 keeps all (default: 100).",
    )
    parser.add_argument(
        "--balance",
        action="store_true",
        help="Downsample equal numbers of real and fake points.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.features is not None:
        if not args.features.is_dir():
            raise SystemExit(f"Feature directory does not exist: {args.features}")
        for flag, value in (
            ("--config", args.config),
            ("--weights", args.weights),
            ("--dataset", args.dataset),
            ("--save-features", args.save_features),
        ):
            if value is not None:
                raise SystemExit(f"{flag} cannot be combined with --features.")
        if args.extract_only:
            raise SystemExit("--extract-only cannot be combined with --features.")
        if args.max_samples is not None:
            raise SystemExit("--max-samples cannot be combined with --features.")
        if args.batch_size is not None or args.workers is not None:
            raise SystemExit(
                "--batch-size and --workers cannot be combined with --features."
            )
    else:
        if args.config is None:
            raise SystemExit("--config is required unless --features is used.")
        if args.dataset is None:
            raise SystemExit("--dataset is required unless --features is used.")
        if not args.config.is_file():
            raise SystemExit(f"Config file does not exist: {args.config}")
        if args.model_type == "detector":
            if args.weights is None:
                raise SystemExit("--weights is required when --model-type is detector.")
            if not args.weights.is_file():
                raise SystemExit(f"Weights file does not exist: {args.weights}")
        elif args.weights is not None:
            raise SystemExit("--weights is only valid with --model-type detector.")
        elif not args.clip_path.is_dir():
            raise SystemExit(f"Local CLIP directory does not exist: {args.clip_path}")
        if args.extract_only and args.save_features is None:
            raise SystemExit("--extract-only requires --save-features.")

    if not args.extract_only and args.output_dir is None:
        raise SystemExit("--output-dir is required unless --extract-only is used.")
    if args.max_samples is not None and args.max_samples < 1:
        raise SystemExit("--max-samples must be at least 1.")
    if args.batch_size is not None and args.batch_size < 1:
        raise SystemExit("--batch-size must be at least 1.")
    if args.workers is not None and args.workers < 0:
        raise SystemExit("--workers cannot be negative.")
    if args.perplexity <= 0:
        raise SystemExit("--perplexity must be positive.")
    if args.pca_components < 0:
        raise SystemExit("--pca-components cannot be negative.")
    if args.pca_components == 1:
        raise SystemExit("--pca-components must be 0 or at least 2.")
    if args.max_iter < 250:
        raise SystemExit("--max-iter must be at least 250.")
    if not 0 <= args.angle <= 1:
        raise SystemExit("--angle must be between 0 and 1.")
    if args.n_jobs == 0 or args.n_jobs < -1:
        raise SystemExit("--n-jobs must be -1 or a positive integer.")
    if args.max_points < 0:
        raise SystemExit("--max-points cannot be negative.")
    if args.max_points == 1:
        raise SystemExit("--max-points must be 0 or at least 2.")


def get_device(requested: str):
    import torch

    if requested == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available.")
    return torch.device(
        "cuda"
        if requested == "cuda" or (requested == "auto" and torch.cuda.is_available())
        else "cpu"
    )


def extract_features(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    # Imported here so ``--features`` mode stays lightweight (no torch/dataset).
    from detectors import DETECTOR
    from eval.pca.extract_features import (
        VanillaClipFeatures,
        extract,
        load_config,
        load_weights,
        prepare_loader,
    )

    config = load_config(args.config, args.dataset)
    device = get_device(args.device)
    print(f"Feature extraction device: {device}", flush=True)
    if device.type == "cpu" and args.device == "auto":
        print(
            "WARNING: CUDA is not available to PyTorch; feature extraction is "
            "running on CPU and may be very slow. Check torch.cuda.is_available().",
            flush=True,
        )

    if args.model_type == "vanilla-clip":
        model = VanillaClipFeatures(args.clip_path).to(device)
    else:
        model = DETECTOR[config["model_name"]](config).to(device)
        load_weights(model, args.weights, device)

    # The dataset shuffles its sample list with Python's global RNG. Seed it
    # immediately before construction so every model sees the same order.
    random.seed(args.seed)
    np.random.seed(args.seed)
    loader = prepare_loader(config, args.batch_size, args.workers)
    return extract(model, loader, device, args.max_samples)


def save_features(
    output_dir: Path,
    features: np.ndarray,
    labels: np.ndarray,
    metadata: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / FEATURES_FILE, features.astype(np.float32, copy=False))
    np.save(output_dir / LABELS_FILE, labels.astype(np.int64, copy=False))
    (output_dir / EXTRACTION_META_FILE).write_text(
        json.dumps(metadata, indent=2) + "\n"
    )


def load_features(features_dir: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    features_path = features_dir / FEATURES_FILE
    labels_path = features_dir / LABELS_FILE

    for path in (features_path, labels_path):
        if not path.is_file():
            raise SystemExit(f"Missing cached feature file: {path}")

    features = np.load(features_path, allow_pickle=False)
    labels = np.load(labels_path, allow_pickle=False)

    if features.ndim != 2:
        raise SystemExit(
            f"{features_path} must be 2-D (samples, features); "
            f"received shape {features.shape}."
        )
    if labels.ndim != 1:
        raise SystemExit(
            f"{labels_path} must be 1-D (samples,); received shape {labels.shape}."
        )
    if labels.shape[0] != features.shape[0]:
        raise SystemExit(
            f"Feature/label length mismatch: {features.shape[0]} rows in "
            f"{features_path} but {labels.shape[0]} in {labels_path}."
        )

    meta_path = features_dir / EXTRACTION_META_FILE
    metadata: dict[str, Any] = {}
    if meta_path.is_file():
        metadata = json.loads(meta_path.read_text())
    return features, labels, metadata


def build_tsne(args: argparse.Namespace) -> TSNE:
    kwargs: dict[str, Any] = {
        "n_components": 2,
        "perplexity": args.perplexity,
        "learning_rate": "auto",
        "init": "pca",
        "random_state": args.seed,
        "n_jobs": args.n_jobs,
        "angle": args.angle,
    }
    # sklearn renamed n_iter to max_iter in 1.5.
    if "max_iter" in inspect.signature(TSNE).parameters:
        kwargs["max_iter"] = args.max_iter
    else:  # pragma: no cover - compatibility with older scikit-learn
        kwargs["n_iter"] = args.max_iter
    return TSNE(**kwargs)


def compute_embedding(
    features: np.ndarray, args: argparse.Namespace
) -> tuple[np.ndarray, int, int, float | None]:
    features = features.astype(np.float32, copy=False)
    if not np.isfinite(features).all():
        raise SystemExit("Extracted features contain NaN or infinite values.")
    if features.shape[0] < 3:
        raise SystemExit("t-SNE requires at least three samples.")
    if features.shape[1] < 2:
        raise SystemExit("t-SNE requires at least two feature dimensions.")

    original_dimensions = int(features.shape[1])
    pca_components = min(
        args.pca_components, features.shape[0] - 1, original_dimensions
    )
    if pca_components > 0 and pca_components < original_dimensions:
        features = PCA(
            n_components=pca_components, random_state=args.seed
        ).fit_transform(features)
    else:
        pca_components = 0

    max_perplexity = (features.shape[0] - 1) / 3.0
    if args.perplexity > max_perplexity:
        raise SystemExit(
            f"--perplexity {args.perplexity:g} is too large for "
            f"{features.shape[0]} samples; use at most {max_perplexity:.2f}."
        )

    tsne = build_tsne(args)
    coords = tsne.fit_transform(features)
    kl_divergence = (
        float(tsne.kl_divergence_) if hasattr(tsne, "kl_divergence_") else None
    )
    return coords, original_dimensions, pca_components, kl_divergence


def downsample_output(
    coords: np.ndarray,
    labels: np.ndarray,
    max_points: int,
    balance: bool,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Subsample the 2-D embedding after t-SNE.

    The dataset order and this RNG are seeded with the same value for every
    model, so each model's plot uses the same sample indices.
    """
    if max_points == 0 or max_points >= coords.shape[0]:
        return coords, labels

    rng = np.random.default_rng(seed)
    if balance:
        real_indices = np.flatnonzero(labels == 0)
        fake_indices = np.flatnonzero(labels == 1)
        per_class = max_points // 2
        selected = np.concatenate(
            [
                rng.choice(
                    real_indices, min(per_class, real_indices.size), replace=False
                ),
                rng.choice(
                    fake_indices, min(per_class, fake_indices.size), replace=False
                ),
            ]
        )
    else:
        selected = rng.choice(coords.shape[0], max_points, replace=False)

    selected.sort()
    return coords[selected], labels[selected]


def write_csv(path: Path, coords: np.ndarray, labels: np.ndarray) -> None:
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("x", "y", "label", "class"))
        for (x, y), label in zip(coords, labels):
            label = int(label)
            writer.writerow((f"{x:.6f}", f"{y:.6f}", label, LABEL_NAMES[label]))


def write_outputs(
    output_dir: Path,
    coords: np.ndarray,
    labels: np.ndarray,
    metadata: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = labels.astype(np.int64, copy=False)

    write_csv(output_dir / "tsne.csv", coords, labels)
    write_csv(output_dir / "real.csv", coords[labels == 0], labels[labels == 0])
    write_csv(output_dir / "fake.csv", coords[labels == 1], labels[labels == 1])
    (output_dir / "meta.json").write_text(json.dumps(metadata, indent=2) + "\n")


def main() -> None:
    args = parse_args()
    validate_args(args)

    if args.features is not None:
        features, labels, source = load_features(args.features)
        print(
            f"Loaded {features.shape[0]} samples with {features.shape[1]} "
            f"features from {args.features}",
            flush=True,
        )
        model_type = source.get("model_type")
        config = source.get("config")
        weights = source.get("weights")
        dataset = source.get("dataset")
    else:
        start = time.time()
        features, labels = extract_features(args)
        print(
            f"Feature extraction finished in {time.time() - start:.1f}s", flush=True
        )
        model_type = args.model_type
        config = str(args.config)
        weights = str(args.weights) if args.weights is not None else None
        dataset = args.dataset

        if args.save_features is not None:
            save_features(
                args.save_features,
                features,
                labels,
                {
                    "model_type": model_type,
                    "config": config,
                    "weights": weights,
                    "dataset": dataset,
                    "samples": int(features.shape[0]),
                    "dimensions": int(features.shape[1]),
                    "max_samples": args.max_samples,
                    "random_state": args.seed,
                },
            )
            print(
                f"Cached features and labels in {args.save_features}",
                flush=True,
            )
            if args.extract_only:
                return

    unique_labels = np.unique(labels)
    unknown = sorted(set(unique_labels.tolist()) - LABEL_NAMES.keys())
    if unknown:
        raise SystemExit(
            f"Unexpected labels {unknown}; this script expects 0=real and 1=fake."
        )

    start = time.time()
    coords, original_dimensions, pca_components, kl_divergence = compute_embedding(
        features, args
    )
    print(f"PCA + t-SNE finished in {time.time() - start:.1f}s", flush=True)
    label_counts = {
        LABEL_NAMES[int(label)]: int((labels == label).sum()) for label in unique_labels
    }
    output_coords, output_labels = downsample_output(
        coords, labels, args.max_points, args.balance, args.seed
    )
    output_label_counts = {
        LABEL_NAMES[int(label)]: int((output_labels == label).sum())
        for label in np.unique(output_labels)
    }

    metadata = {
        "model_type": model_type,
        "config": config,
        "weights": weights,
        "dataset": dataset,
        "features_dir": str(args.features) if args.features is not None else None,
        "samples": int(features.shape[0]),
        "label_counts": label_counts,
        "output_samples": int(output_coords.shape[0]),
        "output_label_counts": output_label_counts,
        "max_points": args.max_points,
        "balance": args.balance,
        "original_dimensions": original_dimensions,
        "pca_components": pca_components,
        "perplexity": args.perplexity,
        "max_iter": args.max_iter,
        "angle": args.angle,
        "random_state": args.seed,
        "kl_divergence": kl_divergence,
    }
    write_outputs(args.output_dir, output_coords, output_labels, metadata)

    print(f"Samples: {features.shape[0]} ({label_counts})")
    print(f"Plot points: {output_coords.shape[0]} ({output_label_counts})")
    print(f"Feature dimensions: {original_dimensions}")
    print(
        "t-SNE: "
        f"perplexity={args.perplexity:g}, "
        f"pca_components={pca_components}, "
        f"max_iter={args.max_iter}, "
        f"angle={args.angle:g}, "
        f"seed={args.seed}"
    )
    print(f"Data written to: {args.output_dir}")


if __name__ == "__main__":
    main()

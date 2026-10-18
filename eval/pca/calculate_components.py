"""Calculate the number of PCA components needed to retain a variance threshold.

The input file must contain a feature matrix with shape ``(samples, features)``.
NumPy (``.npy``/``.npz``), pickle, and PyTorch files are supported. For mapping
objects, the script looks for a feature-like key (``features``, ``feat``,
``embeddings``, or ``embedding``), then falls back to the only 2-D numeric
value.

Example:
    python eval/pca/calculate_components.py \
        --model-file /path/to/features.npy \
        --threshold 90
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.decomposition import PCA


FEATURE_KEYS = ("features", "feat", "embeddings", "embedding")


def _load_file(path: Path) -> Any:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.load(path, allow_pickle=False)
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            if not archive.files:
                raise ValueError(f"{path} does not contain any arrays")
            if len(archive.files) == 1:
                return archive[archive.files[0]]
            return {key: archive[key] for key in archive.files}
    if suffix in {".pkl", ".pickle"}:
        with path.open("rb") as file:
            return pickle.load(file)
    if suffix in {".pt", ".pth", ".ckpt"}:
        try:
            import torch
        except ImportError as error:
            raise RuntimeError(
                "PyTorch is required to load .pt, .pth, and .ckpt files."
            ) from error
        return torch.load(path, map_location="cpu", weights_only=True)
    raise ValueError(
        f"Unsupported file extension {path.suffix!r}. "
        "Use .npy, .npz, .pkl, .pickle, .pt, .pth, or .ckpt."
    )


def _as_array(value: Any, source: str) -> np.ndarray | None:
    if isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.number):
        return value

    try:
        import torch
    except ImportError:
        torch = None
    if torch is not None and isinstance(value, torch.Tensor):
        if value.dtype != torch.bool:
            return value.detach().cpu().numpy()
    return None


def _find_features(value: Any, source: str = "input") -> np.ndarray:
    array = _as_array(value, source)
    if array is not None:
        return array

    if isinstance(value, dict):
        for key in FEATURE_KEYS:
            if key in value:
                candidate = _as_array(value[key], f"{source}[{key!r}]")
                if candidate is not None:
                    return candidate

        candidates = [
            (key, candidate)
            for key, item in value.items()
            if (candidate := _as_array(item, f"{source}[{key!r}]")) is not None
            and candidate.ndim == 2
        ]
        if len(candidates) == 1:
            return candidates[0][1]
        if candidates:
            names = ", ".join(str(key) for key, _ in candidates)
            raise ValueError(
                f"{source} contains multiple 2-D numeric values ({names}); "
                "store the feature matrix under 'features' or pass a file "
                "containing only the feature matrix."
            )

    raise ValueError(
        f"Could not find a numeric feature matrix in {source}. "
        "Expected an array with shape (samples, features), or a mapping "
        "containing 'features', 'feat', 'embeddings', or 'embedding'."
    )


def calculate_pca_values(features: np.ndarray, threshold: float) -> dict[str, Any]:
    """Return plot values for a feature matrix and variance threshold."""
    if features.ndim != 2:
        raise ValueError(
            f"Expected a 2-D feature matrix, received shape {features.shape}."
        )
    if features.shape[0] < 2 or features.shape[1] < 1:
        raise ValueError(
            f"Feature matrix must contain at least two samples and one feature; "
            f"received shape {features.shape}."
        )
    if not np.isfinite(features).all():
        raise ValueError("Feature matrix contains NaN or infinite values.")

    pca = PCA()
    pca.fit(features.astype(np.float64, copy=False))
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    component_count = min(
        int(np.searchsorted(cumulative, threshold / 100.0) + 1),
        cumulative.size,
    )

    return {
        "samples": int(features.shape[0]),
        "original_dimensions": int(features.shape[1]),
        "threshold_percent": threshold,
        "components": component_count,
        "retained_variance_percent": float(cumulative[component_count - 1] * 100),
        "variance_loss_percent": float((1 - cumulative[component_count - 1]) * 100),
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "cumulative_explained_variance_ratio": cumulative.tolist(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate PCA components needed to retain a variance threshold."
    )
    parser.add_argument(
        "--model-file",
        "--model",
        type=Path,
        required=True,
        help="File containing a (samples, features) matrix.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=90.0,
        help="Cumulative explained variance threshold as a percent (default: 90).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON file for the calculated values.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model_file.is_file():
        raise SystemExit(f"Model file does not exist: {args.model_file}")
    if not 0 < args.threshold <= 100:
        raise SystemExit("--threshold must be greater than 0 and at most 100.")

    features = _find_features(_load_file(args.model_file), str(args.model_file))
    results = calculate_pca_values(features, args.threshold)

    print(f"Samples: {results['samples']}")
    print(f"Original dimensions: {results['original_dimensions']}")
    print(f"Threshold: {results['threshold_percent']:.4g}%")
    print(f"Components required: {results['components']}")
    print(f"Retained variance: {results['retained_variance_percent']:.6f}%")
    print(f"Variance loss: {results['variance_loss_percent']:.6f}%")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2) + "\n")
        print(f"Values written to: {args.output}")


if __name__ == "__main__":
    main()

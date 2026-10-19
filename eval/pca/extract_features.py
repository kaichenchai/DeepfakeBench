"""Extract detector features for PCA analysis.

Example:
    python eval/pca/extract_features.py \
        --config training/config/detector/effort.yaml \
        --weights /path/to/model.pth \
        --dataset Celeb-DF-v2 \
        --output features.npy

    python eval/pca/extract_features.py \
        --model-type vanilla-clip \
        --config training/config/detector/effort.yaml \
        --dataset Celeb-DF-v2 \
        --output vanilla_clip_features.npy
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.utils.data
import yaml
from tqdm import tqdm
from transformers import CLIPModel


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
TRAINING_DIR = REPO_ROOT / "training"
for path in (REPO_ROOT, TRAINING_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

os.chdir(REPO_ROOT)

from dataset.abstract_dataset import DeepfakeAbstractBaseDataset
from detectors import DETECTOR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract model feature vectors for PCA analysis."
    )
    parser.add_argument(
        "--model-type",
        choices=("detector", "vanilla-clip"),
        default="detector",
        help="Extract from a trained detector or local vanilla CLIP (default: detector).",
    )
    parser.add_argument(
        "--config", type=Path, required=True,
        help="Detector YAML config, also used for dataset settings in vanilla-clip mode.",
    )
    parser.add_argument(
        "--weights", type=Path,
        help="Detector checkpoint. Not used by vanilla-clip mode.",
    )
    parser.add_argument(
        "--clip-path",
        type=Path,
        default=REPO_ROOT / "models--openai--clip-vit-large-patch14",
        help="Local Hugging Face CLIP directory for vanilla-clip mode.",
    )
    parser.add_argument("--dataset", required=True, help="Dataset name to process.")
    parser.add_argument(
        "--output", type=Path, required=True, help="Output .npy feature matrix."
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
        "--labels-output",
        type=Path,
        help="Optional .npy file for the corresponding dataset labels.",
    )
    return parser.parse_args()


def load_config(config_path: Path, dataset: str) -> dict:
    with config_path.open() as file:
        config = yaml.safe_load(file)
    with (TRAINING_DIR / "config" / "test_config.yaml").open() as file:
        config.update(yaml.safe_load(file))
    config["test_dataset"] = dataset
    return config


def load_weights(model: torch.nn.Module, weights_path: Path, device: torch.device) -> None:
    checkpoint = torch.load(weights_path, map_location=device)
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unsupported checkpoint format: {weights_path}")

    state_dict = {
        key.removeprefix("module."): value for key, value in checkpoint.items()
    }
    model.load_state_dict(state_dict, strict=True)


class VanillaClipFeatures(torch.nn.Module):
    def __init__(self, model_path: Path):
        super().__init__()
        self.backbone = CLIPModel.from_pretrained(model_path).vision_model

    def forward(self, data_dict: dict, inference: bool = False) -> dict:
        features = self.backbone(data_dict["image"])["pooler_output"]
        return {"feat": features}


def prepare_loader(config: dict, batch_size: int | None, workers: int | None):
    dataset = DeepfakeAbstractBaseDataset(config=config, mode="test")
    return torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size if batch_size is not None else config["test_batchSize"],
        shuffle=False,
        num_workers=workers if workers is not None else int(config["workers"]),
        collate_fn=dataset.collate_fn,
        drop_last=False,
    )


def extract(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    max_samples: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    features: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    sample_count = 0

    model.eval()
    with torch.inference_mode():
        for data_dict in tqdm(loader, desc="Extracting features"):
            batch_labels = data_dict["label"].detach().cpu().numpy()
            model_batch = {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in data_dict.items()
            }
            predictions = model(model_batch, inference=True)
            if "feat" not in predictions:
                raise KeyError(
                    "Detector output does not contain 'feat'; this script "
                    "expects the detector feature representation."
                )

            batch_features = predictions["feat"].detach().cpu().numpy()
            if batch_features.ndim != 2:
                batch_features = batch_features.reshape(batch_features.shape[0], -1)

            if max_samples is not None:
                remaining = max_samples - sample_count
                if remaining <= 0:
                    break
                batch_features = batch_features[:remaining]
                batch_labels = batch_labels[:remaining]

            features.append(batch_features)
            labels.append(batch_labels)
            sample_count += len(batch_features)
            if max_samples is not None and sample_count >= max_samples:
                break

    if not features:
        raise RuntimeError("The dataset produced no feature vectors.")
    return np.concatenate(features), np.concatenate(labels)


def main() -> None:
    args = parse_args()
    for path, name in ((args.config, "Config"),):
        if not path.is_file():
            raise SystemExit(f"{name} file does not exist: {path}")
    if args.model_type == "detector":
        if args.weights is None:
            raise SystemExit("--weights is required when --model-type is detector.")
        if not args.weights.is_file():
            raise SystemExit(f"Weights file does not exist: {args.weights}")
    elif args.weights is not None:
        raise SystemExit("--weights is only valid with --model-type detector.")
    elif not args.clip_path.is_dir():
        raise SystemExit(f"Local CLIP directory does not exist: {args.clip_path}")
    if args.max_samples is not None and args.max_samples < 1:
        raise SystemExit("--max-samples must be at least 1.")
    if args.batch_size is not None and args.batch_size < 1:
        raise SystemExit("--batch-size must be at least 1.")
    if args.workers is not None and args.workers < 0:
        raise SystemExit("--workers cannot be negative.")

    config = load_config(args.config, args.dataset)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available.")
    device = torch.device(
        "cuda" if args.device == "cuda" or (
            args.device == "auto" and torch.cuda.is_available()
        ) else "cpu"
    )

    if args.model_type == "vanilla-clip":
        model = VanillaClipFeatures(args.clip_path).to(device)
    else:
        model = DETECTOR[config["model_name"]](config).to(device)
        load_weights(model, args.weights, device)
    features, labels = extract(model, prepare_loader(config, args.batch_size, args.workers),
                               device, args.max_samples)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, features)
    print(f"Saved {features.shape[0]} samples with {features.shape[1]} features to {args.output}")

    if args.labels_output is not None:
        args.labels_output.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.labels_output, labels)
        print(f"Saved labels to {args.labels_output}")


if __name__ == "__main__":
    main()

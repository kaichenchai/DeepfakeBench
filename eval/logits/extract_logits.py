"""Extract detector logits on all requested test datasets.

Each model is supplied as ``NAME CONFIG WEIGHTS``. For every dataset, the
script writes pgfplots-compatible CSV tables with one row per image.

Example:
    python eval/logits/extract_logits.py \
        --model effort training/config/detector/effort.yaml \
        training/pretrained/effort_clip_L14_trainOn_FaceForensic.pth \
        --model ours training/config/detector/effort_ce.yaml \
        logs/training/effort_ce_effort_custom_2026-08-22-16-45-56/final/ckpt_final.pth \
        --datasets FaceForensics++ UADFV \
        --output-dir eval/logits/results
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Iterable
import numpy as np
import torch
import torch.utils.data
import yaml
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
TRAINING_DIR = REPO_ROOT / "training"
for path in (REPO_ROOT, TRAINING_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dataset.abstract_dataset import DeepfakeAbstractBaseDataset
from detectors import DETECTOR


DEFAULT_DATASETS = (
    "FaceForensics++",
    "UADFV",
    "Celeb-DF-v1",
    "Celeb-DF-v2",
    "DFDC",
    "DFDCP",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract detector logits and plot their real/fake distributions."
    )
    parser.add_argument(
        "--model",
        nargs=3,
        action="append",
        metavar=("NAME", "CONFIG", "WEIGHTS"),
        required=True,
        help="Model name, detector YAML, and checkpoint. Repeat for each model.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DEFAULT_DATASETS),
        help="Datasets to process (default: all standard test datasets).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "eval" / "logits" / "results",
        help="Directory for CSV, NPZ, and PNG outputs.",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    return parser.parse_args()


def load_config(config_path: Path, dataset: str) -> dict:
    with config_path.open() as file:
        config = yaml.safe_load(file)
    with (TRAINING_DIR / "config" / "test_config.yaml").open() as file:
        config.update(yaml.safe_load(file))
    config["test_dataset"] = dataset
    return config


def load_weights(
    model: torch.nn.Module, weights_path: Path, device: torch.device
) -> None:
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


def make_loader(
    config: dict, dataset: str, batch_size: int | None, workers: int
) -> torch.utils.data.DataLoader:
    dataset_config = dict(config)
    dataset_config["test_dataset"] = dataset
    test_set = DeepfakeAbstractBaseDataset(config=dataset_config, mode="test")
    return torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=batch_size or int(dataset_config["test_batchSize"]),
        shuffle=False,
        num_workers=workers,
        collate_fn=test_set.collate_fn,
        drop_last=False,
    )


def extract(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    model_name: str,
    dataset: str,
) -> dict[str, np.ndarray | str]:
    logits: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    image_paths: list[str] = []
    model.eval()
    with torch.inference_mode():
        for data_dict in tqdm(
            loader, desc=f"{model_name} / {dataset}", leave=False
        ):
            model_batch = {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in data_dict.items()
            }
            predictions = model(model_batch, inference=True)
            if "cls" not in predictions:
                raise KeyError(
                    f"Model {model_name!r} did not return 'cls' logits."
                )
            batch_logits = predictions["cls"].detach().cpu().numpy()
            if batch_logits.ndim != 2 or batch_logits.shape[1] != 2:
                raise ValueError(
                    f"Expected [batch, 2] logits from {model_name!r}; "
                    f"got {batch_logits.shape}."
                )
            logits.append(batch_logits)
            labels.append(data_dict["label"].detach().cpu().numpy().astype(np.int64))
            image_paths.extend(str(path) for path in data_dict["image"])

    if not logits:
        raise RuntimeError(f"Dataset {dataset!r} produced no samples.")
    return {
        "model": model_name,
        "dataset": dataset,
        "logits": np.concatenate(logits),
        "labels": np.concatenate(labels),
        "image_paths": np.asarray(image_paths),
    }


def write_table(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(
    results: Iterable[dict[str, np.ndarray | str]], output_dir: Path
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for result in results:
        logits = result["logits"]
        labels = result["labels"]
        image_paths = result["image_paths"]
        model_name = str(result["model"])
        dataset = str(result["dataset"])
        safe_dataset = dataset.replace("/", "_").replace(" ", "_")
        safe_model = model_name.replace("/", "_").replace(" ", "_")
        model_rows = []
        for path, label, pair in zip(image_paths, labels, logits):
            row = {
                "model": model_name,
                "dataset": dataset,
                "image": str(path),
                "label": int(label),
                "logit_real": float(pair[0]),
                "logit_fake": float(pair[1]),
                "logit_margin_fake_minus_real": float(pair[1] - pair[0]),
            }
            rows.append(row)
            model_rows.append(row)
        write_table(output_dir / f"{safe_dataset}__{safe_model}.csv", model_rows)
        write_table(
            output_dir / f"{safe_dataset}__{safe_model}__real.csv",
            [row for row in model_rows if row["label"] == 0],
        )
        write_table(
            output_dir / f"{safe_dataset}__{safe_model}__fake.csv",
            [row for row in model_rows if row["label"] == 1],
        )

    write_table(output_dir / "logits.csv", rows)


def main() -> None:
    args = parse_args()
    if args.workers < 0 or (args.batch_size is not None and args.batch_size < 1):
        raise SystemExit("--workers must be non-negative and --batch-size must be positive.")
    device = torch.device(
        "cuda"
        if args.device == "cuda"
        or (args.device == "auto" and torch.cuda.is_available())
        else "cpu"
    )
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available.")

    models = []
    for name, config_path, weights_path in args.model:
        config = Path(config_path)
        weights = Path(weights_path)
        if not config.is_file():
            raise SystemExit(f"Config file does not exist: {config}")
        if not weights.is_file():
            raise SystemExit(f"Weights file does not exist: {weights}")
        model_config = load_config(config, args.datasets[0])
        model = DETECTOR[model_config["model_name"]](model_config).to(device)
        load_weights(model, weights, device)
        models.append((name, model, model_config))

    results = []
    loader_config = models[0][2]
    for dataset in args.datasets:
        loader = make_loader(loader_config, dataset, args.batch_size, args.workers)
        for name, model, _ in models:
            results.append(extract(model, loader, device, name, dataset))
    write_outputs(results, args.output_dir)
    print(f"Saved pgfplots-compatible logits tables to {args.output_dir}")


if __name__ == "__main__":
    main()

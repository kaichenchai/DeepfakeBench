"""Generate GradCAM heatmaps for a trained DeepfakeBench detector.

The detector's CLIP ViT backbone exposes patch tokens at every transformer
block, so GradCAM is taken from the last encoder layer by default. The class
score is back-propagated to that layer's token activations, the per-channel
gradient is used as the CAM weight, and the resulting 16x16 map (for a 224 px
input with patch size 14) is upsampled to the input resolution.

Outputs (inside ``--output-dir``):
    images/       -- per-sample input, heatmap, and overlay images
    summary.png   -- contact sheet of the first ``--summary-samples`` samples
    results.csv   -- label, prediction, scores, target class, and paths
    meta.json     -- model, layer, sampling, and rendering parameters
    gradcam.npy   -- raw CAM maps, only with ``--save-npy``

Example:
    python eval/gradcam/generate_gradcam.py \
        --config training/config/detector/effort_ce_orthogonal_weight.yaml \
        --weights logs/training/effort_ce_orthogonal_weight_effort_custom_2026-08-23-19-55-41/final/ckpt_final.pth \
        --dataset FaceForensics++ \
        --output-dir eval/gradcam/data/effort \
        --num-samples 32 --balance
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import colormaps
from PIL import Image, ImageDraw

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
TRAINING_DIR = REPO_ROOT / "training"
for path in (REPO_ROOT, TRAINING_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

os.chdir(REPO_ROOT)

from dataset.abstract_dataset import DeepfakeAbstractBaseDataset  # noqa: E402
from detectors import DETECTOR  # noqa: E402
from eval.pca.extract_features import load_config, load_weights  # noqa: E402

LABEL_NAMES = {0: "real", 1: "fake"}
TARGET_CLASSES = ("auto", "fake", "real", "pred")
TOKEN_REDUCTIONS = ("patch", "cls")
CATEGORY_CHOICES = (
    "real_high",
    "fake_high",
    "misclassified_low",
    "misclassified_high",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate GradCAM heatmaps for a trained detector."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Detector YAML config, also used for dataset settings.",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        required=True,
        help="Detector checkpoint.",
    )
    parser.add_argument("--dataset", required=True, help="Dataset name to process.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for images, results.csv, and meta.json.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Inference device (default: auto).",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=16,
        help="Number of samples to render when not using --categories (default: 16).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit the candidate pool to the first N dataset entries.",
    )
    parser.add_argument(
        "--balance",
        action="store_true",
        help="Sample equal numbers of real and fake frames.",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        choices=CATEGORY_CHOICES,
        default=None,
        help=(
            "Render frames by outcome instead of sampling at random: "
            "real_high/fake_high (correct and confident), "
            "misclassified_low (wrong and close to the decision boundary), "
            "misclassified_high (wrong and confident). Requires a scan pass."
        ),
    )
    parser.add_argument(
        "--per-category",
        type=int,
        default=4,
        help="Frames rendered per category (default: 4).",
    )
    parser.add_argument(
        "--scan-samples",
        type=int,
        default=1024,
        help=(
            "Frames to score when using --categories; 0 scans the whole "
            "candidate pool (default: 1024)."
        ),
    )
    parser.add_argument(
        "--scan-batch-size",
        type=int,
        default=16,
        help="Batch size for the confidence scan (default: 16).",
    )
    parser.add_argument(
        "--high-confidence",
        type=float,
        default=0.90,
        help="Minimum predicted-class probability for '*_high' categories (default: 0.90).",
    )
    parser.add_argument(
        "--low-confidence",
        type=float,
        default=0.65,
        help=(
            "Maximum predicted-class probability for misclassified_low "
            "(default: 0.65)."
        ),
    )
    parser.add_argument(
        "--seed", type=int, default=1024, help="Random seed (default: 1024)."
    )
    parser.add_argument(
        "--target-class",
        choices=TARGET_CLASSES,
        default=None,
        help=(
            "Class score to explain: fake (1), real (0), pred, or auto. "
            "Default: fake for random sampling; auto (ground-truth class for "
            "correct high-confidence frames, the prediction for misclassifications) "
            "when --categories is set."
        ),
    )
    parser.add_argument(
        "--token-reduction",
        choices=TOKEN_REDUCTIONS,
        default="patch",
        help=(
            "How token gradients become channel weights: 'patch' averages "
            "over patch tokens, 'cls' uses the CLS token gradient "
            "(default: patch)."
        ),
    )
    parser.add_argument(
        "--layer-index",
        type=int,
        default=-1,
        help="Index into the backbone encoder layers (default: -1, last).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Heatmap opacity in the overlay, 0-1 (default: 0.5).",
    )
    parser.add_argument(
        "--cmap",
        default="jet",
        help="Matplotlib colormap for the heatmap (default: jet).",
    )
    parser.add_argument(
        "--summary-samples",
        type=int,
        default=8,
        help="Samples in summary.png; 0 disables it (default: 8).",
    )
    parser.add_argument(
        "--save-npy",
        action="store_true",
        help="Also write the raw CAM maps to gradcam.npy.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.config.is_file():
        raise SystemExit(f"Config file does not exist: {args.config}")
    if not args.weights.is_file():
        raise SystemExit(f"Weights file does not exist: {args.weights}")
    if args.num_samples < 1:
        raise SystemExit("--num-samples must be at least 1.")
    if args.max_samples is not None and args.max_samples < 1:
        raise SystemExit("--max-samples must be at least 1.")
    if args.per_category < 1:
        raise SystemExit("--per-category must be at least 1.")
    if args.scan_samples < 0:
        raise SystemExit("--scan-samples cannot be negative.")
    if args.scan_batch_size < 1:
        raise SystemExit("--scan-batch-size must be at least 1.")
    if not 0.5 < args.low_confidence <= 1.0:
        raise SystemExit("--low-confidence must be in (0.5, 1].")
    if not 0.5 <= args.high_confidence <= 1.0:
        raise SystemExit("--high-confidence must be in [0.5, 1].")
    if args.low_confidence > args.high_confidence:
        raise SystemExit("--low-confidence cannot exceed --high-confidence.")
    if args.categories is not None and args.balance:
        raise SystemExit(
            "Use --per-category with --categories instead of --balance."
        )
    if not 0.0 <= args.alpha <= 1.0:
        raise SystemExit("--alpha must be between 0 and 1.")
    if args.summary_samples < 0:
        raise SystemExit("--summary-samples cannot be negative.")
    if args.cmap not in colormaps:
        raise SystemExit(f"Unknown matplotlib colormap: {args.cmap}")


def get_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available.")
    if requested == "auto" and not torch.cuda.is_available():
        print("[gradcam] CUDA is unavailable; falling back to CPU.", flush=True)
    return torch.device(
        "cuda"
        if requested == "cuda" or (requested == "auto" and torch.cuda.is_available())
        else "cpu"
    )


def sample_path(data_dict: dict[str, Any], index: int) -> str:
    paths = data_dict["image"][index]
    if isinstance(paths, (list, tuple)):
        return str(paths[len(paths) // 2])
    return str(paths)


def select_indices(
    labels: np.ndarray,
    pool: np.ndarray,
    num_samples: int,
    balance: bool,
    rng: np.random.Generator,
) -> list[int]:
    if not balance:
        count = min(num_samples, pool.size)
        return sorted(rng.choice(pool, size=count, replace=False).tolist())

    quotas = {0: num_samples // 2, 1: num_samples - num_samples // 2}
    selected: list[int] = []
    leftovers: list[np.ndarray] = []
    for label in (0, 1):
        class_pool = pool[labels[pool] == label]
        take = min(quotas[label], class_pool.size)
        if take > 0:
            chosen = rng.choice(class_pool, size=take, replace=False)
            selected.extend(int(value) for value in chosen)
        leftovers.append(class_pool)

    remaining = num_samples - len(selected)
    if remaining > 0:
        rest = np.concatenate(leftovers)
        rest = np.setdiff1d(rest, np.asarray(selected, dtype=np.int64))
        selected.extend(
            int(value)
            for value in rng.choice(rest, size=min(remaining, rest.size), replace=False)
        )
    return sorted(selected)


def scan_candidates(
    model: torch.nn.Module,
    dataset: DeepfakeAbstractBaseDataset,
    indices: list[int],
    labels: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            chunk = indices[start : start + batch_size]
            images = torch.stack([dataset[index][0] for index in chunk]).to(device)
            logits = model({"image": images}, inference=True)["cls"]
            probabilities = torch.softmax(logits.float(), dim=-1).cpu().numpy()
            for index, prob_fake in zip(chunk, probabilities[:, 1]):
                prob_fake = float(prob_fake)
                prediction = int(prob_fake >= 0.5)
                records.append(
                    {
                        "index": int(index),
                        "label": int(labels[index]),
                        "prob_fake": prob_fake,
                        "pred": prediction,
                        "predicted_prob": (
                            prob_fake if prediction == 1 else 1.0 - prob_fake
                        ),
                    }
                )
            done = min(start + batch_size, len(indices))
            if (start // batch_size) % 16 == 0 or done == len(indices):
                print(f"[gradcam] Scored {done}/{len(indices)} frames", flush=True)
    return records


def classify_record(
    record: dict[str, Any], high_confidence: float, low_confidence: float
) -> str | None:
    correct = record["pred"] == record["label"]
    if correct:
        if record["predicted_prob"] >= high_confidence:
            return f"{LABEL_NAMES[record['label']]}_high"
        return None
    if record["predicted_prob"] <= low_confidence:
        return "misclassified_low"
    if record["predicted_prob"] >= high_confidence:
        return "misclassified_high"
    return None


def select_category_samples(
    records: list[dict[str, Any]],
    categories: list[str],
    per_category: int,
    high_confidence: float,
    low_confidence: float,
) -> tuple[list[tuple[dict[str, Any], str]], dict[str, int]]:
    grouped: dict[str, list[dict[str, Any]]] = {
        category: [] for category in categories
    }
    counts = {category: 0 for category in CATEGORY_CHOICES}
    for record in records:
        category = classify_record(record, high_confidence, low_confidence)
        if category is None:
            continue
        counts[category] += 1
        if category in grouped:
            grouped[category].append(record)

    selected: list[tuple[dict[str, Any], str]] = []
    for category in categories:
        items = grouped[category]
        # Low-confidence errors closest to 0.5 first; confident frames first.
        if category == "misclassified_low":
            items.sort(key=lambda record: (record["predicted_prob"], record["index"]))
        else:
            items.sort(key=lambda record: (-record["predicted_prob"], record["index"]))
        selected.extend((record, category) for record in items[:per_category])
        if len(items) < per_category:
            print(
                f"[gradcam] Only {len(items)} '{category}' frame(s) found "
                f"(requested {per_category}).",
                flush=True,
            )
    return selected, counts


def resolve_target(requested: str, category: str | None) -> str:
    if requested != "auto":
        return requested
    if category == "real_high":
        return "real"
    if category == "fake_high":
        return "fake"
    return "pred"


def resolve_layer(model: torch.nn.Module, layer_index: int) -> tuple[torch.nn.Module, str]:
    encoder = getattr(model.backbone, "encoder", None)
    layers = getattr(encoder, "layers", None)
    if layers is None:
        raise SystemExit(
            "The model backbone has no 'encoder.layers'; this script expects a "
            "CLIP ViT backbone."
        )
    if not -len(layers) <= layer_index < len(layers):
        raise SystemExit(
            f"--layer-index {layer_index} is out of range for {len(layers)} layers."
        )

    # The classifier only reads the CLS token, so gradients with respect to
    # the layer's output are zero for every patch token. The pre-attention
    # LayerNorm output is the last point where all tokens affect the CLS
    # prediction through the layer's attention.
    layer = layers[layer_index]
    for attribute in ("layer_norm1", "norm1"):
        norm = getattr(layer, attribute, None)
        if isinstance(norm, torch.nn.Module):
            return norm, f"backbone.encoder.layers[{layer_index}].{attribute}"
    return layer, f"backbone.encoder.layers[{layer_index}]"


def resolve_grid(model: torch.nn.Module, resolution: int) -> tuple[int, int]:
    patch_size = getattr(model.backbone.config, "patch_size", None) or 14
    image_size = getattr(model.backbone.config, "image_size", None) or resolution
    if isinstance(image_size, (list, tuple)):
        image_size = image_size[0]
    grid = int(image_size) // int(patch_size)
    return grid, grid


def denormalize(image: torch.Tensor, mean: list[float], std: list[float]) -> np.ndarray:
    mean_tensor = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
    std_tensor = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)
    image = (image.detach().cpu().float() * std_tensor + mean_tensor).clamp(0.0, 1.0)
    return (image.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)


def compute_cam(
    model: torch.nn.Module,
    layer: torch.nn.Module,
    image: torch.Tensor,
    target_class: str,
    token_reduction: str,
    resolution: int,
) -> tuple[np.ndarray, np.ndarray, int, tuple[int, int]]:
    captured: dict[str, torch.Tensor] = {}

    def hook(module, inputs, output):
        captured["activation"] = output if torch.is_tensor(output) else output[0]

    handle = layer.register_forward_hook(hook)
    try:
        predictions = model({"image": image}, inference=True)
        logits = predictions["cls"]
        if logits.ndim != 2 or logits.shape[0] != 1:
            raise RuntimeError(
                f"Expected logits of shape [1, classes], got {tuple(logits.shape)}."
            )
        probabilities = torch.softmax(logits.detach().float(), dim=-1)
        prediction = int(logits.argmax(dim=-1))
        target = {"fake": 1, "real": 0, "pred": prediction}[target_class]
        activation = captured.get("activation")
        if activation is None:
            raise RuntimeError("The target layer produced no activation.")
        if activation.ndim != 3:
            raise RuntimeError(
                "Expected activation of shape [batch, tokens, channels], got "
                f"{tuple(activation.shape)}."
            )
        gradients = torch.autograd.grad(logits[0, target], activation)[0]
    finally:
        handle.remove()

    tokens = activation.shape[1]
    grid_h, grid_w = resolve_grid(model, resolution)
    num_patches = grid_h * grid_w
    if tokens < num_patches:
        side = math.isqrt(tokens)
        if side * side != tokens:
            raise RuntimeError(
                f"Activation has {tokens} tokens but the expected grid is "
                f"{grid_h}x{grid_w}."
            )
        grid_h = grid_w = side
        num_patches = tokens
    prefix = tokens - num_patches
    if prefix not in (0, 1):
        raise RuntimeError(
            f"Activation has {tokens} tokens, but the backbone grid is "
            f"{grid_h}x{grid_w}; the input resolution likely does not match the "
            "backbone's image size."
        )

    patch_activations = activation[0, prefix:]
    if token_reduction == "cls" and prefix > 0:
        weights = gradients[0, 0]
    else:
        weights = gradients[0, prefix:].mean(dim=0)

    cam = (patch_activations * weights.unsqueeze(0)).sum(dim=-1).clamp(min=0.0)
    cam = cam.reshape(1, 1, grid_h, grid_w)
    height, width = image.shape[-2:]
    cam = F.interpolate(cam, size=(height, width), mode="bilinear", align_corners=False)
    cam = cam[0, 0].detach().cpu().numpy()

    minimum = float(cam.min())
    maximum = float(cam.max())
    if maximum > minimum:
        cam = (cam - minimum) / (maximum - minimum)
    else:
        cam = np.zeros_like(cam)

    return cam, probabilities[0].detach().cpu().numpy(), target, (grid_h, grid_w)


def colorize(cam: np.ndarray, cmap_name: str) -> np.ndarray:
    rgba = colormaps[cmap_name](cam)
    return (rgba[..., :3] * 255.0).round().astype(np.uint8)


def blend(image: np.ndarray, heatmap: np.ndarray, alpha: float) -> np.ndarray:
    return (
        image.astype(np.float32) * (1.0 - alpha) + heatmap.astype(np.float32) * alpha
    ).clip(0, 255).astype(np.uint8)


def sample_stem(index: int, label: int, path: str, category: str | None = None) -> str:
    path_obj = Path(path)
    parts = [f"{index:05d}", LABEL_NAMES[label]]
    if category:
        parts.append(category)
    parts.extend((path_obj.parent.name, path_obj.stem))
    return "_".join(parts)


def write_summary(
    path: Path,
    rows: list[dict[str, Any]],
    cmap_name: str,
    alpha: float,
) -> None:
    gap = 8
    text_height = 20
    tiles = []
    for row in rows:
        original = np.asarray(Image.open(row["input_image"]).convert("RGB"))
        heatmap = colorize(row["cam"], cmap_name)
        overlay = blend(original, heatmap, alpha)
        height, width = original.shape[:2]
        tile = Image.new("RGB", (3 * width + 2 * gap, height + text_height), "white")
        draw = ImageDraw.Draw(tile)
        caption = (
            f"idx {row['index']}"
            + (f" | {row['category']}" if row["category"] else "")
            + f" | truth {row['class']} | pred {row['pred_class']} "
            f"| P(fake) {row['prob_fake']:.3f} | target {row['target_class']}"
        )
        draw.text((2, 2), caption, fill="black")
        for column, array in enumerate((original, heatmap, overlay)):
            tile.paste(Image.fromarray(array), (column * (width + gap), text_height))
        tiles.append(tile)

    width = max(tile.width for tile in tiles)
    height = sum(tile.height for tile in tiles) + gap * (len(tiles) - 1)
    canvas = Image.new("RGB", (width, height), "white")
    offset = 0
    for tile in tiles:
        canvas.paste(tile, (0, offset))
        offset += tile.height + gap
    canvas.save(path)


def main() -> None:
    args = parse_args()
    validate_args(args)
    if args.target_class is None:
        args.target_class = "auto" if args.categories else "fake"

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    config = load_config(args.config, args.dataset)
    if config.get("video_mode", False):
        raise SystemExit("Video-level datasets are not supported by GradCAM.")
    device = get_device(args.device)

    print(f"[gradcam] Device: {device}", flush=True)
    model = DETECTOR[config["model_name"]](config).to(device)
    load_weights(model, args.weights, device)
    model.eval()

    layer, layer_name = resolve_layer(model, args.layer_index)
    print(f"[gradcam] Target layer: {layer_name}", flush=True)

    dataset = DeepfakeAbstractBaseDataset(config=config, mode="test")
    labels = np.asarray(dataset.data_dict["label"], dtype=np.int64)
    pool_size = len(dataset)
    if args.max_samples is not None:
        pool_size = min(pool_size, args.max_samples)
    pool = np.arange(pool_size)
    rng = np.random.default_rng(args.seed)
    categories = (
        list(dict.fromkeys(args.categories)) if args.categories is not None else []
    )
    category_by_index: dict[int, str] = {}
    category_counts: dict[str, int] = {}
    scanned = 0

    if categories:
        scan_count = (
            pool_size if args.scan_samples == 0 else min(args.scan_samples, pool_size)
        )
        scan_indices = sorted(
            int(value) for value in rng.choice(pool, size=scan_count, replace=False)
        )
        print(
            f"[gradcam] Scoring {len(scan_indices)} frames for "
            f"{', '.join(categories)} ...",
            flush=True,
        )
        records = scan_candidates(
            model, dataset, scan_indices, labels, device, args.scan_batch_size
        )
        selected, category_counts = select_category_samples(
            records,
            categories,
            args.per_category,
            args.high_confidence,
            args.low_confidence,
        )
        indices = [record["index"] for record, _ in selected]
        category_by_index = {
            record["index"]: category for record, category in selected
        }
        scanned = len(scan_indices)
        print(
            "[gradcam] Scan counts: "
            + ", ".join(f"{name}={count}" for name, count in category_counts.items()),
            flush=True,
        )
        if not indices:
            raise SystemExit("No samples matched the requested categories.")
    else:
        indices = select_indices(labels, pool, args.num_samples, args.balance, rng)
        if len(indices) < args.num_samples:
            print(
                f"[gradcam] Requested {args.num_samples} samples but only "
                f"{len(indices)} are available.",
                flush=True,
            )

    output_dir = args.output_dir
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    mean = config["mean"]
    std = config["std"]
    rows: list[dict[str, Any]] = []
    cam_maps: list[np.ndarray] = []

    start = time.time()
    for position, index in enumerate(indices, start=1):
        image, label, _, _ = dataset[index]
        label = int(label)
        image = image.to(device).unsqueeze(0)
        image.requires_grad_(True)
        category = category_by_index.get(index)
        target_choice = resolve_target(args.target_class, category)

        cam, probabilities, target, grid = compute_cam(
            model,
            layer,
            image,
            target_choice,
            args.token_reduction,
            int(config["resolution"]),
        )
        prediction = int(np.argmax(probabilities))
        path = sample_path(dataset.data_dict, index)
        stem = sample_stem(index, label, path, category)
        original = denormalize(image[0], mean, std)
        heatmap = colorize(cam, args.cmap)
        overlay = blend(original, heatmap, args.alpha)

        input_path = image_dir / f"{stem}_input.png"
        heatmap_path = image_dir / f"{stem}_heatmap.png"
        overlay_path = image_dir / f"{stem}_overlay.png"
        Image.fromarray(original).save(input_path)
        Image.fromarray(heatmap).save(heatmap_path)
        Image.fromarray(overlay).save(overlay_path)

        row = {
            "index": index,
            "path": path,
            "label": label,
            "class": LABEL_NAMES[label],
            "category": category or "",
            "pred": prediction,
            "pred_class": LABEL_NAMES[prediction],
            "prob_fake": float(probabilities[1]),
            "target": target,
            "target_class": LABEL_NAMES[target],
            "grid": grid,
            "cam": cam,
            "input_image": str(input_path),
            "heatmap_image": str(heatmap_path),
            "overlay_image": str(overlay_path),
        }
        rows.append(row)
        cam_maps.append(cam)
        print(
            f"[gradcam] {position}/{len(indices)} idx={index} "
            f"{category + ' ' if category else ''}"
            f"truth={LABEL_NAMES[label]} pred={LABEL_NAMES[prediction]} "
            f"P(fake)={probabilities[1]:.4f}",
            flush=True,
        )

    elapsed = time.time() - start

    with (output_dir / "results.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            (
                "index",
                "path",
                "label",
                "class",
                "category",
                "pred",
                "pred_class",
                "prob_fake",
                "target",
                "target_class",
                "grid_h",
                "grid_w",
                "input_image",
                "heatmap_image",
                "overlay_image",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row["index"],
                    row["path"],
                    row["label"],
                    row["class"],
                    row["category"],
                    row["pred"],
                    row["pred_class"],
                    f"{row['prob_fake']:.6f}",
                    row["target"],
                    row["target_class"],
                    row["grid"][0],
                    row["grid"][1],
                    row["input_image"],
                    row["heatmap_image"],
                    row["overlay_image"],
                )
            )

    if args.save_npy:
        np.save(output_dir / "gradcam.npy", np.stack(cam_maps))
        np.save(output_dir / "gradcam_indices.npy", np.asarray(indices))

    if args.summary_samples > 0 and rows:
        write_summary(
            output_dir / "summary.png",
            rows[: args.summary_samples],
            args.cmap,
            args.alpha,
        )

    metadata = {
        "config": str(args.config),
        "weights": str(args.weights),
        "dataset": args.dataset,
        "device": str(device),
        "model_name": config["model_name"],
        "layer": layer_name,
        "target_class": args.target_class,
        "token_reduction": args.token_reduction,
        "categories": categories,
        "per_category": args.per_category,
        "high_confidence": args.high_confidence,
        "low_confidence": args.low_confidence,
        "scan_samples": args.scan_samples,
        "scanned": scanned,
        "category_counts": category_counts,
        "num_samples": len(rows),
        "num_requested": (
            args.per_category * len(categories) if categories else args.num_samples
        ),
        "max_samples": args.max_samples,
        "balance": args.balance,
        "seed": args.seed,
        "alpha": args.alpha,
        "cmap": args.cmap,
        "resolution": int(config["resolution"]),
        "grid": list(rows[0]["grid"]) if rows else None,
        "elapsed_seconds": round(elapsed, 3),
        "samples": [
            {key: value for key, value in row.items() if key != "cam"} for row in rows
        ],
    }
    (output_dir / "meta.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(
        f"[gradcam] Wrote {len(rows)} GradCAM samples to {output_dir} "
        f"in {elapsed:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()

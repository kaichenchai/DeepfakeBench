"""
Cross-Dataset Validation Script
================================
Evaluates a trained model on multiple deepfake detection datasets to assess
generalization performance.

Datasets tested:
    - FaceForensics++
    - UADFV
    - Celeb-DF-v1
    - Celeb-DF-v2
    - DFDC
    - DFDCP

Usage:
    python training/cross_dataset_test.py \
        --detector_path training/config/detector/effort_ce.yaml \
        --weights_path /path/to/checkpoint.pth \
        [--datasets FaceForensics++ UADFV Celeb-DF-v1 Celeb-DF-v2 DFDC DFDCP] \
        [--output_dir ./results/cross_dataset/] \
        [--wandb_project deepfakebench] \
        [--wandb_name my_cross_test] \
        [--wandb_tags cross_dataset effort_ce] \
        [--checkpoint_type best|final] \
        [--no_wandb]
"""

import os
import sys
import argparse
import time
import json
import traceback
import yaml
import wandb
from datetime import datetime
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data
from tqdm import tqdm

print("[cross_dataset_test] Script started. Importing modules...", flush=True)

from dataset.abstract_dataset import DeepfakeAbstractBaseDataset
from detectors import DETECTOR
from metrics.utils import get_test_metrics

print("[cross_dataset_test] All imports complete.", flush=True)

# ── Default cross-dataset test suite ─────────────────────────────────────────
CROSS_DATASETS = [
    "FaceForensics++",
    "UADFV",
    "Celeb-DF-v1",
    "Celeb-DF-v2",
    "DFDC",
    "DFDCP",
]

# ── Argument parsing ─────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Cross-dataset evaluation for deepfake detectors."
)
parser.add_argument(
    "--detector_path", type=str, required=True,
    help="Path to detector YAML config file (e.g. training/config/detector/effort_ce.yaml)."
)
parser.add_argument(
    "--weights_path", type=str, required=True,
    help="Path to the trained model weights (.pth checkpoint)."
)
parser.add_argument(
    "--datasets", type=str, nargs="+", default=CROSS_DATASETS,
    help="List of dataset names to evaluate (default: all 6 cross-dataset benchmarks)."
)
parser.add_argument(
    "--output_dir", type=str, default="./logs/cross_dataset_test/",
    help="Directory to save the results JSON and summary."
)
parser.add_argument(
    "--device", type=str, default="auto",
    help="Device to run on: 'auto', 'cuda', 'cpu', or a specific device index."
)

# ── Wandb arguments ──────────────────────────────────────────────────────
parser.add_argument(
    "--wandb_project", type=str, default=None,
    help="Wandb project name (default: read from detector config, fallback 'deepfakebench')."
)
parser.add_argument(
    "--wandb_name", type=str, default=None,
    help="Display name for this wandb run. If not provided, uses config's run_name."
)
parser.add_argument(
    "--wandb_tags", type=str, nargs="+", default=None,
    help="Tags for wandb run (default: ['cross_dataset', config['model_name']])."
)
parser.add_argument(
    "--checkpoint_type", type=str, choices=["best", "final"], default="final",
    help="Checkpoint type being tested: 'best' (test/avg/ckpt_best.pth) or 'final' (final/ckpt_final.pth). "
         "Prepended to the wandb run name. Default: 'final'."
)
parser.add_argument(
    "--no_wandb", action="store_true", default=False,
    help="Disable wandb logging entirely."
)
args = parser.parse_args()

print(f"[cross_dataset_test] Args parsed. detector_path={args.detector_path}", flush=True)
print(f"[cross_dataset_test] weights_path={args.weights_path}", flush=True)
print(f"[cross_dataset_test] datasets={args.datasets}", flush=True)
print(f"[cross_dataset_test] output_dir={args.output_dir}", flush=True)
print(f"[cross_dataset_test] no_wandb={args.no_wandb}", flush=True)

# ── Device selection ─────────────────────────────────────────────────────────
if args.device == "auto":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
elif args.device == "cpu":
    device = torch.device("cpu")
else:
    device = torch.device(args.device)

print(f"Using device: {device}")


# ── Helpers ──────────────────────────────────────────────────────────────────
def init_seed(config):
    """Initialize random seeds for reproducibility."""
    import random
    if config.get("manualSeed") is None:
        config["manualSeed"] = random.randint(1, 10000)
    random.seed(config["manualSeed"])
    torch.manual_seed(config["manualSeed"])
    if config.get("cuda", False):
        torch.cuda.manual_seed_all(config["manualSeed"])


def load_config(detector_path: str) -> dict:
    """Load the detector YAML configuration and merge with test_config."""
    print(f"[cross_dataset_test] Loading detector config from: {detector_path}", flush=True)
    with open(detector_path, "r") as f:
        config = yaml.safe_load(f)
    print(f"[cross_dataset_test] Detector config loaded. keys={list(config.keys())}", flush=True)

    test_config_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "config", "test_config.yaml"
    )
    print(f"[cross_dataset_test] Loading test config from: {test_config_path}", flush=True)
    with open(test_config_path, "r") as f:
        test_config = yaml.safe_load(f)

    config.update(test_config)
    print(f"[cross_dataset_test] Configs merged. Final keys={list(config.keys())}", flush=True)

    # Carry over label_dict from test_config if present in original config
    if "label_dict" in config:
        test_config["label_dict"] = config["label_dict"]

    print(f"[cross_dataset_test] Config loaded successfully. model_name={config.get('model_name', 'UNKNOWN')}", flush=True)
    return config


def prepare_test_loader(config: dict, dataset_name: str) -> torch.utils.data.DataLoader:
    """Build a DataLoader for a single test dataset."""
    cfg = config.copy()
    cfg["test_dataset"] = dataset_name

    test_set = DeepfakeAbstractBaseDataset(config=cfg, mode="test")

    loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=cfg["test_batchSize"],
        shuffle=False,
        num_workers=int(cfg.get("workers", 4)),
        collate_fn=test_set.collate_fn,
        drop_last=False,
    )
    return loader


def load_model(config: dict, weights_path: str) -> nn.Module:
    """Instantiate the detector and load pretrained weights."""
    print(f"[cross_dataset_test] Instantiating model: {config.get('model_name', 'UNKNOWN')}", flush=True)
    model_class = DETECTOR[config["model_name"]]
    model = model_class(config).to(device)
    print(f"[cross_dataset_test] Model instantiated on {device}. Loading weights...", flush=True)

    ckpt = torch.load(weights_path, map_location=device)
    print(f"[cross_dataset_test] Weights file loaded. ckpt keys count={len(ckpt)}", flush=True)

    # Strip 'module.' prefix if the checkpoint was saved from DDP
    if any(k.startswith("module.") for k in ckpt.keys()):
        print("[cross_dataset_test] Stripping 'module.' prefix from checkpoint keys.", flush=True)
        ckpt = {k[7:]: v for k, v in ckpt.items()}

    model.load_state_dict(ckpt, strict=True)
    print(f"[cross_dataset_test] Weights loaded successfully from {weights_path}", flush=True)
    return model


@torch.no_grad()
def inference(model: nn.Module, data_dict: dict) -> dict:
    """Run inference on a single batch. Returns predictions dict with 'prob' and 'feat'."""
    return model(data_dict, inference=True)


def evaluate_one_dataset(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    dataset_name: str,
) -> dict:
    """Run inference over a full dataset and compute metrics."""
    print(f"[cross_dataset_test] Starting evaluation on {dataset_name}. Num batches={len(loader)}", flush=True)
    model.eval()

    preds, labels, feats = [], [], []

    for batch_idx, data_dict in enumerate(tqdm(loader, desc=f"Testing {dataset_name}", leave=False)):
        images = data_dict["image"].to(device)
        # Binarise labels: 0 vs 1
        target = torch.where(data_dict["label"] != 0, 1, 0).to(device)

        # Build input dict (mask / landmark are optional)
        batch = {"image": images, "label": target}
        if data_dict.get("mask") is not None:
            batch["mask"] = data_dict["mask"].to(device)
        if data_dict.get("landmark") is not None:
            batch["landmark"] = data_dict["landmark"].to(device)

        output = inference(model, batch)

        preds.extend(output["prob"].cpu().numpy().tolist())
        labels.extend(target.cpu().numpy().tolist())
        feats.extend(output["feat"].cpu().numpy().tolist())

        if batch_idx == 0:
            print(f"[cross_dataset_test] First batch of {dataset_name}: "
                  f"images.shape={images.shape}, prob.shape={output['prob'].shape}", flush=True)

    print(f"[cross_dataset_test] Inference done for {dataset_name}. "
          f"Total samples: preds={len(preds)}, labels={len(labels)}", flush=True)

    preds_np = np.array(preds)
    labels_np = np.array(labels)

    print(f"[cross_dataset_test] Computing metrics for {dataset_name}...", flush=True)
    metrics = get_test_metrics(
        y_pred=preds_np,
        y_true=labels_np,
        img_names=loader.dataset.data_dict["image"],
    )
    print(f"[cross_dataset_test] Metrics computed for {dataset_name}: {list(metrics.keys())}", flush=True)
    return metrics


# ── Wandb helpers ────────────────────────────────────────────────────────────
def init_wandb(config: dict, weights_dir: str):
    """Initialise wandb run with config, tags, and metadata."""
    project = args.wandb_project or config.get("wandb", {}).get("project", "deepfakebench")

    # Determine run name: prepend checkpoint_type, then CLI arg > config run_name > model_name + timestamp
    ckpt_prefix = f"{args.checkpoint_type}_"
    if args.wandb_name:
        run_name = ckpt_prefix + args.wandb_name
    elif config.get("run_name"):
        run_name = ckpt_prefix + "cross_dataset_test_" + config["run_name"]
    else:
        run_name = ckpt_prefix + "cross_dataset_test_" + config.get("model_name", "unknown")

    # Tags: CLI arg > default tags with model_name
    if args.wandb_tags:
        tags = args.wandb_tags
    else:
        tags = ["test", config.get("model_name", "unknown")]

    wandb.init(
        project=project,
        name=run_name,
        config=config,                    # uploads the full merged config
        dir=config.get("wandb", {}).get("save_dir", "./logs/wandb/"),
        mode="disabled" if args.no_wandb else config.get("wandb", {}).get("mode", "online"),
        tags=tags,
        notes=f"Weights directory: {weights_dir}",
    )

    # Log the weights directory as a wandb summary entry
    wandb.summary["weights_dir"] = weights_dir
    wandb.summary["weights_path"] = args.weights_path
    wandb.summary["detector_config"] = args.detector_path


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    # ── Derive weights_dir early (before wandb init) ─────────────────────
    weights_dir = os.path.dirname(os.path.abspath(args.weights_path))

    print("=" * 80)
    print("  Cross-Dataset Evaluation")
    print(f"  Started at:     {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Config:         {args.detector_path}")
    print(f"  Weights:        {args.weights_path}")
    print(f"  Weights dir:    {weights_dir}")
    print(f"  Datasets:       {args.datasets}")
    print(f"  Device:         {device}")
    print(f"  Wandb:          {'disabled' if args.no_wandb else 'enabled'}")
    print("=" * 80)

    # 1. Load config
    print("[cross_dataset_test] Step 1: Loading config...", flush=True)
    config = load_config(args.detector_path)
    config["cuda"] = (device.type == "cuda")
    config["cudnn"] = config["cuda"]

    init_seed(config)
    if config["cudnn"]:
        cudnn.deterministic = True
    print(f"[cross_dataset_test] Config ready. cuda={config['cuda']}, seed={config.get('manualSeed')}", flush=True)

    # 2. Init wandb (uploads config + metadata)
    print("[cross_dataset_test] Step 2: Initializing wandb...", flush=True)
    init_wandb(config, weights_dir)
    print("[cross_dataset_test] Wandb initialized.", flush=True)

    # 3. Build model + load weights
    print("[cross_dataset_test] Step 3: Building model and loading weights...", flush=True)
    model = load_model(config, args.weights_path)
    print("[cross_dataset_test] Model ready.", flush=True)

    # 4. Evaluate each dataset
    all_results = {}
    successful_datasets = []
    total_start = time.time()

    for dataset_name in args.datasets:
        print(f"\n{'─' * 60}")
        print(f"  Evaluating: {dataset_name}")
        print(f"{'─' * 60}")

        # ── Pre-check: verify dataset directory exists on disk ────────────
        rgb_dir = config.get("rgb_dir", "./datasets/rgb")
        dataset_path = os.path.join(rgb_dir, dataset_name)
        if not os.path.isdir(dataset_path):
            print(f"  [SKIP] Dataset directory not found: {dataset_path}")
            all_results[dataset_name] = {"error": f"directory not found: {dataset_path}"}
            continue

        try:
            loader = prepare_test_loader(config, dataset_name)
            print(f"[cross_dataset_test] DataLoader ready for {dataset_name}. "
                  f"Dataset size={len(loader.dataset)}, batches={len(loader)}", flush=True)
        except Exception as e:
            print(f"  [SKIP] Could not load dataset '{dataset_name}': {e}")
            traceback.print_exc()
            all_results[dataset_name] = {"error": str(e)}
            continue

        try:
            metrics = evaluate_one_dataset(model, loader, dataset_name)
            all_results[dataset_name] = metrics
            successful_datasets.append(dataset_name)

            # ── Log per-dataset metrics to wandb ──────────────────────────
            wandb_metrics = {}
            for k, v in metrics.items():
                if isinstance(v, (int, float, np.floating, np.integer)):
                    wandb_metrics[f"{dataset_name}/{k}"] = float(v)
            if wandb_metrics:
                wandb.log(wandb_metrics)

            # Pretty-print per-dataset results
            print(f"  Results for {dataset_name}:")
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")
                else:
                    print(f"    {k}: {v}")
        except Exception as e:
            print(f"  [FAIL] Error evaluating '{dataset_name}': {e}")
            traceback.print_exc()
            all_results[dataset_name] = {"error": str(e)}

    total_time = time.time() - total_start

    # 5. Summary table
    print("\n" + "=" * 80)
    print("  Cross-Dataset Evaluation Summary")
    print("=" * 80)

    # Collect scalar metrics for table
    metric_keys = set()
    for ds, res in all_results.items():
        if "error" not in res:
            metric_keys.update(k for k, v in res.items() if isinstance(v, (int, float)))

    metric_keys = sorted(metric_keys)

    # Header
    header = f"{'Dataset':<22}"
    for mk in metric_keys:
        header += f" {mk:>10}"
    print(header)
    print("-" * len(header))

    # ── Build wandb summary table ────────────────────────────────────────
    summary_table_data = []

    for ds in args.datasets:
        res = all_results.get(ds, {})
        if "error" in res:
            print(f"{ds:<22} {'ERROR':>10}")
        else:
            row = f"{ds:<22}"
            row_dict = {"model": config.get("run_name"), "dataset": ds}
            for mk in metric_keys:
                val = res.get(mk, float("nan"))
                if isinstance(val, float):
                    row += f" {val:10.4f}"
                else:
                    row += f" {str(val):>10}"
                row_dict[mk] = val
            print(row)
            summary_table_data.append(row_dict)

    print("-" * len(header))
    print(f"\n  Total evaluation time: {total_time:.1f} s")

    # ── Log summary table, datasets tested & total time to wandb ─────────
    wandb.summary["datasets_tested"] = successful_datasets
    if summary_table_data:
        columns = ["model", "dataset"] + metric_keys
        table = wandb.Table(columns=columns)
        for row in summary_table_data:
            table.add_data(*[row.get(c, float("nan")) for c in columns])
        wandb.log({"cross_dataset_summary": table})
    wandb.summary["total_time_s"] = total_time

    # 6. Save results to disk
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    result_file = os.path.join(args.output_dir, f"cross_dataset_results_{timestamp}.json")

    # Convert numpy types for JSON serialisation
    def convert(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return o

    with open(result_file, "w") as f:
        json.dump(all_results, f, default=convert, indent=2)

    print(f"\n  Results saved to: {result_file}")
    print("=" * 80)

    # 7. Finish wandb
    if hasattr(wandb, "run") and wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    print("[cross_dataset_test] Entering main()...", flush=True)
    try:
        main()
        print("[cross_dataset_test] main() completed successfully.", flush=True)
    except Exception as e:
        print(f"[cross_dataset_test] FATAL ERROR in main(): {e}", flush=True)
        traceback.print_exc()
        sys.exit(1)

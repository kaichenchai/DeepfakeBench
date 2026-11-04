"""
Post-hoc check of real-image similarity for the fake-only-loss checkpoint.

Runs inference with a checkpoint trained using ONLY the fake branch of the
masked-counterfactual-backbone loss (no real-branch supervision ever applied),
then measures — on REAL images — the quantity the real branch would have
enforced: the similarity between the learned pooler features
``full_features`` and the frozen counterfactual (pristine CLIP) pooler
features ``cf_features``.

Per real sample this reports:
  * cos_sim         = cosine_similarity(full_features, cf_features)
  * mse             = mean((full_features - cf_features)^2) over the 1024-d dim
  * normalized_mse  = mse / (mean ||cf_features||^2 + 1e-8)   (the exact form
                      the old real branch minimised)

Interpreting the result
-----------------------
* If ``cos_sim ~= 1`` and ``mse ~= 0`` WITHOUT the real branch ever being
  supervised -> explanation (2): near-identity on real images is an emergent
  property of the trained model, so the real-branch supervision was not needed.
* If they have drifted noticeably but cross-dataset AUC still holds ->
  explanation (1): "real features should equal cf features" does not matter
  much for generalization, even though it sounds intuitively important.

Usage (run from the repo root so relative config paths resolve):
    python eval/reformulate-cf-loss/fake_loss_only_probe_real_image_similarity.py \
        --config training/config/detector/<fake_only_config>.yaml \
        --weights /path/to/fake_only_loss_checkpoint.pth \
        --dataset Celeb-DF-v2 \
        --n_probe 1000
"""

import os
import sys
import json
import argparse
import yaml

import numpy as np
import torch
import torch.utils.data

from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Path setup. This script lives in <repo>/eval/reformulate-cf-loss/ and the
# training package (detectors/metrics/loss/dataset) lives in <repo>/training/.
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
TRAINING_DIR = os.path.join(REPO_ROOT, "training")

for _p in (SCRIPT_DIR, TRAINING_DIR, REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# The configs reference CLIP weights, rgb_dir and dataset_json_folder by
# relative paths, so everything downstream assumes cwd == repo root.
os.chdir(REPO_ROOT)

from dataset.abstract_dataset import DeepfakeAbstractBaseDataset
from effort_custom_detector_probe import Effort_Custom_Detector_Probe


def parse_args():
    parser = argparse.ArgumentParser(
        description="Probe real-image cos_sim / MSE between learned and "
                    "counterfactual pooler features (fake-only-loss checkpoint)."
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to the detector YAML config (must match the checkpoint's "
             "svd_trainable_ranks / architecture).",
    )
    parser.add_argument(
        "--weights", type=str, required=True,
        help="Path to the fake-only-loss .pth checkpoint.",
    )
    parser.add_argument(
        "--dataset", type=str, required=True,
        help="Dataset name to probe (e.g. Celeb-DF-v2, DFDC, UADFV).",
    )
    parser.add_argument(
        "--n_probe", type=int, default=1000,
        help="Number of real samples to collect (default: 1000).",
    )
    parser.add_argument(
        "--batch_size", type=int, default=None,
        help="Override the config's test_batchSize for probing.",
    )
    parser.add_argument(
        "--svd_trainable_ranks", type=int, default=None,
        help="Override config['svd_trainable_ranks'] (must match the "
             "checkpoint's architecture).",
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to run on (default: auto).",
    )
    parser.add_argument(
        "--out_dir", type=str,
        default=os.path.join(SCRIPT_DIR, "output"),
        help="Directory to save histograms and summary (default: "
             "eval/reformulate-cf-loss/output).",
    )
    return parser.parse_args()


def load_config(detector_path: str) -> dict:
    with open(detector_path, "r") as f:
        config = yaml.safe_load(f)

    test_config_path = os.path.join(TRAINING_DIR, "config", "test_config.yaml")
    with open(test_config_path, "r") as f:
        test_config = yaml.safe_load(f)

    # test_config supplies rgb_dir / dataset_json_folder / label_dict / mode.
    config.update(test_config)
    return config


def prepare_test_loader(config: dict, dataset_name: str, batch_size: int):
    cfg = config.copy()
    cfg["test_dataset"] = dataset_name

    test_set = DeepfakeAbstractBaseDataset(config=cfg, mode="test")

    bs = batch_size if batch_size is not None else cfg["test_batchSize"]
    loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=bs,
        shuffle=False,
        num_workers=int(cfg.get("workers", 0)),
        collate_fn=test_set.collate_fn,
        drop_last=False,
    )
    return loader


def load_checkpoint_state_dict(path: str) -> dict:
    ckpt = torch.load(path, map_location="cpu")

    # Accept raw state_dicts or wrapped dicts saved by other trainers.
    if isinstance(ckpt, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in ckpt and isinstance(ckpt[key], dict):
                ckpt = ckpt[key]
                break

    if not isinstance(ckpt, dict):
        raise ValueError(f"Unrecognised checkpoint format at {path}: {type(ckpt)}")

    # Strip DataParallel 'module.' prefix if present.
    if any(k.startswith("module.") for k in ckpt.keys()):
        ckpt = {k[len("module."):]: v for k, v in ckpt.items()}

    return ckpt


@torch.no_grad()
def collect_real_similarity(model, loader, device, n_probe):
    """Run the model over the loader and collect similarity metrics for real samples."""
    cos_sims, mses, norm_mses = [], [], []
    pbar = tqdm(loader, desc="Probing real-image similarity", leave=True)

    for data_dict in pbar:
        if len(cos_sims) >= n_probe:
            break

        images = data_dict["image"].to(device)
        # Binarise labels: fake = 1, real = 0.
        labels = torch.where(data_dict["label"] != 0, 1, 0).to(device)

        batch = {"image": images, "label": labels}
        if data_dict.get("mask") is not None:
            batch["mask"] = data_dict["mask"].to(device)
        if data_dict.get("landmark") is not None:
            batch["landmark"] = data_dict["landmark"].to(device)

        pred_dict = model(batch, inference=True)
        cos_sim = model.compute_cos_sim(batch, pred_dict)
        mse = model.compute_mse(batch, pred_dict)
        norm_mse = model.compute_normalized_mse(batch, pred_dict)

        mask_real = labels == 0
        if mask_real.any():
            cos_sims.extend(cos_sim[mask_real].cpu().numpy().tolist())
            mses.extend(mse[mask_real].cpu().numpy().tolist())
            norm_mses.extend(norm_mse[mask_real].cpu().numpy().tolist())
            pbar.set_postfix(n_real=len(cos_sims))

    cos_sims = np.asarray(cos_sims[:n_probe], dtype=np.float64)
    mses = np.asarray(mses[:n_probe], dtype=np.float64)
    norm_mses = np.asarray(norm_mses[:n_probe], dtype=np.float64)

    if cos_sims.size == 0:
        raise RuntimeError("No real samples collected — check the dataset name "
                           "and labels.")
    return cos_sims, mses, norm_mses


def compute_stats(cos_sim, mse, norm_mse):
    return {
        "n": int(cos_sim.size),
        "cos_sim_mean": float(cos_sim.mean()),
        "cos_sim_std": float(cos_sim.std()),
        "cos_sim_median": float(np.median(cos_sim)),
        "cos_sim_min": float(cos_sim.min()),
        "cos_sim_max": float(cos_sim.max()),
        "cos_sim_frac_gt_0.9": float((cos_sim > 0.9).mean()),
        "mse_mean": float(mse.mean()),
        "mse_std": float(mse.std()),
        "mse_median": float(np.median(mse)),
        "mse_min": float(mse.min()),
        "mse_max": float(mse.max()),
        "normalized_mse_mean": float(norm_mse.mean()),
        "normalized_mse_std": float(norm_mse.std()),
    }


def plot_histograms(cos_sim, mse, stats, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.hist(cos_sim, bins=60, color="#4C72B0", alpha=0.85, edgecolor="white")
    ax.axvline(1.0, color="black", linestyle="--", linewidth=1.0,
               label="cos_sim = 1 (perfect alignment)")
    ax.axvline(stats["cos_sim_mean"], color="green", linestyle="-", linewidth=1.2,
               label=f"mean = {stats['cos_sim_mean']:.3f}")
    ax.set_xlabel("cos_sim(full_features, cf_features)")
    ax.set_ylabel("count")
    ax.set_title(f"Real-image cos_sim\nmean={stats['cos_sim_mean']:.3f}, "
                 f"std={stats['cos_sim_std']:.3f}, "
                 f"frac>0.9={stats['cos_sim_frac_gt_0.9']:.3f}")
    ax.legend(loc="best")

    ax = axes[1]
    ax.hist(mse, bins=60, color="#DD8452", alpha=0.85, edgecolor="white")
    ax.axvline(stats["mse_mean"], color="green", linestyle="-", linewidth=1.2,
               label=f"mean = {stats['mse_mean']:.4f}")
    ax.set_xlabel("MSE(full_features, cf_features)")
    ax.set_ylabel("count")
    ax.set_title(f"Real-image MSE\nmean={stats['mse_mean']:.4f}, "
                 f"std={stats['mse_std']:.4f}, "
                 f"norm_mean={stats['normalized_mse_mean']:.4f}")
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto"
        else args.device
    )
    print(f"Using device: {device}")

    config = load_config(args.config)
    if args.svd_trainable_ranks is not None:
        config["svd_trainable_ranks"] = args.svd_trainable_ranks
    print(f"svd_trainable_ranks = {config.get('svd_trainable_ranks', 1)} "
          "(must match the checkpoint architecture)")

    loader = prepare_test_loader(config, args.dataset, args.batch_size)
    print(f"Loaded dataset '{args.dataset}': {len(loader.dataset)} samples, "
          f"{len(loader)} batches.")

    print("Building probe detector and loading weights...")
    model = Effort_Custom_Detector_Probe(config).to(device)
    model.eval()

    state_dict = load_checkpoint_state_dict(args.weights)
    model.load_state_dict(state_dict)
    print(f"Weights loaded from {args.weights}")

    cos_sim, mse, norm_mse = collect_real_similarity(
        model, loader, device, args.n_probe
    )
    stats = compute_stats(cos_sim, mse, norm_mse)

    print("\n" + "=" * 60)
    print("  Real-image full_features vs cf_features summary")
    print("=" * 60)
    for k, v in stats.items():
        print(f"  {k:>22}: {v}")
    print("=" * 60)

    # Soft interpretation based on the scale-free cos_sim metric.
    if stats["cos_sim_mean"] > 0.9:
        print("\n  cos_sim near 1 without real-branch supervision -> "
              "explanation (2): near-identity on real images is an emergent "
              "property; the real-branch loss was not needed.")
    else:
        print("\n  cos_sim has drifted below 0.9 despite no real-branch "
              "supervision -> more consistent with explanation (1): "
              "'real features == cf features' does not matter much for "
              "generalization.")

    os.makedirs(args.out_dir, exist_ok=True)
    cos_path = os.path.join(args.out_dir, "cos_sim_real.npy")
    mse_path = os.path.join(args.out_dir, "mse_real.npy")
    png_path = os.path.join(args.out_dir, "real_image_similarity_histograms.png")
    json_path = os.path.join(args.out_dir, "real_image_similarity_summary.json")

    np.save(cos_path, cos_sim)
    np.save(mse_path, mse)
    plot_histograms(cos_sim, mse, stats, png_path)
    with open(json_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nSaved:\n  {cos_path}\n  {mse_path}\n  {png_path}\n  {json_path}")


if __name__ == "__main__":
    main()

"""
Probe the counterfactual cos_sim distribution on fake samples.

Confirms (or refutes) the hypothesis that, under the OLD ``relu(cos_sim)``
fake-branch loss, a mass of fake samples settles at meaningfully *negative*
cosine similarity between the detector's learned pooler features and the
frozen counterfactual (pristine CLIP) pooler features.

The old loss ``F.relu(cos_sim)`` is free for ``cos_sim <= 0`` (zero loss, zero
gradient), so the model is free to push fake residuals to negative similarity
at no cost. The replacements have no such free region — e.g. the
temperature-scaled softplus ``F.softplus(beta * cos_sim) / beta`` used by the
current ``masked_counterfactual_backbone`` fake branch, or the earlier
``cos_sim ** 2`` — which is the leading explanation for the observed AUC/AP
regression.

What this script does
---------------------
1. Loads a trained ``effort_custom`` checkpoint (the OLD-loss checkpoint).
2. Probes ``n_probe`` fake samples from the requested dataset.
3. For each fake sample, computes ``cos_sim = cosine_similarity(cf_features,
   pred_dict['feat'])`` — the exact quantity the loss acts on.
4. Prints mean/std and the fraction with ``cos_sim < -0.1``, and saves a
   histogram.

Reading the result
------------------
* A real mass of fake samples at meaningfully negative ``cos_sim`` (not
  clustered near zero) CONFIRMS the hypothesis.
* If most fake samples sit near ``cos_sim ~= 0``, the hypothesis is weaker and
  the AUC/AP regression likely has a different cause.

Usage (run from the repo root so relative config paths resolve):
    python eval/reformulate-cf-loss/probe_cos_sim.py \
        --config training/config/detector/effort_ce_masked_counterfactual_backbone.yaml \
        --weights /path/to/old_loss_checkpoint.pth \
        --dataset Celeb-DF-v2 \
        --n_probe 1000 \
        --out_dir eval/reformulate-cf-loss/output
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
        description="Probe cos_sim (fake samples) between learned and "
                    "counterfactual pooler features."
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to the detector YAML config (must match the checkpoint's "
             "svd_trainable_ranks / architecture).",
    )
    parser.add_argument(
        "--weights", type=str, required=True,
        help="Path to the trained .pth checkpoint.",
    )
    parser.add_argument(
        "--dataset", type=str, required=True,
        help="Dataset name to probe (e.g. Celeb-DF-v2, DFDC, UADFV).",
    )
    parser.add_argument(
        "--n_probe", type=int, default=1000,
        help="Number of fake samples to collect (default: 1000).",
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
        help="Directory to save histogram and summary (default: "
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
def collect_fake_cos_sim(model, loader, device, n_probe):
    """Run the model over the loader and collect cos_sim for fake samples."""
    cos_sims = []
    pbar = tqdm(loader, desc="Probing fake cos_sim", leave=True)

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
        cos_sim = model.compute_cos_sim(batch, pred_dict)  # [batch_size]

        mask_fake = labels == 1
        if mask_fake.any():
            cos_sims.extend(cos_sim[mask_fake].cpu().numpy().tolist())
            pbar.set_postfix(n_fake=len(cos_sims))

    cos_sims = np.asarray(cos_sims[:n_probe], dtype=np.float64)
    if cos_sims.size == 0:
        raise RuntimeError("No fake samples collected — check the dataset name "
                           "and labels.")
    return cos_sims


def compute_stats(cos_sim):
    stats = {
        "n": int(cos_sim.size),
        "mean": float(cos_sim.mean()),
        "std": float(cos_sim.std()),
        "median": float(np.median(cos_sim)),
        "min": float(cos_sim.min()),
        "max": float(cos_sim.max()),
        "frac_lt_neg_01": float((cos_sim < -0.1).mean()),
        "frac_lt_0": float((cos_sim < 0).mean()),
    }
    return stats


def plot_histogram(cos_sim, stats, out_path):
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(cos_sim, bins=60, color="#4C72B0", alpha=0.85, edgecolor="white")

    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.0,
               label="cos_sim = 0 (new-loss minimum)")
    ax.axvline(-0.1, color="red", linestyle=":", linewidth=1.2,
               label="cos_sim = -0.1 (negative-mass threshold)")
    ax.axvline(stats["mean"], color="green", linestyle="-", linewidth=1.2,
               label=f"mean = {stats['mean']:.3f}")

    ax.set_xlabel("cos_sim(full_features, cf_features)")
    ax.set_ylabel("count")
    ax.set_title(
        f"Fake-sample counterfactual cos_sim\n"
        f"n={stats['n']}, mean={stats['mean']:.3f}, std={stats['std']:.3f}, "
        f"frac<-0.1={stats['frac_lt_neg_01']:.3f}"
    )
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
    # Routes through the detector's custom load_state_dict, which strips
    # counterfactual_backbone keys and tolerates their absence.
    model.load_state_dict(state_dict)
    print(f"Weights loaded from {args.weights}")

    cos_sim = collect_fake_cos_sim(model, loader, device, args.n_probe)
    stats = compute_stats(cos_sim)

    print("\n" + "=" * 60)
    print("  Fake-sample counterfactual cos_sim summary")
    print("=" * 60)
    for k, v in stats.items():
        print(f"  {k:>16}: {v}")
    print("=" * 60)
    print(f"\n  frac with cos_sim < -0.1 : {stats['frac_lt_neg_01']:.4f}")
    print(f"  frac with cos_sim <  0.0 : {stats['frac_lt_0']:.4f}")

    if stats["frac_lt_neg_01"] > 0.10:
        print("\n  [CONFIRMED] A real mass of fake samples sits at meaningfully "
              "negative cosine similarity.\n"
              "              -> the old relu(cos_sim) loss had a free region "
              "that cos_sim**2 does not.")
    else:
        print("\n  [NOT CONFIRMED] Most fake samples sit near cos_sim ~= 0. The "
              "regression likely has a different cause.")

    os.makedirs(args.out_dir, exist_ok=True)
    npy_path = os.path.join(args.out_dir, "cos_sim_fakes.npy")
    png_path = os.path.join(args.out_dir, "cos_sim_fakes_histogram.png")
    json_path = os.path.join(args.out_dir, "cos_sim_fakes_summary.json")

    np.save(npy_path, cos_sim)
    plot_histogram(cos_sim, stats, png_path)
    with open(json_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nSaved:\n  {npy_path}\n  {png_path}\n  {json_path}")


if __name__ == "__main__":
    main()

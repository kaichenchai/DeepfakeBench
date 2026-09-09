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
2. Probes ``n_probe`` samples from the requested dataset, filtered by
   ``--type`` (``real`` / ``fake`` / ``both``).
3. For each sample, computes ``cos_sim = cosine_similarity(cf_features,
   pred_dict['feat'])`` — the exact quantity the loss acts on.
4. Prints mean/std and the fraction with ``cos_sim < -0.1``, and saves one or
   two histograms (overlaid when ``--type both``).

Reading the result
------------------
* A real mass of fake samples at meaningfully negative ``cos_sim`` (not
  clustered near zero) CONFIRMS the hypothesis.
* If most fake samples sit near ``cos_sim ~= 0``, the hypothesis is weaker and
  the AUC/AP regression likely has a different cause.
* Comparing the real vs fake overlaid histograms (``--type both``) shows how
  the two distributions separate under the learned pooler.

Usage (run from the repo root so relative config paths resolve):
    python eval/reformulate-cf-loss/probe_cos_sim.py \
        --config training/config/detector/effort_ce_masked_counterfactual_backbone.yaml \
        --weights /path/to/old_loss_checkpoint.pth \
        --dataset Celeb-DF-v2 \
        --type both \
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
        "--type", type=str, default="fake",
        choices=["real", "fake", "both"],
        help="Which samples to probe: 'real', 'fake' or 'both'. When 'both' "
             "is given, real and fake cosine similarities are collected "
             "separately and plotted as two overlaid histograms "
             "(default: fake).",
    )
    parser.add_argument(
        "--n_probe", type=int, default=1000,
        help="Number of samples per class to collect (default: 1000).",
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
def collect_cos_sim(model, loader, device, n_probe, data_type):
    """Run the model over the loader and collect cos_sim per class.

    Returns a tuple ``(cos_sims_real, cos_sims_fake)`` of numpy arrays. The
    class(es) requested by ``data_type`` are populated up to ``n_probe`` each;
    the class that was not requested is returned as an empty array.

    Args:
        data_type: one of 'real', 'fake' or 'both'.
    """
    need_real = data_type in ("real", "both")
    need_fake = data_type in ("fake", "both")

    cos_sims_real = []
    cos_sims_fake = []
    pbar = tqdm(loader, desc=f"Probing {data_type} cos_sim", leave=True)

    def enough(arr):
        return len(arr) >= n_probe

    for data_dict in pbar:
        # Stop once each requested class has reached n_probe. For 'both' we
        # must keep going until BOTH classes are full.
        if need_real and need_fake:
            done = enough(cos_sims_real) and enough(cos_sims_fake)
        elif need_real:
            done = enough(cos_sims_real)
        else:  # need_fake
            done = enough(cos_sims_fake)
        if done:
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
        cos_sim_np = cos_sim.cpu().numpy()

        mask_fake = labels == 1
        mask_real = ~mask_fake
        mask_fake_np = mask_fake.cpu().numpy()
        mask_real_np = mask_real.cpu().numpy()

        if need_real and mask_real.any():
            cos_sims_real.extend(cos_sim_np[mask_real_np].tolist())
        if need_fake and mask_fake.any():
            cos_sims_fake.extend(cos_sim_np[mask_fake_np].tolist())
        pbar.set_postfix(n_real=len(cos_sims_real), n_fake=len(cos_sims_fake))

    cos_sims_real = np.asarray(cos_sims_real[:n_probe], dtype=np.float64)
    cos_sims_fake = np.asarray(cos_sims_fake[:n_probe], dtype=np.float64)

    if not need_real:
        cos_sims_real = np.empty(0)
    if not need_fake:
        cos_sims_fake = np.empty(0)

    if cos_sims_real.size == 0 and cos_sims_fake.size == 0:
        raise RuntimeError("No samples collected — check the dataset name "
                           "and labels.")

    return cos_sims_real, cos_sims_fake


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


def plot_histogram(cos_sim_real, cos_sim_fake, stats_real, stats_fake,
                   out_path, data_type):
    """Plot cos_sim histogram(s). When ``data_type`` is 'both', the real and
    fake distributions are overlaid on the same axes."""
    COLOR_REAL = "#4C72B0"
    COLOR_FAKE = "#DD8452"

    fig, ax = plt.subplots(figsize=(8, 5))

    if data_type == "both":
        ax.hist(cos_sim_real, bins=60, color=COLOR_REAL, alpha=0.6,
                edgecolor="white",
                label=f"real (n={stats_real['n']}, mean={stats_real['mean']:.3f})")
        ax.hist(cos_sim_fake, bins=60, color=COLOR_FAKE, alpha=0.6,
                edgecolor="white",
                label=f"fake (n={stats_fake['n']}, mean={stats_fake['mean']:.3f})")
        ax.axvline(stats_real["mean"], color=COLOR_REAL, linestyle="--",
                   linewidth=1.2)
        ax.axvline(stats_fake["mean"], color=COLOR_FAKE, linestyle="--",
                   linewidth=1.2)
        title = (
            f"Real vs fake counterfactual cos_sim (overlaid)\n"
            f"real: n={stats_real['n']}, mean={stats_real['mean']:.3f}, "
            f"std={stats_real['std']:.3f}\n"
            f"fake: n={stats_fake['n']}, mean={stats_fake['mean']:.3f}, "
            f"std={stats_fake['std']:.3f}, "
            f"frac<-0.1={stats_fake['frac_lt_neg_01']:.3f}"
        )
    else:
        is_fake = data_type == "fake"
        cos_sims = cos_sim_fake if is_fake else cos_sim_real
        stats = stats_fake if is_fake else stats_real
        color = COLOR_FAKE if is_fake else COLOR_REAL
        label = "Fake" if is_fake else "Real"
        ax.hist(cos_sims, bins=60, color=color, alpha=0.85,
                edgecolor="white", label=f"{label} samples")
        ax.axvline(stats["mean"], color="green", linestyle="-", linewidth=1.2,
                   label=f"mean = {stats['mean']:.3f}")
        title = (
            f"{label}-sample counterfactual cos_sim\n"
            f"n={stats['n']}, mean={stats['mean']:.3f}, "
            f"std={stats['std']:.3f}, "
            f"frac<-0.1={stats['frac_lt_neg_01']:.3f}"
        )

    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.0,
               label="cos_sim = 0 (new-loss minimum)")
    ax.axvline(-0.1, color="red", linestyle=":", linewidth=1.2,
               label="cos_sim = -0.1 (negative-mass threshold)")

    ax.set_xlabel("cos_sim(full_features, cf_features)")
    ax.set_ylabel("count")
    ax.set_title(title)
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

    def print_summary(name, stats):
        print("\n" + "=" * 60)
        print(f"  {name} counterfactual cos_sim summary")
        print("=" * 60)
        for k, v in stats.items():
            print(f"  {k:>16}: {v}")
        print("=" * 60)
        print(f"\n  frac with cos_sim < -0.1 : {stats['frac_lt_neg_01']:.4f}")
        print(f"  frac with cos_sim <  0.0 : {stats['frac_lt_0']:.4f}")

    def print_confirmation(stats):
        if stats["frac_lt_neg_01"] > 0.10:
            print("\n  [CONFIRMED] A real mass of fake samples sits at "
                  "meaningfully negative cosine similarity.\n"
                  "              -> the old relu(cos_sim) loss had a free "
                  "region that cos_sim**2 does not.")
        else:
            print("\n  [NOT CONFIRMED] Most fake samples sit near cos_sim ~= 0. "
                  "The regression likely has a different cause.")

    cos_sim_real, cos_sim_fake = collect_cos_sim(
        model, loader, device, args.n_probe, args.type
    )

    os.makedirs(args.out_dir, exist_ok=True)

    saved = []

    if args.type == "both":
        stats_real = compute_stats(cos_sim_real)
        stats_fake = compute_stats(cos_sim_fake)

        print_summary("Real-sample", stats_real)
        print_summary("Fake-sample", stats_fake)
        print_confirmation(stats_fake)

        npy_real_path = os.path.join(args.out_dir, "cos_sim_reals.npy")
        npy_fake_path = os.path.join(args.out_dir, "cos_sim_fakes.npy")
        png_path = os.path.join(args.out_dir, "cos_sim_both_histogram.png")
        json_path = os.path.join(args.out_dir, "cos_sim_both_summary.json")

        np.save(npy_real_path, cos_sim_real)
        np.save(npy_fake_path, cos_sim_fake)
        plot_histogram(cos_sim_real, cos_sim_fake, stats_real, stats_fake,
                       png_path, "both")
        with open(json_path, "w") as f:
            json.dump({"real": stats_real, "fake": stats_fake}, f, indent=2)

        saved.extend([npy_real_path, npy_fake_path, png_path, json_path])
    else:
        stats = compute_stats(cos_sim_fake if args.type == "fake" else cos_sim_real)

        label = "Fake" if args.type == "fake" else "Real"
        print_summary(f"{label}-sample", stats)
        if args.type == "fake":
            print_confirmation(stats)

        stem = "fakes" if args.type == "fake" else "reals"
        npy_path = os.path.join(args.out_dir, f"cos_sim_{stem}.npy")
        png_path = os.path.join(args.out_dir, f"cos_sim_{stem}_histogram.png")
        json_path = os.path.join(args.out_dir, f"cos_sim_{stem}_summary.json")

        cos_sims = cos_sim_fake if args.type == "fake" else cos_sim_real
        np.save(npy_path, cos_sims)
        plot_histogram(cos_sims, np.empty(0), stats, None, png_path, args.type)
        with open(json_path, "w") as f:
            json.dump(stats, f, indent=2)

        saved.extend([npy_path, png_path, json_path])

    print("\nSaved:")
    for p in saved:
        print(f"  {p}")


if __name__ == "__main__":
    main()

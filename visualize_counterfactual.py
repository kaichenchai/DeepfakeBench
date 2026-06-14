#!/usr/bin/env python3
"""
Counterfactual Visualization Script for Effort_Custom_Detector.

Compares the full backbone (main + residual) against the counterfactual backbone
(main weights only) to localize where the model focuses on deepfake artifacts.

Three visualization levels:
  Level 1 — Attention Map Difference:
      Difference in last-layer self-attention (CLS token → spatial patches).
      Shows where the residual components redirect the model's attention.

  Level 2 — Residual Feature Norm (per patch):
      Magnitude of the residual contribution at each spatial position in the
      last SVDResidualLinear layer. Highlights patches most affected by the
      learned deepfake-specific adaptation.

  Level 3 — Hidden State Difference (per patch):
      L2 norm of the difference between full and counterfactual last hidden
      states at each patch position. Captures the aggregate effect of all
      residual layers on the final representation.

Usage:
  python visualize_counterfactual.py \
      --config training/config/detector/effort_ce.yaml \
      --checkpoint /path/to/ckpt.pth \
      --image /path/to/image.jpg \
      --output ./output_heatmaps/

  # Or pass a directory of images:
  python visualize_counterfactual.py \
      --config training/config/detector/effort_ce.yaml \
      --checkpoint /path/to/ckpt.pth \
      --image_dir /path/to/images/ \
      --output ./output_heatmaps/
"""

import os
import sys
import argparse
import math

import numpy as np
import cv2
import yaml
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Path setup — add training/ to sys.path so we can import the detector
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.join(SCRIPT_DIR, 'training')
sys.path.insert(0, TRAINING_DIR)

# Import the registry and register all detectors (including effort_custom)
from metrics.registry import DETECTOR
from detectors.effort_custom_detector import SVDResidualLinear, Effort_Custom_Detector  # noqa: F401 — ensure registration

# CLIP ViT-L/14 normalisation constants
CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
CLIP_STD  = np.array([0.26862954, 0.26130258, 0.27577711],  dtype=np.float32)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_image(path: str, resolution: int = 224) -> torch.Tensor:
    """Load an image, resize to (resolution, resolution), normalise with CLIP stats."""
    img = Image.open(path).convert('RGB')
    img = img.resize((resolution, resolution), Image.BICUBIC)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - CLIP_MEAN[None, None, :]) / CLIP_STD[None, None, :]
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
    return tensor


def denormalise(tensor: torch.Tensor) -> np.ndarray:
    """Convert a normalised [C, H, W] tensor back to uint8 numpy (H, W, 3)."""
    arr = tensor.detach().cpu().permute(1, 2, 0).numpy()
    arr = arr * CLIP_STD + CLIP_MEAN
    arr = np.clip(arr, 0, 1) * 255
    return arr.astype(np.uint8)


def patch_heatmap_to_image(
    patch_values: np.ndarray,       # [16, 16]  or  [14, 14]
    original_img: np.ndarray,       # [H, W, 3] uint8
    cmap: int = cv2.COLORMAP_JET,
    alpha: float = 0.55,
) -> np.ndarray:
    """
    Resize a patch-grid heatmap to the original image size and overlay it.
    Returns a uint8 BGR image (for saving with OpenCV).
    """
    # Normalise to [0, 1]
    vmin, vmax = patch_values.min(), patch_values.max()
    if vmax - vmin > 1e-8:
        patch_norm = (patch_values - vmin) / (vmax - vmin)
    else:
        patch_norm = np.zeros_like(patch_values)

    # Upsample to image resolution
    h, w = original_img.shape[:2]
    heatmap = cv2.resize(patch_norm.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    heatmap = (heatmap * 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap, cmap)

    # Overlay
    img_bgr = cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(img_bgr, 1 - alpha, heatmap_colored, alpha, 0)
    return overlay


# ---------------------------------------------------------------------------
# Counterfactual context manager
# ---------------------------------------------------------------------------

class CounterfactualMode:
    """
    Context manager that temporarily nullifies residual components across
    every SVDResidualLinear in the backbone, giving a pure pre-trained pass.
    """

    def __init__(self, backbone: nn.Module):
        self.backbone = backbone
        self.saved: dict = {}

    def __enter__(self):
        for module in self.backbone.modules():
            if isinstance(module, SVDResidualLinear) and module.S_residual is not None:
                self.saved[module] = (
                    module.S_residual,
                    module.U_residual,
                    module.V_residual,
                )
                module.S_residual = None
                module.U_residual = None
                module.V_residual = None
        return self

    def __exit__(self, *args):
        for module, (S, U, V) in self.saved.items():
            module.S_residual = S
            module.U_residual = U
            module.V_residual = V


# ---------------------------------------------------------------------------
# Core visualisation logic
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_visualizations(model: nn.Module, image_tensor: torch.Tensor):
    """
    Run both the full and counterfactual forward passes and return three
    per-patch heatmap arrays (each shape [grid, grid]) along with the
    original image as uint8 numpy.

    Returns
    -------
    orig_img : np.ndarray  [H, W, 3] uint8
    heatmap_attn_diff : np.ndarray  [grid, grid]
    heatmap_residual_norm : np.ndarray  [grid, grid]
    heatmap_hidden_diff : np.ndarray  [grid, grid]
    probs : dict with 'full' and 'counterfactual' keys
    """
    device = next(model.parameters()).device
    backbone: nn.Module = model.backbone
    head: nn.Linear = model.head

    # ------ 1. Full forward pass (with attention outputs) ------------------
    output_full = backbone(image_tensor, output_attentions=True)
    full_hidden = output_full.last_hidden_state              # [1, 257, 1024]
    full_pooler = output_full.pooler_output                  # [1, 1024]
    full_attentions = output_full.attentions                 # tuple of [1, heads, 257, 257]

    logits_full = head(full_pooler)
    prob_full = torch.softmax(logits_full, dim=1)

    # ------ 2. Counterfactual forward pass (main weights only) -------------
    with CounterfactualMode(backbone):
        output_cf = backbone(image_tensor, output_attentions=True)
        cf_hidden = output_cf.last_hidden_state              # [1, 257, 1024]
        cf_pooler = output_cf.pooler_output                  # [1, 1024]
        cf_attentions = output_cf.attentions                 # tuple of [1, heads, 257, 257]

        logits_cf = head(cf_pooler)
        prob_cf = torch.softmax(logits_cf, dim=1)

    # ------ 3. Residual feature norm (need a separate pass with hooks) -----
    # We capture the residual features from the *last* SVDResidualLinear in
    # the last transformer block so we can measure per-patch residual magnitude.
    last_residual_features: list = []

    def hook_residual(module, input, output):
        # module.cached_residual_features was set during the forward pass
        if module.cached_residual_features is not None:
            last_residual_features.append(module.cached_residual_features.clone())

    # Find the last SVDResidualLinear
    hooks = []
    for module in backbone.modules():
        if isinstance(module, SVDResidualLinear):
            # Remove any previous hook reference (we'll re-register)
            pass
    # Register on every SVDResidualLinear so we capture the last one naturally
    last_svd_module = None
    for module in backbone.modules():
        if isinstance(module, SVDResidualLinear):
            last_svd_module = module

    if last_svd_module is not None:
        handle = last_svd_module.register_forward_hook(hook_residual)
        _ = backbone(image_tensor)          # one more full forward to capture residual
        handle.remove()

    # ------ 4. Build the three heatmaps -----------------------------------

    # --- Level 1: Attention Map Difference (last layer, CLS → patches) ---
    last_attn_full = full_attentions[-1]              # [1, heads, 257, 257]
    last_attn_cf   = cf_attentions[-1]                # [1, heads, 257, 257]
    # CLS token (idx 0) attending to spatial patches (idx 1:257)
    cls_attn_full = last_attn_full[0, :, 0, 1:]       # [heads, 256]
    cls_attn_cf   = last_attn_cf[0, :, 0, 1:]         # [heads, 256]
    attn_diff = (cls_attn_full - cls_attn_cf).mean(dim=0)  # [256]

    # Reshape to square grid (ViT-L/14 with 224×224 → 16×16 patches)
    grid_size = int(math.sqrt(attn_diff.shape[0]))
    heatmap_attn_diff = attn_diff.reshape(grid_size, grid_size).cpu().numpy()

    # --- Level 2: Residual Feature Norm (per patch) -----------------------
    if last_residual_features:
        # Shape: [1, 257, 1024] — CLS at idx 0, then 256 spatial patches
        res_feat = last_residual_features[-1][0]       # [257, 1024]
        res_norm = torch.norm(res_feat[1:], dim=1)     # [256]  exclude CLS
        heatmap_residual_norm = res_norm.reshape(grid_size, grid_size).cpu().numpy()
    else:
        heatmap_residual_norm = np.zeros((grid_size, grid_size), dtype=np.float32)

    # --- Level 3: Hidden State Difference (per patch) ---------------------
    hidden_diff = full_hidden[0, 1:, :] - cf_hidden[0, 1:, :]  # [256, 1024]
    hidden_diff_norm = torch.norm(hidden_diff, dim=1)           # [256]
    heatmap_hidden_diff = hidden_diff_norm.reshape(grid_size, grid_size).cpu().numpy()

    # ------ 5. Original image (denormalised) ------------------------------
    orig_img = denormalise(image_tensor[0])

    probs = {
        'full': prob_full[0, 1].item(),
        'counterfactual': prob_cf[0, 1].item(),
    }

    return orig_img, heatmap_attn_diff, heatmap_residual_norm, heatmap_hidden_diff, probs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Counterfactual visualisation for Effort_Custom_Detector'
    )
    parser.add_argument('--config', type=str, required=True,
                        help='Path to detector YAML config (e.g. training/config/detector/effort_ce.yaml)')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained model checkpoint (.pth)')
    parser.add_argument('--image', type=str, default=None,
                        help='Single image to visualise')
    parser.add_argument('--image_dir', type=str, default=None,
                        help='Directory of images to visualise (processed in batch)')
    parser.add_argument('--output', type=str, default='./counterfactual_output',
                        help='Output directory for heatmap images')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device: "cuda", "cpu", or "auto" (default)')
    parser.add_argument('--levels', type=str, nargs='+',
                        default=['attn', 'residual', 'hidden'],
                        choices=['attn', 'residual', 'hidden'],
                        help='Which visualisation levels to generate '
                             '(default: all three)')
    args = parser.parse_args()

    # --- Device ---
    if args.device == 'auto':
        device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device_str = args.device
    device = torch.device(device_str)

    # --- Load config ---
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    # Ensure model_name is set
    if 'model_name' not in config:
        config['model_name'] = 'effort_custom'

    # Override dataset_json_folder for local paths if needed
    if 'dataset_json_folder' not in config:
        config['dataset_json_folder'] = os.path.join(SCRIPT_DIR, 'preprocessing/dataset_json')

    # --- Build model ---
    print(f"Loading model '{config['model_name']}' from config: {args.config}")
    model_class = DETECTOR[config['model_name']]
    model = model_class(config).to(device)
    model.eval()

    # --- Load checkpoint ---
    ckpt = torch.load(args.checkpoint, map_location=device)
    # Strip 'module.' prefix if present (from DataParallel)
    state_dict = ckpt
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    print(f"Loaded checkpoint: {args.checkpoint}")

    # --- Collect images ---
    image_paths = []
    if args.image:
        image_paths.append(args.image)
    if args.image_dir:
        for fname in sorted(os.listdir(args.image_dir)):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                image_paths.append(os.path.join(args.image_dir, fname))

    if not image_paths:
        print("Error: No images provided. Use --image or --image_dir.")
        sys.exit(1)

    # --- Create output directory ---
    os.makedirs(args.output, exist_ok=True)

    # --- Process each image ---
    for img_path in image_paths:
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        print(f"\nProcessing: {img_path}")

        image_tensor = load_image(img_path, resolution=224).to(device)

        orig_img, hm_attn, hm_res, hm_hid, probs = compute_visualizations(model, image_tensor)

        # Print prediction info
        print(f"  Full backbone prob (fake):       {probs['full']:.4f}")
        print(f"  Counterfactual prob (fake):      {probs['counterfactual']:.4f}")
        print(f"  Δ prob (full − counterfactual):  {probs['full'] - probs['counterfactual']:+.4f}")

        # Save each requested level
        level_configs = {
            'attn': ('level1_attn_diff', hm_attn, 'Attention Map Difference'),
            'residual': ('level2_residual_norm', hm_res, 'Residual Feature Norm'),
            'hidden': ('level3_hidden_diff', hm_hid, 'Hidden State Difference'),
        }

        for level_key in args.levels:
            prefix, heatmap, title = level_configs[level_key]
            out_path = os.path.join(args.output, f'{img_name}_{prefix}.png')

            overlay = patch_heatmap_to_image(heatmap, orig_img, alpha=0.55)

            # Add title text
            cv2.putText(
                overlay, title,
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
                cv2.LINE_AA,
            )
            cv2.putText(
                overlay,
                f"Full: {probs['full']:.3f}  CF: {probs['counterfactual']:.3f}",
                (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                cv2.LINE_AA,
            )

            cv2.imwrite(out_path, overlay)
            print(f"  Saved: {out_path}")

    print("\nDone.")


if __name__ == '__main__':
    main()

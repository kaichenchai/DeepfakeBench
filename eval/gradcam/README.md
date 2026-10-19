# GradCAM heatmaps

`generate_gradcam.py` runs a trained detector on the test split, back-propagates
the selected class score to the last CLIP ViT transformer block, and writes
GradCAM heatmaps over the input frames.

Samples can be drawn at random (balanced or not) or selected by outcome:
high-confidence correct real/fake frames and low-confidence misclassifications.
The outcome mode first scores a scan set with inference only, then renders
GradCAM for the selected frames.

GradCAM is computed from the **pre-attention LayerNorm output** of the last
encoder layer (`backbone.encoder.layers[-1].layer_norm1`). This is the last
point where every patch token still affects the CLS token that the classifier
reads; using the layer's output would give zero gradients for all patch tokens.
The 16x16 patch map (224 px input, patch size 14) is upsampled to 224x224.

## Run on a detector config and checkpoint

```bash
python eval/gradcam/generate_gradcam.py \
    --config training/config/detector/effort_ce_orthogonal_weight.yaml \
    --weights logs/training/effort_ce_orthogonal_weight_effort_custom_2026-08-23-19-55-41/final/ckpt_final.pth \
    --dataset FaceForensics++ \
    --output-dir eval/gradcam/data/effort \
    --num-samples 32 \
    --balance
```

[`get_gradcam.job`](../../get_gradcam.job) runs the EFFORT and CE + HSIC + CF
models as a Slurm array job, one GPU per model, with the same seed and sample
selection. It renders confident correct real/fake frames plus low- and
high-confidence errors. Keep `#rj array=0-1` in sync with the number of entries
in `PAIRS`.

## Sampling by outcome

```bash
python eval/gradcam/generate_gradcam.py \
    --config training/config/detector/effort_ce_orthogonal_weight.yaml \
    --weights logs/training/effort_ce_orthogonal_weight_effort_custom_2026-08-23-19-55-41/final/ckpt_final.pth \
    --dataset FaceForensics++ \
    --output-dir eval/gradcam/data/effort_categories \
    --categories real_high fake_high misclassified_low misclassified_high \
    --per-category 4 \
    --scan-samples 4096
```

The scan pass scores `--scan-samples` random frames (0 scans the whole
candidate pool) without gradients. Categories are mutually exclusive:

| Category | Definition |
|----------|------------|
| `real_high` | Correct real, with `P(real) >= --high-confidence` |
| `fake_high` | Correct fake, with `P(fake) >= --high-confidence` |
| `misclassified_low` | Wrong prediction, with predicted-class probability `<= --low-confidence` (closest to 0.5 first) |
| `misclassified_high` | Wrong prediction, with predicted-class probability `>= --high-confidence` |

The most confident frames are selected first for the `*_high` categories, and
the frames closest to the decision boundary first for `misclassified_low`. If a
category contains fewer than `--per-category` frames, all of them are rendered
and a warning is printed. In-domain detectors can be very accurate, so finding
misclassifications may require increasing `--scan-samples`.

When `--target-class` is `auto` (the default in category mode), correct
real/fake frames explain their ground-truth class and misclassifications
explain the class the model predicted.

## Output files

For an output directory `eval/gradcam/data/<model>/`:

| File | Contents |
|------|----------|
| `images/<stem>_input.png` | Input frame after denormalization |
| `images/<stem>_heatmap.png` | GradCAM heatmap |
| `images/<stem>_overlay.png` | Heatmap blended over the input |
| `summary.png` | Input / heatmap / overlay contact sheet |
| `results.csv` | Ground-truth label, category, prediction, `prob_fake`, target class, and file paths |
| `meta.json` | Config, checkpoint, dataset, layer, sampling, scan counts, and rendering settings |
| `gradcam.npy` | Raw CAM maps, `(N, 224, 224)`, only with `--save-npy` |
| `gradcam_indices.npy` | Dataset indices for the saved maps, only with `--save-npy` |

The ground truth comes from the dataset's test split; `prob_fake` is the model's
softmax score for class 1. `stem` is
`<index>_<truth>[_<category>]_<video>_<frame>`.

## Options

| Option | Default | Meaning |
|--------|---------|---------|
| `--num-samples` | `16` | Number of frames to render when not using `--categories` |
| `--max-samples` | all | Restrict the candidate pool to the first N dataset entries |
| `--balance` | off | Sample equal numbers of real and fake frames |
| `--categories` | off | Render by outcome: `real_high`, `fake_high`, `misclassified_low`, `misclassified_high` |
| `--per-category` | `4` | Frames rendered per category |
| `--scan-samples` | `1024` | Frames scored before category selection; `0` scans the whole pool |
| `--scan-batch-size` | `16` | Batch size for the confidence scan |
| `--high-confidence` | `0.90` | Minimum predicted-class probability for `*_high` |
| `--low-confidence` | `0.65` | Maximum predicted-class probability for `misclassified_low` |
| `--seed` | `1024` | Dataset-order and sampling seed |
| `--target-class` | `fake` / `auto` | Explain the fake, real, or predicted class; `auto` is the default with `--categories` |
| `--token-reduction` | `patch` | Average gradients over patch tokens (`patch`) or use the CLS token gradient (`cls`) |
| `--layer-index` | `-1` | Encoder layer to hook; `-1` is the last block |
| `--alpha` | `0.5` | Heatmap opacity in the overlay |
| `--cmap` | `jet` | Matplotlib colormap |
| `--summary-samples` | `8` | Frames in `summary.png`; `0` disables it |
| `--save-npy` | off | Write raw CAM maps |
| `--device` | `auto` | `auto`, `cpu`, or `cuda` |

`--balance` and `--categories` cannot be combined. The script seeds Python,
NumPy, and PyTorch before building the dataset, so the same seed scans and
selects the same frames for every model. `get_gradcam.job` relies on this to
make the model panels directly comparable.

Use `--target-class real` to visualise evidence for the real class, or
`--target-class pred` to always explain the model's decision. Video-level
configs (`video_mode: true`) are not supported.
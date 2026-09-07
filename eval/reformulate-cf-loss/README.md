# Reformulate CF loss — Step 1 diagnosis probe

Probes the cosine similarity between the detector's learned pooler features
and the frozen counterfactual (pristine CLIP ViT) pooler features, **on the
fake subset only**, to confirm whether the old `relu(cos_sim)` loss was
exploiting a free region at `cos_sim < 0`.

## Files

| File | Purpose |
|------|---------|
| `probe_cos_sim.py` | Probes **fake** samples, histograms `cos_sim` vs `cf_features` (confirms the `relu` free-region hypothesis). |
| `fake_loss_only_probe_real_image_similarity.py` | Probes **real** samples, histograms `cos_sim` and `MSE` vs `cf_features` for the fake-only-loss checkpoint. |
| `effort_custom_detector_probe.py` | Probe wrapper that **subclasses** `training/detectors/effort_custom_detector.py` and adds `compute_cos_sim()`, `compute_mse()`, `compute_normalized_mse()` (registers as `effort_custom_probe`). Automatically stays in sync with the newest training detector — no duplicate copy to maintain. |

## Usage

Run from the **repo root** (relative config paths — CLIP weights, `rgb_dir`,
`dataset_json_folder` — resolve against the repo root):

```bash
python eval/reformulate-cf-loss/probe_cos_sim.py \
    --config training/config/detector/effort_ce_masked_counterfactual_backbone.yaml \
    --weights /path/to/old_loss_checkpoint.pth \
    --dataset Celeb-DF-v2 \
    --n_probe 1000 \
    --out_dir eval/reformulate-cf-loss/output
```

### Arguments

| Flag | Description |
|------|-------------|
| `--config` | Detector YAML config. Must match the checkpoint's architecture. |
| `--weights` | Path to the `.pth` checkpoint (the **old-loss** one for this diagnosis). |
| `--dataset` | Dataset name, e.g. `Celeb-DF-v2`, `DFDC`, `UADFV`, `FaceForensics++`. |
| `--n_probe` | Number of fake samples to collect (default `1000`). |
| `--batch_size` | Optional override of the config's `test_batchSize`. |
| `--svd_trainable_ranks` | Optional override of `svd_trainable_ranks`. **Must match the checkpoint** or loading will fail on shape mismatch. |
| `--device` | `auto` / `cpu` / `cuda` (default `auto`). |
| `--out_dir` | Output directory (default `eval/reformulate-cf-loss/output`). |

> **Important:** the checkpoint architecture depends on `svd_trainable_ranks`
> (`r = 1024 - svd_trainable_ranks`). If the checkpoint was trained with
> e.g. `svd_trainable_ranks: 16`, pass `--svd_trainable_ranks 16` (or use a
> config that sets it).

## Output

- `cos_sim_fakes_histogram.png` — histogram with reference lines at `0`,
  `-0.1`, and the mean.
- `cos_sim_fakes.npy` — raw per-sample `cos_sim` values (fake subset).
- `cos_sim_fakes_summary.json` — mean / std / median / min / max and the
  fraction with `cos_sim < -0.1` and `< 0`.

## Interpreting the result

- **`frac < -0.1` is substantial (e.g. > 10%)** → a real mass of fake samples
  sits at meaningfully negative cosine similarity, so the old `relu` loss had a
  free region that `cos_sim ** 2` does not. **Hypothesis confirmed.**
- **Most fake samples cluster near `cos_sim ≈ 0`** → the hypothesis is weak;
  the AUC/AP regression has a different cause and should be investigated
  before changing the loss.

## Fake-only real-image probe

For the fake-only-loss checkpoint (no real branch ever applied), measure the
quantity the real branch would have enforced — `cos_sim` / `MSE` between
`full_features` and `cf_features` — on **real** images:

```bash
python eval/reformulate-cf-loss/fake_loss_only_probe_real_image_similarity.py \
    --config training/config/detector/<fake_only_config>.yaml \
    --weights /path/to/fake_only_loss_checkpoint.pth \
    --dataset Celeb-DF-v2 \
    --n_probe 1000
```

Outputs `real_image_similarity_histograms.png` (cos_sim + MSE subplots),
`cos_sim_real.npy`, `mse_real.npy`, and `real_image_similarity_summary.json`.
See the script's docstring for how to read the result (emergent property vs.
"real == cf" not mattering for generalization).

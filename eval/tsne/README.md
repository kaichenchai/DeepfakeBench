# t-SNE feature plots

`extract_tsne.py` runs a detector (or vanilla CLIP) on a dataset, takes the
model's `feat` output, computes a 2-D t-SNE embedding, and writes CSV tables
that can be plotted directly with `pgfplots`.

Feature extraction and checkpoint loading are shared with
[`eval/pca/extract_features.py`](../pca/extract_features.py), so the features
are the same ones used for the PCA analysis.

The pipeline can run in one pass, or in two cached stages so the expensive GPU
forward passes happen once and t-SNE can be re-run cheaply (or locally):

```bash
# stage 1 (GPU): extract features + ground-truth labels and cache them
python eval/tsne/extract_tsne.py \
    --config training/config/detector/effort_ce_hsic_masked_counterfactual_backbone_clip_cf.yaml \
    --weights logs/training/effort_ce_hsic_masked_counterfactual_backbone_clip_cf_effort_custom_seed_1024_2026-10-04-23-07-22/final/ckpt_final.pth \
    --dataset FaceForensics++ \
    --save-features eval/tsne/features/ce_hsic_cf \
    --extract-only

# stage 2 (CPU only): t-SNE from the cache
python eval/tsne/extract_tsne.py \
    --features eval/tsne/features/ce_hsic_cf \
    --output-dir eval/tsne/data/ce_hsic_cf
```

Stage 2 imports only NumPy and scikit-learn, so it runs anywhere — no GPU,
checkpoint, or dataset needed.

## Extract from a detector config and checkpoint

One-pass extraction and embedding:

```bash
python eval/tsne/extract_tsne.py \
    --config training/config/detector/effort_ce_hsic_masked_counterfactual_backbone_clip_cf.yaml \
    --weights logs/training/effort_ce_hsic_masked_counterfactual_backbone_clip_cf_effort_custom_seed_1024_2026-10-04-23-07-22/final/ckpt_final.pth \
    --dataset FaceForensics++ \
    --output-dir eval/tsne/data/ce_hsic_cf
```

Vanilla CLIP does not need a detector checkpoint:

```bash
python eval/tsne/extract_tsne.py \
    --model-type vanilla-clip \
    --config training/config/detector/effort.yaml \
    --dataset FaceForensics++ \
    --output-dir eval/tsne/data/vanilla_clip
```

The `--config` is still required in vanilla-clip mode because it supplies the
dataset paths, preprocessing, resolution, and test loader settings. Override
the local CLIP directory with `--clip-path` if needed.

## Slurm jobs

The vanilla CLIP, EFFORT, and CE + HSIC + CF models used in the PCA comparison
run as two array jobs (three tasks each, `#rj array=0-2`):

```bash
rj get_tsne_features.job   # GPU: cache features + labels
rj get_tsne_data.job       # CPU: t-SNE + CSV output (no GPU requested)
```

Keep `#rj array` in sync with the number of entries in each job's `PAIRS`.
The feature caches live in `eval/tsne/features/<model>/`; they are plain
`.npy` files you can also copy off the cluster and embed locally with:

```bash
python eval/tsne/extract_tsne.py \
    --features eval/tsne/features/effort \
    --output-dir eval/tsne/data/effort
```

t-SNE is computed on all extracted samples. The coordinates written for
plotting are then reduced to `--max-points` points, so LaTeX does not have to
render the full 5,600-point set. The same seeded subset is used for every
model. Uniform downsampling keeps the dataset ratio (about 20% real); add
`--balance` for an even real/fake split.

## Cached features

`--save-features <dir>` writes:

| File | Shape | Contents |
|------|-------|----------|
| `features.npy` | `(samples, dimensions)` | Model `feat` output, float32 |
| `labels.npy` | `(samples,)` | Ground truth per point: `0` real, `1` fake |
| `extraction_meta.json` | | Config, checkpoint, dataset, sample count |

`--features <dir>` reads those files and skips model construction entirely.
Row order is identical between the two files, and the dataset shuffle is seeded
with `--seed`, so every model's cache is aligned index-for-index.

## Output files

For an output directory `eval/tsne/data/<model>/`:

| File | Columns | Contents |
|------|---------|----------|
| `tsne.csv` | `x,y,label,class` | Downsampled points; `label` is `0` (real) or `1` (fake), `class` is `real`/`fake` |
| `real.csv` | `x,y,label,class` | Downsampled real samples only (label 0) |
| `fake.csv` | `x,y,label,class` | Downsampled fake samples only (label 1) |
| `meta.json` | | Config, checkpoint, dataset, full/plotted sample counts, and t-SNE parameters |

The labels are the dataset ground truth from the test loader, not model
predictions.

## Plotting with pgfplots

Plot real and fake as separate series:

```latex
\begin{tikzpicture}
\begin{axis}[
    width=0.48\linewidth,
    xlabel={$t_1$},
    ylabel={$t_2$},
    xtick=\empty,
    ytick=\empty,
    legend style={draw=none, fill=none, at={(0.02,0.98)}, anchor=north west},
]
    \addplot[only marks, mark=*, mark size=0.9pt, blue, opacity=0.55]
        table[x=x, y=y, col sep=comma] {eval/tsne/data/ce_hsic_cf/real.csv};
    \addlegendentry{Real}
    \addplot[only marks, mark=*, mark size=0.9pt, red, opacity=0.55]
        table[x=x, y=y, col sep=comma] {eval/tsne/data/ce_hsic_cf/fake.csv};
    \addlegendentry{Fake}
\end{axis}
\end{tikzpicture}
```

Alternatively, use the combined file with the ground-truth class column:

```latex
\addplot[
    only marks,
    mark=*,
    mark size=0.9pt,
    scatter,
    scatter src=explicit symbolic,
    scatter/classes={
        0={mark=*, blue},
        1={mark=*, red}
    },
] table[x=x, y=y, meta=class, col sep=comma]
    {eval/tsne/data/ce_hsic_cf/tsne.csv};
```

## Device and performance

Feature extraction uses `--device auto` by default: CUDA when
`torch.cuda.is_available()` is true, otherwise CPU. t-SNE itself is CPU-only
(`--n-jobs`), so the GPU only speeds up the model forward passes.

The script prints the selected device, a warning when `auto` falls back to CPU,
and timing for feature extraction and t-SNE. `get_tsne_data.job` passes
`--device cuda` and checks CUDA before starting, so a misconfigured GPU
environment fails immediately instead of running on CPU for hours.

### t-SNE speed

Benchmark on the existing 5,600 × 1,024 vanilla-CLIP features (local CPU,
50 PCA components, perplexity 30):

| Setting | Fit time | Trustworthiness@10 |
|---------|----------|--------------------|
| 1,000 iterations | 8.3 s | 0.9998 |
| 750 iterations | 6.5 s | 0.9998 |
| 500 iterations | 4.4 s | 0.9998 |
| 500 iterations + `--angle 0.8` | 3.5 s | 0.9997 |
| 250 iterations | 2.4 s | 0.9973 |
| 2,000 points pre-sampled, 1,000 iterations | 3.2 s | — |

The optimisation loop is effectively single-threaded: `n_jobs=1` took 8.8 s
versus 8.2 s for `n_jobs=8`, so extra CPUs do not help much. Reducing PCA from
50 to 30 dimensions also made no measurable difference. The job therefore uses
`--max-iter 500 --angle 0.8`, which is about 2.4× faster than the original
1,000-iteration, `angle=0.5` run with no measurable loss in local neighbourhood
preservation on this data.

For a larger speedup, pre-sample before t-SNE (for example
`--max-samples 2000`), at the cost of computing a different embedding: the
neighbour graph is then built only from that subset.

## t-SNE parameters

Default settings:

| Option | Default | Meaning |
|--------|---------|---------|
| `--perplexity` | `30` | t-SNE perplexity |
| `--pca-components` | `50` | PCA dimensions before t-SNE; `0` disables PCA |
| `--max-iter` | `1000` | Optimisation iterations (the job uses `500`) |
| `--angle` | `0.5` | Barnes-Hut accuracy/speed trade-off; higher is faster (job uses `0.8`) |
| `--seed` | `1024` | Random seed |
| `--n-jobs` | `-1` | Worker processes; `-1` uses all cores |
| `--max-points` | `100` | Points written for plotting; `0` keeps all 5,600 |
| `--balance` | off | Sample equal numbers of real and fake points |
| `--save-features` | — | Cache `features.npy` + `labels.npy` for later runs |
| `--extract-only` | off | Stop after `--save-features`; skip t-SNE |
| `--features` | — | Reuse a cache instead of running a model |

Use the same parameters for every model so the panels are comparable. Because
the dataset order and the downsampling RNG are seeded with `--seed`, every
model writes exactly the same sample indices. If you use `--max-samples` to
extract a small subset, lower `--perplexity` accordingly (it must be below one
third of the sample count).

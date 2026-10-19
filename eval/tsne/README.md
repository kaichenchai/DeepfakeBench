# t-SNE feature plots

`extract_tsne.py` runs a detector (or vanilla CLIP) on a dataset, takes the
model's `feat` output, computes a 2-D t-SNE embedding, and writes plain-text
tables that can be plotted directly with `pgfplots`.

Feature extraction and checkpoint loading are shared with
[`eval/pca/extract_features.py`](../pca/extract_features.py), so the features
are the same ones used for the PCA analysis.

## Extract from a detector config and checkpoint

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

[`get_tsne_data.job`](../../get_tsne_data.job) runs the vanilla CLIP, EFFORT,
and CE + HSIC + CF models used in the PCA comparison.

t-SNE is computed on all extracted samples. The coordinates written for
plotting are then reduced to `--max-points` points (default `2000`), so LaTeX
does not have to render the full 5,600-point set. The same seeded subset is
used for every model. Uniform downsampling keeps the dataset ratio (about 20%
real); add `--balance` for an even real/fake split.

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

## t-SNE parameters

Default settings:

| Option | Default | Meaning |
|--------|---------|---------|
| `--perplexity` | `30` | t-SNE perplexity |
| `--pca-components` | `50` | PCA dimensions before t-SNE; `0` disables PCA |
| `--max-iter` | `1000` | Optimisation iterations |
| `--seed` | `1024` | Random seed |
| `--n-jobs` | `-1` | Worker processes; `-1` uses all cores |
| `--max-points` | `2000` | Points written for plotting; `0` keeps all 5,600 |
| `--balance` | off | Sample equal numbers of real and fake points |

Use the same parameters for every model so the panels are comparable. Because
the dataset order and the downsampling RNG are seeded with `--seed`, every
model writes exactly the same sample indices. If you use `--max-samples` to
extract a small subset, lower `--perplexity` accordingly (it must be below one
third of the sample count).

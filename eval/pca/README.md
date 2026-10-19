# PCA component count

`calculate_components.py` reports the minimum number of principal components
needed to retain a requested percentage of the variance in a feature matrix.
The default threshold is 90%, matching the PCA plot convention.

To extract the detector's feature output directly from a config and checkpoint:

```bash
python eval/pca/extract_features.py \
    --config training/config/detector/effort.yaml \
    --weights /path/to/model.pth \
    --dataset Celeb-DF-v2 \
    --output features.npy
```

The extractor runs the detector in evaluation mode and saves its `feat` output,
not the model weights. For the `effort` detector this is the 1024-dimensional
backbone representation. Use `--labels-output labels.npy` to save matching
dataset labels.

Vanilla CLIP can be extracted without a detector checkpoint. By default it
loads the local ViT-L/14 weights from
`models--openai--clip-vit-large-patch14/`:

```bash
python eval/pca/extract_features.py \
    --model-type vanilla-clip \
    --config training/config/detector/effort.yaml \
    --dataset Celeb-DF-v2 \
    --output vanilla_clip_features.npy
```

The `--config` is still required in vanilla-clip mode because it supplies the
dataset paths, preprocessing, resolution, and test loader settings. Override
the local CLIP directory with `--clip-path` if needed.

Then calculate the PCA component count:

```bash
python eval/pca/calculate_components.py \
    --model-file features.npy \
    --threshold 90 \
    --output eval/pca/results.json
```

The input must contain a matrix shaped `(samples, features)`. Supported files
are `.npy`, `.npz`, `.pkl`, `.pickle`, `.pt`, `.pth`, and `.ckpt`. For mapping
objects, the script first checks `features`, `feat`, `embeddings`, and
`embedding`; if none is present, exactly one 2-D numeric value must exist.

The output includes the component count, retained variance, variance loss, and
the per-component and cumulative explained-variance arrays. The printed
`Components required` value is the bar value for a model in the plot.

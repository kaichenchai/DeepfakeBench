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

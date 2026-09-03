#!/usr/bin/env python3
"""
Render per-metric results tables as PNG images.

Reads the per-metric CSVs produced by compute_mean_std.py (eval/<metric>.csv),
then draws a model (rows) x dataset (columns) table for each of the 5 metrics:
acc, ap, auc, eer, video_auc.

Each cell shows "mean ± std" across the 5 seeds. Within each dataset column:
  - the best model  (highest value; lowest for EER)          -> BOLD
  - the second-best model                                    -> UNDERLINED

Outputs: eval/<metric>.png

Run with the repo's uv venv (matplotlib is required):
    source .venv/bin/activate
    python eval/make_metric_tables_png.py
"""

import argparse
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import lines as mlines

HERE = os.path.dirname(os.path.abspath(__file__))

METRICS = ["acc", "ap", "auc", "eer", "video_auc"]
HIGHER_BETTER = {m: True for m in METRICS}
HIGHER_BETTER["eer"] = False  # EER: lower is better

# (display label, raw dataset key used in the CSVs)
DATASETS = [
    ("FF++", "FaceForensics++"),
    ("UADFV", "UADFV"),
    ("CelebDFv1", "Celeb-DF-v1"),
    ("CelebDFv2", "Celeb-DF-v2"),
    ("DFDCP", "DFDCP"),
    ("DFDC", "DFDC"),
    ("DFD", "DeepFakeDetection"),
]

MODELS = [
    "ce",
    "ce_hsic",
    "ce_masked_counterfactual_backbone",
    "ce_hsic_masked_counterfactual_backbone",
    "ce_weight",
    "ce_orthogonal",
    "ce_orthogonal_weight",
]

N_DECIMALS = 3  # decimal places for mean and std in the cells


def load_metric(metric, datasets):
    """Return {model: {display_dataset: (mean, std)}} for one metric."""
    path = os.path.join(HERE, f"{metric}.csv")
    data = {m: {} for m in MODELS}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            model = row["model"]
            if model not in data:
                continue
            for disp, raw in datasets:
                mean = row.get(f"{raw}_mean")
                std = row.get(f"{raw}_std")
                if mean not in (None, ""):
                    data[model][disp] = (float(mean),
                                         float(std) if std not in (None, "") else 0.0)
    return data


def rank_models(values, higher_better):
    """Order models by performance for one dataset column. Returns (best, second)."""
    items = [(m, v) for m, v in values.items() if v is not None]
    items.sort(key=lambda kv: kv[1], reverse=higher_better)
    best = items[0][0]
    second = items[1][0] if len(items) > 1 else None
    return best, second


def render_metric(metric, data, datasets, out_path, show_avg=True):
    n_rows = len(MODELS)
    n_cols = len(datasets)

    fig, ax = plt.subplots(figsize=(16.0, 4.0 + 0.62 * n_rows))
    ax.axis("off")

    # ----- geometry (data coords) -----
    col_w = 1.7
    row_h = 1.0
    x0 = 0.0
    y_bottom = 0.0
    header_y = y_bottom + n_rows * row_h + 1.0          # dataset headers
    title_y = header_y + 1.1                            # figure title
    label_x = -0.35                                     # right edge of model labels
    n_cols_total = n_cols + (1 if show_avg else 0)      # + overall-average column

    ax.set_xlim(label_x - 4.2, x0 + n_cols_total * col_w + 0.4)
    ax.set_ylim(y_bottom - 0.7, title_y + 0.4)

    # ----- title -----
    ax.text((label_x + x0 + n_cols_total * col_w) / 2, title_y,
            f"Cross-dataset {metric.upper()}  (mean $\\pm$ std over seeds)",
            ha="center", va="center", fontsize=15, fontweight="bold")

    # ----- dataset header row -----
    for j, (disp, _) in enumerate(datasets):
        cx = x0 + j * col_w + col_w / 2
        ax.add_patch(plt.Rectangle((x0 + j * col_w, header_y - 0.42),
                                   col_w, 0.84, facecolor="#e6eef6",
                                   edgecolor="black", linewidth=0.6))
        ax.text(cx, header_y, disp, ha="center", va="center",
                fontsize=11, fontweight="bold")

    if show_avg:
        # overall-average header cell (rightmost column)
        cx_avg = x0 + n_cols * col_w + col_w / 2
        ax.add_patch(plt.Rectangle((x0 + n_cols * col_w, header_y - 0.42),
                                   col_w, 0.84, facecolor="#e6eef6",
                                   edgecolor="black", linewidth=0.6))
        ax.text(cx_avg, header_y, "Avg.", ha="center", va="center",
                fontsize=11, fontweight="bold")

    # ----- per-cell highlighting: best (bold) and second-best (underline) -----
    best_cells = {}      # (row, col) -> True  (bold)
    under_cells = {}     # (row, col) -> True  (underline)
    for j, (disp, _) in enumerate(datasets):
        values = {}
        for i, model in enumerate(MODELS):
            cell = data.get(model, {}).get(disp)
            values[model] = cell[0] if cell else None
        best, second = rank_models(values, HIGHER_BETTER[metric])
        best_i = MODELS.index(best)
        best_cells[(best_i, j)] = True
        if second is not None:
            under_cells[(MODELS.index(second), j)] = True

    # ----- overall average per model: mean over the included dataset means -----
    avg_mean = {}
    for model in MODELS:
        vals = [data.get(model, {}).get(disp) for disp, _ in datasets]
        vals = [v[0] for v in vals if v is not None]
        avg_mean[model] = (sum(vals) / len(vals)) if vals else None
    best_avg, second_avg = (None, None)
    if show_avg:
        best_avg, second_avg = rank_models(avg_mean, HIGHER_BETTER[metric])

    # ----- body cells -----
    second_texts = []
    for i, model in enumerate(MODELS):
        yc = y_bottom + (n_rows - 1 - i) * row_h + row_h / 2
        # model label
        ax.text(label_x, yc, model, ha="right", va="center",
                fontsize=10, fontweight="bold")
        for j, (disp, _) in enumerate(datasets):
            cx = x0 + j * col_w + col_w / 2
            ax.add_patch(plt.Rectangle((x0 + j * col_w, yc - row_h / 2),
                                       col_w, row_h, facecolor="none",
                                       edgecolor="#9aa5b1", linewidth=0.4))
            cell = data.get(model, {}).get(disp)
            if cell is None:
                ax.text(cx, yc, "-", ha="center", va="center", fontsize=9)
                continue
            mean, std = cell
            txt = f"{mean:.{N_DECIMALS}f} $\\pm$ {std:.{N_DECIMALS}f}"
            is_best = best_cells.get((i, j), False)
            is_second = under_cells.get((i, j), False)
            t = ax.text(cx, yc, txt, ha="center", va="center", fontsize=9.5,
                        fontweight="bold" if is_best else "normal")
            if is_second:
                second_texts.append(t)

        if show_avg:
            # overall-average cell (rightmost column)
            cx_avg = x0 + n_cols * col_w + col_w / 2
            ax.add_patch(plt.Rectangle((x0 + n_cols * col_w, yc - row_h / 2),
                                       col_w, row_h, facecolor="none",
                                       edgecolor="#9aa5b1", linewidth=0.4))
            avgv = avg_mean.get(model)
            if avgv is None:
                ax.text(cx_avg, yc, "-", ha="center", va="center", fontsize=9)
            else:
                is_best = (model == best_avg)
                is_second = (model == second_avg)
                t = ax.text(cx_avg, yc, f"{avgv:.{N_DECIMALS}f}",
                            ha="center", va="center", fontsize=9.5,
                            fontweight="bold" if is_best else "normal")
                if is_second:
                    second_texts.append(t)

    # ----- underline second-best cells (drawn precisely under the text) -----
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = ax.transData.inverted()
    for t in second_texts:
        bb = t.get_window_extent(renderer=renderer)
        (x0p, y0p), (x1p, y1p) = inv.transform([(bb.x0, bb.y0), (bb.x1, bb.y1)])
        y_line = min(y0p, y1p) - 0.05
        ax.add_line(mlines.Line2D([x0p, x1p], [y_line, y_line],
                                  color="black", linewidth=1.1,
                                  solid_capstyle="butt"))

    fig.canvas.draw()
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {out_path}")


def main():
    p = argparse.ArgumentParser(description="Render per-metric result tables as PNG.")
    p.add_argument("--exclude", nargs="+", default=[],
                   help="Display dataset labels to drop, e.g. UADFV.")
    p.add_argument("--suffix", type=str, default="",
                   help="Suffix appended to output filenames, e.g. _noUADFV.")
    p.add_argument("--no-avg", action="store_true",
                   help="Omit the overall-average (Avg.) column.")
    args = p.parse_args()

    datasets = [d for d in DATASETS if d[0] not in args.exclude]
    for metric in METRICS:
        data = load_metric(metric, datasets)
        out_path = os.path.join(HERE, f"{metric}{args.suffix}.png")
        render_metric(metric, data, datasets, out_path, show_avg=not args.no_avg)


if __name__ == "__main__":
    main()

"""
Compare relu vs. temperature-scaled softplus  g(x) = softplus(beta*x)/beta
over the cosine-similarity domain x in [-1, 1].

    softplus(z) = log(1 + exp(z))
    g(x)        = log(1 + exp(beta*x)) / beta
    g'(x)       = sigmoid(beta*x)
    g''(x)      = beta * sigmoid(beta*x) * (1 - sigmoid(beta*x))

As beta -> infinity, g(x) -> relu(x) = max(0, x),
so beta controls how "sharp" the smooth approximation is.
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def relu(x):
    return np.maximum(0.0, x)


def g(x, beta):
    return np.log1p(np.exp(beta * x)) / beta


def dg(x, beta):
    s = 1.0 / (1.0 + np.exp(-beta * x))  # sigmoid(beta x)
    return s


def d2g(x, beta):
    s = 1.0 / (1.0 + np.exp(-beta * x))
    return beta * s * (1.0 - s)


x = np.linspace(-1.0, 1.0, 2001)
dx = x[1] - x[0]
betas = [1, 3, 10]
labels = ["relu"] + [f"softplus({b}x)/{b}" for b in betas]
colors = ["tab:red", "tab:blue", "tab:green", "tab:purple"]

# --- quantitative comparison ------------------------------------------------
print("=" * 88)
print("relu vs temperature-scaled softplus over cos_sim in [-1, 1]")
print("=" * 88)
print(f"{'metric':38s}" + "".join(f"{l:>16s}" for l in labels))
print("-" * 88)


def fmt(v, digits=6):
    return f"{v:>16.{digits}f}"


# f at key points
for xx, name in [(0.0, "f(0)"), (-1.0, "f(-1)"), (1.0, "f(+1)")]:
    row = [relu(xx)] + [g(xx, b) for b in betas]
    print(f"{name:38s}" + "".join(fmt(v) for v in row))

# derivative jump at x=0 (finite difference across the transition)
i0 = int(np.searchsorted(x, 0.0))
jumps = [np.nan] + [abs(dg(x, b)[i0 + 1] - dg(x, b)[i0 - 1]) for b in betas]
print(f"{'derivative jump at x=0':38s}" + "".join(
    ">16s" if False else f"{'1.000000' if l == 'relu' else fmt(j):>16s}"
    for j, l in zip(jumps, labels)
))

# max curvature (relu = unbounded delta; softplus = beta/4 at x=0)
curv = [np.inf] + [b / 4.0 for b in betas]
print(f"{'max |f''| curvature':38s}" + "".join(
    '>16s' if False else (f"{'inf':>16}" if not np.isfinite(c) else fmt(c))
    for c in curv
))

# min gradient on the negative half (dead-zone check)
neg = x <= 0
row = [0.0] + [np.min(dg(x, b)[neg]) for b in betas]
print(f"{'min gradient on x<=0':38s}" + "".join(fmt(v) for v in row))

# max |g - relu| over the whole domain (how close the smooth gate is to relu)
approx_err = [0.0] + [np.max(np.abs(g(x, b) - relu(x))) for b in betas]
print(f"{'max |g - relu| on [-1,1]':38s}" + "".join(fmt(v) for v in approx_err))

print()
print("Interpretation:")
print("  * beta -> infinity recovers relu exactly (max |g-relu| shrinks).")
print("  * Higher beta lowers the constant penalty at negative cos_sim (g(-1) -> 0),")
print("    closer to the intent 'ignore zero/negative similarity'.")
print("  * All betas stay C^inf smooth: derivative sigmoid(beta x) is continuous and")
print("    gradient > 0 everywhere, so there is never a hard dead zone.")
print("  * Tradeoff: curvature grows as beta/4 and the transition band narrows to")
print("    ~[-1/beta, 1/beta], so large beta feels more like relu's sharp kink.")

# --- plots -------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)

axes[0].plot(x, relu(x), label="relu", lw=2, color=colors[0])
for b, c in zip(betas, colors[1:]):
    axes[0].plot(x, g(x, b), label=f"softplus({b}x)/{b}", lw=2, color=c)
axes[0].axvline(0, color="gray", ls=":", lw=1)
axes[0].set_title("loss gate f(x)")
axes[0].set_xlabel("cos_sim (x)")
axes[0].legend(fontsize=8)
axes[0].grid(alpha=0.3)

axes[1].plot(x, (x > 0).astype(float), label="relu f'", lw=2, color=colors[0])
for b, c in zip(betas, colors[1:]):
    axes[1].plot(x, dg(x, b), label=f"{b} f'", lw=2, color=c)
axes[1].axvline(0, color="gray", ls=":", lw=1)
axes[1].set_title("gradient f'(x)")
axes[1].set_xlabel("cos_sim (x)")
axes[1].legend(fontsize=8)
axes[1].grid(alpha=0.3)

for b, c in zip(betas, colors[1:]):
    axes[2].plot(x, d2g(x, b), label=f"{b} f''", lw=2, color=c)
axes[2].axvline(0, color="gray", ls=":", lw=1)
axes[2].set_title("curvature f''(x)  (peak = beta/4)")
axes[2].set_xlabel("cos_sim (x)")
axes[2].legend(fontsize=8)
axes[2].grid(alpha=0.3)

out_path = "eval/reformulate-cf-loss/softplus_scaled_vs_relu.png"
fig.savefig(out_path, dpi=150)
print(f"\nPlot saved to: {out_path}")

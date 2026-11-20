"""make_fig_index.py -- Fig. S1: reference-leakage index against the Monte Carlo null AUC (sim_index_grid.py)."""
import os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = os.path.join("results", "index_grid")
A = pd.read_csv(os.path.join(D, "grid_A.csv")); B = pd.read_csv(os.path.join(D, "grid_B.csv"))
C = ["#2a78d6", "#eb6834", "#1baf7a"]          # categorical slots 1-3 (all-pairs safe)
MK = ["o", "s", "^"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"

plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.linewidth": 0.6})
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), constrained_layout=True)

def panel(ax, groups, title):
    ax.plot([0.35, 1.0], [0.35, 1.0], color=MUTED, lw=0.8, ls="--", zorder=1)
    for (lab, df), c, mk in zip(groups, C, MK):
        ax.scatter(df.pred, df.obs, s=12, marker=mk, facecolor=c, edgecolor="white", linewidth=0.4,
                   alpha=0.9, label=lab, zorder=2)
    ax.set_xlim(0.35, 1.0); ax.set_ylim(0.35, 1.0); ax.set_aspect("equal")
    ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
    ax.set_xlabel("Index $F_{d,d}(\\rho)$"); ax.set_title(title, fontsize=8, color=INK, loc="left")
    for s in ["top", "right"]: ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=7, loc="upper left", handletextpad=0.2)

am = A[A.reference == "mean"]
panel(axes[0], [(f"{d}", am[am.design == k]) for d, k in
                [("Leave-self-out", "LSO"), ("Unit-wise", "UW"), ("Size-matched", "SM")]],
      "(a) Mean reference, independent")
axes[0].set_ylabel("Monte Carlo null AUC")
panel(axes[1], [(f"r = {r}", B[B.r == r]) for r in [0.0, 0.5, 0.9]], "(b) Correlated features")
med = A[(A.reference == "median") & (A.design == "LSO")]
panel(axes[2], [("$n_h=1$", med[med.n_h == 1]), ("$n_h=2$", med[med.n_h == 2]), ("$n_h\\geq3$", med[med.n_h >= 3])],
      "(c) Median reference, leave-self-out")
out = os.path.join(D, "FigS1_Index_Validation.png")
fig.savefig(out, dpi=300); print(out)

"""make_fig_s2.py -- Fig. S2: (a) index vs classifier null BA (simulation + real data); (b-d) semi-synthetic optimism."""
import os, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C = pd.read_csv("results/index_grid/calibration.csv")
S = pd.read_csv("results/noneeg/semisynthetic_summary.csv")
COL = {"LSO": "#2a78d6", "UW": "#eb6834", "SM": "#1baf7a"}
MK = {"LSO": "o", "UW": "s", "SM": "^"}
NAME = {"LSO": "Leave-self-out", "UW": "Unit-wise", "SM": "Size-matched"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.linewidth": 0.6})
fig, ax = plt.subplots(1, 4, figsize=(7.4, 2.35), constrained_layout=True, gridspec_kw={"width_ratios": [1.25, 1, 1, 1]})

a = ax[0]
a.plot([0.5, 1], [0.5, 1], ls="--", lw=0.8, color=MUTED, zorder=1)
for d in ["LSO", "UW", "SM"]:
    x = C[C.design == d]
    a.scatter(0.5 + (x["index"] - 0.5).abs(), x.null_BA, s=10, marker=MK[d], facecolor=COL[d], edgecolor="white",
              linewidth=0.3, alpha=0.85, label=NAME[d], zorder=2)
real = [  # (label, index, null BA logistic, null BA boosting)
    ("H&N", 0.744, 0.733, 0.743), ("Wearable", 0.603, 0.552, 0.566), (None, 0.599, 0.550, None),
    (None, 0.592, 0.543, None), (None, 0.483, 0.505, 0.518), (None, 0.500, 0.501, 0.518)]
for lab, ix, b1, b2 in real:
    xx = 0.5 + abs(ix - 0.5)
    for b in [b1, b2]:
        if b is not None:
            a.scatter(xx, b, s=46, marker="*", facecolor=INK, edgecolor="white", linewidth=0.4, zorder=3)
    if lab:
        a.annotate(lab, (xx, max(v for v in [b1, b2] if v)), xytext=(4, 4), textcoords="offset points", fontsize=7, color=INK)
a.scatter([], [], s=46, marker="*", facecolor=INK, label="Real data")
a.set_xlim(0.48, 1.0); a.set_ylim(0.45, 1.0)
a.set_xlabel(r"$0.5+|\mathrm{AUC}_0-0.5|$"); a.set_ylabel("Classifier null BA")
a.set_title("(a) Index vs null accuracy", loc="left", fontsize=8, color=INK)
a.legend(frameon=False, fontsize=6.5, loc="upper left", handletextpad=0.2)

for k, nh in enumerate([1, 2, 3]):
    b = ax[k + 1]
    b.axhline(0, color=MUTED, lw=0.8, ls="--")
    for d in ["LSO", "UW", "SM"]:
        x = S[(S.n_h == nh) & (S.design == d)].sort_values("delta")
        if len(x):
            b.errorbar(x.delta, x.optimism, yerr=1.96 * x.se, color=COL[d], marker=MK[d], ms=4, lw=1.5,
                       capsize=2, label=NAME[d])
    b.set_ylim(-0.1, 0.42); b.set_xticks([0, 0.5, 1, 2])
    b.set_xlabel(r"Injected effect $\Delta$ ($\sigma_b$)")
    b.set_title(f"({'bcd'[k]}) $n_h={nh}$", loc="left", fontsize=8, color=INK)
    if k == 0:
        b.set_ylabel("Optimism (estimate $-$ truth)")
    if k == 1:
        b.legend(frameon=False, fontsize=6.5, loc="upper right", handletextpad=0.2)
for x in ax:
    x.grid(color=GRID, lw=0.5); x.set_axisbelow(True)
    for s in ["top", "right"]:
        x.spines[s].set_visible(False)
fig.savefig("results/noneeg/FigS2_Index_Calibration_Semisynthetic.png", dpi=300)
print("saved")

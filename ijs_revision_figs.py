"""Figures 6 and 7 of the revised manuscript: permutation null and simulation study."""
import json
import os

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

import ijs_figures  # noqa: F401  (shared rcParams)
from ijs_figures import C_NAIVE, C_SA

REV = "outputs_ijs/revision"
FIG = "outputs_ijs/figs"
os.makedirs(FIG, exist_ok=True)
C_MATCH = "#3E7C59"


def permutation_figure():
    df = pd.read_csv(f"{REV}/permutation_null.csv")
    res = json.load(open(f"{REV}/permutation_results.json"))
    null = df[df.perm > 0]
    panels = [("cole|{}|abs", "(a) Cole, absolute"),
              ("cole|{}|fusion", "(b) Cole, equal fusion"),
              ("published|{}|fusion", "(c) Published set, fusion")]
    fig, axes = plt.subplots(1, 4, figsize=(11.0, 3.5), sharex=True)
    bins = np.linspace(0, 1, 26)
    for ax, (key, title) in zip(axes[:3], panels):
        for dn, col in (("naive", C_NAIVE), ("sa", C_SA)):
            k = key.format(dn)
            ax.hist(null[k], bins=bins, color=col, alpha=0.65)
            ax.axvline(res[k]["observed"], color=col, lw=2.6, ls="-")
        ax.set_title(title, fontsize=13.5)
    path = f"{REV}/selection_permutation_null.csv"
    ax = axes[3]
    if os.path.exists(path):
        sp = pd.read_csv(path)
        sn, so = sp[sp.perm > 0], sp[sp.perm == 0].iloc[0]
        for c, col in (("BA_prespecified", "#4F6D8F"), ("BA_selected", C_SA)):
            ax.hist(sn[c], bins=bins, color=col, alpha=0.65)
            ax.axvline(so[c], color=col, lw=2.6)
    ax.set_title("(d) Selection-aware", fontsize=13.5)
    for ax in axes:
        ax.axvline(0.5, color="0.2", lw=1.2, ls="--")
        ax.set_xlim(0, 1.02); ax.set_xticks([0, 0.5, 1]); ax.tick_params(labelsize=12)
        ax.set_xlabel("Balanced accuracy", fontsize=12.5)
    axes[0].set_ylabel("Permutations", fontsize=13)
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    h = [Patch(color=C_NAIVE, alpha=.65), Patch(color=C_SA, alpha=.65), Patch(color="#4F6D8F", alpha=.65),
         Line2D([], [], color="0.3", lw=2.6), Line2D([], [], color="0.2", lw=1.2, ls="--")]
    fig.legend(h, ["null, spectrum-wise (a-c)", "null, sample-wise (a-c); fully nested (d)", "null, pre-specified (d)",
                   "observed", "chance"], loc="upper center", ncol=5, fontsize=11.5, bbox_to_anchor=(0.5, 1.10),
               handlelength=1.4, columnspacing=1.0)
    fig.tight_layout()
    fig.savefig(f"{FIG}/Fig4_Permutation_Null.png", bbox_inches="tight")
    plt.close(fig)


def simulation_figure():
    g = pd.read_csv(f"{REV}/sim_grid.csv")
    pr = pd.read_csv(f"{REV}/sim_precision.csv")
    eff = g.delta.max()
    fig = plt.figure(figsize=(9.6, 6.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.1], hspace=0.55, wspace=0.42)
    norm = TwoSlopeNorm(vmin=-0.15, vcenter=0, vmax=0.45)
    iccs = sorted(g.icc.unique())
    ms = sorted(g.m.unique())
    for k, (dn, title) in enumerate((("naive", "(a) Spectrum-wise"), ("sa", "(b) Sample-wise"),
                                     ("matched", "(c) Size-matched"))):
        ax = fig.add_subplot(gs[0, k])
        d = g[g.delta == eff].assign(opt=lambda t: t[f"{dn}_est"] - t[f"{dn}_true"])
        mat = d.groupby(["m", "icc"]).opt.mean().unstack().loc[ms, iccs].values
        im = ax.imshow(mat, cmap="RdBu_r", norm=norm, aspect="auto", origin="lower")
        ax.set_xticks(range(len(iccs)), [f"{v:g}" for v in iccs], fontsize=11)
        ax.set_yticks(range(len(ms)), [str(v) for v in ms], fontsize=12.5)
        ax.set_xlabel("ICC", fontsize=13)
        if k == 0:
            ax.set_ylabel("Spectra per sample, m", fontsize=13.5)
        ax.set_title(title, fontsize=14)
        ax.grid(False)
    cax = fig.add_axes([0.92, 0.57, 0.012, 0.3])
    cb = fig.colorbar(im, cax=cax); cb.set_label("optimism (estimate - truth)", fontsize=13); cb.ax.tick_params(labelsize=12)
    # (d) null distribution of the estimate
    ax = fig.add_subplot(gs[1, 0])
    d0 = g[(g.delta == 0) & (g.m == 15)]
    for dn, col, lab in (("naive", C_NAIVE, "spectrum-wise"), ("sa", C_SA, "sample-wise"),
                         ("matched", C_MATCH, "size-matched")):
        s = d0.groupby("icc")[f"{dn}_est"].mean()
        ax.plot(s.index, s.values, "o-", color=col, label=lab, ms=6)
    s = d0.groupby("icc")["sa_true"].mean()
    ax.plot(s.index, s.values, "k--", lw=1, label="truth")
    ax.set_ylim(0.3, 1.02)
    ax.set_xlabel("ICC")
    ax.set_ylabel("Estimated BA", fontsize=13)
    ax.set_title("(d) No class difference", fontsize=14)
    ax.set_ylim(0.3, 1.05)
    ax.legend(fontsize=12, loc="upper left")
    # (e) precision vs number of patients
    ax = fig.add_subplot(gs[1, 1:])
    for nh, mk in ((2, "s"), (3, "o")):
        for dn, col, off in (("sa", C_SA, -0.04), ("matched", C_MATCH, 0.04)):
            d = pr[pr.n_h == nh]
            q = d.groupby("n_pat")[f"{dn}_est"].quantile([.025, .5, .975]).unstack()
            x = np.log2(q.index.values) + off + (0.12 if nh == 3 else -0.12)
            ax.errorbar(x, q[0.5], yerr=[q[0.5] - q[0.025], q[0.975] - q[0.5]], fmt=mk, color=col, ms=6,
                        capsize=2.5, lw=1.2, label=f"{'sample-wise' if dn == 'sa' else 'size-matched'}, {nh} healthy samples")
    t = pr.groupby("n_pat")[["sa_true", "matched_true"]].mean().mean(axis=1)
    ax.plot(np.log2(t.index.values), t.values, "k--", lw=1, label="mean true BA")
    ax.set_xticks(np.log2(sorted(pr.n_pat.unique())), [str(v) for v in sorted(pr.n_pat.unique())])
    ax.set_xlabel("Number of patients")
    ax.set_ylabel("Estimated BA", fontsize=13)
    ax.set_title("(e) Precision (ICC 0.95, m = 15)", fontsize=14)
    ax.set_ylim(0.5, 1.0)
    ax.legend(fontsize=11.5, ncol=2, loc="upper center", bbox_to_anchor=(0.45, -0.17), columnspacing=1.0, handletextpad=0.4)
    fig.savefig(f"{FIG}/Fig5_Simulation.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    if os.path.exists(f"{REV}/permutation_results.json"):
        permutation_figure()
    if os.path.exists(f"{REV}/sim_precision.csv"):
        simulation_figure()

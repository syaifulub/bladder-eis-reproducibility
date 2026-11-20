"""Figures for the revised manuscript (Iranian Journal of Science)."""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle

matplotlib.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 12.5, "axes.titlesize": 13.5, "axes.labelsize": 12.5,
    "xtick.labelsize": 11.5, "ytick.labelsize": 11.5, "legend.fontsize": 11,
    "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": ":", "legend.frameon": False,
    "figure.dpi": 110, "savefig.dpi": 400,
})
C_H, C_T = "#2E7DD1", "#D1442E"
C_NAIVE, C_SA = "#8A94A6", "#E07A2E"
C_ABS, C_MAH, C_FUS = "#4C8BE2", "#9B7BD4", "#3E7C59"


# ------------------------------------------------------------------- Fig. 1 schematic
def schematic():
    """Fig. 1: spectrum-wise, sample-wise, and size-matched sample-wise referencing (one patient, two healthy samples)."""
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 2.45))
    REF, REF2 = "#F2C14E", "#9AD1A8"

    def sample_box(ax, x, col, label, query=None, ref=lambda k: False, ref_col=REF, faded=False):
        w = h = 0.9
        y = 0.55
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                    fc="white", ec=col, lw=1.6, alpha=0.5 if faded else 1))
        ax.text(x + w / 2, y + h + 0.07, label, ha="center", va="bottom", fontsize=12.5, color=col)
        for k in range(9):
            cx = x + 0.15 + (k % 3) * (w - 0.30) / 2
            cy = y + 0.15 + (k // 3) * (h - 0.30) / 2
            fc, ec = (ref_col if ref(k) else "white"), col
            if query is not None and k == query:
                fc, ec = "black", "black"
            ax.add_patch(Circle((cx, cy), 0.06, fc=fc, ec=ec, lw=1.0, alpha=0.5 if faded else 1))

    specs = [("(a) Spectrum-wise (naive)", "naive",
              "Healthy query vs. all other healthy\nspectra, incl. repeats of its own sample"),
             ("(b) Sample-wise (corrected)", "sample",
              "Healthy query vs. healthy spectra\nof the other samples only"),
             ("(c) Size-matched sample-wise", "matched",
              "Tumor query vs. one healthy sample\nper version (H1 or H2), averaged")]
    for ax, (title, mode, txt) in zip(axes, specs):
        ax.set_xlim(0, 3.5); ax.set_ylim(-0.42, 1.9); ax.axis("off"); ax.set_aspect("equal")
        ax.set_title(title, fontsize=13.5, pad=1)
        if mode == "naive":
            sample_box(ax, 0.1, C_H, "H1", query=4, ref=lambda k: k != 4)
            sample_box(ax, 1.3, C_H, "H2", ref=lambda k: True)
            sample_box(ax, 2.5, C_T, "T1", faded=True)
        elif mode == "sample":
            sample_box(ax, 0.1, C_H, "H1", query=4)
            sample_box(ax, 1.3, C_H, "H2", ref=lambda k: True)
            sample_box(ax, 2.5, C_T, "T1", faded=True)
        else:
            sample_box(ax, 0.1, C_H, "H1", ref=lambda k: True, ref_col=REF2)
            sample_box(ax, 1.3, C_H, "H2", ref=lambda k: True)
            sample_box(ax, 2.5, C_T, "T1", query=4)
        ax.text(1.75, 0.18, txt, ha="center", va="top", fontsize=11.5)
    h = [plt.Line2D([], [], marker="o", ls="", ms=8, mfc="black", mec="black"),
         plt.Line2D([], [], marker="o", ls="", ms=8, mfc=REF, mec="0.3"),
         plt.Line2D([], [], marker="o", ls="", ms=8, mfc=REF2, mec="0.3")]
    fig.legend(h, ["query spectrum", "reference spectrum", "reference of the second version in (c)"],
               loc="lower center", ncol=3, fontsize=11.5, bbox_to_anchor=(0.5, -0.03), handletextpad=0.3)
    fig.tight_layout(rect=(0, 0.09, 1, 1), w_pad=0.4)
    return fig


# --------------------------------------------------- Fig. 2 repeated-spectra structure
def sample_structure(specs_p1, dist_df):
    """specs_p1: list of (label, sample, f, z) for one patient; dist_df: kind, d."""
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.7))
    fig.subplots_adjust(wspace=0.28)
    ax = axes[0]
    hs = sorted({s for l, s, *_ in specs_p1 if l == 0}); ts = sorted({s for l, s, *_ in specs_p1 if l == 1})
    cmh = plt.cm.Blues(np.linspace(0.45, 0.95, len(hs))); cmt = plt.cm.Reds(np.linspace(0.45, 0.95, len(ts)))
    col = {**{s: cmh[i] for i, s in enumerate(hs)}, **{s: cmt[i] for i, s in enumerate(ts)}}
    for lab, s, f, z in specs_p1:
        m = (f >= 1e3) & (f <= 1e6)
        ax.loglog(f[m] / 1e3, np.abs(z[m]), "-", lw=0.6, alpha=0.55, color=col[s])
    ax.set_xlabel("Frequency [kHz]", fontsize=13); ax.set_ylabel(r"$|Z|$  [$\Omega$]", fontsize=13); ax.tick_params(labelsize=12)
    ax.set_title("(a) P1: spectra by tissue sample", fontsize=13.5)
    ax.plot([], [], color=cmh[-1], label=f"healthy ({len(hs)} samples)")
    ax.plot([], [], color=cmt[-1], label=f"tumor ({len(ts)} samples)")
    ax.set_yticks([2, 3, 5, 10, 20, 30]); ax.set_yticklabels(["2", "3", "5", "10", "20", "30"])
    ax.minorticks_off()
    ax.legend(fontsize=11.5, loc="upper right")
    ax = axes[1]
    kinds = (("within", "own sample", C_H), ("between", "other healthy sample(s)", "#7FA7D9"), ("tumor", "tumor", C_T))
    pats = [p for p in ("P1", "P4", "P5") if (dist_df.Patient == p).any()]
    pos, data, cols = [], [], []
    for i, p in enumerate(pats):
        for k, (kind, _, c) in enumerate(kinds):
            d = dist_df[(dist_df.Patient == p) & (dist_df.kind == kind)].d.values
            pos.append(i * 4 + k); data.append(d); cols.append(c)
    bp = ax.boxplot(data, positions=pos, widths=0.7, patch_artist=True, showfliers=False,
                    medianprops=dict(color="black", lw=1.2))
    for patch, c in zip(bp["boxes"], cols):
        patch.set_facecolor(c); patch.set_alpha(0.6)
    for x_, d, c in zip(pos, data, cols):
        jit = np.random.default_rng(int(x_)).uniform(-0.2, 0.2, d.size)
        ax.plot(x_ + jit, d, "o", ms=1.8, alpha=0.35, color=c)
    ax.set_yscale("log")
    ax.set_xticks([i * 4 + 1 for i in range(len(pats))])
    ax.set_xticklabels([f"{p}" for p in pats])
    ax.set_ylabel("Mahalanobis distance", fontsize=13); ax.tick_params(labelsize=12)
    for kind, lab, c in kinds:
        ax.plot([], [], "s", color=c, alpha=0.6, label=("healthy vs. " + lab) if kind != "tumor" else "tumor vs. healthy samples")
    ax.set_ylim(top=ax.get_ylim()[1] * 30)
    ax.legend(fontsize=11.5, loc="upper left", ncol=1)
    ax.set_title("(b) Distance to healthy reference", fontsize=13.5)
    fig.tight_layout()
    return fig


# --------------------------------------------------------- Fig. 3 factorial design
def factorial(vals, cis, counts):
    """vals/cis/counts keyed by (referencing, selection)."""
    fig, ax = plt.subplots(figsize=(5.8, 4.0))
    xs = {"leaky": 0, "nested": 1}
    for refk, lab, c, off in (("naive", "Spectrum-wise referencing (naive)", C_NAIVE, -0.04),
                              ("sa", "Sample-wise referencing (corrected)", C_SA, 0.04)):
        xx = [xs["leaky"] + off, xs["nested"] + off]
        yy = [vals[(refk, "leaky")], vals[(refk, "nested")]]
        lo = [vals[(refk, s)] - cis[(refk, s)][0] for s in ("leaky", "nested")]
        hi = [cis[(refk, s)][1] - vals[(refk, s)] for s in ("leaky", "nested")]
        ax.errorbar(xx, yy, yerr=[lo, hi], fmt="-o", color=c, ms=7, lw=1.8, capsize=4, label=lab)
        for x_, s in zip(xx, ("leaky", "nested")):
            ax.annotate(f"{vals[(refk, s)]:.3f}  ({counts[(refk, s)]})", (x_, vals[(refk, s)]),
                        textcoords="offset points", xytext=(0, 9) if refk == "naive" else (0, -16),
                        ha="center", fontsize=9, color=c,
                        bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.5))
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Leaky selection\n(all six patients)", "Nested selection\n(training patients only)"])
    ax.set_xlim(-0.45, 1.45); ax.set_ylim(0.4, 1.08)
    ax.set_ylabel("Sample-level balanced accuracy")
    ax.axhline(0.5, color="0.5", lw=0.8, ls="--")
    ax.legend(fontsize=10, loc="lower left")
    fig.tight_layout()
    return fig


# ------------------------------------------------- Fig. 4 representation reversal
def representations(vals, cis):
    """vals[(features, referencing, branch)]"""
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.8), sharey=True)
    branches = (("abs", "Absolute", C_ABS), ("mah", "Mahalanobis", C_MAH), ("fusion", "Equal fusion", C_FUS))
    for ax, (feat, title) in zip(axes, (("freq", "(a) Nested frequency selection"), ("cole", "(b) Cole parameters"))):
        for k, (b, lab, c) in enumerate(branches):
            for j, (refk, mk) in enumerate((("naive", "o"), ("sa", "s"))):
                x = j + (k - 1) * 0.18
                v = vals[(feat, refk, b)]
                lo, hi = cis[(feat, refk, b)]
                ax.errorbar([x], [v], yerr=[[v - lo], [hi - v]], fmt=mk, color=c, ms=7, capsize=3,
                            label=lab if j == 0 else None)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Spectrum-wise\n(naive)", "Sample-wise\n(corrected)"])
        ax.set_xlim(-0.5, 1.5); ax.set_ylim(0, 1.05)
        ax.axhline(0.5, color="0.5", lw=0.8, ls="--")
        ax.set_title(title)
    axes[0].set_ylabel("Sample-level balanced accuracy")
    axes[0].legend(fontsize=10, loc="lower left")
    fig.tight_layout()
    return fig


# --------------------------------------------------------- Fig. 5 selected frequencies
def frequencies(rows):
    """rows: list of (label, list_of_kHz, colour, group)."""
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    yt, yl = [], []
    yv = 0
    last = None
    for lab, fk, c, grp in rows:
        if last is not None and grp != last:
            yv += 0.6
        ax.plot(fk, [yv] * len(fk), "o", color=c, ms=6)
        yt.append(yv); yl.append(lab)
        yv += 1
        last = grp
    ax.set_xscale("log"); ax.set_xlim(0.8, 1300)
    ax.set_yticks(yt); ax.set_yticklabels(yl, fontsize=10.5); ax.invert_yaxis()
    ax.set_xlabel("Frequency [kHz]")
    ax.axvspan(10, 34, color="0.9", lw=0, zorder=0)
    ax.axvspan(100, 1000, color="#FBE3CF", lw=0, zorder=0)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------- Fig. S1 drift
def drift(fits):
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    for s_, g in fits.groupby("Sample", sort=False):
        if len(g) < 6:
            continue
        x = np.linspace(0, 1, len(g))
        ax.plot(x, np.log(g.R0.values) - np.log(g.R0.values[0]), lw=0.8, alpha=0.7,
                color=C_H if g.Label.iloc[0] == 0 else C_T)
    ax.plot([], [], color=C_H, label="healthy samples"); ax.plot([], [], color=C_T, label="tumor samples")
    ax.set_xlabel("Normalized acquisition order within tissue sample")
    ax.set_ylabel(r"$\ln R_0 - \ln R_0^{\mathrm{first}}$")
    ax.legend(fontsize=10)
    fig.tight_layout()
    return fig


def frequencies_icc(rows, f_khz, icc):
    """(a) between-sample ICC of ln|Z| among healthy spectra; (b) selected frequency sets (side by side)."""
    fig, (ax0, ax) = plt.subplots(1, 2, figsize=(9.6, 4.3), gridspec_kw=dict(width_ratios=[1, 1.45], wspace=0.08))
    ax0.semilogx(f_khz, icc, "-", color="0.2", lw=2.2)
    ax0.set_ylabel("ICC of ln|Z| (healthy spectra)", fontsize=13)
    ax0.set_ylim(0.5, 1.0)
    ax0.set_title("(a) Between-sample share of variance", fontsize=13.5)
    for a in (ax0, ax):
        a.axvspan(10, 34, color="0.88", lw=0, zorder=0)
        a.axvspan(100, 1000, color="#FBE3CF", lw=0, zorder=0)
        a.set_xscale("log"); a.set_xlim(0.8, 1300)
        a.set_xlabel("Frequency [kHz]", fontsize=13)
        a.tick_params(labelsize=12)
    yt, yl = [], []
    yv, last = 0, None
    for lab, fk, c, grp in rows:
        if last is not None and grp != last:
            yv += 0.6
        ax.plot(fk, [yv] * len(fk), "o", color=c, ms=7)
        yt.append(yv); yl.append(lab)
        yv += 1
        last = grp
    ax.set_yticks(yt); ax.set_yticklabels(yl, fontsize=12); ax.invert_yaxis()
    ax.yaxis.tick_right()
    ax.set_title("(b) Frequencies retained", fontsize=13.5)
    return fig

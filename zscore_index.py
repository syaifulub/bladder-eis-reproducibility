"""
zscore_index.py -- validation of the reference-leakage index for subject-wise z-scoring (Remark 1 of the article).

When a deviation from a reference mean is divided by the standard deviation of the same reference, the
null variances of Eq. (2) are divided by the expected reference variance of each class,
    E s^2(N, k) = [(N - 1) sigma_w^2 + m (k - 1) sigma_b^2] / N     (N windows from k units, ddof = 0),
so rho_z = (V_t / E s_t^2) / (V_r / E s_r^2). Designs: INC (query in its own reference), LSO, UW, SM.
Part 1: Monte Carlo grid (ICC x m x n_h x d x design; 20,000 subjects per condition).
Part 2: wearable stress data, null AUC of sum_f z_f^2 / V'_{r,f} over 200 within-subject permutations
        for the z-score practices of noneeg_practices.py, against the index.
Outputs: results/index_grid/zscore_grid.csv, results/noneeg/zscore_index_real.csv
usage: python zscore_index.py [grid|real|both]
"""
import os, sys, itertools
import numpy as np, pandas as pd
from scipy import stats

S = 20000


from leakage_index import z_variances


def index(vr, vt, d):
    rho = float(np.exp(np.mean(np.log(vt / vr))))
    return rho, float(stats.f.cdf(rho, d, d))


def grid():
    rng = np.random.default_rng(20261120); rows = []
    for icc, m, nh, d, design in itertools.product([0.3, 0.6, 0.9], [5, 15], [2, 4], [1, 4], ["INC", "LSO", "UW", "SM"]):
        b = rng.standard_normal((S, nh + 1, d)) * np.sqrt(icc)
        e = rng.standard_normal((S, nh + 1, m, d)) * np.sqrt(1 - icc)
        x = b[:, :, None, :] + e
        H = x[:, :nh].reshape(S, nh * m, d); q, t = H[:, 0], x[:, nh, 0]
        refs = {"INC": (H, H), "LSO": (H[:, 1:], H), "UW": (H[:, m:], H), "SM": (H[:, m:], H[:, m:])}[design]
        zr = (q - refs[0].mean(1)) / refs[0].std(1); zt = (t - refs[1].mean(1)) / refs[1].std(1)
        vr, vt = z_variances(np.full(d, icc), np.full(d, 1 - icc), nh, m, design)
        sr, st = (zr ** 2 / vr).sum(1), (zt ** 2 / vr).sum(1)
        rho, pred = index(vr, vt, d)
        obs = stats.mannwhitneyu(st, sr).statistic / S ** 2
        rows.append(dict(ICC=icc, m=m, n_h=nh, d=d, design=design, rho=rho, index=pred, null_AUC=obs, diff=obs - pred))
    T = pd.DataFrame(rows); os.makedirs(os.path.join("results", "index_grid"), exist_ok=True)
    T.to_csv(os.path.join("results", "index_grid", "zscore_grid.csv"), index=False)
    print(T.groupby("design")["diff"].agg(["mean", lambda v: np.abs(v).max()]).round(4))
    print("overall max |diff|", round(T["diff"].abs().max(), 4))


def real(n_perm=200):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import noneeg_practices as npx
    W = pd.read_csv(os.path.join("results", "noneeg", "noneeg_windows.csv"))
    X = W[npx.FEATS].values; subj = W.subject.values; unit = W.unit.values; y = W.y.values
    Xs = X.copy()
    for s in np.unique(subj):
        i = subj == s; Xs[i] = (X[i] - X[i].mean(0)) / (X[i].std(0) + 1e-12)
    df = pd.DataFrame(Xs); f = list(df.columns); df["unit"] = unit; df["subject"] = subj
    um = df.groupby("unit")[f].transform("mean")
    sw2 = ((df[f] - um) ** 2).sum().values / (len(df) - df.unit.nunique())
    g = df.groupby(["subject", "unit"])[f].mean(); ns = df.groupby(["subject", "unit"]).size()
    sb2 = np.clip(np.mean([g.loc[s].var(ddof=1).values - sw2 * (1.0 / ns.loc[s]).mean() for s in np.unique(subj)], 0), 1e-6, None)
    nt = {s: len(np.unique(unit[(subj == s) & (y == 1)])) for s in np.unique(subj)}
    n_h = float(np.mean([len(np.unique(unit[subj == s])) - nt[s] for s in np.unique(subj)])); m = float(df.groupby("unit").size().mean())
    rng = np.random.default_rng(20261120); perms = []
    for b in range(n_perm):
        yp = np.zeros(len(y), int)
        for s in np.unique(subj):
            idx = np.flatnonzero(subj == s); us = rng.choice(np.unique(unit[idx]), nt[s], replace=False)
            yp[idx[np.isin(unit[idx], us)]] = 1
        perms.append(yp)
    rows = []
    for design, practice in [("INC", "BASE_INC"), ("LSO", "BASE_LSO"), ("UW", "BASE_UW"), ("SM", "BASE_SM")]:
        vr, vt = z_variances(sb2, sw2, n_h, m, design)
        Z0, src0 = npx.normalize(X, subj, unit, y, practice)
        C = np.nan_to_num(np.corrcoef(Z0[y[src0] == 0].T)); np.fill_diagonal(C, 1)
        lam = np.clip(np.linalg.eigvalsh(C), 0, None); deff = float(lam.sum() ** 2 / (lam ** 2).sum())
        rho = float(np.exp(np.mean(np.log(vt / vr)))); pred = float(stats.f.cdf(rho, deff, deff))
        aucs = []
        for yp in perms:
            Z, src = npx.normalize(X, subj, unit, yp, practice)
            sc = (Z ** 2 / vr).sum(1); t, r = sc[yp[src] == 1], sc[yp[src] == 0]
            aucs.append(stats.mannwhitneyu(t, r).statistic / (len(t) * len(r)))
        rows.append(dict(practice=practice, design=design, rho=rho, d_eff=deff, index=pred, null_AUC=np.mean(aucs),
                         null_lo=np.percentile(aucs, 2.5), null_hi=np.percentile(aucs, 97.5), B=n_perm))
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(os.path.join("results", "noneeg", "zscore_index_real.csv"), index=False)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "both"
    if what in ("grid", "both"): grid()
    if what in ("real", "both"): real()

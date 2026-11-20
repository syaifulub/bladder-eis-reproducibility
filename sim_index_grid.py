"""
sim_index_grid.py -- Monte Carlo validation of the reference-leakage index (Corollary 1).

For every condition, S independent subjects are simulated, each with n_h reference-class units and one
target-class unit (no class difference), m repeats per unit and d features. One reference-class query and
one target-class query are drawn per subject, referenced under the leave-self-out (LSO), unit-wise (UW)
or size-matched (SM) design, and scored by s = sum_f z_f^2 / V_{r,f}. The empirical null AUC P(s_t > s_r)
is compared with the index AUC0 = F_{d,d}(rho), where rho is the geometric mean over features of
V_t/V_r from Eq. (2) and d is the effective dimension of the correlation matrix of the reference-class
deviations (estimated from the simulated data, as it would be in practice).

Grid A: independent features with equal ICC (ICC x m x n_h x d x design), mean and median references.
Grid B: d = 4 correlated features (equicorrelation r) with equal or unequal ICCs.

Outputs: results/index_grid/grid_A.csv, grid_B.csv, summary.json
"""
import os, json, itertools, time
import numpy as np, pandas as pd
from scipy import stats

OUT = os.path.join("results", "index_grid"); os.makedirs(OUT, exist_ok=True)
S = 20000
SEED = 20261119


def variances(icc, m, nh, design):
    sb2, sw2 = icc, 1.0 - icc
    vt_all = sb2 * (1 + 1 / nh) + sw2 * (1 + 1 / (nh * m))
    if design == "LSO":
        n = nh * m - 1
        vr = sb2 * m * m * nh * (nh - 1) / n ** 2 + sw2 * (1 + 1 / n)
        vt = vt_all
    else:
        vr = sb2 * (1 + 1 / (nh - 1)) + sw2 * (1 + 1 / ((nh - 1) * m))
        vt = vt_all if design == "UW" else vr
    return vr, vt


def deff(D):
    lam = np.linalg.eigvalsh(np.corrcoef(D.T)) if D.shape[1] > 1 else np.array([1.0])
    return float(lam.sum() ** 2 / (lam ** 2).sum())


def auc(st, sr):
    return float(stats.mannwhitneyu(st, sr).statistic / (len(st) * len(sr)))


def run(icc_f, m, nh, d, r, design, rng, median=False):
    icc_f = np.asarray(icc_f, float)
    R = np.full((d, d), r) + (1 - r) * np.eye(d)
    L = np.linalg.cholesky(R)
    b = rng.standard_normal((S, nh + 1, d)) @ L.T * np.sqrt(icc_f)
    e = rng.standard_normal((S, nh + 1, m, d)) @ L.T * np.sqrt(1 - icc_f)
    x = b[:, :, None, :] + e
    H = x[:, :nh].reshape(S, nh * m, d)          # reference-class observations, query = H[:, 0]
    q, t = H[:, 0], x[:, nh, 0]
    stat = (lambda a: np.median(a, axis=1)) if median else (lambda a: a.mean(axis=1))
    if design == "LSO":
        ref_r, ref_t = H[:, 1:], H
    elif design == "UW":
        ref_r, ref_t = H[:, m:], H
    else:                                            # SM: target referenced against the same n_h-1 units
        ref_r, ref_t = H[:, m:], H[:, m:]
    zr, zt = q - stat(ref_r), t - stat(ref_t)
    V = np.array([variances(i, m, nh, design) for i in icc_f])
    vr, vt = V[:, 0], V[:, 1]
    sr, st = (zr ** 2 / vr).sum(1), (zt ** 2 / vr).sum(1)
    rho = float(np.exp(np.mean(np.log(vt / vr))))
    de = deff(zr)
    return dict(rho=rho, d_eff=de, pred=float(stats.f.cdf(rho, de, de)), obs=auc(st, sr))


def main():
    rng = np.random.default_rng(SEED)
    t0 = time.time()
    rows = []
    for icc, m, nh, d in itertools.product([0.1, 0.25, 0.5, 0.75, 0.9, 0.97], [2, 5, 15, 30], [1, 2, 3, 6], [1, 4]):
        for design in ["LSO", "UW", "SM"]:
            if design != "LSO" and nh < 2:
                continue
            for med in [False, True]:
                r = run([icc] * d, m, nh, d, 0.0, design, rng, median=med)
                rows.append(dict(grid="A", ICC=icc, m=m, n_h=nh, d=d, r=0.0, icc_pattern="equal",
                                 design=design, reference="median" if med else "mean", **r))
    A = pd.DataFrame(rows); A.to_csv(os.path.join(OUT, "grid_A.csv"), index=False)
    print("grid A", len(A), "conditions", round(time.time() - t0), "s")
    rows = []
    for icc, m, nh, r, pat in itertools.product([0.25, 0.5, 0.9, 0.97], [5, 15], [1, 2, 3], [0.0, 0.5, 0.9], ["equal", "unequal"]):
        icc_f = [icc] * 4 if pat == "equal" else [icc * 0.4, icc * 0.7, icc * 0.9, icc]
        for design in ["LSO", "UW"]:
            if design == "UW" and nh < 2:
                continue
            res = run(icc_f, m, nh, 4, r, design, rng)
            rows.append(dict(grid="B", ICC=icc, m=m, n_h=nh, d=4, r=r, icc_pattern=pat, design=design,
                             reference="mean", **res))
    B = pd.DataFrame(rows); B.to_csv(os.path.join(OUT, "grid_B.csv"), index=False)
    print("grid B", len(B), "conditions", round(time.time() - t0), "s")

    def summ(df):
        err = (df.pred - df.obs).abs()
        return dict(n=int(len(df)), mean_abs_err=float(err.mean()), p95_abs_err=float(err.quantile(0.95)),
                    max_abs_err=float(err.max()), corr=float(np.corrcoef(df.pred, df.obs)[0, 1]),
                    obs_range=[float(df.obs.min()), float(df.obs.max())])
    out = {
        "S_per_condition": S, "seed": SEED,
        "A_mean_all": summ(A[A.reference == "mean"]),
        "A_mean_by_design": {k: summ(g) for k, g in A[A.reference == "mean"].groupby("design")},
        "A_median_all": summ(A[A.reference == "median"]),
        "A_median_by_design": {k: summ(g) for k, g in A[A.reference == "median"].groupby("design")},
        "B_all": summ(B),
        "B_by_r": {str(k): summ(g) for k, g in B.groupby("r")},
        "B_by_pattern": {k: summ(g) for k, g in B.groupby("icc_pattern")},
        "B_LSO": summ(B[B.design == "LSO"]),
    }
    json.dump(out, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

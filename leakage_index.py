"""
leakage_index.py -- reference implementation of the reference-leakage index (Corollary 1 of the article).

Given repeated measurements X (rows = observations), the subject and unit of every row, and the class of every
unit (0 = reference class, 1 = target class), the index returns, for a referencing design, the null probability
AUC0 that a target-class query deviates more from its reference than a reference-class query when no class
difference exists. AUC0 = 0.5 means that the design hands no null discrimination to a classifier.

    from leakage_index import reference_leakage_index
    res = reference_leakage_index(X, subject, unit, y, design="LSO")   # "INC", "LSO", "UW" or "SM"
    print(res["AUC0"])
    res = reference_leakage_index(X, subject, unit, y, design="INC", zscore=True)   # baseline z-scoring

Designs: INC = query included in its own reference (e.g. z-scoring against all baseline windows); LSO = leave-self-out (only the query is removed from its reference), UW = unit-wise (the query's whole
unit is removed), SM = size-matched (UW for reference-class queries; target-class queries referenced against
n_h - 1 units, n_h versions averaged). Variance components are estimated within subjects from features
standardized within subject, treating all units of a subject as exchangeable (the permutation null). Correlated
features are handled by the effective dimension of the reference-class deviations; the index is exact for mean
references with independent features and conservative for correlated features (Note S1 of the article).
With zscore=True, deviations are divided by the standard deviation of the same reference (Remark 1 of the
article): the null variances are divided by the expected reference variance of each class; this first-order
index is exact for SM and a lower bound for INC and LSO when few reference units carry a high ICC.
"""
import numpy as np
import pandas as pd
from scipy import stats

__all__ = ["reference_leakage_index", "deviation_variances", "z_variances"]


def deviation_variances(sb2, sw2, n_h, m, design):
    """Per-feature null variances (V_r, V_t) of Eq. (2) for mean references."""
    sb2, sw2 = np.asarray(sb2, float), np.asarray(sw2, float)
    vt = sb2 * (1 + 1 / n_h) + sw2 * (1 + 1 / (n_h * m))
    if design in ("LSO", "INC"):
        n = n_h * m - 1
        vr = sb2 * m * m * n_h * (n_h - 1) / n ** 2 + sw2 * (1 + 1 / n)
        if design == "INC":                               # query included: deviation shrinks by (1 - 1/(n_h m))
            vr = vr * (1 - 1 / (n_h * m)) ** 2
    elif design in ("UW", "SM"):
        if n_h < 2:
            raise ValueError("unit-wise and size-matched referencing need at least two reference units")
        vr = sb2 * (1 + 1 / (n_h - 1)) + sw2 * (1 + 1 / ((n_h - 1) * m))
        if design == "SM":
            vt = vr.copy()
    else:
        raise ValueError("design must be 'INC', 'LSO', 'UW' or 'SM'")
    return vr, vt


def _es2(sb2, sw2, N, k, m):
    """expected variance (ddof = 0) of N reference observations from k units with m repeats each"""
    return ((N - 1) * sw2 + m * (k - 1) * sb2) / N


def z_variances(sb2, sw2, n_h, m, design):
    """null variances (V'_r, V'_t) of deviations divided by the standard deviation of the same reference"""
    vr, vt = deviation_variances(sb2, sw2, n_h, m, design)
    sb2, sw2 = np.asarray(sb2, float), np.asarray(sw2, float)
    n = n_h * m
    st = _es2(sb2, sw2, n, n_h, m)
    if design in ("LSO", "INC"):
        sr = st if design == "INC" else _es2(sb2, sw2, n - 1, n_h, m)
    else:
        sr = _es2(sb2, sw2, (n_h - 1) * m, n_h - 1, m)
        if design == "SM":
            st = sr
    return vr / sr, vt / st


def _effective_dimension(D):
    if D.shape[1] == 1:
        return 1.0
    C = np.nan_to_num(np.corrcoef(D.T)); np.fill_diagonal(C, 1.0)
    lam = np.clip(np.linalg.eigvalsh(C), 0, None)
    return float(lam.sum() ** 2 / (lam ** 2).sum())


def reference_leakage_index(X, subject, unit, y, design="LSO", zscore=False):
    X = np.asarray(X, float); subject = np.asarray(subject); unit = np.asarray(unit); y = np.asarray(y).astype(int)
    if X.ndim == 1:
        X = X[:, None]
    Xs = np.empty_like(X)
    for s in np.unique(subject):                                   # standardize within subject
        i = subject == s
        Xs[i] = (X[i] - X[i].mean(0)) / (X[i].std(0) + 1e-12)
    keep = Xs.std(0) > 1e-12
    Xs = Xs[:, keep]
    df = pd.DataFrame(Xs); f = list(df.columns); df["unit"] = unit; df["subject"] = subject
    um = df.groupby("unit")[f].transform("mean")
    sw2 = ((df[f] - um) ** 2).sum().values / (len(df) - df.unit.nunique())          # pooled within-unit
    g = df.groupby(["subject", "unit"])[f].mean(); ns = df.groupby(["subject", "unit"]).size()
    sb = [g.loc[s].var(ddof=1).values - sw2 * (1.0 / ns.loc[s]).mean()
          for s in np.unique(subject) if g.loc[s].shape[0] > 1]
    sb2 = np.clip(np.mean(sb, axis=0), 1e-6, None)                                 # between units, within subject
    units = pd.DataFrame({"subject": subject, "unit": unit, "y": y}).drop_duplicates("unit")
    n_h = float(units[units.y == 0].groupby("subject").size().mean())
    m = float(df.groupby("unit").size().mean())
    vr, vt = (z_variances if zscore else deviation_variances)(sb2, sw2, n_h, m, design)
    rho = float(np.exp(np.mean(np.log(vt / vr))))
    # effective dimension from leave-self-out mean deviations of reference-class observations
    D = []
    for s in np.unique(subject):
        idx = np.flatnonzero((subject == s) & (y == 0))
        if len(idx) < 2:
            continue
        tot = Xs[idx].sum(0)
        D.append(Xs[idx] - (tot - Xs[idx]) / (len(idx) - 1))
    d_eff = _effective_dimension(np.vstack(D))
    return {"design": design, "AUC0": float(stats.f.cdf(rho, d_eff, d_eff)), "rho": rho, "d_eff": d_eff,
            "n_h": n_h, "m": m, "ICC": (sb2 / (sb2 + sw2)).tolist()}


if __name__ == "__main__":   # small self-check on simulated data with ICC 0.9, n_h = 2, m = 15, d = 1
    rng = np.random.default_rng(0); rows = []
    for s in range(300):
        for u in range(3):
            b = rng.normal(0, np.sqrt(0.9))
            for j in range(15):
                rows.append((s, f"{s}_{u}", int(u == 2), b + rng.normal(0, np.sqrt(0.1))))
    a = np.array(rows, dtype=object)
    r = reference_leakage_index(a[:, 3].astype(float), a[:, 0], a[:, 1], a[:, 2].astype(int), "LSO")
    vr, vt = deviation_variances(0.9, 0.1, 2, 15, "LSO")
    print("estimated AUC0 %.3f, theoretical %.3f" % (r["AUC0"], stats.f.cdf(vt / vr, 1, 1)))

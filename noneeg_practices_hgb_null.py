"""
noneeg_practices_hgb_null.py -- permutation null of gradient boosting for the z-score practices (wearable data).

Same practices, validation, weights and within-subject permutations as Part A of noneeg_practices.py (same seed,
so the permutations coincide with the first B logistic-regression permutations); boosting = scikit-learn
HistGradientBoosting (180 iterations, depth 3, learning rate 0.04, minimum leaf 5, L2 2). Results are appended
after every permutation, so an interrupted run resumes where it stopped.
Outputs: results/noneeg/practices_hgb_null_raw.csv, practices_hgb_null.csv
usage: python noneeg_practices_hgb_null.py [B]
"""
import os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import noneeg_practices as npx

PRACT = ["BASE_INC", "BASE_LSO", "BASE_UW", "BASE_SM"]
OUT = os.path.join("results", "noneeg")


def main(B):
    W = pd.read_csv(os.path.join(OUT, "noneeg_windows.csv"))
    X = W[npx.FEATS].values; subj = W.subject.values; unit = W.unit.values; y = W.y.values
    rng = np.random.default_rng(20261121)                       # as in noneeg_practices.part_a
    nt = {s: len(np.unique(unit[(subj == s) & (y == 1)])) for s in np.unique(subj)}
    raw = os.path.join(OUT, "practices_hgb_null_raw.csv")
    done = set(pd.read_csv(raw).perm) if os.path.exists(raw) else set()
    t0 = time.time()
    for b in range(B):
        yp = np.zeros(len(y), int)
        for s in np.unique(subj):
            idx = np.flatnonzero(subj == s); us = rng.choice(np.unique(unit[idx]), nt[s], replace=False)
            yp[idx[np.isin(unit[idx], us)]] = 1
        if b in done:
            continue
        row = {"perm": b}
        for p in PRACT:
            row[p] = npx.loso(X, subj, unit, yp, p, "hgb")
        pd.DataFrame([row]).to_csv(raw, mode="a", header=not os.path.exists(raw), index=False)
        print(b, f"{time.time()-t0:.0f}s", flush=True)
    R = pd.read_csv(raw); obs = pd.read_csv(os.path.join(OUT, "practices_real.csv")).set_index("practice")
    rows = []
    for p in PRACT:
        v = R[p].values; o = obs.loc[p, "BA_hgb"]
        rows.append(dict(practice=p, BA_hgb=o, null_mean=v.mean(), null_lo=np.percentile(v, 2.5),
                         null_hi=np.percentile(v, 97.5), p=(1 + (v >= o).sum()) / (1 + len(v)), B=len(v)))
    T = pd.DataFrame(rows); T.to_csv(os.path.join(OUT, "practices_hgb_null.csv"), index=False); print(T.round(3).to_string(index=False))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 100)

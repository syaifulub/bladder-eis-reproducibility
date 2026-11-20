"""
sim_index_calibration.py -- does the reference-leakage index predict the null balanced accuracy of a trained classifier?

Simulated subjects with n_h reference-class units and one target-class unit (no class difference), m repeats and
d features. Referenced features are absolute deviations from the reference mean (leave-self-out, unit-wise or
size-matched design); L2 logistic regression is trained and evaluated with subject-grouped five-fold
cross-validation; the unit-level balanced accuracy (BA) under the null is compared with the index AUC0.
Outputs: results/index_grid/calibration.csv, calibration_summary.json
"""
import os, json, itertools, time
import numpy as np, pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold

OUT = os.path.join("results", "index_grid"); os.makedirs(OUT, exist_ok=True)
SEED, N_SUBJ, REPS = 20261120, 40, 20


def variances(icc, m, nh, design):
    sb2, sw2 = icc, 1 - icc
    vt = sb2 * (1 + 1 / nh) + sw2 * (1 + 1 / (nh * m))
    if design == "LSO":
        n = nh * m - 1; vr = sb2 * m * m * nh * (nh - 1) / n ** 2 + sw2 * (1 + 1 / n)
    else:
        vr = sb2 * (1 + 1 / (nh - 1)) + sw2 * (1 + 1 / ((nh - 1) * m))
        if design == "SM": vt = vr
    return vr, vt


def dataset(rng, icc, m, nh, d, design):
    Z, ylab, subj, unit = [], [], [], []
    for s in range(N_SUBJ):
        b = rng.normal(0, np.sqrt(icc), (nh + 1, d))
        x = b[:, None, :] + rng.normal(0, np.sqrt(1 - icc), (nh + 1, m, d))
        H = x[:nh]                                         # reference-class units
        for u in range(nh):
            for j in range(m):
                if design == "LSO":
                    ref = (H.reshape(-1, d).sum(0) - H[u, j]) / (nh * m - 1)
                else:
                    ref = np.delete(H, u, axis=0).reshape(-1, d).mean(0)
                Z.append(np.abs(H[u, j] - ref)); ylab.append(0); subj.append(s); unit.append(f"{s}_{u}")
        for j in range(m):
            q = x[nh, j]
            if design == "SM":
                for v in range(nh):
                    ref = np.delete(H, v, axis=0).reshape(-1, d).mean(0)
                    Z.append(np.abs(q - ref)); ylab.append(1); subj.append(s); unit.append(f"{s}_t")
            else:
                Z.append(np.abs(q - H.reshape(-1, d).mean(0))); ylab.append(1); subj.append(s); unit.append(f"{s}_t")
    return np.array(Z), np.array(ylab), np.array(subj), np.array(unit)


def null_ba(Z, y, subj, unit):
    prob = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(Z, y, subj):
        w = np.where(y[tr] == 1, 0.5 / y[tr].mean(), 0.5 / (1 - y[tr].mean()))
        sc = StandardScaler().fit(Z[tr]); m = LogisticRegression(C=1.0, max_iter=2000)
        m.fit(sc.transform(Z[tr]), y[tr], sample_weight=w); prob[te] = m.predict_proba(sc.transform(Z[te]))[:, 1]
    u = pd.DataFrame({"u": unit, "y": y, "p": prob}).groupby("u").agg(y=("y", "first"), p=("p", "mean"))
    yh = (u.p.values >= 0.5).astype(int); yy = u.y.values
    return 0.5 * ((yh[yy == 1] == 1).mean() + (yh[yy == 0] == 0).mean())


def main():
    rng = np.random.default_rng(SEED); rows = []; t0 = time.time()
    for icc, m, nh, d in itertools.product([0.25, 0.5, 0.75, 0.9, 0.97], [5, 15], [1, 2, 4], [1, 4]):
        for design in ["LSO", "UW", "SM"]:
            if design != "LSO" and nh < 2:
                continue
            vr, vt = variances(icc, m, nh, design)
            idx = float(stats.f.cdf(vt / vr, d, d))
            bas = [null_ba(*dataset(rng, icc, m, nh, d, design)) for _ in range(REPS)]
            rows.append(dict(ICC=icc, m=m, n_h=nh, d=d, design=design, index=idx,
                             null_BA=float(np.mean(bas)), null_BA_sd=float(np.std(bas))))
        print(icc, m, nh, d, f"{time.time()-t0:.0f}s", flush=True)
    C = pd.DataFrame(rows); C.to_csv(os.path.join(OUT, "calibration.csv"), index=False)
    x, y = C["index"].values - 0.5, C["null_BA"].values - 0.5
    slope = float((x * y).sum() / (x * x).sum())                       # line through (0.5, 0.5)
    resid = y - slope * x
    out = dict(n_conditions=int(len(C)), reps=REPS, n_subjects=N_SUBJ,
               spearman=float(stats.spearmanr(C["index"], C["null_BA"]).correlation),
               pearson=float(np.corrcoef(C["index"], C["null_BA"])[0, 1]),
               slope_through_half=slope, rmse_line=float(np.sqrt(np.mean(resid ** 2))),
               ba_le_index_share=float((C.null_BA <= C["index"] + 0.01).mean()))
    json.dump(out, open(os.path.join(OUT, "calibration_summary.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

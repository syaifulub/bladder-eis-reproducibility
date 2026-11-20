"""
sim_zscore_calibration.py -- does the index for z-scored references (Remark 1) predict the null balanced accuracy
of trained classifiers on signed z-scores?

Simulated subjects (40 per data set) with n_h reference-class units and one target-class unit without class
difference, m repeats, d features; every observation is z-scored against its reference (INC: all reference-class
observations including the query; LSO: all except the query; UW: the other reference-class units; SM: UW for
reference-class queries, target-class queries n_h times against n_h - 1 units). Signed z-scores are classified by
L2 logistic regression and by gradient boosting (HistGradientBoosting, 180 iterations, depth 3, learning rate 0.04,
minimum leaf 5, L2 2) with subject-grouped five-fold cross-validation; the unit-level null balanced accuracy is
compared with the index. Outputs: results/index_grid/zscore_calibration.csv, zscore_calibration_summary.json
usage: python sim_zscore_calibration.py [reps]
"""
import os, sys, json, itertools, time
import numpy as np, pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from leakage_index import z_variances

OUT = os.path.join("results", "index_grid"); N_SUBJ = 40; SEED = 20261120


def zs(q, R):
    return (q - R.mean(0)) / (R.std(0) + 1e-12)


def dataset(rng, icc, m, nh, d, design):
    Z, ylab, subj, unit = [], [], [], []
    for s in range(N_SUBJ):
        b = rng.normal(0, np.sqrt(icc), (nh + 1, d))
        x = b[:, None, :] + rng.normal(0, np.sqrt(1 - icc), (nh + 1, m, d))
        H = x[:nh]; Hf = H.reshape(-1, d)
        for u in range(nh):
            for j in range(m):
                if design == "INC":
                    R = Hf
                elif design == "LSO":
                    R = np.delete(Hf, u * m + j, axis=0)
                else:
                    R = np.delete(H, u, axis=0).reshape(-1, d)
                Z.append(zs(H[u, j], R)); ylab.append(0); subj.append(s); unit.append(f"{s}_{u}")
        for j in range(m):
            q = x[nh, j]
            if design == "SM":
                for v in range(nh):
                    Z.append(zs(q, np.delete(H, v, axis=0).reshape(-1, d))); ylab.append(1); subj.append(s); unit.append(f"{s}_t")
            else:
                Z.append(zs(q, Hf)); ylab.append(1); subj.append(s); unit.append(f"{s}_t")
    return np.array(Z), np.array(ylab), np.array(subj), np.array(unit)


def null_ba(Z, y, subj, unit, kind):
    prob = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(Z, y, subj):
        w = np.where(y[tr] == 1, 0.5 / y[tr].mean(), 0.5 / (1 - y[tr].mean()))
        if kind == "lr":
            sc = StandardScaler().fit(Z[tr]); m = LogisticRegression(C=1.0, max_iter=2000)
            m.fit(sc.transform(Z[tr]), y[tr], sample_weight=w); prob[te] = m.predict_proba(sc.transform(Z[te]))[:, 1]
        else:
            m = HistGradientBoostingClassifier(max_iter=180, max_depth=3, learning_rate=0.04, min_samples_leaf=5,
                                               l2_regularization=2.0, random_state=0)
            m.fit(Z[tr], y[tr], sample_weight=w); prob[te] = m.predict_proba(Z[te])[:, 1]
    u = pd.DataFrame({"u": unit, "y": y, "p": prob}).groupby("u").agg(y=("y", "first"), p=("p", "mean"))
    yh = (u.p.values >= 0.5).astype(int); yy = u.y.values
    return 0.5 * ((yh[yy == 1] == 1).mean() + (yh[yy == 0] == 0).mean())


def main(reps):
    rng = np.random.default_rng(SEED); rows = []; t0 = time.time()
    for icc, m, nh, d in itertools.product([0.25, 0.5, 0.75, 0.9], [5, 15], [2, 4], [1, 4]):
        for design in ["INC", "LSO", "UW", "SM"]:
            vr, vt = z_variances(np.full(d, icc), np.full(d, 1 - icc), nh, m, design)
            rho = float(np.exp(np.mean(np.log(vt / vr)))); idx = float(stats.f.cdf(rho, d, d))
            lr, hg = [], []
            for _ in range(reps):
                D = dataset(rng, icc, m, nh, d, design)
                lr.append(null_ba(*D, "lr")); hg.append(null_ba(*D, "hgb"))
            rows.append(dict(ICC=icc, m=m, n_h=nh, d=d, design=design, index=idx, pred_BA=0.5 + abs(idx - 0.5),
                             null_BA_lr=float(np.mean(lr)), null_BA_hgb=float(np.mean(hg))))
        print(icc, m, nh, d, f"{time.time()-t0:.0f}s", flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(OUT, "zscore_calibration.csv"), index=False)
    C = pd.DataFrame(rows)
    out = {"n_conditions": len(C), "reps": reps}
    for k in ["lr", "hgb"]:
        x = C.pred_BA.values - 0.5; yv = C[f"null_BA_{k}"].values - 0.5
        slope = float((x * yv).sum() / (x * x).sum())
        out[k] = dict(pearson=float(np.corrcoef(C.pred_BA, C[f"null_BA_{k}"])[0, 1]), slope_through_half=slope,
                      rmse_identity=float(np.sqrt(np.mean((yv - x) ** 2))), mean_diff=float(np.mean(yv - x)),
                      sm_range=[float(C[C.design == "SM"][f"null_BA_{k}"].min()), float(C[C.design == "SM"][f"null_BA_{k}"].max())])
    json.dump(out, open(os.path.join(OUT, "zscore_calibration_summary.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 10)

"""
noneeg_practices.py -- common per-subject normalization practices on the PhysioNet Non-EEG data.

Practices (features z-scored with a per-subject mean and standard deviation computed from):
  RAW       no normalization
  FULL      all windows of the subject (all stages, including the query)          -- common practice
  BASE_INC  all reference-class (relaxation) windows of the subject, query included -- common practice
  BASE_LSO  reference-class windows except the query                            (leave-self-out)
  BASE_UW   reference-class windows of the other reference units                 (unit-wise)
  BASE_SM   as UW for reference-class queries; target queries normalized n_h times against n_h - 1 units
Part A (real labels): leave-one-subject-out unit-level balanced accuracy, logistic regression and boosting, and a
  within-subject permutation null (logistic regression) that recomputes every label-dependent normalization.
Part B (semi-synthetic): relaxation stages as real units, known effect injected (see semisynthetic_noneeg.py);
  estimate vs deployment truth on fresh units normalized with the statistics of the observed reference units
  (for BASE_SM, as in training: n_h normalizations against n_h - 1 units, predictions averaged).
usage: python noneeg_practices.py A [n_perm]   |   python noneeg_practices.py B [reps]
"""
import os, sys, json, time
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import HistGradientBoostingClassifier

OUT = os.path.join("results", "noneeg")
FEATS = ["EDA_mean", "EDA_std", "EDA_slope", "TEMP_mean", "TEMP_slope", "ACC_std", "HR_mean", "HR_std", "SpO2_mean"]
PRACTICES = ["RAW", "FULL", "BASE_INC", "BASE_LSO", "BASE_UW", "BASE_SM"]
EPS = 1e-6


def z(x, R):
    return (x - R.mean(0)) / (R.std(0) + EPS)


def normalize(X, subj, unit, y, practice):
    """returns Z (rows), src (window index); BASE_SM gives n_h rows per target window"""
    if practice == "RAW":
        return X.copy(), np.arange(len(X))
    Z, src = [], []
    for s in np.unique(subj):
        idx = np.flatnonzero(subj == s); ref = idx[y[idx] == 0]; ru = np.unique(unit[ref])
        if practice == "FULL":
            st = (X[idx].mean(0), X[idx].std(0) + EPS)
            for i in idx: Z.append((X[i] - st[0]) / st[1]); src.append(i)
            continue
        excl = {u: (X[ref[unit[ref] != u]].mean(0), X[ref[unit[ref] != u]].std(0) + EPS) for u in ru}
        full = (X[ref].mean(0), X[ref].std(0) + EPS)
        for i in idx:
            if practice == "BASE_INC":
                st = full
            elif y[i] == 0 and practice == "BASE_LSO":
                r = ref[ref != i]; st = (X[r].mean(0), X[r].std(0) + EPS)
            elif y[i] == 0:
                st = excl[unit[i]]
            elif practice == "BASE_SM":
                for u in ru:
                    Z.append((X[i] - excl[u][0]) / excl[u][1]); src.append(i)
                continue
            else:
                st = full
            Z.append((X[i] - st[0]) / st[1]); src.append(i)
    return np.asarray(Z), np.asarray(src)


def model(kind):
    if kind == "lr":
        return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000))
    return HistGradientBoostingClassifier(max_iter=180, max_depth=3, learning_rate=0.04, min_samples_leaf=5,
                                          l2_regularization=2.0, random_state=0)


def weights(subj, y):
    w = np.zeros(len(y))
    for s in np.unique(subj):
        for c in (0, 1):
            m = (subj == s) & (y == c); k = m.sum()
            if k: w[m] = 1.0 / k
    return w * len(y) / w.sum()


def fit(m, kind, Z, y, w):
    if kind == "lr": m.fit(Z, y, logisticregression__sample_weight=w)
    else: m.fit(Z, y, sample_weight=w)


def unit_ba(units, y, p):
    d = pd.DataFrame({"u": units, "y": y, "p": p}).groupby("u").agg(y=("y", "first"), p=("p", "mean"))
    yh = (d.p.values >= 0.5).astype(int); yy = d.y.values
    return 0.5 * ((yh[yy == 1] == 1).mean() + (yh[yy == 0] == 0).mean())


def loso(X, subj, unit, y, practice, kind):
    Z, src = normalize(X, subj, unit, y, practice); ys, ss = y[src], subj[src]; prob = np.zeros(len(src))
    for s in np.unique(subj):
        te = ss == s; tr = ~te
        m = model(kind); fit(m, kind, Z[tr], ys[tr], weights(ss[tr], ys[tr])); prob[te] = m.predict_proba(Z[te])[:, 1]
    pw = pd.Series(prob).groupby(src).mean()
    return unit_ba(unit[pw.index.values], y[pw.index.values], pw.values)


def part_a(B):
    W = pd.read_csv(os.path.join(OUT, "noneeg_windows.csv"))
    X = W[FEATS].values; subj = W.subject.values; unit = W.unit.values; y = W.y.values
    rng = np.random.default_rng(20261121)
    obs = {k: {p: loso(X, subj, unit, y, p, k) for p in PRACTICES} for k in ["lr", "hgb"]}
    print("observed", json.dumps(obs, indent=1), flush=True)
    nt = {s: len(np.unique(unit[(subj == s) & (y == 1)])) for s in np.unique(subj)}
    null = {p: [] for p in PRACTICES}; t0 = time.time()
    for b in range(B):
        yp = np.zeros(len(y), int)
        for s in np.unique(subj):
            idx = np.flatnonzero(subj == s); us = rng.choice(np.unique(unit[idx]), nt[s], replace=False)
            yp[idx[np.isin(unit[idx], us)]] = 1
        for p in PRACTICES:
            null[p].append(loso(X, subj, unit, yp, p, "lr"))
        if (b + 1) % max(1, B // 10) == 0: print(f"{b+1}/{B} {time.time()-t0:.0f}s", flush=True)
    rows = []
    for p in PRACTICES:
        v = np.array(null[p]); o = obs["lr"][p]
        rows.append(dict(practice=p, BA_lr=o, BA_hgb=obs["hgb"][p], null_mean=v.mean(), null_lo=np.percentile(v, 2.5),
                         null_hi=np.percentile(v, 97.5), p=(1 + (v >= o).sum()) / (1 + B), B=B))
    T = pd.DataFrame(rows); T.to_csv(os.path.join(OUT, "practices_real.csv"), index=False); print(T.round(3).to_string(index=False))


def part_b(reps):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import semisynthetic_noneeg as ss
    blocks, sb = ss.load_blocks(); rows = []; t0 = time.time()
    for nh in [2, 3]:
        rng = np.random.default_rng(20261121 + nh)
        for delta in [0.0, 1.0]:
            for r in range(reps):
                order = rng.permutation(len(blocks)); k = nh + 3; nsub = len(blocks) // k; shift = delta * sb
                X, subj, unit, y, fresh = [], [], [], [], {}
                for s in range(nsub):
                    b = [blocks[i] for i in order[s * k:(s + 1) * k]]
                    for u in range(nh):
                        X.append(b[u]); subj += [s] * len(b[u]); unit += [f"{s}r{u}"] * len(b[u]); y += [0] * len(b[u])
                    X.append(b[nh] + shift); subj += [s] * len(b[nh]); unit += [f"{s}t"] * len(b[nh]); y += [1] * len(b[nh])
                    fresh[s] = (b[nh + 1], b[nh + 2] + shift, b[:nh])
                X = np.vstack(X); subj = np.array(subj); unit = np.array(unit); y = np.array(y)
                for p in PRACTICES:
                    Z, src = normalize(X, subj, unit, y, p); ys, sss = y[src], subj[src]
                    est = np.zeros(len(src)); tu, ty, tp = [], [], []
                    for s in range(nsub):
                        te = sss == s; tr = ~te
                        m = model("lr"); fit(m, "lr", Z[tr], ys[tr], weights(sss[tr], ys[tr]))
                        est[te] = m.predict_proba(Z[te])[:, 1]
                        fr, ft, Rb = fresh[s]; R = np.vstack(Rb)
                        for lab, blk in [(0, fr), (1, ft)]:
                            if p == "RAW":
                                pp = m.predict_proba(blk)[:, 1]
                            elif p == "BASE_SM":                                   # deployment consistent with training:
                                pp = np.mean([m.predict_proba(z(blk, np.vstack([Rb[w] for w in range(nh) if w != v])))[:, 1]
                                              for v in range(nh)], axis=0)        # n_h references of n_h - 1 units, averaged
                            else:
                                pp = m.predict_proba(z(blk, R))[:, 1]              # deployment: observed reference units only
                            tp += list(pp); ty += [lab] * len(blk); tu += [f"{s}f{lab}"] * len(blk)
                    pw = pd.Series(est).groupby(src).mean()
                    e = unit_ba(unit[pw.index.values], y[pw.index.values], pw.values)
                    t = unit_ba(np.array(tu), np.array(ty), np.array(tp))
                    rows.append(dict(n_h=nh, delta=delta, rep=r, practice=p, estimate=e, truth=t, optimism=e - t))
            print(nh, delta, f"{time.time()-t0:.0f}s", flush=True)
    R = pd.DataFrame(rows); R.to_csv(os.path.join(OUT, "practices_semisynthetic.csv"), index=False)
    g = R.groupby(["n_h", "delta", "practice"]).agg(estimate=("estimate", "mean"), truth=("truth", "mean"),
                                                   optimism=("optimism", "mean"), sd=("optimism", "std"), n=("optimism", "size")).reset_index()
    g["se"] = g.sd / np.sqrt(g.n); g.to_csv(os.path.join(OUT, "practices_semisynthetic_summary.csv"), index=False)
    print(g.round(3).to_string(index=False))


if __name__ == "__main__":
    if sys.argv[1] == "A":
        part_a(int(sys.argv[2]) if len(sys.argv) > 2 else 1000)
    else:
        part_b(int(sys.argv[2]) if len(sys.argv) > 2 else 200)

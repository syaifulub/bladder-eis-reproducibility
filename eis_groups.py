"""
eis_groups.py -- tissue-sample-aware patient-referenced evaluation.

The public dataset contains repeated spectra of a small number of tissue samples
per patient. Two referencing designs are implemented with the same code:

* spectrum-wise (naive):  every spectrum is its own group, so the healthy
  reference of a healthy query excludes only the query spectrum itself
  (the leave-self-out rule of the original analysis);
* sample-wise (corrected): the reference excludes ALL spectra of the query's
  tissue sample (leave-one-tissue-sample-out).

The same referencing design is used for frequency ranking, classifier training
and testing. Performance is aggregated at the tissue-sample level.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score

import eis_pipeline as ep


# ----------------------------------------------------------------------------- data
def sample_ids(raw):
    """Tissue-sample identifier for every spectrum, in ep.harmonize order."""
    out = []
    for p_index, pkey in enumerate(sorted(raw.keys(), key=lambda s: (len(s), s)), start=1):
        for ckey, meas in raw[pkey].items():
            if not isinstance(meas, (list, tuple)):
                continue
            for m in meas:
                t = m["tissue"] if isinstance(m, dict) else getattr(m, "tissue")
                out.append(f"P{p_index}:{t}")
    return np.asarray(out)


def spectrum_groups(n):
    return np.asarray([f"s{i}" for i in range(n)])


# ------------------------------------------------------------------ representations
def build_rep_g(X, pid, y, groups, patients):
    """Absolute-deviation and Ledoit-Wolf Mahalanobis representations relative to the
    same-patient healthy spectra that do NOT share the query's group.
    Returns A, M and a mask of rows that have at least one reference spectrum."""
    n, d = X.shape
    A = np.full((n, d), np.nan)
    M = np.full((n, 1), np.nan)
    for p in patients:
        rows = np.flatnonzero(pid == p)
        healthy = np.flatnonzero((pid == p) & (y == 0))
        cache = {}
        for i in rows:
            pool = healthy[groups[healthy] != groups[i]]
            if pool.size == 0:
                continue
            key = pool.tobytes()
            if key not in cache:
                ref = X[pool]
                med = np.median(ref, axis=0)
                if ref.shape[0] >= 2:
                    lw = LedoitWolf().fit(ref)
                    cache[key] = (med, lw.location_, lw.precision_)
                else:
                    cache[key] = (med, None, None)
            med, mu, prec = cache[key]
            A[i] = np.abs(X[i] - med)
            M[i, 0] = 0.0 if mu is None else float(np.sqrt(max((X[i] - mu) @ prec @ (X[i] - mu), 0.0)))
    valid = ~np.isnan(A).any(1)
    return A, M, valid


# ---------------------------------------------------------------------- weighting
def class_balanced_weights(pid, y):
    """Each class receives half of the total weight; within a class every patient
    contributes equally."""
    w = np.zeros(len(y))
    for c in (0, 1):
        pats = np.unique(pid[y == c])
        for p in pats:
            m = (pid == p) & (y == c)
            w[m] = 1.0 / (len(pats) * m.sum())
    return w * len(y) / w.sum()


def weight_fn(kind):
    return ep.patient_class_weights if kind == "patient_class" else class_balanced_weights


# --------------------------------------------------------------- frequency ranking
def rank_frequencies_g(R, NX, pid, y, groups, train_patients, seed=ep.SEED0):
    """Rank candidate frequencies with inner leave-one-patient-out over train_patients.
    Inner patients without both classes evaluable are not scored (they still train)."""
    rows = []
    for j in range(R.shape[1]):
        X = ep.build_vectors(R, NX, [j])
        A, M, ok = build_rep_g(X, pid, y, groups, train_patients)
        bas, aucs = [], []
        for q in train_patients:
            te = (pid == q) & ok
            tr = np.isin(pid, train_patients) & (pid != q) & ok
            if len(np.unique(y[te])) < 2 or len(np.unique(y[tr])) < 2:
                continue
            w = ep.patient_class_weights(pid[tr], y[tr])
            _, _, pf = ep.fit_predict(A[tr], M[tr], y[tr], w, A[te], M[te], seed)
            mm = ep.patient_metrics(y[te], pf)
            bas.append(mm["BA"]); aucs.append(mm["AUC"])
        ba = np.asarray(bas)
        mean_ba, worst = float(ba.mean()), float(ba.min())
        sd = float(ba.std(ddof=1)) if ba.size > 1 else 0.0
        rows.append(dict(index=j, MeanBA=mean_ba, WorstBA=worst, SD=sd, MeanAUC=float(np.nanmean(aucs)),
                         Score=float(np.sqrt((1 - mean_ba) ** 2 + (1 - worst) ** 2 + sd ** 2))))
    return pd.DataFrame(rows).sort_values(["Score", "MeanBA", "WorstBA", "SD", "MeanAUC"],
                                          ascending=[True, False, False, True, False]).reset_index(drop=True)


def nested_selection_g(R, NX, pid, y, groups, patients, seed=ep.SEED0):
    return {p: sorted(rank_frequencies_g(R, NX, pid, y, groups, [q for q in patients if q != p], seed)
                      ["index"].to_numpy()[:ep.TOP_M].tolist()) for p in patients}


def leaky_selection_g(R, NX, pid, y, groups, patients, seed=ep.SEED0):
    return sorted(rank_frequencies_g(R, NX, pid, y, groups, list(patients), seed)["index"].to_numpy()[:ep.TOP_M].tolist())


# ----------------------------------------------------------------------- evaluation
def evaluate_g(Xfun, pid, y, ref_groups, patients, sample_groups, weights=None,
               n_repeats=ep.N_REPEATS, weighting="patient_class"):
    """Outer leave-one-patient-out evaluation. Xfun(p) gives the feature matrix of fold p.
    weights: None (equal fusion) or dict fold -> fusion weight on the absolute branch.
    Returns one record per (repeat, evaluable test spectrum)."""
    wfun = weight_fn(weighting)
    recs = []
    for p in patients:
        X = Xfun(p)
        trp = [q for q in patients if q != p]
        A, M, ok = build_rep_g(X, pid, y, ref_groups, patients)
        tr = np.isin(pid, trp) & ok
        te = (pid == p) & ok
        if te.sum() == 0:
            continue
        w_tr = wfun(pid[tr], y[tr])
        wf = 0.5 if weights is None else weights[p]
        for rep in range(n_repeats):
            a, b, _ = ep.fit_predict(A[tr], M[tr], y[tr], w_tr, A[te], M[te], seed=rep)
            recs.append(pd.DataFrame(dict(Repeat=rep, Patient=p, Sample=sample_groups[te], y=y[te],
                                          Row=np.flatnonzero(te), p_abs=a, p_mah=b,
                                          p=wf * a + (1 - wf) * b)))
    return pd.concat(recs, ignore_index=True)


def oof_g(X, pid, y, groups, train_patients, seed=ep.SEED0):
    """Inner-LOPO out-of-fold branch probabilities for the outer-training patients."""
    A, M, ok = build_rep_g(X, pid, y, groups, train_patients)
    idx = np.flatnonzero(np.isin(pid, train_patients) & ok)
    pa, pm = np.full(len(y), np.nan), np.full(len(y), np.nan)
    for q in train_patients:
        te = idx[pid[idx] == q]
        tr = idx[pid[idx] != q]
        if te.size == 0:
            continue
        w = ep.patient_class_weights(pid[tr], y[tr])
        a, b, _ = ep.fit_predict(A[tr], M[tr], y[tr], w, A[te], M[te], seed)
        pa[te], pm[te] = a, b
    return pa, pm, idx


def inner_weights_g(Xfun, pid, y, groups, sample_groups, patients):
    """Fusion weight per outer fold chosen on inner out-of-fold predictions (sample-level BA,
    ties toward 0.5); the held-out patient is never used."""
    W = {}
    for p in patients:
        trp = [q for q in patients if q != p]
        pa, pm, idx = oof_g(Xfun(p), pid, y, groups, trp)
        best = None
        for w in np.round(np.linspace(0, 1, 11), 2):
            st = pd.DataFrame(dict(Sample=sample_groups[idx], y=y[idx], p=w * pa[idx] + (1 - w) * pm[idx])) \
                .groupby("Sample").agg(y=("y", "first"), p=("p", "mean"))
            ba = pooled(st)["BA"]
            cand = (ba, -abs(w - 0.5), w)
            best = cand if best is None or cand > best else best
        W[p] = float(best[2])
    return W


# -------------------------------------------------------------------------- metrics
def sample_table(recs, col="p", keep=None):
    """One row per tissue sample: label and probability averaged over spectra and repeats."""
    r = recs if keep is None else recs[recs.Sample.isin(keep)]
    return r.groupby(["Patient", "Sample"]).agg(y=("y", "first"), p=(col, "mean")).reset_index()


def pooled(st, thr=0.5):
    pred = (st.p >= thr).astype(int)
    tp = int(((pred == 1) & (st.y == 1)).sum()); n1 = int((st.y == 1).sum())
    tn = int(((pred == 0) & (st.y == 0)).sum()); n0 = int((st.y == 0).sum())
    sens = tp / n1 if n1 else np.nan
    spec = tn / n0 if n0 else np.nan
    auc = roc_auc_score(st.y, st.p) if st.y.nunique() == 2 else np.nan
    return dict(BA=0.5 * (sens + spec), Sens=sens, Spec=spec, TP=tp, N1=n1, TN=tn, N0=n0, AUC=auc)


def spectrum_ba(recs, col="p", keep=None):
    r = recs if keep is None else recs[recs.Sample.isin(keep)]
    vals = []
    for _, g in r.groupby("Repeat"):
        pred = (g[col] >= 0.5).astype(int)
        vals.append(0.5 * ((pred[g.y == 1] == 1).mean() + (pred[g.y == 0] == 0).mean()))
    return float(np.mean(vals))


def repeat_sd(recs, col="p", keep=None):
    r = recs if keep is None else recs[recs.Sample.isin(keep)]
    return float(np.std([pooled(sample_table(g, col))["BA"] for _, g in r.groupby("Repeat")], ddof=1))


def summary(recs, col="p", keep=None):
    st = sample_table(recs, col, keep)
    out = pooled(st)
    out.update(SpectrumBA=spectrum_ba(recs, col, keep), RepeatSD=repeat_sd(recs, col, keep))
    return out, st


def cluster_boot(tables, B=10000, seed=ep.SEED0, thr=0.5):
    """Patient-level cluster bootstrap of the pooled sample-level balanced accuracy.
    tables: dict name -> sample table (all tables contain the same samples).
    The same resampled patients are used for every table, so differences are paired.
    Resamples without any healthy or any tumor sample are discarded.
    Returns percentile 95% CIs of each BA and of all pairwise differences."""
    names = list(tables)
    pats = np.asarray(sorted(tables[names[0]].Patient.unique()))
    counts = {}
    for n in names:
        t = tables[n]
        pred = (t.p >= thr).astype(int)
        c = np.zeros((len(pats), 4))
        for k, q in enumerate(pats):
            m = (t.Patient == q).values
            c[k] = [((pred == 1) & (t.y == 1) & m).sum(), ((t.y == 1) & m).sum(),
                    ((pred == 0) & (t.y == 0) & m).sum(), ((t.y == 0) & m).sum()]
        counts[n] = c
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(pats), size=(B, len(pats)))
    draws = {}
    for n in names:
        s = counts[n][idx].sum(1)                     # B x 4
        with np.errstate(invalid="ignore", divide="ignore"):
            draws[n] = 0.5 * (s[:, 0] / s[:, 1] + s[:, 2] / s[:, 3])
    valid = np.all([np.isfinite(v) for v in draws.values()], axis=0)
    draws = {n: v[valid] for n, v in draws.items()}
    ci = {n: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for n, v in draws.items()}
    diff = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            d = draws[a] - draws[b]
            diff[f"{a} - {b}"] = (float(np.mean(d)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)))
    return ci, diff, int(valid.sum())

"""
rev_common.py -- shared helpers for the revision analyses.

* lw_fit / fast_rep : numerically identical, faster re-implementation of
  eg.build_rep_g (closed-form Ledoit-Wolf as in scikit-learn);
* run_eval          : outer leave-one-patient-out evaluation on precomputed
  representations (identical to eg.evaluate_g for fold-independent features);
* LogitClf / backend: L2 logistic-regression backend that can replace XGBoost
  everywhere ep.fit_predict is used.
"""
from contextlib import contextmanager

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import eis_pipeline as ep
import eis_groups as eg


# ------------------------------------------------------------------ Ledoit-Wolf
def lw_fit(X):
    """Ledoit-Wolf location and precision (same estimator as sklearn.covariance.LedoitWolf)."""
    n, d = X.shape
    mu = X.mean(0)
    Xc = X - mu
    emp = Xc.T @ Xc / n
    if d == 1:
        s, m = 0.0, float(emp[0, 0])
    else:
        X2 = Xc ** 2
        tr = X2.sum(0) / n
        m = tr.sum() / d
        beta_ = (X2.T @ X2).sum()
        delta_ = ((Xc.T @ Xc) ** 2).sum() / n ** 2
        beta = (beta_ / n - delta_) / (d * n)
        delta = (delta_ - 2.0 * m * tr.sum() + d * m * m) / d
        beta = min(beta, delta)
        s = 0.0 if beta == 0 else beta / delta
    cov = (1.0 - s) * emp
    cov[np.diag_indices(d)] += s * m
    return mu, np.linalg.pinv(cov, hermitian=True)


def fast_rep(X, pid, y, groups, patients):
    """Same output as eg.build_rep_g."""
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
                cache[key] = (med, *lw_fit(ref)) if ref.shape[0] >= 2 else (med, None, None)
            med, mu, prec = cache[key]
            A[i] = np.abs(X[i] - med)
            if mu is None:
                M[i, 0] = 0.0
            else:
                dd = X[i] - mu
                M[i, 0] = float(np.sqrt(max(dd @ prec @ dd, 0.0)))
    return A, M, ~np.isnan(A).any(1)


# ------------------------------------------------------------------ evaluation
def run_eval(A, M, ok, pid, y, sgroups, patients, seeds, weights=None, test_patients=None):
    recs = []
    for p in (patients if test_patients is None else test_patients):
        trp = [q for q in patients if q != p]
        tr = np.isin(pid, trp) & ok
        te = (pid == p) & ok
        if te.sum() == 0:
            continue
        w = ep.patient_class_weights(pid[tr], y[tr])
        wf = 0.5 if weights is None else weights[p]
        for s in seeds:
            a, b, _ = ep.fit_predict(A[tr], M[tr], y[tr], w, A[te], M[te], seed=int(s))
            recs.append(pd.DataFrame(dict(Repeat=int(s), Patient=p, Sample=sgroups[te], y=y[te],
                                          Row=np.flatnonzero(te), p_abs=a, p_mah=b, p=wf * a + (1 - wf) * b)))
    return pd.concat(recs, ignore_index=True)


def sample_ba(recs, col, keep):
    st = eg.sample_table(recs, col, keep)
    return eg.pooled(st)["BA"], st


# ------------------------------------------------------------------ classifier backends
class LogitClf:
    """L2 logistic regression on log(1 + feature), standardized; deterministic."""

    def __init__(self, seed=0, C=1.0):
        self.C = C

    def fit(self, X, y, sample_weight=None):
        Z = np.log1p(np.abs(X))
        self.sc = StandardScaler().fit(Z)
        self.m = LogisticRegression(C=self.C, max_iter=5000).fit(self.sc.transform(Z), y, sample_weight=sample_weight)
        return self

    def predict_proba(self, X):
        return self.m.predict_proba(self.sc.transform(np.log1p(np.abs(X))))


@contextmanager
def backend(kind):
    old = ep.make_classifier
    if kind == "logit":
        ep.make_classifier = lambda seed: LogitClf(seed)
    try:
        yield
    finally:
        ep.make_classifier = old


def matched_rep(X, pid, y, G, patients):
    """Reference-size-matched sample-aware representation.
    A healthy query is referenced against the patient's other healthy samples (n_h - 1 samples).
    A tumor query is referenced n_h times, each time leaving one healthy sample out, so that both
    classes see references built from the same number of samples. Returns (row index, A, M) for
    every (query, reference version); predictions are averaged over versions."""
    idx, As, Ms = [], [], []
    for p in patients:
        healthy = np.flatnonzero((pid == p) & (y == 0))
        hs = np.unique(G[healthy])
        cache = {}
        for i in np.flatnonzero(pid == p):
            if y[i] == 0:
                pools = [healthy[G[healthy] != G[i]]]
            elif hs.size >= 2:
                pools = [healthy[G[healthy] != s] for s in hs]
            else:
                pools = [healthy]
            for pool in pools:
                if pool.size == 0:
                    continue
                key = pool.tobytes()
                if key not in cache:
                    ref = X[pool]
                    cache[key] = (np.median(ref, axis=0), *lw_fit(ref)) if ref.shape[0] >= 2 else (np.median(ref, axis=0), None, None)
                med, mu, prec = cache[key]
                dd = X[i] - (mu if mu is not None else med)
                idx.append(i)
                As.append(np.abs(X[i] - med))
                Ms.append(0.0 if mu is None else float(np.sqrt(max(dd @ prec @ dd, 0.0))))
    return np.asarray(idx), np.asarray(As), np.asarray(Ms)[:, None]


def run_eval_matched(X, pid, y, G, patients, seeds, test_patients=None):
    idx, A, M = matched_rep(X, pid, y, G, patients)
    pv, yv = pid[idx], y[idx]
    recs = []
    for p in (patients if test_patients is None else test_patients):
        trp = [q for q in patients if q != p]
        tr = np.isin(pv, trp)
        te = pv == p
        if te.sum() == 0:
            continue
        w = ep.patient_class_weights(pv[tr], yv[tr])
        for s in seeds:
            a, b, _ = ep.fit_predict(A[tr], M[tr], yv[tr], w, A[te], M[te], seed=int(s))
            d = pd.DataFrame(dict(Row=idx[te], p_abs=a, p_mah=b)).groupby("Row").mean().reset_index()
            d["p"] = 0.5 * d.p_abs + 0.5 * d.p_mah
            d["Repeat"], d["Patient"], d["Sample"], d["y"] = int(s), p, G[d.Row], y[d.Row]
            recs.append(d)
    return pd.concat(recs, ignore_index=True)


def permute_within_patient(y, pid, G, rng):
    """Shuffle class labels among the tissue samples of each patient (spectra follow their sample)."""
    y2 = y.copy()
    for p in np.unique(pid):
        m = pid == p
        samples = np.unique(G[m])
        lab = np.asarray([y[m & (G == s)][0] for s in samples])
        new = dict(zip(samples, rng.permutation(lab)))
        for s in samples:
            y2[m & (G == s)] = new[s]
    return y2


def evaluable(pid, y, G, patients):
    """Tissue samples with at least one same-patient healthy reference sample other than themselves."""
    _, _, ok = fast_rep(np.zeros((len(y), 1)) + np.arange(len(y))[:, None] * 0.0, pid, y, G, patients)
    return sorted(set(G[ok]))

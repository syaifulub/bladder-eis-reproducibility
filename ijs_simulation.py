"""
ijs_simulation.py -- does the pseudo-replication inflation generalize beyond the bladder data?

Hierarchical generator (whitened four-dimensional feature space, analogous to the four Cole
parameters):
    patient offset      mu_p  ~ N(0, 4 I)                          (removed by referencing)
    healthy sample      theta_s = mu_p + b_s,            b_s ~ N(0, I)
    tumor sample        theta_s = mu_p + delta_p + b_s,  delta_p ~ N(delta, 0.25 I)
    spectrum            x_ij    = theta_s + e_ij,        e_ij ~ N(0, (1 - ICC)/ICC I)
with |delta| = Delta. Each patient contributes n_h healthy and n_t tumor tissue samples with
m spectra each. ICC is the share of the within-patient variance that is between samples; the
between-sample variance is held at 1 so that the separability of tissue samples (the quantity
that matters clinically) does not change with ICC; only the redundancy of repeated spectra does.

For every replicate the patient-referenced pipeline (absolute + Mahalanobis branches,
equal fusion, patient-class weights, leave-one-patient-out) is run with
    naive referencing        (leave-one-spectrum-out) and
    sample-aware referencing (leave-one-tissue-sample-out).
The cross-validated sample-level BA (the estimate) is compared with the TRUE deployment BA of
the same pipeline: the classifier trained on all simulated patients is applied to 40 fresh
patients, in which a NEW query tissue sample (healthy or tumor) is referenced against all of the
patient's existing healthy samples, as it would be in clinical use. Optimism = estimate - truth.
A third design, reference-size-matched sample-aware referencing (rc.matched_rep), removes the
asymmetry that a healthy query sees n_h - 1 reference samples whereas a tumor query sees n_h.
"""
import json
import os
import sys
import time
from itertools import product
from multiprocessing import Pool

import numpy as np
import pandas as pd

import eis_pipeline as ep
import eis_groups as eg
import rev_common as rc

OUT = "outputs_ijs/revision"
os.makedirs(OUT, exist_ok=True)
D = 4
N_FRESH = 40


def simulate(rng, n_pat, n_h, n_t, m, icc, delta, offset=0):
    dvec = np.full(D, delta / np.sqrt(D))
    X, pid, y, G = [], [], [], []
    for p in range(n_pat):
        mu = rng.normal(0, 2.0, D)
        dp = dvec + rng.normal(0, 0.5, D)
        for c, ns in ((0, n_h), (1, n_t)):
            for s in range(ns):
                th = mu + c * dp + rng.normal(0, 1.0, D)
                X.append(th + rng.normal(0, np.sqrt((1 - icc) / icc), (m, D)))
                pid += [f"P{p + offset}"] * m
                y += [c] * m
                G += [f"P{p + offset}:{c}{s}"] * m
    return np.vstack(X), np.asarray(pid), np.asarray(y), np.asarray(G)


def fit_branches(A, M, y, w, seed):
    a = ep.make_classifier(seed).fit(A, y, sample_weight=w)
    b = ep.make_classifier(seed + 1).fit(M, y, sample_weight=w)
    return a, b


def pooled_ba(G, y, p):
    st = pd.DataFrame(dict(Patient=0, Sample=G, y=y, p=p)).groupby("Sample").agg(y=("y", "first"), p=("p", "mean"))
    return eg.pooled(st)["BA"]


def deployment_set(rng, n_h, n_t, m, icc, delta):
    """Fresh patients as in clinical use: n_h healthy REFERENCE samples plus one new healthy and
    n_t tumor QUERY samples; every query is referenced against all n_h reference samples."""
    X, pid, y, G = simulate(rng, N_FRESH, n_h + 1, n_t, m, icc, delta, offset=10_000)
    query_h = np.asarray([g.endswith(f":0{n_h}") for g in G])          # last healthy sample = query
    yref = y.copy()
    yref[query_h] = 2                                                  # excluded from the reference pool
    A, M, ok = rc.fast_rep(X, pid, yref, G, np.unique(pid))
    q = ok & (query_h | (y == 1))
    return A[q], M[q], y[q], G[q]


def replicate(args):
    cell, rep, kind = args
    ep.XGB_PARAMS["n_jobs"] = 1
    n_pat, n_h, n_t, m, icc, delta = cell
    rng = np.random.default_rng([ep.SEED0, rep, int(1000 * icc), m, int(100 * delta), n_pat, n_h])
    X, pid, y, G = simulate(rng, n_pat, n_h, n_t, m, icc, delta)
    Af, Mf, yf, Gf = deployment_set(rng, n_h, n_t, m, icc, delta)
    pats = np.unique(pid)
    keep = sorted(set(G[rc.fast_rep(X, pid, y, G, pats)[2]]))
    out = dict(n_pat=n_pat, n_h=n_h, n_t=n_t, m=m, icc=icc, delta=delta, rep=rep, backend=kind)
    with rc.backend(kind):
        for dn in ("naive", "sa", "matched"):
            if dn == "matched":
                r = rc.run_eval_matched(X, pid, y, G, pats, seeds=[rep])
                idx, A, M = rc.matched_rep(X, pid, y, G, pats)
                yt, pt = y[idx], pid[idx]
            else:
                grp = np.asarray([f"s{i}" for i in range(len(y))]) if dn == "naive" else G
                A, M, ok = rc.fast_rep(X, pid, y, grp, pats)
                r = rc.run_eval(A, M, ok, pid, y, G, pats, seeds=[rep])
                A, M, yt, pt = A[ok], M[ok], y[ok], pid[ok]
            out[f"{dn}_est"] = rc.sample_ba(r, "p", keep)[0]
            out[f"{dn}_est_abs"] = rc.sample_ba(r, "p_abs", keep)[0]
            ca, cm = fit_branches(A, M, yt, ep.patient_class_weights(pt, yt), rep)
            pa, pm = ca.predict_proba(Af)[:, 1], cm.predict_proba(Mf)[:, 1]
            out[f"{dn}_true"] = pooled_ba(Gf, yf, 0.5 * pa + 0.5 * pm)
            out[f"{dn}_true_abs"] = pooled_ba(Gf, yf, pa)
    return out


def run(jobs, name, procs=2):
    """Run replicates in parallel; results are appended to a partial file so that an interrupted
    run resumes where it stopped."""
    t0 = time.time()
    part = f"{OUT}/sim_{name}_partial.csv"
    keycols = ["n_pat", "n_h", "n_t", "m", "icc", "delta", "rep", "backend"]
    done = set()
    if os.path.exists(part):
        done = set(map(tuple, pd.read_csv(part)[keycols].astype(str).values.tolist()))
    todo = [j for j in jobs if tuple(map(str, (*j[0], j[1], j[2]))) not in done]
    print(f"{name}: {len(jobs) - len(todo)} done, {len(todo)} to go", flush=True)
    with Pool(procs) as pool:
        for i, r in enumerate(pool.imap_unordered(replicate, todo, chunksize=2)):
            pd.DataFrame([r]).to_csv(part, mode="a", header=not os.path.exists(part), index=False)
            if i % 200 == 0:
                print(f"[{time.time() - t0:6.0f}s] {name} {i}/{len(todo)}", flush=True)
    df = pd.read_csv(part).drop_duplicates(keycols)
    df.to_csv(f"{OUT}/sim_{name}.csv", index=False)
    return df


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    DELTA = 4.0
    if which in ("grid", "all"):
        cells = [(6, 3, 3, m, icc, d) for icc, m, d in product([0.1, 0.25, 0.5, 0.75, 0.9, 0.97], [1, 5, 15, 30], [0.0, DELTA])]
        run([(c, r, "logit") for c in cells for r in range(30)], "grid")
    if which in ("xgb", "all"):
        cells = [(6, 3, 3, 15, icc, d) for icc in (0.1, 0.5, 0.9, 0.97) for d in (0.0, DELTA)]
        run([(c, r, "xgb") for c in cells for r in range(20)], "xgb")
    if which in ("precision", "all"):
        cells = [(n, nh, 3, 15, 0.95, DELTA) for n in (6, 12, 24, 48) for nh in (2, 3)]
        run([(c, r, "logit") for c in cells for r in range(100)], "precision")

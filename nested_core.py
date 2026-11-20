"""
nested_core.py -- fully nested pipeline selection as a reusable function.

full_nested(yk, ...) runs, for labels yk:
  outer leave-one-patient-out; in every outer fold
    * frequencies ranked on the five outer-training patients (for the outer model),
    * inner leave-one-patient-out over the five training patients, in which the frequency
      candidate is re-ranked on the four inner-training patients of every inner fold
      (patient-level ranking when at least two of them have >= 2 healthy samples, otherwise a
      fallback chosen by `fallback`),
    * six candidates (frequencies | Cole) x (fusion | absolute | Mahalanobis) compared by inner
      sample-level BA (ties: inner AUC, then the pre-specified frequency fusion),
    * the chosen candidate refitted on the five training patients and applied once to the
      held-out patient.
Options
  fallback : "loso"    leave-one-tissue-sample-out ranking within the four patients (main analysis)
             "outer"   reuse the outer-fold ranking (favours the frequency candidate)
             "generic" fixed decade-spaced set (no data-driven ranking)
  design   : "sa" sample-wise referencing; "matched" size-matched sample-wise referencing
             (ranking, inner comparison, and outer evaluation all use the same design)
  patients : subset of patients (e.g. without P2)
With design="sa", fallback="loso" and seeds=range(30) the result is identical to ijs_revision.py R1.
"""
import numpy as np
import pandas as pd

import eis_pipeline as ep
import eis_groups as eg
import rev_common as rc

GRID = ep.candidate_grid()
CANDS = [(rep, br) for rep in ("freq", "cole") for br in ("fusion", "abs", "mah")]
BRCOL = {"fusion": "p", "abs": "p_abs", "mah": "p_mah"}


class Data:
    def __init__(self, raw, fits):
        self.R, self.NX, self.pid, self.y, _ = ep.harmonize(raw, GRID)
        self.G = eg.sample_ids(raw)
        self.THETA = np.column_stack([np.log(fits.R0), np.log(fits.Rinf), np.log10(fits.fc), fits.alpha])
        self.generic = sorted({int(np.argmin(np.abs(GRID - f))) for f in (1e3, 1e4, 1e5, 5e5, 1e6)})


def reps(D, X, yk, patients, design):
    """Row index, absolute-deviation and Mahalanobis representations (one row per query version)."""
    if design == "sa":
        A, M, ok = rc.fast_rep(X, D.pid, yk, D.G, patients)
        idx = np.flatnonzero(ok)
        return idx, A[ok], M[ok]
    return rc.matched_rep(X, D.pid, yk, D.G, patients)


def fit_eval(D, rep, yk, train_patients, test_mask_fn, seed):
    """Fit on rows of train_patients, predict rows selected by test_mask_fn(row_pid, row_group);
    predictions of query versions are averaged per spectrum."""
    idx, A, M = rep
    pv, gv, yv = D.pid[idx], D.G[idx], yk[idx]
    te = test_mask_fn(pv, gv)
    tr = np.isin(pv, train_patients) & ~te
    if te.sum() == 0 or len(np.unique(yv[tr])) < 2:
        return None
    w = ep.patient_class_weights(pv[tr], yv[tr])
    a, b, pf = ep.fit_predict(A[tr], M[tr], yv[tr], w, A[te], M[te], seed=int(seed))
    d = pd.DataFrame(dict(Row=idx[te], p_abs=a, p_mah=b, p_rank=pf)).groupby("Row").mean().reset_index()
    d["p"] = 0.5 * d.p_abs + 0.5 * d.p_mah
    d["y"], d["Patient"], d["Sample"] = yk[d.Row], D.pid[d.Row], D.G[d.Row]
    return d


def _score(bas, aucs, j):
    ba = np.asarray(bas)
    sd = float(ba.std(ddof=1)) if ba.size > 1 else 0.0
    return dict(index=j, MeanBA=float(ba.mean()), WorstBA=float(ba.min()), SD=sd, MeanAUC=float(np.nanmean(aucs)),
                Score=float(np.sqrt((1 - ba.mean()) ** 2 + (1 - ba.min()) ** 2 + sd ** 2)))


def _top(rows):
    df = pd.DataFrame(rows).sort_values(["Score", "MeanBA", "WorstBA", "SD", "MeanAUC"],
                                        ascending=[True, False, False, True, False])
    return sorted(int(v) for v in df["index"].to_numpy()[:ep.TOP_M])


def rank_lopo(D, yk, train, design, seed=ep.SEED0):
    rows = []
    for j in range(D.R.shape[1]):
        rep = reps(D, ep.build_vectors(D.R, D.NX, [j]), yk, train, design)
        bas, aucs = [], []
        for q in train:
            if len(np.unique(yk[rep[0]][D.pid[rep[0]] == q])) < 2:
                continue
            d = fit_eval(D, rep, yk, train, lambda pv, gv: pv == q, seed)
            if d is None:
                continue
            m = ep.patient_metrics(d.y.to_numpy(), d.p_rank.to_numpy())
            bas.append(m["BA"]); aucs.append(m["AUC"])
        rows.append(_score(bas, aucs, j))
    return _top(rows)


def rank_loso(D, yk, train, design, seed=ep.SEED0):
    rows = []
    for j in range(D.R.shape[1]):
        rep = reps(D, ep.build_vectors(D.R, D.NX, [j]), yk, train, design)
        parts = []
        for smp in np.unique(D.G[rep[0]]):
            d = fit_eval(D, rep, yk, train, lambda pv, gv: gv == smp, seed)
            if d is not None:
                parts.append(d)
        d = pd.concat(parts, ignore_index=True)
        bas, aucs = [], []
        for q in train:
            dq = d[d.Patient == q]
            if dq.y.nunique() < 2:
                continue
            m = ep.patient_metrics(dq.y.to_numpy(), dq.p_rank.to_numpy())
            bas.append(m["BA"]); aucs.append(m["AUC"])
        rows.append(_score(bas, aucs, j))
    return _top(rows)


def scorable(D, train, yk):
    multi = [v for v in train if len(np.unique(D.G[(D.pid == v) & (yk == 0)])) >= 2]
    return len(multi) >= 2


def full_nested(D, yk, patients, fallback="loso", design="sa", seeds=(ep.SEED0,), cache=None, keep=None):
    cache = {} if cache is None else cache
    pts = list(patients)
    keep = rc.evaluable(D.pid, yk, D.G, pts) if keep is None else keep
    chosen, inner_rows, recs, inner_method = {}, [], [], {}
    for p in pts:
        trp = [q for q in pts if q != p]
        k_out = ("outer", design, p)
        if k_out not in cache:
            cache[k_out] = rank_lopo(D, yk, trp, design)
        Fout = cache[k_out]
        scores = {}
        for rep_name in ("freq", "cole"):
            parts = []
            for q in trp:
                tr4 = [v for v in trp if v != q]
                if rep_name == "cole":
                    X = D.THETA
                else:
                    if scorable(D, tr4, yk):
                        k_in, method = ("lopo", design, p, q), "LOPO"
                        if k_in not in cache:
                            cache[k_in] = rank_lopo(D, yk, tr4, design)
                        F = cache[k_in]
                    elif fallback == "loso":
                        k_in, method = ("loso", design, p, q), "LOSO"
                        if k_in not in cache:
                            cache[k_in] = rank_loso(D, yk, tr4, design)
                        F = cache[k_in]
                    elif fallback == "outer":
                        F, method = Fout, "outer"
                    elif fallback == "generic":
                        F, method = D.generic, "generic"
                    else:
                        raise ValueError(fallback)
                    inner_method[f"{p}|{q}"] = method
                    X = ep.build_vectors(D.R, D.NX, F)
                rep = reps(D, X, yk, trp, design)
                d = fit_eval(D, rep, yk, trp, lambda pv, gv: pv == q, ep.SEED0)
                if d is not None:
                    parts.append(d)
            r = pd.concat(parts, ignore_index=True)
            keep_in = sorted(set(D.G[np.isin(D.pid, trp)]) & set(r.Sample))
            for br, col in BRCOL.items():
                m = eg.pooled(eg.sample_table(r.assign(Repeat=0), col, keep_in))
                scores[(rep_name, br)] = (m["BA"], m["AUC"] if np.isfinite(m["AUC"]) else 0.0)
                inner_rows.append(dict(Fold=p, Representation=rep_name, Branch=br, InnerBA=m["BA"], InnerAUC=m["AUC"]))
        best = max(CANDS, key=lambda c: (round(scores[c][0], 10), round(scores[c][1], 10), -CANDS.index(c)))
        chosen[p] = dict(representation=best[0], branch=best[1], inner_BA=scores[best][0])
        X = ep.build_vectors(D.R, D.NX, Fout) if best[0] == "freq" else D.THETA
        rep = reps(D, X, yk, pts, design)
        for s in seeds:
            d = fit_eval(D, rep, yk, trp, lambda pv, gv: pv == p, s)
            if d is not None:
                d["Repeat"] = int(s)
                d["p_sel"] = d[BRCOL[best[1]]]
                recs.append(d)
    recs = pd.concat(recs, ignore_index=True)
    st = eg.sample_table(recs, "p_sel", keep)
    return dict(choice=chosen, inner=pd.DataFrame(inner_rows), records=recs, sample_table=st,
                metrics=eg.pooled(st), inner_method=inner_method)

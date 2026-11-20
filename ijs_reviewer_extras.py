"""
ijs_reviewer_extras.py -- additional robustness analyses requested in review.

E1 per-patient confusion matrices and leave-one-patient-out influence (pooled BA recomputed
   with each patient's samples removed) for the main configurations
E2 stability of frequency selection (band shares, selection counts, pairwise Jaccard index)
E3 Cole identifiability: absolute Cole branch re-evaluated without the spectra whose
   characteristic frequency lies outside the fitted window
E4 stability of predictions across the 30 XGBoost seeds (per-seed BA, samples changing class)
E5 design effect and effective sample size for every Cole parameter and class, with the
   Kish correction for unequal numbers of spectra per sample
Output: outputs_ijs/revision/reviewer_extras.json and CSV tables.
"""
import json
from itertools import combinations

import numpy as np
import pandas as pd

import eis_pipeline as ep
import eis_groups as eg
import rev_common as rc
from cole_analysis import FMIN, FMAX

OUT = "outputs_ijs"
REV = f"{OUT}/revision"
raw = ep.load_raw_dataset("electrical-impedance-data-bladder/dataset_without_modules.pickle")
GRID = ep.candidate_grid()
R, NX, pid, y, _ = ep.harmonize(raw, GRID)
G = eg.sample_ids(raw)
P = [f"P{i}" for i in range(1, 7)]
fits = pd.read_csv(f"{OUT}/cole_fits.csv")
THETA = np.column_stack([np.log(fits.R0), np.log(fits.Rinf), np.log10(fits.fc), fits.alpha])
SEL = json.load(open(f"{OUT}/selections_idx.json"))
EVAL = rc.evaluable(pid, y, G, P)
X = {}

# ------------------------------------------------------------------ E1 per patient and influence
tables = {"Spectrum-wise, nested frequencies": f"{OUT}/sampletable_naive_nested_fusion.csv",
          "Sample-wise, nested frequencies": f"{OUT}/sampletable_sa_nested_fusion.csv",
          "Sample-wise, published set": f"{OUT}/sampletable_sa_published_fusion.csv",
          "Fully nested pipeline": f"{REV}/sampletable_nested_pipeline_selected.csv"}
rows, infl = [], []
for name, path in tables.items():
    st = pd.read_csv(path)
    st["pred"] = (st.p >= 0.5).astype(int)
    for p in P:
        s = st[st.Patient == p]
        tp, fn = int(((s.y == 1) & (s.pred == 1)).sum()), int(((s.y == 1) & (s.pred == 0)).sum())
        tn, fp = int(((s.y == 0) & (s.pred == 0)).sum()), int(((s.y == 0) & (s.pred == 1)).sum())
        rows.append(dict(Config=name, Patient=p, TP=tp, FN=fn, TN=tn, FP=fp,
                         BA=(tp / (tp + fn) + tn / (tn + fp)) / 2 if (tp + fn) and (tn + fp) else np.nan))
        rest = st[st.Patient != p]
        infl.append(dict(Config=name, Removed=p, BA=eg.pooled(rest)["BA"]))
per_patient = pd.DataFrame(rows)
influence = pd.DataFrame(infl)
per_patient.to_csv(f"{REV}/per_patient_confusion.csv", index=False)
influence.to_csv(f"{REV}/leave_one_patient_out_influence.csv", index=False)
X["influence"] = {c: dict(min=float(g.BA.min()), max=float(g.BA.max()),
                          min_patient=g.loc[g.BA.idxmin(), "Removed"], max_patient=g.loc[g.BA.idxmax(), "Removed"])
                  for c, g in influence.groupby("Config")}

# ------------------------------------------------------------------ E2 frequency stability
inner = json.load(open(f"{REV}/selections_inner.json"))
sets = {"sample-wise outer": list(SEL["sa_nested"].values()), "sample-wise inner": list(inner.values()),
        "spectrum-wise outer": list(SEL["naive_nested"].values())}
stab = {}
for k, ss in sets.items():
    f = np.concatenate([GRID[s] for s in ss]) / 1e3
    jac = [len(set(a) & set(b)) / len(set(a) | set(b)) for a, b in combinations(ss, 2)]
    counts = pd.Series(np.concatenate(ss)).value_counts()
    stab[k] = dict(n_sets=len(ss), share_below_10k=float((f < 10).mean()), share_10_100k=float(((f >= 10) & (f < 100)).mean()),
                   share_above_100k=float((f >= 100).mean()), mean_jaccard=float(np.mean(jac)),
                   n_distinct=int(counts.size), max_count=int(counts.max()),
                   most_frequent_khz=float(GRID[counts.index[0]] / 1e3))
X["frequency_stability"] = stab

# ------------------------------------------------------------------ E3 Cole identifiability
inwin = ((fits.fc >= FMIN) & (fits.fc <= FMAX)).to_numpy()
rows_ = np.flatnonzero(inwin)
X["cole_out_of_window"] = dict(n=int((~inwin).sum()),
                               per_sample={g: int(n) for g, n in pd.Series(G[~inwin]).value_counts().items()})
p_, y_, G_, T_ = pid[rows_], y[rows_], G[rows_], THETA[rows_]
A, M, ok = rc.fast_rep(T_, p_, y_, G_, P)
keep_ = rc.evaluable(p_, y_, G_, P)
r = rc.run_eval(A, M, ok, p_, y_, G_, P, seeds=range(ep.N_REPEATS))
m = eg.pooled(eg.sample_table(r, "p_abs", keep_))
st_main = pd.read_csv(f"{OUT}/sampletable_sa_cole_abs.csv")
mis = st_main[(st_main.p >= 0.5).astype(int) != st_main.y].Sample.tolist()
X["cole_in_window_only"] = dict(BA=m["BA"], TP=m["TP"], N1=m["N1"], TN=m["TN"], N0=m["N0"], n_samples=len(keep_),
                                misclassified_main=mis,
                                misclassified_main_out_of_window=int(sum(X["cole_out_of_window"]["per_sample"].get(s, 0) for s in mis)))

# ------------------------------------------------------------------ E4 seed stability
def per_seed(recs, col):
    bas, preds = [], []
    for s, g in recs.groupby("Repeat"):
        st = eg.sample_table(g, col, EVAL)
        bas.append(eg.pooled(st)["BA"])
        preds.append(st.set_index("Sample").p.ge(0.5).astype(int))
    pr = pd.concat(preds, axis=1)
    return dict(BA_min=float(min(bas)), BA_max=float(max(bas)), BA_sd=float(np.std(bas, ddof=1)),
                n_samples_changing_class=int((pr.nunique(axis=1) > 1).sum()))

A, M, ok = rc.fast_rep(THETA, pid, y, G, P)
rc_cole = rc.run_eval(A, M, ok, pid, y, G, P, seeds=range(ep.N_REPEATS))
rf = eg.evaluate_g(lambda p: ep.build_vectors(R, NX, SEL["sa_nested"][p]), pid, y, G, P, G)
X["seed_stability"] = {"Cole absolute (fully nested choice)": per_seed(rc_cole, "p_abs"),
                       "Nested frequencies, equal fusion": per_seed(rf, "p")}

# ------------------------------------------------------------------ E5 design effect
icc = json.load(open(f"{OUT}/ijs_results.json"))["icc"]
m_s = pd.Series(G).value_counts()
mbar, cv2 = float(m_s.mean()), float(m_s.var(ddof=0) / m_s.mean() ** 2)
de = {}
for k, v in icc.items():
    d0 = 1 + (mbar - 1) * v
    d1 = 1 + ((1 + cv2) * mbar - 1) * v
    de[k] = dict(ICC=v, DE=d0, N_eff=len(y) / d0, DE_kish=d1, N_eff_kish=len(y) / d1)
X["design_effect"] = dict(m_bar=mbar, cv2=cv2, per_parameter=de,
                          N_eff_min=min(v["N_eff"] for v in de.values()), N_eff_max=max(v["N_eff"] for v in de.values()),
                          N_eff_kish_min=min(v["N_eff_kish"] for v in de.values()),
                          N_eff_kish_max=max(v["N_eff_kish"] for v in de.values()))

# ------------------------------------------------------------------ E6 published set without P2
P5 = [p for p in P if p != "P2"]
XPUB = ep.build_vectors(*ep.harmonize(raw, np.array([3.55e3, 568.99e3, 655.13e3, 754.31e3, 868.51e3]))[:2], [0, 1, 2, 3, 4])
rp = eg.evaluate_g(lambda p: XPUB, pid, y, G, P5, G)
mp = eg.pooled(eg.sample_table(rp, "p", rc.evaluable(pid, y, G, P5)))
X["published_without_P2"] = dict(BA=mp["BA"], TP=mp["TP"], N1=mp["N1"], TN=mp["TN"], N0=mp["N0"])

json.dump(json.loads(json.dumps(X, default=float)), open(f"{REV}/reviewer_extras.json", "w"), indent=1)
print(json.dumps(json.loads(json.dumps(X, default=float)), indent=1))
print(per_patient.to_string())

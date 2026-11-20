"""
ijs_revision.py -- additional real-data analyses for the revised IJS manuscript.

R1  Fully nested pipeline selection: inside every outer fold, the representation
    (nested-selected frequencies or Cole parameters) and the branch (fusion, absolute,
    Mahalanobis) are chosen by inner leave-one-patient-out on the five training patients only;
    the chosen pipeline is then applied once to the held-out patient. Within the inner loop,
    the frequencies are re-ranked in every inner fold on the four inner-training patients, so
    every data-driven choice is nested at both levels. With --partial-nesting, the outer-fold
    frequency ranking is reused in the inner folds instead (incomplete nesting, for comparison);
    results go to *_v2_outer_ranking files and the checkpoint is not touched. This turns the
    exploratory Cole-absolute result into a selection-honest estimate.
R2  Reference-size-matched sample-aware referencing (rc.matched_rep).
R3  P2 label-swap sensitivity: the complete factorial re-run with the P2 class labels exchanged.
R4  Logistic-regression backend: the complete factorial (ranking and evaluation) re-run with
    L2 logistic regression instead of XGBoost.
Outputs: outputs_ijs/revision/revision_results.json and CSV tables.
"""
import json
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd

import eis_pipeline as ep
import eis_groups as eg
import rev_common as rc

eg.build_rep_g = rc.fast_rep          # identical results (verified), several times faster
OUT = "outputs_ijs/revision"
os.makedirs(OUT, exist_ok=True)
T0 = time.time()
tick = lambda m: print(f"[{time.time() - T0:6.0f}s] {m}", flush=True)

raw = ep.load_raw_dataset("electrical-impedance-data-bladder/dataset_without_modules.pickle")
GRID = ep.candidate_grid()
R, NX, pid, y, meta = ep.harmonize(raw, GRID)
P = [f"P{i}" for i in range(1, 7)]
G = eg.sample_ids(raw)
GS = eg.spectrum_groups(len(y))
fits = pd.read_csv("outputs_ijs/cole_fits.csv")
THETA = np.column_stack([np.log(fits.R0), np.log(fits.Rinf), np.log10(fits.fc), fits.alpha])
Rv, NXv, *_ = ep.harmonize(raw, np.array([3.55e3, 568.99e3, 655.13e3, 754.31e3, 868.51e3]))
XPUB = ep.build_vectors(Rv, NXv, [0, 1, 2, 3, 4])
SEL = json.load(open("outputs_ijs/selections_idx.json"))
EVAL = rc.evaluable(pid, y, G, P)
SEEDS = range(ep.N_REPEATS)
khz = lambda idx: [round(float(GRID[j]) / 1e3, 3) for j in idx]
Xsel = lambda s: (lambda p: ep.build_vectors(R, NX, s[p] if isinstance(s, dict) else s))
RES, TAB = {}, {}


def summarize(name, recs, keep, cols=(("p", "fusion"), ("p_abs", "abs"), ("p_mah", "mah"))):
    for col, tag in cols:
        s, st = eg.summary(recs, col, keep=keep)
        RES.setdefault("summary", {})[f"{name}|{tag}"] = s
        TAB[f"{name}|{tag}"] = st


def boot(names, key):
    ci, diff, nb = eg.cluster_boot({k: TAB[k] for k in names})
    RES.setdefault("ci", {}).update({k: v for k, v in ci.items()})
    RES.setdefault("diff", {})[key] = diff
    RES.setdefault("boot_valid", {})[key] = nb


def factorial_selections(yy, cache_key):
    """Four selections of the factorial design, cached per outer fold so that an interrupted
    run resumes where it stopped."""
    path = f"{OUT}/selections_{cache_key}.json"
    S = json.load(open(path)) if os.path.exists(path) else {}
    for key, grp in (("naive_nested", GS), ("naive_leaky", GS), ("sa_nested", G), ("sa_leaky", G)):
        if key.endswith("leaky"):
            if key not in S:
                S[key] = [int(v) for v in eg.leaky_selection_g(R, NX, pid, yy, grp, P)]
        else:
            S.setdefault(key, {})
            for p in P:
                if p not in S[key]:
                    trp = [q for q in P if q != p]
                    S[key][p] = sorted(int(v) for v in eg.rank_frequencies_g(R, NX, pid, yy, grp, trp)["index"].to_numpy()[:ep.TOP_M])
                    json.dump(S, open(path, "w"), indent=1)
        json.dump(S, open(path, "w"), indent=1)
        tick(f"  selection {cache_key}/{key} ready")
    return S


CKPT = f"{OUT}/revision_checkpoint.pkl"
if os.path.exists(CKPT):
    RES, TAB = pickle.load(open(CKPT, "rb"))
    print("resuming; completed sections:", RES.get("_done", []))


def todo(name):
    return name not in RES.get("_done", [])


def done(name):
    RES.setdefault("_done", []).append(name)
    pickle.dump((RES, TAB), open(CKPT, "wb"))


# =========================================================== R1 nested pipeline selection
INNER_SEL_PATH = f"{OUT}/selections_inner.json"
INNER_SEL = json.load(open(INNER_SEL_PATH)) if os.path.exists(INNER_SEL_PATH) else {}
INNER_HOW = {}


def rank_frequencies_lsoo(train_patients, seed=ep.SEED0):
    """Fallback ranking when patient-level inner validation is impossible (only one of the
    training patients has two or more healthy samples): leave-one-tissue-sample-out validation
    within the training patients, scored with the same criterion D(f) over the patients whose
    held-out predictions contain both classes."""
    rows = []
    for j in range(R.shape[1]):
        X = ep.build_vectors(R, NX, [j])
        A, M, ok = rc.fast_rep(X, pid, y, G, train_patients)
        pool = np.isin(pid, train_patients) & ok
        pred = np.full(len(y), np.nan)
        for smp in np.unique(G[pool]):
            te = pool & (G == smp)
            tr = pool & (G != smp)
            if len(np.unique(y[tr])) < 2:
                continue
            w = ep.patient_class_weights(pid[tr], y[tr])
            _, _, pf = ep.fit_predict(A[tr], M[tr], y[tr], w, A[te], M[te], seed)
            pred[te] = pf
        bas, aucs = [], []
        for q in train_patients:
            m = (pid == q) & np.isfinite(pred)
            if len(np.unique(y[m])) < 2:
                continue
            mm = ep.patient_metrics(y[m], pred[m])
            bas.append(mm["BA"]); aucs.append(mm["AUC"])
        ba = np.asarray(bas)
        sd = float(ba.std(ddof=1)) if ba.size > 1 else 0.0
        rows.append(dict(index=j, MeanBA=float(ba.mean()), WorstBA=float(ba.min()), SD=sd, MeanAUC=float(np.nanmean(aucs)),
                         Score=float(np.sqrt((1 - ba.mean()) ** 2 + (1 - ba.min()) ** 2 + sd ** 2))))
    return pd.DataFrame(rows).sort_values(["Score", "MeanBA", "WorstBA", "SD", "MeanAUC"],
                                          ascending=[True, False, False, True, False]).reset_index(drop=True)


def n_scorable(train_patients):
    """Number of training patients that can be scored by patient-level inner validation."""
    multi = [v for v in train_patients if len(np.unique(G[(pid == v) & (y == 0)])) >= 2]
    return len(multi) if len(multi) >= 2 else 0


def inner_selection(p, q):
    """Top-5 frequencies ranked on the outer training patients of fold p without inner test
    patient q (sample-wise referencing): by leave-one-patient-out when at least two of them have
    two or more healthy samples, otherwise by leave-one-tissue-sample-out; cached."""
    key = f"{p}|{q}"
    if key not in INNER_SEL:
        tr4 = [v for v in P if v not in (p, q)]
        if n_scorable(tr4):
            ranking, how = eg.rank_frequencies_g(R, NX, pid, y, G, tr4), "LOPO"
        else:
            ranking, how = rank_frequencies_lsoo(tr4), "LOSO"
        INNER_SEL[key] = sorted(int(v) for v in ranking["index"].to_numpy()[:ep.TOP_M])
        INNER_HOW[key] = how
        json.dump(INNER_SEL, open(INNER_SEL_PATH, "w"), indent=1)
        tick(f"  inner selection {key} ({INNER_HOW[key]}): {khz(INNER_SEL[key])} kHz")
    return INNER_SEL[key]


PARTIAL = "--partial-nesting" in sys.argv
if PARTIAL:
    RES, TAB = {}, {}

if PARTIAL or todo("R1"):
    tick("R1 nested pipeline selection ...")
    CANDS = [(rep, br) for rep in ("freq", "cole") for br in ("fusion", "abs", "mah")]   # original design first
    chosen, recs1, inner_tab = {}, [], []
    for p in P:
        trp = [q for q in P if q != p]
        feats = {"freq": ep.build_vectors(R, NX, SEL["sa_nested"][p]), "cole": THETA}
        scores = {}
        for rep in feats:
            parts = []
            for q in trp:
                # inner fold q: frequencies are re-ranked on the four inner-training patients only,
                # so q influences neither the frequencies, nor the classifier, nor the reference
                if rep == "freq" and PARTIAL:
                    Xq = feats["freq"]          # outer-fold ranking reused: q helped choose the frequencies
                elif rep == "freq":
                    Xq = ep.build_vectors(R, NX, inner_selection(p, q))
                else:
                    Xq = THETA
                A, M, ok = rc.fast_rep(Xq, pid, y, G, trp)
                parts.append(rc.run_eval(A, M, ok, pid, y, G, trp, seeds=[ep.SEED0], test_patients=[q]))
            r = pd.concat(parts, ignore_index=True)
            keep_in = sorted(set(G[np.isin(pid, trp)]) & set(r.Sample))
            for col, br in (("p", "fusion"), ("p_abs", "abs"), ("p_mah", "mah")):
                m = eg.pooled(eg.sample_table(r, col, keep_in))
                scores[(rep, br)] = (m["BA"], m["AUC"] if np.isfinite(m["AUC"]) else 0.0)
                inner_tab.append(dict(Fold=p, Representation=rep, Branch=br, InnerBA=m["BA"], InnerAUC=m["AUC"]))
        best = max(CANDS, key=lambda c: (round(scores[c][0], 10), round(scores[c][1], 10), -CANDS.index(c)))
        chosen[p] = dict(representation=best[0], branch=best[1], inner_BA=scores[best][0])
        X = feats[best[0]]
        A, M, ok = rc.fast_rep(X, pid, y, G, P)
        r = rc.run_eval(A, M, ok, pid, y, G, P, seeds=SEEDS, test_patients=[p])
        r["p_sel"] = r[{"fusion": "p", "abs": "p_abs", "mah": "p_mah"}[best[1]]]
        recs1.append(r)
    sfx = "_v2_outer_ranking" if PARTIAL else ""
    pd.DataFrame(inner_tab).to_csv(f"{OUT}/nested_pipeline_inner_scores{sfx}.csv", index=False)
    recs1 = pd.concat(recs1, ignore_index=True)
    summarize("nested_pipeline", recs1, EVAL, cols=(("p_sel", "selected"),))
    RES["nested_pipeline_choice"] = chosen
    if PARTIAL:
        ci, _, _ = eg.cluster_boot({"nested_pipeline|selected": TAB["nested_pipeline|selected"]})
        RES["ci"] = ci
        json.dump(json.loads(json.dumps(RES, default=float)), open(f"{OUT}/revision_results_v2_outer_ranking.json", "w"), indent=1)
        TAB["nested_pipeline|selected"].to_csv(f"{OUT}/sampletable_partial_pipeline_selected.csv", index=False)
        print("partial nesting:", RES["summary"]["nested_pipeline|selected"], ci)
        sys.exit(0)
    RES["inner_selected_khz"] = {k: khz(v) for k, v in INNER_SEL.items()}
    RES["inner_selection_method"] = {k: ("LOPO" if n_scorable([v for v in P if v not in k.split("|")]) else "LOSO") for k in INNER_SEL}
    print(chosen)
    done("R1")

# ======================================================= R2 reference-size-matched design
if todo("R2"):
    tick("R2 reference-size-matched referencing ...")
    for name, fx in (("matched_nested", Xsel(SEL["sa_nested"])), ("matched_cole", lambda p: THETA),
                     ("matched_published", lambda p: XPUB)):
        if name == "matched_nested":
            r = pd.concat([rc.run_eval_matched(fx(p), pid, y, G, P, SEEDS, test_patients=[p]) for p in P], ignore_index=True)
        else:
            r = rc.run_eval_matched(fx(None), pid, y, G, P, SEEDS)
        summarize(name, r, EVAL)

    # reference the main-analysis tables for the bootstrap
    for k in ("naive_nested|fusion", "naive_leaky|fusion", "sa_nested|fusion", "sa_leaky|fusion",
              "sa_cole|abs", "sa_cole|fusion", "sa_published|fusion", "sa_nested|abs", "sa_nested|mah"):
        TAB[k] = pd.read_csv(f"outputs_ijs/sampletable_{k.replace('|', '_')}.csv")
    boot(["nested_pipeline|selected", "sa_nested|fusion", "sa_cole|abs", "naive_nested|fusion",
          "matched_nested|fusion", "matched_nested|abs", "matched_nested|mah",
          "matched_cole|fusion", "matched_cole|abs", "matched_published|fusion", "sa_published|fusion"], "main_extra")
    done("R2")

# ======================================================= R3 P2 label swap
if todo("R3"):
    tick("R3 P2 label swap: selections ...")
    y2 = y.copy()
    y2[pid == "P2"] = 1 - y2[pid == "P2"]
    EVAL2 = rc.evaluable(pid, y2, G, P)
    ev2 = pd.DataFrame(dict(G=G, y=y2, pid=pid)).drop_duplicates("G").set_index("G").loc[EVAL2]
    RES["p2swap_evaluable"] = dict(n=len(EVAL2), healthy=int((ev2.y == 0).sum()), tumor=int((ev2.y == 1).sum()),
                                   healthy_patients=sorted(ev2[ev2.y == 0].pid.unique().tolist()))
    S2 = factorial_selections(y2, "p2swap")
    RES["p2swap_selected_khz"] = {k: ({p: khz(v) for p, v in s.items()} if isinstance(s, dict) else khz(s)) for k, s in S2.items()}
    tick("R3 P2 label swap: evaluations ...")
    for name, fx, grp in (("p2_naive_nested", Xsel(S2["naive_nested"]), GS), ("p2_naive_leaky", Xsel(S2["naive_leaky"]), GS),
                          ("p2_sa_nested", Xsel(S2["sa_nested"]), G), ("p2_sa_leaky", Xsel(S2["sa_leaky"]), G),
                          ("p2_sa_cole", lambda p: THETA, G), ("p2_sa_published", lambda p: XPUB, G)):
        summarize(name, eg.evaluate_g(fx, pid, y2, grp, P, G), EVAL2)
    boot(["p2_naive_leaky|fusion", "p2_naive_nested|fusion", "p2_sa_leaky|fusion", "p2_sa_nested|fusion",
          "p2_sa_cole|abs", "p2_sa_published|fusion"], "p2swap")
    done("R3")

# ======================================================= R4 logistic-regression backend
if todo("R4"):
    tick("R4 logistic regression: selections ...")
    with rc.backend("logit"):
        S3 = factorial_selections(y, "logit")
        RES["logit_selected_khz"] = {k: ({p: khz(v) for p, v in s.items()} if isinstance(s, dict) else khz(s)) for k, s in S3.items()}
        tick("R4 logistic regression: evaluations ...")
        for name, fx, grp in (("lr_naive_nested", Xsel(S3["naive_nested"]), GS), ("lr_naive_leaky", Xsel(S3["naive_leaky"]), GS),
                              ("lr_sa_nested", Xsel(S3["sa_nested"]), G), ("lr_sa_leaky", Xsel(S3["sa_leaky"]), G),
                              ("lr_naive_cole", lambda p: THETA, GS), ("lr_sa_cole", lambda p: THETA, G),
                              ("lr_sa_published", lambda p: XPUB, G)):
            summarize(name, eg.evaluate_g(fx, pid, y, grp, P, G, n_repeats=1), EVAL)
    boot(["lr_naive_leaky|fusion", "lr_naive_nested|fusion", "lr_sa_leaky|fusion", "lr_sa_nested|fusion",
          "lr_naive_cole|abs", "lr_sa_cole|abs", "lr_naive_cole|fusion", "lr_sa_cole|fusion", "lr_sa_published|fusion"], "logit")
    done("R4")


# ======================================================= save
def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return o


json.dump(clean(RES), open(f"{OUT}/revision_results.json", "w"), indent=1, default=float)
pd.DataFrame([dict(Config=k, **v) for k, v in RES["summary"].items()]).to_csv(f"{OUT}/revision_summary.csv", index=False)
for k, st in TAB.items():
    st.to_csv(f"{OUT}/sampletable_{k.replace('|', '_')}.csv", index=False)
for k, s in RES["summary"].items():
    ci = RES["ci"].get(k)
    print(f"{k:32s} BA={s['BA']:.3f} ({s['TP']}/{s['N1']}, {s['TN']}/{s['N0']}) AUC={s['AUC']:.3f}"
          + (f"  CI {ci[0]:.3f}-{ci[1]:.3f}" if ci else ""))
tick("done")

"""
ijs_analysis.py -- all analyses of the revised manuscript (Iranian Journal of Science).

Design: 2 x 2 factorial of
    referencing  : spectrum-wise (naive leave-one-spectrum-out) vs
                   sample-wise (leave-one-tissue-sample-out)
    selection    : leaky (ranking on all six patients) vs nested (ranking inside each outer fold)
evaluated by patient-referenced leave-one-patient-out validation, plus comparator frequency
sets, Cole-parameter features, sensitivity analyses, a variance decomposition of repeated
spectra, and acquisition-order drift. Primary unit of analysis: the tissue sample.
"""
import json
import os
import time

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import least_squares
from sklearn.covariance import LedoitWolf

import eis_pipeline as ep
import eis_groups as eg
from cole_analysis import iterate_spectra, fit_cole, cole, FMIN, FMAX

OUT = "outputs_ijs"
os.makedirs(OUT, exist_ok=True)
T0 = time.time()


def tick(m):
    print(f"[{time.time() - T0:6.0f}s] {m}", flush=True)


raw = ep.load_raw_dataset("electrical-impedance-data-bladder/dataset_without_modules.pickle")
GRID = ep.candidate_grid()
R, NX, pid, y, meta = ep.harmonize(raw, GRID)
P = [f"P{i}" for i in range(1, 7)]
G = eg.sample_ids(raw)                 # tissue samples
GS = eg.spectrum_groups(len(y))        # every spectrum its own group
khz = lambda idx: [round(GRID[j] / 1e3, 3) for j in idx]
RES = {}

# ------------------------------------------------------------------ dataset structure
df = pd.DataFrame(dict(Patient=pid, y=y, G=G))
t1 = []
for p in P:
    d = df[df.Patient == p]
    t1.append(dict(Patient=p,
                   HealthySamples=int(d[d.y == 0].G.nunique()), HealthySpectra=int((d.y == 0).sum()),
                   TumorSamples=int(d[d.y == 1].G.nunique()), TumorSpectra=int((d.y == 1).sum()),
                   FrequencyPoints=meta.set_index("Patient").loc[p, "FrequencyPoints"]))
T1 = pd.DataFrame(t1)
T1.to_csv(f"{OUT}/Table1_structure.csv", index=False)
print(T1)
spp = df.groupby("G").size()
RES["structure"] = dict(n_spectra=int(len(y)), n_samples=int(df.G.nunique()),
                        n_healthy_samples=int(df[df.y == 0].G.nunique()),
                        n_tumor_samples=int(df[df.y == 1].G.nunique()),
                        spectra_per_sample_median=float(spp.median()), spectra_per_sample_min=int(spp.min()),
                        spectra_per_sample_max=int(spp.max()))

_, _, ok_any = eg.build_rep_g(R[:, :1], pid, y, G, P)
EVAL = sorted(set(G[ok_any]))
ev = df[df.G.isin(EVAL)].drop_duplicates("G")
RES["evaluable"] = dict(n=len(EVAL), healthy=int((ev.y == 0).sum()), tumor=int((ev.y == 1).sum()),
                        healthy_patients=sorted(ev[ev.y == 0].Patient.unique().tolist()))
print("evaluable samples:", RES["evaluable"])

# ------------------------------------------------------------------ frequency selection
SEL_CACHE = f"{OUT}/selections_idx.json"
if os.path.exists(SEL_CACHE):
    SEL = json.load(open(SEL_CACHE))
    print("selections loaded from cache")
else:
    tick("nested ranking, spectrum-wise referencing ...")
    SEL = {}
    SEL["naive_nested"] = eg.nested_selection_g(R, NX, pid, y, GS, P)
    tick("leaky ranking, spectrum-wise referencing ...")
    SEL["naive_leaky"] = eg.leaky_selection_g(R, NX, pid, y, GS, P)
    tick("nested ranking, sample-wise referencing ...")
    SEL["sa_nested"] = eg.nested_selection_g(R, NX, pid, y, G, P)
    tick("leaky ranking, sample-wise referencing ...")
    SEL["sa_leaky"] = eg.leaky_selection_g(R, NX, pid, y, G, P)
    tick("nested ranking without P2 (sensitivity) ...")
    SEL["sa_nested_noP2"] = eg.nested_selection_g(R, NX, pid, y, G, [p for p in P if p != "P2"])
    SEL = json.loads(json.dumps(SEL, default=int))
    json.dump(SEL, open(SEL_CACHE, "w"), indent=1)
RES["selected_khz"] = {k: ({p: khz(v) for p, v in s.items()} if isinstance(s, dict) else khz(s)) for k, s in SEL.items()}
CACHE_FITS = f"{OUT}/cole_fits.csv"
print(json.dumps(RES["selected_khz"], indent=1))

VEIL_HZ = np.array([3.55e3, 568.99e3, 655.13e3, 754.31e3, 868.51e3])
Rv, NXv, *_ = ep.harmonize(raw, VEIL_HZ)
GEN = sorted({ep.nearest_grid_index(GRID, f) for f in ep.FIXED_FIVE_HZ})

Xsel = lambda s: (lambda p: ep.build_vectors(R, NX, s[p] if isinstance(s, dict) else s))

# ------------------------------------------------------------------ Cole fits
tick("fitting the Cole model ...")
specs = list(iterate_spectra(raw))
rows = []
for (p, lab, f, z), p2, y2 in zip(specs, pid, y):
    assert p == p2 and lab == y2
    d = fit_cole(f, z)
    d.update(Patient=p, Label=lab)
    rows.append(d)
fits = pd.DataFrame(rows)
fits["Sample"] = G
fits.to_csv(f"{OUT}/cole_fits.csv", index=False)
THETA = np.column_stack([np.log(fits.R0), np.log(fits.Rinf), np.log10(fits.fc), fits.alpha])
q = 100 * fits.rel_rmse.quantile([0.5, 0.95, 1.0]).values
RES["cole_fit"] = dict(rmse_median=q[0], rmse_p95=q[1], rmse_max=q[2])

# identifiability: f_c outside the fitted window and Jacobian standard errors
se = []
for r, (_, _, f, z) in zip(fits.itertuples(), specs):
    m = (f >= FMIN * .999) & (f <= FMAX * 1.001)
    f, z = f[m], z[m]
    w, sc = 2 * np.pi * f, np.abs(z)
    res = lambda prm: np.concatenate([((cole(prm, w) - z) / sc).real, ((cole(prm, w) - z) / sc).imag])
    o = least_squares(res, x0=[r.Rinf, r.R0 - r.Rinf, np.log10(1 / (2 * np.pi * r.fc)), r.alpha],
                      bounds=([1e-6, 1e-6, -9, .1], [1e4, 1e4, -1, 1]), max_nfev=200)
    cov = np.linalg.pinv(o.jac.T @ o.jac) * (2 * o.cost / max(o.jac.shape[0] - 4, 1))
    se.append(np.sqrt(np.clip(np.diag(cov), 0, None)))
se = np.asarray(se)
outside = ((fits.fc < 1e3) | (fits.fc > 1e6)).values
RES["cole_ident"] = dict(outside_pct=float(100 * outside.mean()), outside_n=int(outside.sum()),
                         outside_by={f"{a}-{'H' if b == 0 else 'T'}": int(v) for (a, b), v in fits[outside].groupby(["Patient", "Label"]).size().items()},
                         se_log10fc_median=float(np.median(se[:, 2])), se_alpha_median=float(np.median(se[:, 3])))
print("Cole identifiability:", RES["cole_ident"])

# ------------------------------------------------------------------ evaluations
CONF = {
    # name: (feature function, referencing groups, weights, weighting, patients)
    "naive_nested": (Xsel(SEL["naive_nested"]), GS),
    "naive_leaky": (Xsel(SEL["naive_leaky"]), GS),
    "sa_nested": (Xsel(SEL["sa_nested"]), G),
    "sa_leaky": (Xsel(SEL["sa_leaky"]), G),
    "sa_published": (lambda p: ep.build_vectors(Rv, NXv, [0, 1, 2, 3, 4]), G),
    "sa_generic": (Xsel(GEN), G),
    "naive_cole": (lambda p: THETA, GS),
    "sa_cole": (lambda p: THETA, G),
}
RECS, SUM, TAB = {}, {}, {}
for name, (fx, grp) in CONF.items():
    tick(f"evaluate {name} ...")
    RECS[name] = eg.evaluate_g(fx, pid, y, grp, P, G)
    for col, tag in (("p", "fusion"), ("p_abs", "abs"), ("p_mah", "mah")):
        s, st = eg.summary(RECS[name], col, keep=EVAL)
        SUM[f"{name}|{tag}"], TAB[f"{name}|{tag}"] = s, st

tick("inner-fold fusion weights (sample-wise) ...")
W = {}
for name in ("sa_nested", "sa_cole"):
    fx, grp = CONF[name]
    W[name] = eg.inner_weights_g(fx, pid, y, grp, G, P)
    r = eg.evaluate_g(fx, pid, y, grp, P, G, weights=W[name])
    SUM[f"{name}|innerw"], TAB[f"{name}|innerw"] = eg.summary(r, "p", keep=EVAL)
RES["inner_weights"] = W

# bridge to the originally reported metric (spectrum-wise, all 767 spectra, mean patient BA)
bridge = []
for _, g in RECS["naive_nested"].groupby("Repeat"):
    bridge.append(np.mean([ep.patient_metrics(h.y.values, h.p.values)["BA"] for _, h in g.groupby("Patient")]))
s_all, _ = eg.summary(RECS["naive_nested"], "p", keep=None)
RES["bridge"] = dict(original_metric=float(np.mean(bridge)), naive_all29=s_all)

# ------------------------------------------------------------------ sensitivity analyses
tick("sensitivity: class-balanced weighting ...")
for name in ("sa_nested", "sa_cole"):
    fx, grp = CONF[name]
    r = eg.evaluate_g(fx, pid, y, grp, P, G, weighting="class_balanced")
    for col, tag in (("p", "fusion"), ("p_abs", "abs")):
        SUM[f"{name}|{tag}|cb"], TAB[f"{name}|{tag}|cb"] = eg.summary(r, col, keep=EVAL)

tick("sensitivity: Cole resistances only ...")
TH2 = THETA[:, :2]
r = eg.evaluate_g(lambda p: TH2, pid, y, G, P, G)
for col, tag in (("p", "fusion"), ("p_abs", "abs"), ("p_mah", "mah")):
    SUM[f"sa_coleR|{tag}"], TAB[f"sa_coleR|{tag}"] = eg.summary(r, col, keep=EVAL)

tick("sensitivity: without P2 ...")
P5 = [p for p in P if p != "P2"]
keep2 = np.asarray([p != "P2" for p in pid])
EVAL_noP2 = [s for s in EVAL if not s.startswith("P2:")]
sel5 = SEL["sa_nested_noP2"]
RES["selected_khz"]["sa_nested_noP2"] = {p: khz(v) for p, v in sel5.items()}
for name, fx, grp in (("noP2_sa_nested", Xsel(sel5), G), ("noP2_sa_cole", lambda p: THETA, G),
                      ("noP2_naive_nested", Xsel({p: SEL["naive_nested"][p] for p in P5}), GS)):
    r = eg.evaluate_g(fx, pid, y, grp, P5, G)
    for col, tag in (("p", "fusion"), ("p_abs", "abs")):
        SUM[f"{name}|{tag}"], TAB[f"{name}|{tag}"] = eg.summary(r, col, keep=EVAL_noP2)

# ------------------------------------------------------------------ bootstrap
tick("patient-level cluster bootstrap ...")
main = ["naive_leaky|fusion", "naive_nested|fusion", "sa_leaky|fusion", "sa_nested|fusion",
        "sa_published|fusion", "sa_generic|fusion",
        "naive_nested|abs", "naive_nested|mah", "sa_nested|abs", "sa_nested|mah",
        "naive_cole|fusion", "naive_cole|abs", "naive_cole|mah",
        "sa_cole|fusion", "sa_cole|abs", "sa_cole|mah", "sa_cole|innerw", "sa_nested|innerw"]
CI, DIFF, NB = eg.cluster_boot({k: TAB[k] for k in main})
RES["boot_valid"] = NB

# ------------------------------------------------------------------ per-patient outcomes
pp = []
for k in ("naive_nested|fusion", "sa_nested|fusion", "sa_published|fusion", "sa_cole|abs", "sa_cole|innerw"):
    st = TAB[k]
    pred = (st.p >= .5).astype(int)
    for p in P:
        m = st.Patient == p
        pp.append(dict(Config=k, Patient=p,
                       TumorCorrect=int(((pred == 1) & (st.y == 1) & m).sum()), Tumor=int(((st.y == 1) & m).sum()),
                       HealthyCorrect=int(((pred == 0) & (st.y == 0) & m).sum()), Healthy=int(((st.y == 0) & m).sum())))
PP = pd.DataFrame(pp)
PP.to_csv(f"{OUT}/per_patient_samples.csv", index=False)

# ------------------------------------------------------------------ variance decomposition
tick("within- versus between-sample structure ...")
dist = []
for p in P:
    h = np.flatnonzero((pid == p) & (y == 0))
    samples = sorted(set(G[h]))
    for i in np.flatnonzero(pid == p):
        own = h[(G[h] == G[i]) & (h != i)]
        other = h[G[h] != G[i]]
        if y[i] == 0 and len(samples) >= 2 and own.size >= 2 and other.size >= 2:
            for kind, pool in (("within", own), ("between", other)):
                lw = LedoitWolf().fit(THETA[pool])
                dd = THETA[i] - lw.location_
                dist.append(dict(Patient=p, kind=kind, d=float(np.sqrt(dd @ lw.precision_ @ dd))))
        if y[i] == 1 and h.size >= 2:
            lw = LedoitWolf().fit(THETA[h])
            dd = THETA[i] - lw.location_
            dist.append(dict(Patient=p, kind="tumor", d=float(np.sqrt(dd @ lw.precision_ @ dd))))
DIST = pd.DataFrame(dist)
DIST.to_csv(f"{OUT}/mahalanobis_within_between.csv", index=False)
dmed = DIST.groupby("kind").d.median()
RES["distances"] = dict(within=float(dmed["within"]), between=float(dmed["between"]), tumor=float(dmed["tumor"]),
                        patients_with_between=sorted(DIST[DIST.kind == "between"].Patient.unique().tolist()))
print("median Mahalanobis distance:", RES["distances"])


def icc1(values, groups):
    """One-way random-effects ICC(1) with unequal group sizes."""
    dfv = pd.DataFrame(dict(v=values, g=groups))
    k = dfv.g.nunique(); N = len(dfv)
    if k < 2:
        return np.nan
    gm = dfv.groupby("g").v.agg(["mean", "size"])
    grand = dfv.v.mean()
    msb = (gm["size"] * (gm["mean"] - grand) ** 2).sum() / (k - 1)
    msw = ((dfv.v - dfv.g.map(gm["mean"])) ** 2).sum() / (N - k)
    n0 = (N - (gm["size"] ** 2).sum() / N) / (k - 1)
    return float((msb - msw) / (msb + (n0 - 1) * msw))


icc = {}
names = ["ln R0", "ln Rinf", "log10 fc", "alpha"]
for c, lab in ((0, "healthy"), (1, "tumor")):
    for j, nm in enumerate(names):
        vals, grps = [], []
        for p in P:
            m = (pid == p) & (y == c)
            if len(set(G[m])) >= 2:
                vals.extend(THETA[m, j] - THETA[m, j].mean())
                grps.extend(G[m])
        icc[f"{lab}|{nm}"] = icc1(np.asarray(vals), np.asarray(grps))
m_bar = float(spp.mean())
RES["icc"] = icc
RES["design_effect"] = dict(m_bar=m_bar,
                            deff_healthy_lnR0=1 + (m_bar - 1) * icc["healthy|ln R0"],
                            n_eff=float(len(y) / (1 + (m_bar - 1) * icc["healthy|ln R0"])))
print("ICC:", {k: round(v, 3) for k, v in icc.items()})

# ------------------------------------------------------------------ acquisition-order drift
rh = []
for s_, g in fits.groupby("Sample", sort=False):
    if len(g) >= 6:
        idx = np.arange(len(g))
        k = max(1, len(g) // 5)
        rh.append(dict(Sample=s_, n=len(g), rho=stats.spearmanr(idx, np.log(g.R0)).statistic,
                       drift=float(np.log(g.R0.iloc[-k:]).median() - np.log(g.R0.iloc[:k]).median())))
RH = pd.DataFrame(rh)
RH.to_csv(f"{OUT}/drift_per_sample.csv", index=False)
medR0 = fits.groupby(["Patient", "Label"]).R0.median().unstack()
RES["drift"] = dict(n_samples=int(len(RH)), median_abs_rho=float(RH.rho.abs().median()),
                    frac_abs_rho_gt_08=float((RH.rho.abs() > 0.8).mean()),
                    median_abs_drift_lnR0=float(RH.drift.abs().median()),
                    median_abs_contrast_lnR0=float(np.abs(np.log(medR0[1] / medR0[0])).median()))
print("drift:", RES["drift"])

# ------------------------------------------------------------------ ICC of ln|Z| by frequency
L = np.log(np.hypot(R, NX))
icc_f = []
for j, fr in enumerate(GRID):
    vals, grps = [], []
    for p in ("P1", "P4", "P5"):
        h = (pid == p) & (y == 0)
        vals += list(L[h, j] - L[h, j].mean()); grps += list(G[h])
    icc_f.append(dict(f_kHz=fr / 1e3, ICC=icc1(np.asarray(vals), np.asarray(grps))))
ICCF = pd.DataFrame(icc_f)
ICCF.to_csv(f"{OUT}/frequency_icc.csv", index=False)
RES["icc_freq"] = dict(first=float(ICCF.ICC.iloc[0]), last=float(ICCF.ICC.iloc[-1]),
                       monotone=bool(np.all(np.diff(ICCF.ICC.values) < 0)))
print("ICC of ln|Z|: 1 kHz", round(RES["icc_freq"]["first"], 3), "| 1 MHz", round(RES["icc_freq"]["last"], 3),
      "| monotone decreasing:", RES["icc_freq"]["monotone"])

# ------------------------------------------------------------------ misclassified samples (main design)
st = TAB["sa_nested|fusion"].copy()
st["n"] = st.Sample.map(fits.groupby("Sample").size())
st["ok"] = (st.p >= .5).astype(int) == st.y
p1t = st[(st.Patient == "P1") & (st.y == 1)].p
p1h_wrong = st[(st.Patient == "P1") & (st.y == 0) & (~st.ok)].p
RES["misclassified"] = dict(n_min=int(st[~st.ok].n.min()), n_max=int(st[~st.ok].n.max()),
                            mean_n_correct=float(st[st.ok].n.mean()), n_correct=int(st.ok.sum()),
                            p1_tumor_min=float(p1t.min()), p1_tumor_max=float(p1t.max()),
                            p1_healthy_wrong=float(p1h_wrong.iloc[0]) if len(p1h_wrong) else None)
print("misclassified:", RES["misclassified"])

# ------------------------------------------------------------------ save
def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return o


RES["summary"] = SUM
RES["ci"] = CI
RES["diff"] = DIFF
json.dump(clean(RES), open(f"{OUT}/ijs_results.json", "w"), indent=1, default=float)
pd.DataFrame([dict(Config=k, **{m: v for m, v in s.items()}) for k, s in SUM.items()]).to_csv(f"{OUT}/summary_all.csv", index=False)
for k, st in TAB.items():
    st.to_csv(f"{OUT}/sampletable_{k.replace('|', '_')}.csv", index=False)
tick("done")

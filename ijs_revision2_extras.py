"""Additional analyses requested in the second internal review (no new model fitting).

1. 2x2 interaction of referencing design and frequency selection, with the paired patient-level cluster bootstrap.
2. Patient-level summaries: per-patient sensitivity and specificity, macro-averaged over patients.
3. Acquisition-order drift versus tissue class (is drift a class-linked confounder?).
4. Exact Clopper-Pearson bound for the selection-aware permutation p-value.
5. Sample-level BA of all size-matched configurations (for Table 3 of the article).

Inputs: per-sample prediction tables and drift/Cole outputs of the main pipeline (folder `results/` of the package,
or `outputs_ijs/` when run inside the notebook). Output: JSON with every value printed in the revised article.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

RES = sys.argv[1] if len(sys.argv) > 1 else "outputs_ijs"
REV = f"{RES}/revision"
SEED0 = 20260921
out = {}


def st(path):
    return pd.read_csv(path)


def pooled_ba(t, thr=0.5):
    pred = (t.p >= thr).astype(int)
    sens = ((pred == 1) & (t.y == 1)).sum() / (t.y == 1).sum()
    spec = ((pred == 0) & (t.y == 0)).sum() / (t.y == 0).sum()
    return 0.5 * (sens + spec)


def counts(t, pats, thr=0.5):
    pred = (t.p >= thr).astype(int)
    c = np.zeros((len(pats), 4))
    for k, q in enumerate(pats):
        m = (t.Patient == q).values
        c[k] = [((pred == 1) & (t.y == 1) & m).sum(), ((t.y == 1) & m).sum(),
                ((pred == 0) & (t.y == 0) & m).sum(), ((t.y == 0) & m).sum()]
    return c


def boot_draws(tables, B=10000, seed=SEED0):
    """Identical resampling scheme to eis_groups.cluster_boot (same seed, same discarding rule)."""
    names = list(tables)
    pats = np.asarray(sorted(tables[names[0]].Patient.unique()))
    C = {n: counts(tables[n], pats) for n in names}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(pats), size=(B, len(pats)))
    D = {}
    for n in names:
        s = C[n][idx].sum(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            D[n] = 0.5 * (s[:, 0] / s[:, 1] + s[:, 2] / s[:, 3])
    valid = np.all([np.isfinite(v) for v in D.values()], axis=0)
    return {n: v[valid] for n, v in D.items()}, int(valid.sum())


# ------------------------------------------------------------------ 1. interaction in the 2x2 factorial design
T = {"SW_leaky": st(f"{RES}/sampletable_naive_leaky_fusion.csv"), "SW_nested": st(f"{RES}/sampletable_naive_nested_fusion.csv"),
     "SA_leaky": st(f"{RES}/sampletable_sa_leaky_fusion.csv"), "SA_nested": st(f"{RES}/sampletable_sa_nested_fusion.csv")}
ba = {k: pooled_ba(v) for k, v in T.items()}
D, nvalid = boot_draws(T)
inter = (ba["SW_leaky"] - ba["SW_nested"]) - (ba["SA_leaky"] - ba["SA_nested"])
inter_d = (D["SW_leaky"] - D["SW_nested"]) - (D["SA_leaky"] - D["SA_nested"])
ref_nested = D["SW_nested"] - D["SA_nested"]
out["factorial"] = {
    "BA": ba, "valid_resamples": nvalid,
    "referencing_effect_nested_CI_check": [float(np.percentile(ref_nested, 2.5)), float(np.percentile(ref_nested, 97.5))],
    "interaction": inter, "interaction_CI": [float(np.percentile(inter_d, 2.5)), float(np.percentile(inter_d, 97.5))],
    "sum_of_main_effects": (ba["SW_nested"] - ba["SA_nested"]) + (ba["SA_leaky"] - ba["SA_nested"]),
    "combined_effect": ba["SW_leaky"] - ba["SA_nested"]}

# ------------------------------------------------------------------ 2. patient-level summaries
def patient_level(t):
    rows = []
    for q, g in t.groupby("Patient"):
        pred = (g.p >= 0.5).astype(int)
        nt, nh = int((g.y == 1).sum()), int((g.y == 0).sum())
        rows.append(dict(Patient=q, sens=((pred == 1) & (g.y == 1)).sum() / nt if nt else np.nan,
                         spec=((pred == 0) & (g.y == 0)).sum() / nh if nh else np.nan,
                         all_correct=bool((pred == g.y).all())))
    r = pd.DataFrame(rows)
    both = r.dropna()
    return dict(macro_sens=float(r.sens.mean()), macro_spec=float(r.spec.mean()),
                patient_BA=float(0.5 * (r.sens.mean() + r.spec.mean())),
                n_patients_all_correct=int(r.all_correct.sum()), n_patients=int(len(r)),
                n_patients_both_classes=int(len(both)),
                per_patient=r.to_dict(orient="records"))


PL = {"spectrum-wise, leaky (naive)": f"{RES}/sampletable_naive_leaky_fusion.csv",
      "pre-specified (sample-wise, nested)": f"{RES}/sampletable_sa_nested_fusion.csv",
      "Cole absolute, sample-wise": f"{RES}/sampletable_sa_cole_abs.csv",
      "fully nested pipeline": f"{REV}/sampletable_nested_pipeline_selected.csv",
      "fully nested, size-matched": f"{REV}/sampletable_nested_matched.csv",
      "pre-specified, size-matched": f"{REV}/sampletable_matched_nested_fusion.csv"}
out["patient_level"] = {k: patient_level(st(v)) for k, v in PL.items() if os.path.exists(v)}

# ------------------------------------------------------------------ 3. drift versus class
dr = pd.read_csv(f"{RES}/drift_per_sample.csv")
cf = pd.read_csv(f"{RES}/cole_fits.csv")
lab = cf.groupby("Sample").Label.first()
dr["Label"] = dr.Sample.map(lab)
h, t_ = dr[dr.Label == 0], dr[dr.Label == 1]
mw_rho = stats.mannwhitneyu(h.rho.abs(), t_.rho.abs())
mw_drift = stats.mannwhitneyu(h.drift.abs(), t_.drift.abs())
# per-patient tumour-healthy contrast of ln R0 from the first and from the last fifth of every series
cf["lnR0"] = np.log(cf.R0)
cf["pos"] = cf.groupby("Sample").cumcount()
cf["n"] = cf.groupby("Sample").Sample.transform("size")
first = cf[cf.pos < np.ceil(cf.n / 5)]
last = cf[cf.pos >= cf.n - np.ceil(cf.n / 5)]


def contrast(d):
    s = d.groupby(["Patient", "Sample", "Label"]).lnR0.mean().reset_index()
    m = s.groupby(["Patient", "Label"]).lnR0.mean().unstack()
    return (m[1] - m[0]).dropna()


cF, cL = contrast(first), contrast(last)
out["drift_vs_class"] = {
    "n_healthy_samples": int(len(h)), "n_tumor_samples": int(len(t_)),
    "median_abs_rho": {"healthy": float(h.rho.abs().median()), "tumor": float(t_.rho.abs().median())},
    "median_abs_drift_lnR0": {"healthy": float(h.drift.abs().median()), "tumor": float(t_.drift.abs().median())},
    "share_positive_drift": {"healthy": float((h.drift > 0).mean()), "tumor": float((t_.drift > 0).mean())},
    "mannwhitney_p_abs_rho": float(mw_rho.pvalue), "mannwhitney_p_abs_drift": float(mw_drift.pvalue),
    "contrast_first_fifth": cF.to_dict(), "contrast_last_fifth": cL.to_dict(),
    "max_abs_change_in_contrast": float((cF - cL).abs().max()),
    "sign_agreement": int((np.sign(cF) == np.sign(cL)).sum()), "n_patients": int(len(cF))}

# ------------------------------------------------------------------ 4. selection-aware permutation p-value
B, k = 100, 0
out["selection_aware_p"] = {"B": B, "exceed": k, "p_hat": (1 + k) / (1 + B),
                            "clopper_pearson_upper_95": float(stats.beta.ppf(0.95, k + 1, B - k)),
                            "clopper_pearson_two_sided_upper": float(stats.beta.ppf(0.975, k + 1, B - k))}

# ------------------------------------------------------------------ 5. size-matched configurations
out["size_matched_BA"] = {k: pooled_ba(st(f"{REV}/sampletable_matched_{k}.csv")) for k in
                          ("nested_abs", "nested_mah", "nested_fusion", "cole_abs", "cole_mah", "cole_fusion",
                           "published_abs", "published_mah", "published_fusion")}

json.dump(out, open(f"{REV}/revision2_extras.json", "w"), indent=1, default=float)
print(json.dumps({k: v for k, v in out.items() if k != "patient_level"}, indent=1, default=float))
for k, v in out["patient_level"].items():
    print(k, {a: b for a, b in v.items() if a != "per_patient"})

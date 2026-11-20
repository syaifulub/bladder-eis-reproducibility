"""
eis_pipeline.py
================================================================================
Distribution-Aware Patient-Referenced Electrochemical Impedance Sensing with
Absolute-Mahalanobis Fusion for Bladder Tumor Classification.

Reference implementation of the complete computational workflow described in
the manuscript (Sections "Materials and Methods").

Source dataset
--------------
Veil, Krauss, Amend, Fend & Sawodny (2025), Scientific Reports 15, 825.
doi:10.1038/s41598-024-84844-9
Repository: https://github.com/cayobro/electrical-impedance-data-bladder

Everything in this module is deterministic given SEED0.
================================================================================
"""

from __future__ import annotations

import json
import os
import pickle
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.covariance import LedoitWolf
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    roc_auc_score,
)

warnings.filterwarnings("ignore")

# ==============================================================================
# 1. GLOBAL CONFIGURATION  (manuscript Section "Implementation and Reproducibility")
# ==============================================================================

SEED0 = 20260921          # global reproducibility seed s_0
N_REPEATS = 30            # repeated classifier fits / reference subsamples / perturbations
N_FREQ_GRID = 50          # candidate frequencies
F_MIN, F_MAX = 1e3, 1e6   # 1 kHz - 1 MHz
TOP_M = 5                 # retained frequencies per outer fold
FUSION_W = 0.5            # equal-probability fusion
THRESHOLD = 0.5           # fixed decision threshold

REFERENCE_BUDGETS = [1, 3, 5, 10, "All"]
SIGMA_LEVELS = [0.0, 0.01, 0.025, 0.05, 0.10]

# Prespecified XGBoost parameters (manuscript Table 2)
XGB_PARAMS = dict(
    n_estimators=180,
    max_depth=3,
    learning_rate=0.04,
    min_child_weight=3,
    subsample=0.85,
    colsample_bytree=0.85,
    gamma=0.0,
    reg_alpha=0.10,
    reg_lambda=2.00,
    objective="binary:logistic",
    eval_metric="logloss",
    tree_method="hist",
    n_jobs=0,
    verbosity=0,
)

# ------------------------------------------------------------------------------
# Generic decade-spaced baseline ("Generic decade-spaced five" in the paper).
# Snapped to the nearest candidate-grid frequencies (1.000, 9.541, 104.811,
# 494.171, 1000.000 kHz). The configuration published by Veil et al. (2025)
# is evaluated separately in extra_analyses.py (VEIL_HZ), interpolated exactly.
# Values in Hz.
# ------------------------------------------------------------------------------
FIXED_FIVE_HZ = [1.0e3, 1.0e4, 1.0e5, 5.0e5, 1.0e6]


def candidate_grid(n=N_FREQ_GRID, fmin=F_MIN, fmax=F_MAX):
    """Eq. (1): 50 logarithmically spaced frequencies between 1 kHz and 1 MHz."""
    return np.logspace(np.log10(fmin), np.log10(fmax), n)


# ==============================================================================
# 2. CLASSIFIER BACKEND
# ==============================================================================

def make_classifier(seed):
    """Return a fresh XGBoost classifier with the prespecified configuration."""
    from xgboost import XGBClassifier

    return XGBClassifier(random_state=int(seed), **XGB_PARAMS)


# ==============================================================================
# 3. DATA LOADING AND SPECTRAL HARMONIZATION
# ==============================================================================

def _class_label(key: str) -> int:
    """Map a dataset class key to 0 = healthy, 1 = tumor."""
    k = key.lower()
    if "health" in k or k in {"h", "normal", "benign"}:
        return 0
    return 1


def load_raw_dataset(pickle_path: str) -> dict:
    """Load the published pickle (patient-wise dictionary of measurements)."""
    with open(pickle_path, "rb") as fh:
        return pickle.load(fh)


def _measurement_arrays(measurement):
    """Extract (frequency, complex impedance) from either dict or object records."""
    if isinstance(measurement, dict):
        f = np.asarray(measurement["frequency"], dtype=float)
        z = np.asarray(measurement["z"])
    else:
        f = np.asarray(getattr(measurement, "frequency"), dtype=float)
        z = np.asarray(getattr(measurement, "z"))
    return f.ravel(), np.asarray(z, dtype=complex).ravel()


def harmonize(raw: dict, grid=None):
    """
    Spectral harmonization (manuscript Section 2.2).

    Real and negative-imaginary components are interpolated independently onto
    the common grid, linearly in the log10-frequency coordinate.

    Returns
    -------
    R    : (N, 50) real part
    NX   : (N, 50) negative imaginary part
    pid  : (N,)    patient label, e.g. 'P1'
    y    : (N,)    0 = healthy, 1 = tumor
    meta : DataFrame with one row per patient (original resolution, counts)
    """
    if grid is None:
        grid = candidate_grid()
    log_grid = np.log10(grid)

    R_rows, NX_rows, pid_rows, y_rows = [], [], [], []
    meta_rows = []

    patient_keys = sorted(raw.keys(), key=lambda s: (len(s), s))
    for p_index, pkey in enumerate(patient_keys, start=1):
        pname = f"P{p_index}"
        n_points = set()
        counts = {0: 0, 1: 0}

        for ckey, measurements in raw[pkey].items():
            # The public repository also stores a free-text "info" entry per
            # patient; only list-valued entries are measurement records.
            if not isinstance(measurements, (list, tuple)):
                continue
            label = _class_label(ckey)
            for meas in measurements:
                f, z = _measurement_arrays(meas)
                order = np.argsort(f)
                f, z = f[order], z[order]
                n_points.add(int(f.size))

                logf = np.log10(f)
                r = np.interp(log_grid, logf, np.real(z))
                nx = np.interp(log_grid, logf, -np.imag(z))

                R_rows.append(r)
                NX_rows.append(nx)
                pid_rows.append(pname)
                y_rows.append(label)
                counts[label] += 1

        meta_rows.append(
            dict(
                Patient=pname,
                SourceKey=pkey,
                Healthy=counts[0],
                Tumor=counts[1],
                Total=counts[0] + counts[1],
                FrequencyPoints=sorted(n_points)[0] if len(n_points) == 1
                else "/".join(str(v) for v in sorted(n_points)),
            )
        )

    R = np.asarray(R_rows, dtype=float)
    NX = np.asarray(NX_rows, dtype=float)
    pid = np.asarray(pid_rows)
    y = np.asarray(y_rows, dtype=int)
    meta = pd.DataFrame(meta_rows)
    return R, NX, pid, y, meta


def build_vectors(R, NX, freq_idx):
    """
    Eq. (5): x = [R(f1), -X(f1), ..., R(fm), -X(fm)] for the retained frequencies.
    """
    freq_idx = np.atleast_1d(np.asarray(freq_idx, dtype=int))
    blocks = []
    for j in freq_idx:
        blocks.append(R[:, j])
        blocks.append(NX[:, j])
    return np.column_stack(blocks)


def nearest_grid_index(grid, target_hz):
    """Index of the candidate-grid frequency closest to a target frequency."""
    return int(np.argmin(np.abs(np.log10(grid) - np.log10(target_hz))))


# ==============================================================================
# 4. PATIENT-REFERENCED REPRESENTATIONS
# ==============================================================================

def _mahalanobis(x, mu, precision):
    d = x - mu
    val = float(d @ precision @ d)
    return float(np.sqrt(max(val, 0.0)))


def build_representations(
    X,
    pid,
    y,
    patients=None,
    budget=None,
    rng=None,
):
    """
    Construct both patient-referenced representations (manuscript Section 2.3).

    For every sample the reference pool consists of healthy spectra of the SAME
    patient. For a healthy query the query itself is removed (leave-self-out).

    Parameters
    ----------
    budget : None or int
        None / "All" -> all eligible healthy references.
        int K        -> the first K references of a per-patient random ordering.
    rng : np.random.Generator
        Used only when a finite budget is requested.

    Returns
    -------
    A : (N, d)  element-wise absolute deviation, Eq. (7)
    M : (N, 1)  Ledoit-Wolf shrinkage Mahalanobis distance, Eq. (11)
    """
    n, d = X.shape
    A = np.zeros((n, d), dtype=float)
    M = np.zeros((n, 1), dtype=float)

    if patients is None:
        patients = np.unique(pid)
    if isinstance(budget, str) and budget.lower() == "all":
        budget = None

    for p in patients:
        p_mask = pid == p
        p_rows = np.flatnonzero(p_mask)
        healthy_rows = np.flatnonzero(p_mask & (y == 0))
        if healthy_rows.size == 0:
            continue

        # Patient-specific random ordering used by the reference-budget analysis
        if budget is not None:
            order = rng.permutation(healthy_rows)
        else:
            order = healthy_rows

        cache = {}
        for i in p_rows:
            pool = order[order != i] if y[i] == 0 else order
            if budget is not None:
                pool = pool[: int(budget)]

            key = pool.tobytes()
            if key not in cache:
                ref = X[pool]
                med = np.median(ref, axis=0)
                if ref.shape[0] >= 2:
                    lw = LedoitWolf(assume_centered=False).fit(ref)
                    cache[key] = (med, lw.location_, lw.precision_)
                else:
                    # Covariance is not identifiable from a single reference:
                    # the Mahalanobis branch is set to zero (manuscript Sec. 2.9)
                    cache[key] = (med, None, None)

            med, mu, prec = cache[key]
            A[i] = np.abs(X[i] - med)
            M[i, 0] = 0.0 if mu is None else _mahalanobis(X[i], mu, prec)

    return A, M


# ==============================================================================
# 5. PATIENT-CLASS SAMPLE WEIGHTS  (Eqs. 21-22)
# ==============================================================================

def patient_class_weights(pid, y):
    w = np.zeros(len(y), dtype=float)
    for p in np.unique(pid):
        for c in (0, 1):
            m = (pid == p) & (y == c)
            k = int(m.sum())
            if k:
                w[m] = 1.0 / k
    s = w.sum()
    return w * (len(y) / s) if s > 0 else np.ones(len(y))


# ==============================================================================
# 6. METRICS  (manuscript Section 2.11)
# ==============================================================================

def patient_metrics(y_true, prob, threshold=THRESHOLD):
    y_pred = (prob >= threshold).astype(int)
    out = dict(BA=balanced_accuracy_score(y_true, y_pred),
               Accuracy=accuracy_score(y_true, y_pred))

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    out["Sensitivity"] = tp / (tp + fn) if (tp + fn) else np.nan
    out["Specificity"] = tn / (tn + fp) if (tn + fp) else np.nan
    out["Precision"] = precision_score(y_true, y_pred, zero_division=0)
    out["F1"] = f1_score(y_true, y_pred, zero_division=0)
    out["MCC"] = matthews_corrcoef(y_true, y_pred) if len(np.unique(y_pred)) > 1 or len(np.unique(y_true)) > 1 else 0.0
    out["AUC"] = roc_auc_score(y_true, prob) if len(np.unique(y_true)) > 1 else np.nan
    return out


def fit_predict(A_tr, M_tr, y_tr, w_tr, A_te, M_te, seed):
    """Fit both branches and return (p_abs, p_mah, p_fusion) for the test rows."""
    clf_a = make_classifier(seed)
    clf_a.fit(A_tr, y_tr, sample_weight=w_tr)
    p_abs = clf_a.predict_proba(A_te)[:, 1]

    clf_m = make_classifier(seed + 1)
    clf_m.fit(M_tr, y_tr, sample_weight=w_tr)
    p_mah = clf_m.predict_proba(M_te)[:, 1]

    p_fus = FUSION_W * p_abs + (1.0 - FUSION_W) * p_mah
    return p_abs, p_mah, p_fus


# ==============================================================================
# 7. NESTED PATIENT-WISE FREQUENCY RANKING  (manuscript Section 2.4)
# ==============================================================================

def rank_frequencies_for_fold(R, NX, pid, y, train_patients, seed, verbose=False):
    """
    Rank all candidate frequencies using ONLY the outer-training patients.

    Inner validation: each training patient is held out in turn, the remaining
    four are used for fitting. The ranking score is Eq. (18).
    """
    rows = []
    n_cand = R.shape[1]
    tr_mask = np.isin(pid, train_patients)

    for j in range(n_cand):
        Xj = build_vectors(R, NX, [j])[tr_mask]
        pid_j, y_j = pid[tr_mask], y[tr_mask]

        A, M = build_representations(Xj, pid_j, y_j)

        ba_list, auc_list = [], []
        for inner_p in train_patients:
            te = pid_j == inner_p
            tr = ~te
            if len(np.unique(y_j[te])) < 2 or len(np.unique(y_j[tr])) < 2:
                continue
            w = patient_class_weights(pid_j[tr], y_j[tr])
            _, _, p_fus = fit_predict(
                A[tr], M[tr], y_j[tr], w, A[te], M[te], seed
            )
            m = patient_metrics(y_j[te], p_fus)
            ba_list.append(m["BA"])
            auc_list.append(m["AUC"])

        ba = np.asarray(ba_list, dtype=float)
        mean_ba = float(np.mean(ba))
        worst_ba = float(np.min(ba))
        sd_ba = float(np.std(ba, ddof=1)) if ba.size > 1 else 0.0
        mean_auc = float(np.nanmean(auc_list))

        score = float(np.sqrt((1 - mean_ba) ** 2 + (1 - worst_ba) ** 2 + sd_ba ** 2))
        rows.append(dict(index=j, MeanBA=mean_ba, WorstBA=worst_ba,
                         SD=sd_ba, MeanAUC=mean_auc, Score=score))
        if verbose:
            print(f"    f[{j:02d}] score={score:.4f} meanBA={mean_ba:.4f}")

    df = pd.DataFrame(rows)
    # Ties: higher mean BA, higher worst BA, lower SD, higher AUC
    df = df.sort_values(
        by=["Score", "MeanBA", "WorstBA", "SD", "MeanAUC"],
        ascending=[True, False, False, True, False],
    ).reset_index(drop=True)
    return df


def nested_frequency_selection(R, NX, pid, y, patients, seed=SEED0, verbose=True):
    """Run the ranking once per outer LOPO fold and retain the top-five."""
    selected, rankings = {}, {}
    for p in patients:
        train_patients = [q for q in patients if q != p]
        if verbose:
            print(f"  outer fold: held-out {p}")
        df = rank_frequencies_for_fold(R, NX, pid, y, train_patients, seed)
        rankings[p] = df
        selected[p] = sorted(df["index"].to_numpy()[:TOP_M].tolist())
    return selected, rankings


# ==============================================================================
# 8. PERTURBATION MODELS  (manuscript Section 2.10)
# ==============================================================================

MODE_OFFSET = {"gaussian": 0, "common": 1}   # deterministic, unlike hash()


def perturb(X, rows, sigma, mode, rng):
    """Apply a multiplicative perturbation to the selected rows of X."""
    Xp = X.copy()
    if sigma <= 0 or rows.size == 0:
        return Xp
    if mode == "gaussian":            # Eq. (30): independent, component-wise
        eps = rng.normal(0.0, sigma, size=(rows.size, X.shape[1]))
        Xp[rows] = X[rows] * (1.0 + eps)
    elif mode == "common":            # Eq. (31): one gain per spectrum
        g = rng.normal(0.0, sigma, size=(rows.size, 1))
        Xp[rows] = X[rows] * (1.0 + g)
    else:
        raise ValueError(f"unknown perturbation mode: {mode}")
    return Xp


# ==============================================================================
# 9. OUTER LOPO EVALUATION
# ==============================================================================

def evaluate(
    R,
    NX,
    pid,
    y,
    patients,
    selected,
    n_repeats=N_REPEATS,
    budget=None,
    sigma=0.0,
    mode="gaussian",
    seed0=SEED0,
    progress=None,
):
    """
    Patient-referenced leave-one-patient-out evaluation.

    Returns a long DataFrame with one row per (repeat, patient, configuration).
    Configurations: 'Absolute', 'Mahalanobis', 'Fusion'.
    """
    records = []
    train_cache = {}          # clean training-patient representations per fold

    for p in patients:
        freq_idx = selected[p] if isinstance(selected, dict) else selected
        X_clean = build_vectors(R, NX, freq_idx)
        train_patients = [q for q in patients if q != p]
        tr = np.isin(pid, train_patients)
        te = pid == p
        te_rows = np.flatnonzero(te)
        w = patient_class_weights(pid[tr], y[tr])

        # Training patients are never perturbed and always use the full
        # reference, so their representations are identical in every repeat.
        key = (p, tuple(np.atleast_1d(freq_idx).tolist()))
        if key not in train_cache:
            A_tr_all, M_tr_all = build_representations(
                X_clean, pid, y, patients=train_patients
            )
            train_cache[key] = (A_tr_all[tr], M_tr_all[tr])
        A_tr, M_tr = train_cache[key]

        for rep in range(n_repeats):
            rng = np.random.default_rng(
                seed0
                + 1000 * rep
                + 100000 * int(round(sigma * 1e4))
                + 7919 * MODE_OFFSET.get(mode, 0)
                + 13 * list(patients).index(p)
            )
            X = perturb(X_clean, te_rows, sigma, mode, rng) if sigma > 0 else X_clean

            A_te_all, M_te_all = build_representations(
                X, pid, y, patients=[p], budget=budget, rng=rng
            )

            p_abs, p_mah, p_fus = fit_predict(
                A_tr, M_tr, y[tr], w, A_te_all[te], M_te_all[te], seed=rep,
            )

            for name, prob in (("Absolute", p_abs),
                               ("Mahalanobis", p_mah),
                               ("Fusion", p_fus)):
                m = patient_metrics(y[te], prob)
                m.update(Repeat=rep, Patient=p, Configuration=name)
                records.append(m)

        if progress is not None:
            progress(list(patients).index(p) + 1, len(patients))

    return pd.DataFrame.from_records(records)


def summarize(df, configuration="Fusion"):
    """Patient-first aggregation (manuscript Section 2.11)."""
    sub = df[df["Configuration"] == configuration]
    per_patient = sub.groupby("Patient").mean(numeric_only=True).drop(columns=["Repeat"])
    per_repeat_mean_ba = sub.groupby("Repeat")["BA"].mean()

    return dict(
        per_patient=per_patient,
        MeanBA=float(per_patient["BA"].mean()),
        WorstBA=float(per_patient["BA"].min()),
        InterPatientSD=float(per_patient["BA"].std(ddof=1)),
        RepeatSD=float(per_repeat_mean_ba.std(ddof=1)),
        AUC=float(per_patient["AUC"].mean()),
        Sensitivity=float(per_patient["Sensitivity"].mean()),
        Specificity=float(per_patient["Specificity"].mean()),
        Accuracy=float(per_patient["Accuracy"].mean()),
        Precision=float(per_patient["Precision"].mean()),
        F1=float(per_patient["F1"].mean()),
        MCC=float(per_patient["MCC"].mean()),
    )


# ==============================================================================
# 10. AURC  (Eqs. 34-35)
# ==============================================================================

_TRAPZ = getattr(np, "trapezoid", None) or np.trapz  # NumPy 1.x / 2.x compatible


def aurc(sigmas, values):
    s = np.asarray(sigmas, dtype=float)
    v = np.asarray(values, dtype=float)
    return float(_TRAPZ(v, s) / (s[-1] - s[0]))


# ==============================================================================
# 11. STATISTICS  (manuscript Section 2.12)
# ==============================================================================

def wilcoxon_paired(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if np.allclose(a, b):
        return dict(W=0.0, p=1.0)
    res = stats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
    return dict(W=float(res.statistic), p=float(res.pvalue))


def friedman(*groups):
    res = stats.friedmanchisquare(*groups)
    return dict(chi2=float(res.statistic), p=float(res.pvalue))


def holm(pvalues, labels=None):
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    m = p.size
    adjusted = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * p[idx]
        running = max(running, val)
        adjusted[idx] = min(running, 1.0)
    if labels is None:
        return adjusted
    return pd.DataFrame(dict(Comparison=labels, p_raw=p, p_holm=adjusted))


# ==============================================================================
# 12. EXPORT HELPERS
# ==============================================================================

@dataclass
class Exporter:
    outdir: str = "outputs"
    tables: dict = field(default_factory=dict)

    def __post_init__(self):
        for sub in ("figures", "tables", "latex"):
            os.makedirs(os.path.join(self.outdir, sub), exist_ok=True)

    def table(self, name, df, caption="", label="", float_format="%.4f"):
        self.tables[name] = df
        df.to_csv(os.path.join(self.outdir, "tables", f"{name}.csv"), index=False)
        tex = df.to_latex(index=False, escape=True, float_format=float_format,
                          caption=caption or name, label=label or f"tab:{name}")
        with open(os.path.join(self.outdir, "latex", f"{name}.tex"), "w") as fh:
            fh.write(tex)
        return df

    def figure(self, fig, filename, dpi=400):
        path = os.path.join(self.outdir, "figures", filename)
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        return path

    def workbook(self, filename="all_tables.xlsx"):
        path = os.path.join(self.outdir, filename)
        with pd.ExcelWriter(path, engine="openpyxl") as xl:
            for name, df in self.tables.items():
                df.to_excel(xl, sheet_name=name[:31], index=False)
        return path

    def json(self, obj, filename="results_summary.json"):
        path = os.path.join(self.outdir, filename)
        with open(path, "w") as fh:
            json.dump(obj, fh, indent=2, default=float)
        return path
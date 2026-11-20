"""
cole_analysis.py -- physics-based (Cole-model) patient-referenced representation.

Each measured spectrum is fitted, on its ORIGINAL frequency points between 1 kHz and
1 MHz (no interpolation), with the Cole model

    Z(w) = R_inf + (R_0 - R_inf) / (1 + (j w tau)^alpha),

giving theta = [ln R_0, ln R_inf, log10 f_c, alpha] with f_c = 1 / (2 pi tau).
The four parameters replace the selected frequencies in the patient-referenced
framework (absolute deviation + Ledoit-Wolf Mahalanobis + equal fusion, LOPO,
30 repetitions). No data-driven frequency selection is involved.
"""
import json
import os

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import least_squares

import eis_pipeline as ep

OUT = "outputs_cole"
os.makedirs(OUT, exist_ok=True)
FMIN, FMAX = 1e3, 1e6


def iterate_spectra(raw):
    """Same ordering as ep.harmonize (patients by (len, name); list-valued classes)."""
    for p_index, pkey in enumerate(sorted(raw.keys(), key=lambda s: (len(s), s)), start=1):
        for ckey, meas in raw[pkey].items():
            if not isinstance(meas, (list, tuple)):
                continue
            label = ep._class_label(ckey)
            for m in meas:
                f, z = ep._measurement_arrays(m)
                o = np.argsort(f)
                yield f"P{p_index}", label, f[o], z[o]


def cole(params, w):
    r_inf, d_r, log_tau, alpha = params
    return r_inf + d_r / (1.0 + (1j * w * 10.0 ** log_tau) ** alpha)


def fit_cole(f, z):
    m = (f >= FMIN * 0.999) & (f <= FMAX * 1.001)
    f, z = f[m], z[m]
    w = 2 * np.pi * f
    scale = np.abs(z)

    def resid(p):
        d = (cole(p, w) - z) / scale
        return np.concatenate([d.real, d.imag])

    r_inf0 = max(float(np.real(z[-1])) * 0.9, 1e-3)
    d_r0 = max(float(np.real(z[0])) - r_inf0, 1e-3)
    f_peak = f[np.argmax(-np.imag(z))]
    best = None
    for a0 in (0.6, 0.8, 0.95):
        for lt0 in (np.log10(1 / (2 * np.pi * f_peak)), -4.5, -5.5):
            r = least_squares(resid, x0=[r_inf0, d_r0, lt0, a0],
                              bounds=([1e-6, 1e-6, -9.0, 0.1], [1e4, 1e4, -1.0, 1.0]),
                              x_scale=[r_inf0, d_r0, 1.0, 0.1], max_nfev=4000)
            if best is None or r.cost < best.cost:
                best = r
    r_inf, d_r, log_tau, alpha = best.x
    rel_rmse = float(np.sqrt(np.mean(best.fun ** 2)))
    return dict(R0=r_inf + d_r, Rinf=r_inf, fc=1.0 / (2 * np.pi * 10.0 ** log_tau), alpha=alpha,
                rel_rmse=rel_rmse, n_points=int(f.size))


if __name__ == "__main__":
    raw = ep.load_raw_dataset("electrical-impedance-data-bladder/dataset_without_modules.pickle")
    GRID = ep.candidate_grid()
    R, NX, pid, y, _ = ep.harmonize(raw, GRID)
    PATIENTS = [f"P{i}" for i in range(1, 7)]

    rows = []
    for (p, lab, f, z), (pp, yy) in zip(iterate_spectra(raw), zip(pid, y)):
        assert p == pp and lab == yy
        d = fit_cole(f, z)
        d.update(Patient=p, Label=lab)
        rows.append(d)
    fits = pd.DataFrame(rows)
    fits.to_csv(f"{OUT}/cole_fits.csv", index=False)
    print("fit quality (relative RMSE, %):")
    print((100 * fits.groupby("Patient").rel_rmse.describe()[["50%", "max"]]).round(2))
    print("points used per patient:", fits.groupby("Patient").n_points.first().to_dict())

    # --- physiology: per patient medians by class --------------------------------------
    phys = fits.groupby(["Patient", "Label"])[["R0", "Rinf", "fc", "alpha"]].median().unstack("Label")
    print(phys.round(4))
    phys.to_csv(f"{OUT}/cole_medians.csv")
    sign = {}
    for prm in ("R0", "Rinf", "fc", "alpha"):
        h, t = phys[(prm, 0)], phys[(prm, 1)]
        sign[prm] = dict(tumor_lower=int((t < h).sum()), median_ratio=float(np.median(t / h)),
                         wilcoxon_p=float(stats.wilcoxon(t, h).pvalue))
    print("tumor vs healthy across patients:", json.dumps(sign, indent=1))
    pooled = fits.groupby("Label")[["R0", "Rinf", "fc", "alpha"]].median()
    print("pooled medians:\n", pooled.round(4))

    # --- classification in Cole-parameter space -------------------------------------------
    theta = np.column_stack([np.log(fits.R0), np.log(fits.Rinf), np.log10(fits.fc), fits.alpha])
    Rc, NXc = theta[:, [0, 2]], theta[:, [1, 3]]       # build_vectors -> [th0, th1, th2, th3]
    res_cole = ep.evaluate(Rc, NXc, pid, y, PATIENTS, {p: [0, 1] for p in PATIENTS})
    res_cole.to_csv(f"{OUT}/cole_lopo_long.csv", index=False)
    summ = {c: ep.summarize(res_cole, c) for c in ("Absolute", "Mahalanobis", "Fusion")}
    tab = pd.DataFrame([dict(Configuration=c, MeanBA=s["MeanBA"], AUC=s["AUC"], WorstBA=s["WorstBA"],
                             SD=s["InterPatientSD"], Sens=s["Sensitivity"], Spec=s["Specificity"])
                        for c, s in summ.items()])
    print(tab.round(4))
    pp = summ["Fusion"]["per_patient"].loc[PATIENTS, "BA"]
    print("per patient (Cole fusion):", pp.round(4).to_dict())

    sel = {k: [int(v) for v in vals] for k, vals in json.load(open("outputs_extra/selected.json")).items()}
    res_nested = ep.evaluate(R, NX, pid, y, PATIENTS, sel)
    ppn = ep.summarize(res_nested, "Fusion")["per_patient"].loc[PATIENTS, "BA"]
    d = pp.values - ppn.values
    rng = np.random.default_rng(ep.SEED0)
    bs = d[rng.integers(0, 6, size=(10000, 6))].mean(1)
    nz = int(np.sum(np.abs(d) > 1e-12))
    w = stats.wilcoxon(pp.values, ppn.values) if nz else None
    comp = dict(cole_minus_nested=float(d.mean()), ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                W=None if w is None else float(w.statistic), p=None if w is None else float(w.pvalue), n_nontied=nz,
                per_patient=dict(zip(PATIENTS, np.round(d, 4).tolist())))
    print("Cole fusion minus nested-frequency fusion:", json.dumps(comp, indent=1))
    json.dump(dict(sign=sign, table=tab.to_dict(orient="records"), per_patient=pp.to_dict(), comparison=comp,
                   pooled=pooled.to_dict()), open(f"{OUT}/cole_summary.json", "w"), indent=1, default=float)

"""
ijs_simulation_ext.py -- robustness of the simulation to departures from the base model.

Scenarios (logistic regression, six patients with three healthy and three tumor samples,
m = 15 spectra per sample, ICC in {0.5, 0.9, 0.97}, Delta in {0, 4}, 30 replicates):
  base           the generator of ijs_simulation.py
  correlated     between-sample and within-sample deviations correlated across the four
                 features (AR(1) correlation 0.7), marginal variances unchanged
  heteroscedastic within-sample noise scaled per tissue sample by a log-normal factor
                 (log-SD 0.75, unit mean square), so that repeatability differs between samples
  heterogeneous  patient-specific tumor effects: delta_p = u_p * delta + N(0, I), u_p ~ U(0, 2)
Output: outputs_ijs/revision/sim_ext.csv and sim_ext_summary.json
"""
import json
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

import ijs_simulation as S

L = np.linalg.cholesky(np.array([[0.7 ** abs(i - j) for j in range(S.D)] for i in range(S.D)]))
SCEN = {"name": "base"}


def simulate_scen(rng, n_pat, n_h, n_t, m, icc, delta, offset=0):
    sc = SCEN["name"]
    dvec = np.full(S.D, delta / np.sqrt(S.D))
    X, pid, y, G = [], [], [], []
    for p in range(n_pat):
        mu = rng.normal(0, 2.0, S.D)
        if sc == "heterogeneous":
            dp = rng.uniform(0, 2) * dvec + rng.normal(0, 1.0, S.D)
        else:
            dp = dvec + rng.normal(0, 0.5, S.D)
        for c, ns in ((0, n_h), (1, n_t)):
            for s in range(ns):
                b = rng.normal(0, 1.0, S.D)
                e = rng.normal(0, np.sqrt((1 - icc) / icc), (m, S.D))
                if sc == "correlated":
                    b, e = L @ b, e @ L.T
                if sc == "heteroscedastic":
                    e = e * np.exp(0.75 * rng.normal() - 0.75 ** 2)
                X.append(mu + c * dp + b + e)
                pid += [f"P{p + offset}"] * m
                y += [c] * m
                G += [f"P{p + offset}:{c}{s}"] * m
    return np.vstack(X), np.asarray(pid), np.asarray(y), np.asarray(G)


def replicate(args):
    cell, rep, kind, sc = args
    SCEN["name"] = sc
    S.simulate = simulate_scen
    out = S.replicate((cell, rep, kind))
    out["scenario"] = sc
    return out


if __name__ == "__main__":
    jobs = [((6, 3, 3, 15, icc, d), r, "logit", sc)
            for sc in ("base", "correlated", "heteroscedastic", "heterogeneous")
            for icc in (0.5, 0.9, 0.97) for d in (0.0, 4.0) for r in range(30)]
    procs = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    with Pool(procs) as pool:
        rows = pool.map(replicate, jobs, chunksize=4)
    df = pd.DataFrame(rows)
    df.to_csv(f"{S.OUT}/sim_ext.csv", index=False)
    for dn in ("naive", "sa", "matched"):
        df[f"{dn}_opt"] = df[f"{dn}_est"] - df[f"{dn}_true"]
    cell = df.groupby(["scenario", "delta", "icc"])[["naive_opt", "sa_opt", "matched_opt", "naive_est", "matched_true"]].mean()
    cell.reset_index().to_csv(f"{S.OUT}/sim_ext_cell_means.csv", index=False)
    summ = {}
    for sc, g in cell.groupby(level="scenario"):
        summ[sc] = dict(naive_opt_min=float(g.naive_opt.min()), naive_opt_max=float(g.naive_opt.max()),
                        matched_opt_min=float(g.matched_opt.min()), matched_opt_max=float(g.matched_opt.max()),
                        sa_opt_min=float(g.sa_opt.min()), sa_opt_max=float(g.sa_opt.max()),
                        naive_est_null_icc097=float(g.loc[(sc, 0.0, 0.97), "naive_est"]))
    json.dump(summ, open(f"{S.OUT}/sim_ext_summary.json", "w"), indent=1)
    print(cell.round(3).to_string())
    print(json.dumps(summ, indent=1))

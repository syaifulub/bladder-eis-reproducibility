"""sim_summary.py -- numbers of the simulation study quoted in the manuscript."""
import json

import numpy as np
import pandas as pd

REV = "outputs_ijs/revision"
g = pd.read_csv(f"{REV}/sim_grid.csv")
x = pd.read_csv(f"{REV}/sim_xgb.csv")
pr = pd.read_csv(f"{REV}/sim_precision.csv")
for d in (g, x, pr):
    for dn in ("naive", "sa", "matched"):
        d[f"{dn}_opt"] = d[f"{dn}_est"] - d[f"{dn}_true"]

cell = g.groupby(["delta", "icc", "m"])[[c for c in g.columns if c.endswith(("_est", "_true", "_opt"))]].mean()
S = {}
for dl in (0.0, 4.0):
    c = cell.loc[dl]
    rep = c[c.index.get_level_values("m") >= 5]
    tag = "null" if dl == 0 else "effect"
    S[tag] = dict(
        naive_opt_min=float(rep.naive_opt.min()), naive_opt_max=float(rep.naive_opt.max()),
        naive_opt_icc01_m5=float(c.loc[(0.1, 5), "naive_opt"]),
        naive_opt_icc097_m30=float(c.loc[(0.97, 30), "naive_opt"]),
        naive_est_icc097_m15=float(c.loc[(0.97, 15), "naive_est"]),
        naive_true_icc097_m15=float(c.loc[(0.97, 15), "naive_true"]),
        sa_est_icc097_m15=float(c.loc[(0.97, 15), "sa_est"]),
        sa_true_icc097_m15=float(c.loc[(0.97, 15), "sa_true"]),
        matched_est_icc097_m15=float(c.loc[(0.97, 15), "matched_est"]),
        true_icc097_m15=float(c.loc[(0.97, 15), "matched_true"]),
        sa_opt_min=float(rep.sa_opt.min()), sa_opt_max=float(rep.sa_opt.max()),
        sa_opt_m1_min=float(c[c.index.get_level_values("m") == 1].sa_opt.min()),
        sa_opt_m1_max=float(c[c.index.get_level_values("m") == 1].sa_opt.max()),
        matched_opt_min=float(c.matched_opt.min()), matched_opt_max=float(c.matched_opt.max()),
        matched_abs_mean=float(c.matched_opt.abs().mean()),
    )
# monotonic increase of naive optimism with ICC at m = 15 and 30
for m in (15, 30):
    for dl in (0.0, 4.0):
        v = cell.loc[dl].xs(m, level="m").naive_opt.values
        S.setdefault("naive_monotone_in_icc", {})[f"m{m}_d{dl:g}"] = bool(np.all(np.diff(v) > -0.02))

xc = x.groupby(["delta", "icc"])[[c for c in x.columns if c.endswith(("_est", "_true", "_opt"))]].mean()
S["xgb"] = dict(naive_opt_null_icc097=float(xc.loc[(0.0, 0.97), "naive_opt"]),
                naive_est_null_icc097=float(xc.loc[(0.0, 0.97), "naive_est"]),
                naive_opt_effect_icc097=float(xc.loc[(4.0, 0.97), "naive_opt"]),
                matched_opt_min=float(xc.matched_opt.min()), matched_opt_max=float(xc.matched_opt.max()),
                naive_opt_icc01_null=float(xc.loc[(0.0, 0.1), "naive_opt"]))
q = pr.groupby(["n_pat", "n_h"]).matched_est.quantile([.025, .975]).unstack()
w = (q[0.975] - q[0.025])
S["precision"] = {f"n{n}_h{h}": dict(lo=float(q.loc[(n, h), 0.025]), hi=float(q.loc[(n, h), 0.975]), width=float(w.loc[(n, h)]))
                  for n, h in q.index}
S["precision_true_mean"] = float(pr.matched_true.mean())
S["precision_sa_width"] = {f"n{n}_h{h}": float(v) for (n, h), v in
                           (pr.groupby(["n_pat", "n_h"]).sa_est.quantile(.975) - pr.groupby(["n_pat", "n_h"]).sa_est.quantile(.025)).items()}
json.dump(S, open(f"{REV}/simulation_summary.json", "w"), indent=1)
cell.round(3).to_csv(f"{REV}/simulation_cell_means.csv")
print(json.dumps(S, indent=1))

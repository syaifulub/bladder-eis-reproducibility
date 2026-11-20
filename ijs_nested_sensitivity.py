"""
ijs_nested_sensitivity.py -- sensitivity of the fully nested pipeline selection.

V1 fallback="outer"   : the six inner folds without patient-level ranking reuse the outer-fold
                        ranking (favours the frequency candidate)
V2 fallback="generic" : these six inner folds use a fixed decade-spaced frequency set
V3 design="matched"   : size-matched sample-wise referencing in ranking, inner comparison and
                        outer evaluation
V4 P2 labels exchanged
V5 P2 excluded
All with 30 XGBoost seeds for the outer evaluation; patient-level cluster bootstrap CIs.
Rankings already computed by ijs_revision.py (same procedure, verified identical) are reused.
Output: outputs_ijs/revision/nested_sensitivity.json and sample tables.
"""
import json
import os
import time

import numpy as np
import pandas as pd

import eis_pipeline as ep
import eis_groups as eg
import nested_core as nc
import rev_common as rc

OUT = "outputs_ijs/revision"
T0 = time.time()
tick = lambda m: print(f"[{time.time() - T0:6.0f}s] {m}", flush=True)
raw = ep.load_raw_dataset("electrical-impedance-data-bladder/dataset_without_modules.pickle")
D = nc.Data(raw, pd.read_csv("outputs_ijs/cole_fits.csv"))
P = [f"P{i}" for i in range(1, 7)]
SEEDS = list(range(ep.N_REPEATS))
RESF = f"{OUT}/nested_sensitivity.json"
RES = json.load(open(RESF)) if os.path.exists(RESF) else {}

# cache of the main analysis (sample-wise referencing, labels as given)
SEL = json.load(open("outputs_ijs/selections_idx.json"))["sa_nested"]
INNER = json.load(open(f"{OUT}/selections_inner.json"))
METHOD = json.load(open(f"{OUT}/revision_results.json"))["inner_selection_method"]
CACHE_MAIN = {("outer", "sa", p): sorted(SEL[p]) for p in P}
for k, v in INNER.items():
    p, q = k.split("|")
    CACHE_MAIN[(METHOD[k].lower(), "sa", p, q)] = v


def record(name, r, keep):
    ci, _, nb = eg.cluster_boot({name: r["sample_table"]})
    m = r["metrics"]
    RES[name] = dict(choice=r["choice"], inner_method=r["inner_method"], BA=m["BA"], TP=m["TP"], N1=m["N1"],
                     TN=m["TN"], N0=m["N0"], AUC=m["AUC"], CI=list(ci[name]),
                     n_cole=sum(v["representation"] == "cole" for v in r["choice"].values()),
                     n_abs=sum(v["branch"] == "abs" for v in r["choice"].values()))
    r["sample_table"].to_csv(f"{OUT}/sampletable_nested_{name}.csv", index=False)
    r["inner"].to_csv(f"{OUT}/nested_inner_{name}.csv", index=False)
    json.dump(json.loads(json.dumps(RES, default=float)), open(RESF, "w"), indent=1)
    tick(f"{name}: BA={m['BA']:.3f} ({m['TP']}/{m['N1']}, {m['TN']}/{m['N0']}) CI {ci[name]} choice "
         f"{ {p: (v['representation'], v['branch']) for p, v in r['choice'].items()} }")


y2 = D.y.copy()
y2[D.pid == "P2"] = 1 - y2[D.pid == "P2"]
P5 = [p for p in P if p != "P2"]
runs = [("fallback_outer", dict(yk=D.y, patients=P, fallback="outer", design="sa", cache=dict(CACHE_MAIN))),
        ("fallback_generic", dict(yk=D.y, patients=P, fallback="generic", design="sa", cache=dict(CACHE_MAIN))),
        ("matched", dict(yk=D.y, patients=P, fallback="loso", design="matched", cache={})),
        ("p2_exchanged", dict(yk=y2, patients=P, fallback="loso", design="sa", cache={})),
        ("p2_excluded", dict(yk=D.y, patients=P5, fallback="loso", design="sa", cache={}))]
for name, kw in runs:
    if name in RES:
        tick(f"{name}: done before")
        continue
    tick(f"{name} ...")
    keep = rc.evaluable(D.pid, kw["yk"], D.G, kw["patients"])
    r = nc.full_nested(D, kw["yk"], kw["patients"], fallback=kw["fallback"], design=kw["design"],
                       seeds=SEEDS, cache=kw["cache"], keep=keep)
    record(name, r, keep)
tick("done")

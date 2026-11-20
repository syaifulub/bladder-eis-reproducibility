"""
ijs_selection_permutation.py -- selection-aware permutation test.

For every permutation of tissue-sample labels within patients, the COMPLETE fully nested
procedure of nested_core.full_nested is repeated: outer and inner frequency ranking (with the
leave-one-tissue-sample-out fallback), comparison of the six candidate pipelines in the inner
folds, and evaluation of the chosen pipeline on the held-out patient. The statistic is the
sample-level balanced accuracy of the selected pipeline (one XGBoost seed, as for the observed
value computed here as permutation 0). The pre-specified frequency-based classifier (nested
frequencies, equal fusion) is recorded from the same runs.
Usage: python ijs_selection_permutation.py [B=100]; resumes from its checkpoint.
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

import eis_pipeline as ep
import eis_groups as eg
import nested_core as nc
import rev_common as rc

OUT = "outputs_ijs/revision"
B = int(sys.argv[1]) if len(sys.argv) > 1 else 100
raw = ep.load_raw_dataset("electrical-impedance-data-bladder/dataset_without_modules.pickle")
D = nc.Data(raw, pd.read_csv("outputs_ijs/cole_fits.csv"))
P = [f"P{i}" for i in range(1, 7)]


def prespecified(D, yk, cache, keep):
    """Nested frequencies (outer-fold ranking from the cache), equal fusion, sample-wise referencing."""
    recs = []
    for p in P:
        trp = [q for q in P if q != p]
        rep = nc.reps(D, ep.build_vectors(D.R, D.NX, cache[("outer", "sa", p)]), yk, P, "sa")
        d = nc.fit_eval(D, rep, yk, trp, lambda pv, gv: pv == p, ep.SEED0)
        if d is not None:
            recs.append(d.assign(Repeat=0))
    return eg.pooled(eg.sample_table(pd.concat(recs, ignore_index=True), "p", keep))["BA"]


if __name__ == "__main__":
    t0 = time.time()
    rng = np.random.default_rng(ep.SEED0 + 11)
    labels = [D.y.copy()] + [rc.permute_within_patient(D.y, D.pid, D.G, rng) for _ in range(B)]
    part = f"{OUT}/selection_permutation_partial.csv"
    done = set(pd.read_csv(part).perm) if os.path.exists(part) else set()
    print(f"{len(done)} done, {B + 1 - len(done)} to go", flush=True)
    for k, yk in enumerate(labels):
        if k in done:
            continue
        cache = {}
        keep = rc.evaluable(D.pid, yk, D.G, P)
        r = nc.full_nested(D, yk, P, seeds=[ep.SEED0], cache=cache, keep=keep)
        row = dict(perm=k, BA_selected=r["metrics"]["BA"], BA_prespecified=prespecified(D, yk, cache, keep),
                   n_cole=sum(v["representation"] == "cole" for v in r["choice"].values()),
                   n_abs=sum(v["branch"] == "abs" for v in r["choice"].values()),
                   n_mah=sum(v["branch"] == "mah" for v in r["choice"].values()))
        pd.DataFrame([row]).to_csv(part, mode="a", header=not os.path.exists(part), index=False)
        print(f"[{time.time() - t0:7.0f}s] perm {k}: {row}", flush=True)
    df = pd.read_csv(part).drop_duplicates("perm").sort_values("perm").reset_index(drop=True)
    df.to_csv(f"{OUT}/selection_permutation_null.csv", index=False)
    obs, null = df[df.perm == 0].iloc[0], df[df.perm > 0]
    res = dict(n_perm=int(len(null)))
    for c in ("BA_selected", "BA_prespecified"):
        res[c] = dict(observed=float(obs[c]), null_mean=float(null[c].mean()), null_sd=float(null[c].std(ddof=1)),
                      null_q025=float(null[c].quantile(.025)), null_q975=float(null[c].quantile(.975)),
                      null_max=float(null[c].max()),
                      p_value=float((1 + (null[c] >= obs[c] - 1e-12).sum()) / (1 + len(null))))
    res["null_choice"] = dict(mean_folds_cole=float(null.n_cole.mean()), mean_folds_abs=float(null.n_abs.mean()),
                              frac_all_cole_abs=float(((null.n_cole == 6) & (null.n_abs == 6)).mean()))
    json.dump(res, open(f"{OUT}/selection_permutation_results.json", "w"), indent=1)
    print(json.dumps(res, indent=1))

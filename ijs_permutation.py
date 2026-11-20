"""
ijs_permutation.py -- negative control: within-patient permutation of tissue-sample labels.

Under the null hypothesis that impedance carries no class information beyond the patient,
class labels are exchangeable among the tissue samples of a patient. Labels are therefore
shuffled among samples within each patient (all spectra of a sample keep a common label),
and the complete referencing + classification pipeline is re-run. The null distribution of the
sample-level balanced accuracy is obtained for both referencing designs and for the two
selection-free feature sets (Cole parameters; frequencies published by Veil et al.).
A design that is valid under the null must give null BA centered on 0.5.
"""
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

import eis_pipeline as ep
import eis_groups as eg
import rev_common as rc

OUT = "outputs_ijs/revision"
os.makedirs(OUT, exist_ok=True)
N_PERM = int(sys.argv[1]) if len(sys.argv) > 1 else 1000

raw = ep.load_raw_dataset("electrical-impedance-data-bladder/dataset_without_modules.pickle")
R, NX, pid, y, meta = ep.harmonize(raw, ep.candidate_grid())
P = [f"P{i}" for i in range(1, 7)]
G = eg.sample_ids(raw)
GS = eg.spectrum_groups(len(y))
fits = pd.read_csv("outputs_ijs/cole_fits.csv")
FEATS = {
    "cole": np.column_stack([np.log(fits.R0), np.log(fits.Rinf), np.log10(fits.fc), fits.alpha]),
    "published": ep.build_vectors(*ep.harmonize(raw, np.array([3.55e3, 568.99e3, 655.13e3, 754.31e3, 868.51e3]))[:2],
                                  [0, 1, 2, 3, 4]),
}
DESIGNS = {"naive": GS, "sa": G}


def one(args):
    k, yk = args
    ep.XGB_PARAMS["n_jobs"] = 1
    keep = rc.evaluable(pid, yk, G, P)
    out = dict(perm=k, n_eval=len(keep),
               n_eval_healthy=int(len({s for s in keep if yk[G == s][0] == 0})))
    for fn, X in FEATS.items():
        for dn, grp in DESIGNS.items():
            A, M, ok = rc.fast_rep(X, pid, yk, grp, P)
            r = rc.run_eval(A, M, ok, pid, yk, G, P, seeds=[k])
            for col, tag in (("p", "fusion"), ("p_abs", "abs"), ("p_mah", "mah")):
                out[f"{fn}|{dn}|{tag}"] = rc.sample_ba(r, col, keep)[0]
    return out


if __name__ == "__main__":
    t0 = time.time()
    rng = np.random.default_rng(ep.SEED0 + 7)
    jobs = [(0, y.copy())] + [(k, rc.permute_within_patient(y, pid, G, rng)) for k in range(1, N_PERM + 1)]
    part = f"{OUT}/permutation_null_partial.csv"            # checkpoint: resume after interruption
    done = set(pd.read_csv(part).perm) if os.path.exists(part) else set()
    todo = [j for j in jobs if j[0] not in done]
    print(f"{len(done)} permutations already done, {len(todo)} to go", flush=True)
    with Pool(2) as pool:
        for i, r in enumerate(pool.imap_unordered(one, todo, chunksize=4)):
            pd.DataFrame([r]).to_csv(part, mode="a", header=not os.path.exists(part), index=False)
            if i % 50 == 0:
                print(f"[{time.time() - t0:6.0f}s] {i}/{len(todo)}", flush=True)
    rows = pd.read_csv(part).drop_duplicates("perm").to_dict("records")
    df = pd.DataFrame(rows).sort_values("perm").reset_index(drop=True)
    df.to_csv(f"{OUT}/permutation_null.csv", index=False)
    obs = json.load(open("outputs_ijs/ijs_results.json"))["summary"]
    null = df[df.perm > 0]
    res = dict(n_perm=int(len(null)))
    for c in [c for c in df.columns if "|" in c]:
        fn, dn, tag = c.split("|")
        key = {"cole": f"{dn}_cole|{tag}", "published": f"{dn}_published|{tag}"}[fn]
        o = obs.get(key, {}).get("BA", float(df.loc[df.perm == 0, c].iloc[0]))
        res[c] = dict(null_mean=float(null[c].mean()), null_sd=float(null[c].std(ddof=1)),
                      null_q025=float(null[c].quantile(.025)), null_q975=float(null[c].quantile(.975)),
                      frac_null_ge_0p75=float((null[c] >= .75).mean()),
                      observed=float(o), p_value=float((1 + (null[c] >= o - 1e-12).sum()) / (1 + len(null))))
    json.dump(res, open(f"{OUT}/permutation_results.json", "w"), indent=1)
    print(json.dumps(res, indent=1))
    print(f"done in {time.time() - t0:.0f}s")

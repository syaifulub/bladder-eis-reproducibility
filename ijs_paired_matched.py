"""
ijs_paired_matched.py -- paired patient-level cluster bootstrap of the difference between the
fully nested pipeline under ordinary sample-wise referencing and under size-matched referencing
(complete nested procedure repeated with size-matched references), and of the corresponding
difference for the pre-specified classifier and the absolute Cole branch.
Output: outputs_ijs/revision/paired_matched.json
"""
import json

import pandas as pd

import eis_groups as eg

REV = "outputs_ijs/revision"
T = {"nested_sa": pd.read_csv(f"{REV}/sampletable_nested_pipeline_selected.csv"),
     "nested_matched": pd.read_csv(f"{REV}/sampletable_nested_matched.csv"),
     "prespecified_sa": pd.read_csv("outputs_ijs/sampletable_sa_nested_fusion.csv"),
     "prespecified_matched": pd.read_csv(f"{REV}/sampletable_matched_nested_fusion.csv"),
     "coleabs_sa": pd.read_csv("outputs_ijs/sampletable_sa_cole_abs.csv"),
     "coleabs_matched": pd.read_csv(f"{REV}/sampletable_matched_cole_abs.csv")}
ci, diff, nvalid = eg.cluster_boot(T)
out = dict(n_valid=int(nvalid))
for k in ("nested", "prespecified", "coleabs"):
    a, b = f"{k}_sa", f"{k}_matched"
    d = diff[f"{a} - {b}"]
    out[k] = dict(BA_sa=eg.pooled(T[a])["BA"], BA_matched=eg.pooled(T[b])["BA"],
                  diff=eg.pooled(T[a])["BA"] - eg.pooled(T[b])["BA"], boot_mean=d[0], ci_low=d[1], ci_high=d[2],
                  ci_sa=list(ci[a]), ci_matched=list(ci[b]))
json.dump(out, open(f"{REV}/paired_matched.json", "w"), indent=1)
print(json.dumps(out, indent=1))

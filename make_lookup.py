"""
make_lookup.py -- screening values of the reference-leakage index (Table 5 of the article, Table S18).

Leave-self-out referencing with mean references; sigma_b^2 = ICC, sigma_w^2 = 1 - ICC; index F_{d,d}(V_t / V_r)
for d = 1 and d = 4 independent features of equal ICC.  Output: results/index_grid/lookup_LSO.csv
usage: python make_lookup.py
"""
import os
import pandas as pd
from scipy import stats
from leakage_index import deviation_variances

rows = []
for icc in [0.3, 0.6, 0.9]:
    for m in [5, 15, 50]:
        row = {"ICC": icc, "m": m}
        for nh in [1, 2, 4, 8]:
            vr, vt = deviation_variances(icc, 1 - icc, nh, m, "LSO")
            for d in [1, 4]:
                row[f"nh{nh}_d{d}"] = stats.f.cdf(vt / vr, d, d)
        rows.append(row)
os.makedirs(os.path.join("results", "index_grid"), exist_ok=True)
T = pd.DataFrame(rows)
new = os.path.join("results", "index_grid", "lookup_LSO.csv")
if os.path.exists(new):
    old = pd.read_csv(new)
    print("max abs difference to saved table:", float((old.values - T.values).__abs__().max()))
T.to_csv(new, index=False)
print(T.round(3).to_string(index=False))

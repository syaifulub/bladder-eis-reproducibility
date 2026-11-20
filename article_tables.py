"""article_tables.py -- Tables 1-4 of the article, rebuilt from the analysis outputs.

Usage (inside the notebook or from the command line):
    python article_tables.py [outputs_ijs]
Writes outputs_ijs/article_tables.xlsx and outputs_ijs/article_tables.tex, prints every table, and compares
every cell with the value printed in the article (half-up rounding to three decimals).
"""
import json
import os
import sys
from decimal import Decimal, ROUND_HALF_UP

import pandas as pd

OUT = "outputs_ijs"            # set from the command line when run as a script
REV = f"{OUT}/revision"


def h3(v, d=3):
    """Round half up (as printed in the article)."""
    if v is None or (isinstance(v, float) and v != v):
        return "--"
    q = Decimal(1).scaleb(-d)
    return str(Decimal(f"{float(v):.10f}").quantize(q, rounding=ROUND_HALF_UP))


def load():
    r = json.load(open(f"{OUT}/ijs_results.json"))
    R = json.load(open(f"{REV}/revision_results.json"))
    NS = json.load(open(f"{REV}/nested_sensitivity.json"))
    EX = json.load(open(f"{REV}/reviewer_extras.json"))
    return r, R, NS, EX


def table1():
    t = pd.read_csv(f"{OUT}/Table1_structure.csv")
    # a healthy sample is evaluable under sample-wise referencing only if the patient has another healthy sample
    t["EvaluableHealthy"] = t.HealthySamples.where(t.HealthySamples >= 2, 0)
    return t


def table2(r, R):
    S, C = r["summary"], r["ci"]
    RS, RC = R["summary"], R["ci"]
    rows = [("Spectrum-wise", "Leaky", S["naive_leaky|fusion"], C.get("naive_leaky|fusion")),
            ("Spectrum-wise", "Nested", S["naive_nested|fusion"], C.get("naive_nested|fusion")),
            ("Sample-wise", "Leaky", S["sa_leaky|fusion"], C.get("sa_leaky|fusion")),
            ("Sample-wise", "Nested", S["sa_nested|fusion"], C.get("sa_nested|fusion")),
            ("Size-matched", "Nested", RS["matched_nested|fusion"], RC.get("matched_nested|fusion"))]
    out = []
    for ref, sel, s, ci in rows:
        out.append({"Referencing": ref, "Selection": sel, "Tumor correct": f"{s['TP']}/{s['N1']}",
                    "Healthy correct": f"{s['TN']}/{s['N0']}",
                    "Sample BA (95% CI)": f"{h3(s['BA'])} ({h3(ci[0])}--{h3(ci[1])})",
                    "Sample AUC": h3(s["AUC"]), "Spectrum BA": h3(s["SpectrumBA"])})
    return pd.DataFrame(out)


def table3(r, R):
    S, RS = r["summary"], R["summary"]
    g = lambda D, k: D[k]["BA"] if k in D else None
    rows = [("Nested frequencies", "Spectrum-wise", S, "naive_nested", None),
            ("", "Sample-wise", S, "sa_nested", "sa_nested|innerw"),
            ("", "Size-matched", RS, "matched_nested", None),
            ("Leaky frequencies", "Spectrum-wise", S, "naive_leaky", None),
            ("", "Sample-wise", S, "sa_leaky", None),
            ("Published set", "Sample-wise", S, "sa_published", None),
            ("", "Size-matched", RS, "matched_published", None),
            ("Generic decade-spaced", "Sample-wise", S, "sa_generic", None),
            ("Cole parameters", "Spectrum-wise", S, "naive_cole", None),
            ("", "Sample-wise", S, "sa_cole", "sa_cole|innerw"),
            ("", "Size-matched", RS, "matched_cole", None)]
    out = []
    for feat, ref, D, k, iw in rows:
        out.append({"Features": feat, "Referencing": ref, "Absolute": h3(g(D, f"{k}|abs")),
                    "Mahalanobis": h3(g(D, f"{k}|mah")), "Equal fusion": h3(g(D, f"{k}|fusion")),
                    "Inner-fold weight": h3(S[iw]["BA"]) if iw else "--"})
    return pd.DataFrame(out)


def table4(r, R, NS, EX):
    S, RS = r["summary"], R["summary"]
    ba = lambda D, k: D[k]["BA"]
    rows = [("Primary analysis", ba(S, "sa_nested|fusion"), ba(RS, "nested_pipeline|selected"), ba(S, "sa_cole|abs"), ba(S, "sa_published|fusion")),
            ("P2 excluded", ba(S, "noP2_sa_nested|fusion"), NS["p2_excluded"]["BA"], ba(S, "noP2_sa_cole|abs"), EX["published_without_P2"]["BA"]),
            ("P2 labels exchanged", ba(RS, "p2_sa_nested|fusion"), NS["p2_exchanged"]["BA"], ba(RS, "p2_sa_cole|abs"), ba(RS, "p2_sa_published|fusion")),
            ("Size-matched referencing", ba(RS, "matched_nested|fusion"), NS["matched"]["BA"], ba(RS, "matched_cole|abs"), ba(RS, "matched_published|fusion")),
            ("Class-balanced weighting", ba(S, "sa_nested|fusion|cb"), None, ba(S, "sa_cole|abs|cb"), None),
            ("Fallback: outer-fold ranking", None, NS["fallback_outer"]["BA"], None, None),
            ("Fallback: fixed frequency set", None, NS["fallback_generic"]["BA"], None, None),
            ("Logistic regression", ba(RS, "lr_sa_nested|fusion"), None, ba(RS, "lr_sa_cole|abs"), ba(RS, "lr_sa_published|fusion")),
            ("ln R0 and ln Rinf only", None, None, ba(S, "sa_coleR|abs"), None),
            ("fc inside fitted window only", None, None, EX["cole_in_window_only"]["BA"], None)]
    return pd.DataFrame([{"Analysis": a, "Pre-specified": h3(b), "Fully nested pipeline": h3(c),
                          "Cole, absolute": h3(d), "Published set": h3(e)} for a, b, c, d, e in rows])


# values printed in the article (Tables 1-4), used to verify the rebuilt tables cell by cell
PRINTED = {
    "Table 1": [["6", "105", "6", "110", "100", "6"], ["1", "36", "2", "90", "100", "0"], ["1", "45", "4", "180", "50", "0"],
                ["2", "90", "1", "45", "50", "2"], ["2", "16", "2", "20", "50", "2"], ["1", "16", "1", "14", "20", "0"]],
    "Table 2": [["16/16", "10/10", "1.000 (1.000--1.000)", "1.000", "0.984"],
                ["15/16", "10/10", "0.969 (0.857--1.000)", "0.981", "0.965"],
                ["15/16", "8/10", "0.869 (0.667--0.964)", "0.875", "0.930"],
                ["15/16", "7/10", "0.819 (0.429--0.964)", "0.813", "0.912"],
                ["15/16", "7/10", "0.819 (0.429--0.964)", "0.813", "0.908"]],
    "Table 3": [["0.888", "0.969", "0.969", "--"], ["0.681", "0.344", "0.819", "0.681"], ["0.681", "0.344", "0.819", "--"],
                ["0.825", "1.000", "1.000", "--"], ["0.781", "0.344", "0.869", "--"], ["0.813", "0.563", "0.900", "--"],
                ["0.844", "0.719", "0.950", "--"], ["0.813", "0.506", "0.763", "--"], ["0.950", "0.650", "0.888", "--"],
                ["0.950", "0.438", "0.694", "0.856"], ["0.888", "0.306", "0.706", "--"]],
    "Table 4": [["0.819", "0.950", "0.950", "0.900"], ["0.864", "0.914", "0.914", "0.900"], ["0.583", "0.608", "0.842", "0.850"],
                ["0.819", "0.756", "0.888", "0.950"], ["0.869", "--", "0.856", "--"], ["--", "0.713", "--", "--"],
                ["--", "0.950", "--", "--"], ["0.744", "--", "0.850", "0.794"], ["--", "--", "0.950", "--"],
                ["--", "--", "0.950", "--"]],
}


def verify(tabs):
    rows = []
    for name, first_col in (("Table 1", 1), ("Table 2", 2), ("Table 3", 2), ("Table 4", 1)):
        df = tabs[name]
        for i, exp in enumerate(PRINTED[name]):
            got = [str(v) for v in df.iloc[i, first_col:first_col + len(exp)].tolist()]
            for j, (e, g_) in enumerate(zip(exp, got)):
                rows.append(dict(Table=name, Row=i + 1, Column=df.columns[first_col + j], Article=e, Computed=g_,
                                 Status="PASS" if e == g_ else "FAIL"))
    return pd.DataFrame(rows)


def main():
    r, R, NS, EX = load()
    tabs = {"Table 1": table1(), "Table 2": table2(r, R), "Table 3": table3(r, R), "Table 4": table4(r, R, NS, EX)}
    with pd.ExcelWriter(f"{OUT}/article_tables.xlsx") as xw:
        for k, v in tabs.items():
            v.to_excel(xw, sheet_name=k, index=False)
    with open(f"{OUT}/article_tables.tex", "w") as fh:
        for k, v in tabs.items():
            fh.write(f"% {k}\n" + v.to_latex(index=False, escape=False) + "\n")
    chk = verify(tabs)
    chk.to_csv(f"{OUT}/article_tables_check.csv", index=False)
    return tabs, chk


if __name__ == "__main__":
    if len(sys.argv) > 1:
        OUT = sys.argv[1]
        REV = f"{OUT}/revision"
    tabs, chk = main()
    pd.set_option("display.width", 200)
    for k, v in tabs.items():
        print(f"\n{k}\n{v.to_string(index=False)}")
    bad = chk[chk.Status == "FAIL"]
    print(f"\nTables 1-4: {(chk.Status == 'PASS').sum()} / {len(chk)} cells agree with the article")
    if len(bad):
        print(bad.to_string(index=False))

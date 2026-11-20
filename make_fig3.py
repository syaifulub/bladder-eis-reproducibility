"""Figure 3 (selected frequencies and between-sample ICC), compact side-by-side layout."""
import json, os
import matplotlib
matplotlib.use("Agg")
import pandas as pd
import ijs_figures as fg
OUT = "outputs_ijs"; FIG = "outputs_ijs/figs"; os.makedirs(FIG, exist_ok=True)
r = json.load(open(f"{OUT}/ijs_results.json"))
sk = r["selected_khz"]; rows = []
for p in ["P1", "P2", "P3", "P4", "P5", "P6"]:
    rows.append((f"SW nested, {p} out", sk["naive_nested"][p], fg.C_NAIVE, "a"))
rows.append(("SW leaky", sk["naive_leaky"], "0.25", "b"))
for p in ["P1", "P2", "P3", "P4", "P5", "P6"]:
    rows.append((f"SA nested, {p} out", sk["sa_nested"][p], fg.C_SA, "c"))
rows.append(("SA leaky", sk["sa_leaky"], "#9A4A10", "d"))
rows.append(("Published set", [3.55, 568.99, 655.13, 754.31, 868.51], "black", "e"))
ic = pd.read_csv(f"{OUT}/frequency_icc.csv")
fg.frequencies_icc(rows, ic.f_kHz.values, ic.ICC.values).savefig(f"{FIG}/Fig3_Selected_Frequencies.png", bbox_inches="tight")

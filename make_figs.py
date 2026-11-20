import json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import eis_pipeline as ep, eis_groups as eg, ijs_figures as fg
from cole_analysis import iterate_spectra
OUT = "outputs_ijs"; FIG = "outputs_ijs/figs"
import os; os.makedirs(FIG, exist_ok=True)
r = json.load(open(f"{OUT}/ijs_results.json"))
S, CI = r["summary"], r["ci"]
raw = ep.load_raw_dataset("electrical-impedance-data-bladder/dataset_without_modules.pickle")
G = eg.sample_ids(raw)
specs = list(iterate_spectra(raw))
fg.schematic().savefig(f"{FIG}/Fig1_Referencing_Designs.png", bbox_inches="tight")
p1 = [(lab, g, f, z) for (p, lab, f, z), g in zip(specs, G) if p == "P1"]
dist = pd.read_csv(f"{OUT}/mahalanobis_within_between.csv")
fg.sample_structure(p1, dist).savefig(f"{FIG}/Fig2_Sample_Structure.png", bbox_inches="tight")
key = {("naive", "leaky"): "naive_leaky|fusion", ("naive", "nested"): "naive_nested|fusion",
       ("sa", "leaky"): "sa_leaky|fusion", ("sa", "nested"): "sa_nested|fusion"}
vals = {k: S[v]["BA"] for k, v in key.items()}
cis = {k: CI[v] for k, v in key.items()}
cnt = {k: f"{S[v]['TP']}/{S[v]['N1']} T, {S[v]['TN']}/{S[v]['N0']} H" for k, v in key.items()}
fg.factorial(vals, cis, cnt).savefig(f"{FIG}/FigS2_Factorial.png", bbox_inches="tight")
rv, rc = {}, {}
for feat, pre in (("freq", "nested"), ("cole", "cole")):
    for refk in ("naive", "sa"):
        for b in ("abs", "mah", "fusion"):
            k = f"{refk}_{pre}|{b}"
            rv[(feat, refk, b)] = S[k]["BA"]; rc[(feat, refk, b)] = CI[k]
fg.representations(rv, rc).savefig(f"{FIG}/FigS3_Representations.png", bbox_inches="tight")
sk = r["selected_khz"]; rows = []
for p in ["P1", "P2", "P3", "P4", "P5", "P6"]:
    rows.append((f"Spectrum-wise, nested (held-out {p})", sk["naive_nested"][p], fg.C_NAIVE, "a"))
rows.append(("Spectrum-wise, leaky", sk["naive_leaky"], "0.25", "b"))
for p in ["P1", "P2", "P3", "P4", "P5", "P6"]:
    rows.append((f"Sample-wise, nested (held-out {p})", sk["sa_nested"][p], fg.C_SA, "c"))
rows.append(("Sample-wise, leaky", sk["sa_leaky"], "#9A4A10", "d"))
rows.append(("Published set (Veil et al.)", [3.55, 568.99, 655.13, 754.31, 868.51], "black", "e"))
ic = pd.read_csv(f"{OUT}/frequency_icc.csv")
fg.frequencies_icc(rows, ic.f_kHz.values, ic.ICC.values).savefig(f"{FIG}/Fig3_Selected_Frequencies.png", bbox_inches="tight")
fits = pd.read_csv(f"{OUT}/cole_fits.csv")
fg.drift(fits).savefig(f"{FIG}/FigS1_Drift.png", bbox_inches="tight")
print("figures written:", sorted(os.listdir(FIG)))

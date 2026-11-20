"""Figures 1 and 2 (enlarged fonts; Fig. 1 includes size-matched referencing)."""
import os
import matplotlib
matplotlib.use("Agg")
import pandas as pd
import eis_pipeline as ep, eis_groups as eg, ijs_figures as fg
from cole_analysis import iterate_spectra
OUT = "outputs_ijs"; FIG = "outputs_ijs/figs"; os.makedirs(FIG, exist_ok=True)
fg.schematic().savefig(f"{FIG}/Fig1_Referencing_Designs.png", bbox_inches="tight")
raw = ep.load_raw_dataset("electrical-impedance-data-bladder/dataset_without_modules.pickle")
G = eg.sample_ids(raw)
specs = list(iterate_spectra(raw))
p1 = [(lab, g, f, z) for (p, lab, f, z), g in zip(specs, G) if p == "P1"]
dist = pd.read_csv(f"{OUT}/mahalanobis_within_between.csv")
fg.sample_structure(p1, dist).savefig(f"{FIG}/Fig2_Sample_Structure.png", bbox_inches="tight")

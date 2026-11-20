# ---------------------------------------------------------------- consistency check
S_ = SUM
def ba(k): return S_[k]["BA"]
def cnt(k, which): return S_[k]["TP"] if which == "T" else S_[k]["TN"]
def d_ci(a, b):
    key = f"{a} - {b}"
    if key in DIFF:
        return DIFF[key][1], DIFF[key][2]
    lo, hi = DIFF[f"{b} - {a}"][1], DIFF[f"{b} - {a}"][2]
    return -hi, -lo
dd = pd.read_csv(f"{OUT}/mahalanobis_within_between.csv").groupby(["Patient", "kind"]).d.median()
sk = RES["selected_khz"]
C = []
def add(item, printed, computed, dec):
    C.append((item, printed, computed, dec))

# structure
add("spectra", 767, RES["structure"]["n_spectra"], 0); add("tissue samples", 29, RES["structure"]["n_samples"], 0)
add("healthy samples", 13, RES["structure"]["n_healthy_samples"], 0); add("tumor samples", 16, RES["structure"]["n_tumor_samples"], 0)
add("spectra per sample min", 6, RES["structure"]["spectra_per_sample_min"], 0); add("spectra per sample max", 45, RES["structure"]["spectra_per_sample_max"], 0)
add("spectra per sample median", 15, RES["structure"]["spectra_per_sample_median"], 0)
add("evaluable samples", 26, RES["evaluable"]["n"], 0); add("evaluable healthy", 10, RES["evaluable"]["healthy"], 0)
for p, hs, hn, ts, tn in (("P1", 6, 105, 6, 110), ("P2", 1, 36, 2, 90), ("P3", 1, 45, 4, 180), ("P4", 2, 90, 1, 45), ("P5", 2, 16, 2, 20), ("P6", 1, 16, 1, 14)):
    r_ = T1.set_index("Patient").loc[p]
    add(f"T1 {p} healthy samples", hs, r_.HealthySamples, 0); add(f"T1 {p} healthy spectra", hn, r_.HealthySpectra, 0)
    add(f"T1 {p} tumor samples", ts, r_.TumorSamples, 0); add(f"T1 {p} tumor spectra", tn, r_.TumorSpectra, 0)
# variance decomposition
for k_, v in (("healthy|ln R0", .975), ("healthy|ln Rinf", .573), ("healthy|log10 fc", .967), ("healthy|alpha", .905),
              ("tumor|ln R0", .992), ("tumor|ln Rinf", .901), ("tumor|log10 fc", .984), ("tumor|alpha", .971)):
    add(f"ICC {k_}", v, RES["icc"][k_], 3)
add("mean spectra per sample", 26.4, RES["design_effect"]["m_bar"], 1)
add("design effect", 25.8, RES["design_effect"]["deff_healthy_lnR0"], 1); add("effective n", 29.7, RES["design_effect"]["n_eff"], 1)
add("ICC ln|Z| 1 kHz", 0.950, RES["icc_freq"]["first"], 3); add("ICC ln|Z| 1 MHz", 0.601, RES["icc_freq"]["last"], 3)
add("ICC ln|Z| monotone decreasing", 1, float(RES["icc_freq"]["monotone"]), 0)
# Cole fit
add("Cole fit error median (%)", 0.74, RES["cole_fit"]["rmse_median"], 2); add("Cole fit error p95 (%)", 2.03, RES["cole_fit"]["rmse_p95"], 2)
add("Cole fit error max (%)", 6.81, RES["cole_fit"]["rmse_max"], 2); add("SE log10 fc median", 0.016, RES["cole_ident"]["se_log10fc_median"], 3)
add("fc outside window (n)", 59, RES["cole_ident"]["outside_n"], 0); add("fc outside window (%)", 7.7, RES["cole_ident"]["outside_pct"], 1)
add("fc outside P4 healthy", 40, RES["cole_ident"]["outside_by"].get("P4-H", 0), 0); add("fc outside P1 healthy", 16, RES["cole_ident"]["outside_by"].get("P1-H", 0), 0)
add("fc outside P6 healthy", 3, RES["cole_ident"]["outside_by"].get("P6-H", 0), 0)
# distances
add("within P1", 1.40, dd[("P1", "within")], 2); add("within P4", 0.95, dd[("P4", "within")], 2); add("within P5", 1.51, dd[("P5", "within")], 2)
add("between P1", 0.85, dd[("P1", "between")], 2); add("between P5", 2.27, dd[("P5", "between")], 2); add("between P4", 48.7, dd[("P4", "between")], 1)
add("tumor P4", 2.56, dd[("P4", "tumor")], 2); add("P4 between/tumor ratio", 19, dd[("P4", "between")] / dd[("P4", "tumor")], 0)
# drift
add("drift median |rho|", 0.96, RES["drift"]["median_abs_rho"], 2); add("drift |rho|>0.8 (%)", 62, 100 * RES["drift"]["frac_abs_rho_gt_08"], 0)
add("drift median |d ln R0|", 0.057, RES["drift"]["median_abs_drift_lnR0"], 3); add("contrast median |ln R0 ratio|", 0.955, RES["drift"]["median_abs_contrast_lnR0"], 3)
add("contrast fold", 2.6, float(np.exp(RES["drift"]["median_abs_contrast_lnR0"])), 1)
# Table 2 factorial
for k_, tp, tn, b, lo, hi, auc, sp in (("naive_leaky|fusion", 16, 10, 1.000, 1.000, 1.000, 1.000, .984),
                                      ("naive_nested|fusion", 15, 10, .969, .857, 1.000, .981, .965),
                                      ("sa_leaky|fusion", 15, 8, .869, .667, .964, .875, .930),
                                      ("sa_nested|fusion", 15, 7, .819, .429, .964, .812, .912)):
    add(f"T2 {k_} tumor correct", tp, cnt(k_, "T"), 0); add(f"T2 {k_} healthy correct", tn, cnt(k_, "H"), 0)
    add(f"T2 {k_} BA", b, ba(k_), 3); add(f"T2 {k_} CI low", lo, CI[k_][0], 3); add(f"T2 {k_} CI high", hi, CI[k_][1], 3)
    add(f"T2 {k_} AUC", auc, S_[k_]["AUC"], 3); add(f"T2 {k_} spectrum BA", sp, S_[k_]["SpectrumBA"], 3)
# Table 3 effects
for name, a, b, v, lo, hi in (("ref (nested)", "naive_nested|fusion", "sa_nested|fusion", .150, .000, .500),
                              ("ref (leaky)", "naive_leaky|fusion", "sa_leaky|fusion", .131, .036, .333),
                              ("sel (spectrum-wise)", "naive_leaky|fusion", "naive_nested|fusion", .031, .000, .143),
                              ("sel (sample-wise)", "sa_leaky|fusion", "sa_nested|fusion", .050, .000, .250),
                              ("combined", "naive_leaky|fusion", "sa_nested|fusion", .181, .036, .571),
                              ("ref Mahalanobis", "naive_nested|mah", "sa_nested|mah", .625, .500, .794),
                              ("ref absolute", "naive_nested|abs", "sa_nested|abs", .206, .000, .349)):
    l_, h_ = d_ci(a, b)
    add(f"T3 {name}", v, ba(a) - ba(b), 3); add(f"T3 {name} CI low", lo, l_, 3); add(f"T3 {name} CI high", hi, h_, 3)
add("spectrum-sample gap (pp)", 9.4, 100 * (S_["sa_nested|fusion"]["SpectrumBA"] - ba("sa_nested|fusion")), 1)
add("bootstrap discarded", 139, 10000 - RES["boot_valid"], 0)
# frequencies
add("naive P6 fold min (kHz)", 121, sk["naive_nested"]["P6"][0], 0); add("naive P6 fold max (kHz)", 212, sk["naive_nested"]["P6"][-1], 0)
add("naive leaky min (kHz)", 11, sk["naive_leaky"][0], 0); add("naive leaky max (kHz)", 26, sk["naive_leaky"][-1], 0)
add("naive P1 fold two low freqs < 3 kHz", 2, sum(f_ < 3 for f_ in sk["naive_nested"]["P1"]), 0)
add("naive folds with >=3 in 10-34 kHz (P1-P5)", 5, sum(sum(10 <= f_ <= 34 for f_ in sk["naive_nested"][p]) >= 3 for p in ("P1", "P2", "P3", "P4", "P5")), 0)
add("SA folds all >= 139 kHz", 5, sum(min(sk["sa_nested"][p]) >= 138.9 for p in ("P1", "P2", "P3", "P4", "P5", "P6")), 0)
add("SA P5 fold min (kHz)", 8.3, sk["sa_nested"]["P5"][0], 1); add("SA P5 fold max (kHz)", 19.3, sk["sa_nested"]["P5"][-1], 1)
add("SA leaky min (kHz)", 160, sk["sa_leaky"][0], 0); add("SA leaky max (kHz)", 1000, sk["sa_leaky"][-1], 0)
# Table 5 representations
for k_, v in (("naive_nested|abs", .888), ("naive_nested|mah", .969), ("naive_nested|fusion", .969),
              ("sa_nested|abs", .681), ("sa_nested|mah", .344), ("sa_nested|fusion", .819), ("sa_nested|innerw", .681),
              ("naive_leaky|abs", .825), ("naive_leaky|mah", 1.000), ("naive_leaky|fusion", 1.000),
              ("sa_leaky|abs", .781), ("sa_leaky|mah", .344), ("sa_leaky|fusion", .869),
              ("sa_published|abs", .813), ("sa_published|mah", .563), ("sa_published|fusion", .900),
              ("sa_generic|abs", .813), ("sa_generic|mah", .506), ("sa_generic|fusion", .763),
              ("naive_cole|abs", .950), ("naive_cole|mah", .650), ("naive_cole|fusion", .888),
              ("sa_cole|abs", .950), ("sa_cole|mah", .438), ("sa_cole|fusion", .694), ("sa_cole|innerw", .856)):
    add(f"T5 {k_}", v, ba(k_), 3)
add("SA nested mah healthy correct", 0, cnt("sa_nested|mah", "H"), 0)
add("SA nested mah CI low", .150, CI["sa_nested|mah"][0], 3); add("SA nested mah CI high", .472, CI["sa_nested|mah"][1], 3)
add("published CI low", .833, CI["sa_published|fusion"][0], 3); add("published CI high", 1.000, CI["sa_published|fusion"][1], 3)
add("published tumor correct", 16, cnt("sa_published|fusion", "T"), 0); add("published healthy correct", 8, cnt("sa_published|fusion", "H"), 0)
add("generic CI low", .586, CI["sa_generic|fusion"][0], 3); add("generic CI high", 1.000, CI["sa_generic|fusion"][1], 3)
l_, h_ = d_ci("sa_published|fusion", "sa_nested|fusion")
add("published - nested", .081, ba("sa_published|fusion") - ba("sa_nested|fusion"), 3); add("published - nested CI low", -.071, l_, 3); add("published - nested CI high", .571, h_, 3)
add("Cole abs SA CI low", .917, CI["sa_cole|abs"][0], 3); add("Cole abs SA CI high", 1.000, CI["sa_cole|abs"][1], 3)
add("Cole abs tumor correct", 16, cnt("sa_cole|abs", "T"), 0); add("Cole abs healthy correct", 9, cnt("sa_cole|abs", "H"), 0)
add("Cole inner CI low", .700, CI["sa_cole|innerw"][0], 3); add("Cole inner CI high", .962, CI["sa_cole|innerw"][1], 3)
add("Cole weights min", 0.6, min(RES["inner_weights"]["sa_cole"].values()), 1); add("Cole weights max", 1.0, max(RES["inner_weights"]["sa_cole"].values()), 1)
add("freq weights min", 0.5, min(RES["inner_weights"]["sa_nested"].values()), 1); add("freq weights max", 0.7, max(RES["inner_weights"]["sa_nested"].values()), 1)
add("corrected correct samples", 22, cnt("sa_nested|fusion", "T") + cnt("sa_nested|fusion", "H"), 0)
add("misclassified n min", 6, RES["misclassified"]["n_min"], 0); add("misclassified n max", 14, RES["misclassified"]["n_max"], 0)
add("correct mean n", 28.6, RES["misclassified"]["mean_n_correct"], 1)
add("P1 tumor prob min", .508, RES["misclassified"]["p1_tumor_min"], 3); add("P1 tumor prob max", .508, RES["misclassified"]["p1_tumor_max"], 3)
add("P1 wrong healthy prob", .507, RES["misclassified"]["p1_healthy_wrong"], 3)
# Table 6 sensitivity
for k_, tp, tn, b in (("sa_nested|fusion|cb", 15, 8, .869), ("sa_cole|abs|cb", 13, 9, .856), ("sa_cole|fusion|cb", 10, 9, .763),
                      ("noP2_naive_nested|fusion", 13, 10, .964), ("noP2_sa_nested|fusion", 13, 8, .864), ("noP2_sa_cole|abs", 13, 9, .914),
                      ("sa_coleR|abs", 16, 9, .950), ("sa_coleR|mah", 1, 3, .181), ("sa_coleR|fusion", 11, 5, .594)):
    add(f"T6 {k_} tumor", tp, cnt(k_, "T"), 0); add(f"T6 {k_} healthy", tn, cnt(k_, "H"), 0); add(f"T6 {k_} BA", b, ba(k_), 3)
add("noP2 referencing effect (pp)", 10.0, 100 * (ba("noP2_naive_nested|fusion") - ba("noP2_sa_nested|fusion")), 1)

out = []
for item, printed, computed, dec in C:
    ok_ = abs(float(computed) - float(printed)) <= 0.5 * 10 ** (-dec) + 1e-9
    out.append(dict(Item=item, Manuscript=printed, Computed=round(float(computed), dec + 2), Status="PASS" if ok_ else "FAIL"))
CHECK = pd.DataFrame(out)
CHECK.to_csv(f"{OUT}/consistency_check.csv", index=False)
pd.set_option("display.max_rows", 400)
print(CHECK[CHECK.Status == "FAIL"].to_string() if (CHECK.Status == "FAIL").any() else "no failures")
print(f"\n{(CHECK.Status == 'PASS').sum()} / {len(CHECK)} values agree with the manuscript.")
_m = fits.groupby(["Patient", "Label"]).R0.median().unstack()
_r = _m[1] / _m[0]
C2 = [("P2 tumor/healthy R0 ratio", 10.4, _r["P2"], 1),
      ("other patients with lower tumor R0", 4, int((_r.drop("P2") < 1).sum()), 0)]
for item, printed, computed, dec in C2:
    ok_ = abs(float(computed) - float(printed)) <= 0.5 * 10 ** (-dec) + 1e-9
    CHECK.loc[len(CHECK)] = dict(Item=item, Manuscript=printed, Computed=round(float(computed), dec + 2), Status="PASS" if ok_ else "FAIL")
CHECK.to_csv(f"{OUT}/consistency_check.csv", index=False)
print(f"after P2 checks: {(CHECK.Status == 'PASS').sum()} / {len(CHECK)} values agree with the manuscript.")

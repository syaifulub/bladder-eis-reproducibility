# ---------------------------------------------------------------- consistency check: analyses added in review
NS = json.load(open("outputs_ijs/revision/nested_sensitivity.json"))
EX = json.load(open("outputs_ijs/revision/reviewer_extras.json"))
SP = json.load(open("outputs_ijs/revision/selection_permutation_results.json"))
SP["_above_all"] = float(SP["BA_selected"]["null_max"] < SP["BA_selected"]["observed"])
SE = json.load(open("outputs_ijs/revision/sim_ext_summary.json"))
C4 = []
def add4(item, printed, computed, dec):
    C4.append((item, printed, computed, dec))

# design effect ranges
de_ = EX["design_effect"]
add4("N_eff min", 29.2, de_["N_eff_min"], 1); add4("N_eff max", 49.2, de_["N_eff_max"], 1)
add4("N_eff Kish min", 21.2, de_["N_eff_kish_min"], 1); add4("N_eff Kish max", 35.9, de_["N_eff_kish_max"], 1)
add4("Discussion: 21 to 49 observations (min)", 21, de_["N_eff_kish_min"], 0); add4("Discussion: 21 to 49 (max)", 49, de_["N_eff_max"], 0)
# frequency stability
fs_ = EX["frequency_stability"]
add4("share of outer sample-wise selections above 100 kHz (%)", 83, 100 * fs_["sample-wise outer"]["share_above_100k"], 0)
add4("Jaccard sample-wise outer", .14, fs_["sample-wise outer"]["mean_jaccard"], 2)
add4("Jaccard sample-wise inner", .13, fs_["sample-wise inner"]["mean_jaccard"], 2)
add4("Jaccard spectrum-wise outer", .25, fs_["spectrum-wise outer"]["mean_jaccard"], 2)
# nested sensitivity
def nsv(k, tp, n1, tn, n0, ba):
    v = NS[k]
    add4(f"{k} tumor", tp, v["TP"], 0); add4(f"{k} N1", n1, v["N1"], 0)
    add4(f"{k} healthy", tn, v["TN"], 0); add4(f"{k} N0", n0, v["N0"], 0); add4(f"{k} BA", ba, v["BA"], 3)
nsv("fallback_generic", 16, 16, 9, 10, .950); nsv("fallback_outer", 10, 16, 8, 10, .713)
nsv("matched", 13, 16, 7, 10, .756); nsv("p2_excluded", 13, 14, 9, 10, .914); nsv("p2_exchanged", 7, 15, 9, 12, .608)
add4("fallback generic: Cole abs in all folds", 6, NS["fallback_generic"]["n_cole"] * (NS["fallback_generic"]["n_abs"] == 6), 0)
fo = NS["fallback_outer"]["choice"]
add4("fallback outer: frequency abs in P1 and P5", 2, sum(fo[p]["representation"] == "freq" and fo[p]["branch"] == "abs" for p in ("P1", "P5")), 0)
add4("fallback outer equals all-fold reuse (choices)", 1, float(all((fo[p]["representation"], fo[p]["branch"]) == (REVP["nested_pipeline_choice"][p]["representation"], REVP["nested_pipeline_choice"][p]["branch"]) for p in fo)), 0)
add4("fallback outer equals all-fold reuse (BA)", .713, REVP["summary"]["nested_pipeline|selected"]["BA"], 3)
add4("matched CI low", .308, NS["matched"]["CI"][0], 3); add4("matched CI high", .923, NS["matched"]["CI"][1], 3)
add4("matched: folds with Cole abs", 4, sum(v["representation"] == "cole" and v["branch"] == "abs" for v in NS["matched"]["choice"].values()), 0)
_stm = pd.read_csv("outputs_ijs/revision/sampletable_nested_matched.csv")
_err = _stm[(_stm.p >= 0.5).astype(int) != _stm.y]
add4("matched: both P2 tumor samples misclassified", 2, int(((_err.Patient == "P2") & (_err.y == 1)).sum()), 0)
add4("P2 excluded: Cole abs in all five folds", 5, sum(v["representation"] == "cole" and v["branch"] == "abs" for v in NS["p2_excluded"]["choice"].values()), 0)
add4("P2 exchanged: distinct pipelines", 4, len({(v["representation"], v["branch"]) for v in NS["p2_exchanged"]["choice"].values()}), 0)
# robustness table entries not checked elsewhere
add4("T7 published without P2", .900, EX["published_without_P2"]["BA"], 3)
add4("corrected range min", .583, min(RS["p2_sa_nested|fusion"]["BA"], NS["p2_exchanged"]["BA"]), 3)
add4("corrected range max", .950, max(NS["fallback_generic"]["BA"], RS["nested_pipeline|selected"]["BA"]), 3)
add4("conclusion: down to 0.61", .61, NS["p2_exchanged"]["BA"], 2)
# influence and stability
inf_ = EX["influence"]
add4("influence fully nested min", .938, inf_["Fully nested pipeline"]["min"], 3); add4("influence fully nested max", 1.000, inf_["Fully nested pipeline"]["max"], 3)
add4("influence pre-specified min", .700, inf_["Sample-wise, nested frequencies"]["min"], 3)
add4("influence pre-specified min is P1", 1, float(inf_["Sample-wise, nested frequencies"]["min_patient"] == "P1"), 0)
add4("influence pre-specified max", .902, inf_["Sample-wise, nested frequencies"]["max"], 3)
add4("influence pre-specified max is P5", 1, float(inf_["Sample-wise, nested frequencies"]["max_patient"] == "P5"), 0)
ss_ = EX["seed_stability"]
add4("seed Cole abs min", .919, ss_["Cole absolute (fully nested choice)"]["BA_min"], 3)
add4("seed Cole abs max", .950, ss_["Cole absolute (fully nested choice)"]["BA_max"], 3)
add4("seed Cole abs samples changing class", 1, ss_["Cole absolute (fully nested choice)"]["n_samples_changing_class"], 0)
add4("seed pre-specified samples changing class", 0, ss_["Nested frequencies, equal fusion"]["n_samples_changing_class"], 0)
add4("seed pre-specified BA range", 0.0, ss_["Nested frequencies, equal fusion"]["BA_max"] - ss_["Nested frequencies, equal fusion"]["BA_min"], 3)
iw_ = EX["cole_in_window_only"]
add4("in-window Cole abs BA", .950, iw_["BA"], 3); add4("in-window tumor", 16, iw_["TP"], 0); add4("in-window healthy", 9, iw_["TN"], 0)
add4("out-of-window spectra", 59, EX["cole_out_of_window"]["n"], 0)
add4("misclassified sample out-of-window spectra", 14, iw_["misclassified_main_out_of_window"], 0)
add4("misclassified sample is healthy P1", 1, float(len(iw_["misclassified_main"]) == 1 and iw_["misclassified_main"][0].startswith("P1:hbw")), 0)
# paired bootstrap: sample-wise vs size-matched referencing
PM = json.load(open("outputs_ijs/revision/paired_matched.json"))
add4("paired: nested sa - matched difference", .194, PM["nested"]["diff"], 3)
add4("paired: nested difference CI low", .023, PM["nested"]["ci_low"], 3)
add4("paired: nested difference CI high", .692, PM["nested"]["ci_high"], 3)
add4("paired: Cole abs sa - matched difference", .063, PM["coleabs"]["diff"], 3)
add4("paired: Cole abs difference CI low", 0.000, PM["coleabs"]["ci_low"], 3)
add4("paired: Cole abs difference CI high", .250, PM["coleabs"]["ci_high"], 3)
add4("paired: pre-specified unchanged", 0.0, PM["prespecified"]["diff"], 3)
add4("paired: valid bootstrap draws", 9861, PM["n_valid"], 0)
# selection-aware permutation
SEL_PRINTED = globals().get("SEL_PRINTED", [])
for item, printed, keys, dec in SEL_PRINTED:
    v_ = SP
    for k in keys:
        v_ = v_[k]
    add4(item, printed, v_, dec)
# simulation scenarios
SE_PRINTED = globals().get("SE_PRINTED", [])
for item, printed, fn, dec in SE_PRINTED:
    add4(item, printed, fn(SE), dec)

for item, printed, computed, dec in C4:
    ok_ = abs(float(computed) - float(printed)) <= 0.5 * 10 ** (-dec) + 1e-6
    CHECK.loc[len(CHECK)] = dict(Item=item, Manuscript=printed, Computed=round(float(computed), dec + 2), Status="PASS" if ok_ else "FAIL")
CHECK.to_csv(f"{OUT}/consistency_check.csv", index=False)
print(CHECK[CHECK.Status == "FAIL"].to_string() if (CHECK.Status == "FAIL").any() else "no failures")
print(f"after review checks: {(CHECK.Status == 'PASS').sum()} / {len(CHECK)} values agree with the manuscript.")

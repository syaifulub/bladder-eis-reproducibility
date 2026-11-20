# ---------------------------------------------------------------- consistency check: revision analyses
REVR = json.load(open("outputs_ijs/revision/revision_results.json"))
PERM = json.load(open("outputs_ijs/revision/permutation_results.json"))
SIMS = json.load(open("outputs_ijs/revision/simulation_summary.json"))
RS = REVR["summary"]
C3 = []
def add3(item, printed, computed, dec):
    C3.append((item, printed, computed, dec))
def rdiff(group, a, b):
    d = REVR["diff"][group]
    if f"{a} - {b}" in d:
        return d[f"{a} - {b}"][1], d[f"{a} - {b}"][2]
    lo, hi = d[f"{b} - {a}"][1], d[f"{b} - {a}"][2]
    return -hi, -lo

# Table 7 permutation null
for k_, mean_, lo_, hi_, obs_, p_ in (("cole|naive|fusion", .696, .512, .869, .887, .019), ("cole|sa|fusion", .454, .312, .637, .694, .010),
                                      ("cole|naive|abs", .563, .344, .806, .950, .001), ("cole|sa|abs", .459, .312, .700, .950, .001),
                                      ("cole|naive|mah", .764, .619, .906, .650, .957), ("cole|sa|mah", .459, .250, .669, .438, .582),
                                      ("published|naive|fusion", .740, .562, .906, .875, .081), ("published|sa|fusion", .445, .312, .656, .900, .001),
                                      ("published|naive|mah", .823, .719, .938, 1.000, .001), ("published|sa|mah", .424, .250, .625, .563, .088)):
    v_ = PERM[k_]
    add3(f"T7 {k_} null mean", mean_, v_["null_mean"], 3); add3(f"T7 {k_} q025", lo_, v_["null_q025"], 3)
    add3(f"T7 {k_} q975", hi_, v_["null_q975"], 3); add3(f"T7 {k_} observed", obs_, v_["observed"], 3)
    add3(f"T7 {k_} p", p_, v_["p_value"], 3)
add3("permutations", 1000, PERM["n_perm"], 0)
add3("published naive mah null >= 0.75 (%)", 95.8, 100 * PERM["published|naive|mah"]["frac_null_ge_0p75"], 1)
add3("SA null mean min", .424, min(PERM[k]["null_mean"] for k in PERM if "|sa|" in k), 3)
add3("SA null mean max", .484, max(PERM[k]["null_mean"] for k in PERM if "|sa|" in k), 3)
add3("SA null q975 min", .625, min(PERM[k]["null_q975"] for k in PERM if "|sa|" in k), 3)
add3("SA null q975 max", .750, max(PERM[k]["null_q975"] for k in PERM if "|sa|" in k), 3)
# fully nested pipeline selection
ch = REVR["nested_pipeline_choice"]
add3("folds choosing absolute branch", 6, sum(v["branch"] == "abs" for v in ch.values()), 0)
add3("folds choosing Cole", 6, sum(v["representation"] == "cole" for v in ch.values()), 0)
add3("inner BA of chosen pipeline min", .866, min(v["inner_BA"] for v in ch.values()), 3)
add3("inner BA of chosen pipeline max", 1.000, max(v["inner_BA"] for v in ch.values()), 3)
_inn = pd.read_csv("outputs_ijs/revision/nested_pipeline_inner_scores.csv")
_bestf = _inn[_inn.Representation == "freq"].groupby("Fold").InnerBA.max()
_bestc = _inn[(_inn.Representation == "cole") & (_inn.Branch == "abs")].set_index("Fold").InnerBA
add3("folds where Cole abs beats best frequency candidate", 6, int((_bestc > _bestf).sum()), 0)
add3("inner folds with LOSO fallback", 6, sum(v == "LOSO" for v in REVR["inner_selection_method"].values()), 0)
add3("inner folds total", 30, len(REVR["inner_selection_method"]), 0)
s_ = RS["nested_pipeline|selected"]
add3("nested pipeline BA", .950, s_["BA"], 3); add3("nested pipeline tumor", 16, s_["TP"], 0); add3("nested pipeline healthy", 9, s_["TN"], 0)
add3("nested pipeline CI low", .917, REVR["ci"]["nested_pipeline|selected"][0], 3)
add3("nested pipeline CI high", 1.000, REVR["ci"]["nested_pipeline|selected"][1], 3)
add3("nested pipeline - pre-specified", .131, s_["BA"] - ba("sa_nested|fusion"), 3)
lo_, hi_ = rdiff("main_extra", "nested_pipeline|selected", "sa_nested|fusion")
add3("nested pipeline - pre-specified CI low", 0.000, lo_, 3); add3("nested pipeline - pre-specified CI high", .571, hi_, 3)
add3("nested pipeline equals Cole abs", 1, float(abs(s_["BA"] - ba("sa_cole|abs")) < 1e-9), 0)
# incomplete nesting: outer-fold frequency ranking reused in the inner folds
REVP = json.load(open("outputs_ijs/revision/revision_results_v2_outer_ranking.json"))
_innp = pd.read_csv("outputs_ijs/revision/nested_pipeline_inner_scores_v2_outer_ranking.csv")
_bestfp = _innp[_innp.Representation == "freq"].groupby("Fold").InnerBA.max()
add3("partial: max rise of best frequency candidate", .375, (_bestfp - _bestf).max(), 3)
add3("partial: P1 best frequency inner BA", 1.000, _bestfp["P1"], 3); add3("full: P1 best frequency inner BA", .625, _bestf["P1"], 3)
chp = REVP["nested_pipeline_choice"]
add3("partial: P1 chose frequency abs", 1, float(chp["P1"]["representation"] == "freq" and chp["P1"]["branch"] == "abs"), 0)
add3("partial: P5 chose frequency abs", 1, float(chp["P5"]["representation"] == "freq" and chp["P5"]["branch"] == "abs"), 0)
sp_ = REVP["summary"]["nested_pipeline|selected"]
add3("partial pipeline BA", .713, sp_["BA"], 3); add3("partial pipeline tumor", 10, sp_["TP"], 0); add3("partial pipeline healthy", 8, sp_["TN"], 0)
# size-matched referencing
for k_, v in (("matched_nested|fusion", .819), ("matched_cole|abs", .887), ("matched_published|fusion", .950),
              ("matched_nested|mah", .344), ("matched_cole|mah", .306)):
    add3(f"matched {k_}", v, RS[k_]["BA"], 3)
add3("matched nested correct samples", 22, RS["matched_nested|fusion"]["TP"] + RS["matched_nested|fusion"]["TN"], 0)
# P2 labels exchanged
add3("P2 swap evaluable healthy", 12, REVR["p2swap_evaluable"]["healthy"], 0); add3("P2 swap evaluable tumor", 15, REVR["p2swap_evaluable"]["tumor"], 0)
for k_, tp, tn, b in (("p2_naive_nested|fusion", 15, 10, .917), ("p2_sa_nested|fusion", 10, 6, .583),
                      ("p2_sa_cole|abs", 14, 9, .842), ("p2_sa_published|fusion", 13, 10, .850)):
    add3(f"T6 {k_} tumor", tp, RS[k_]["TP"], 0); add3(f"T6 {k_} healthy", tn, RS[k_]["TN"], 0); add3(f"T6 {k_} BA", b, RS[k_]["BA"], 3)
add3("P2 swap referencing effect", .333, RS["p2_naive_nested|fusion"]["BA"] - RS["p2_sa_nested|fusion"]["BA"], 3)
lo_, hi_ = rdiff("p2swap", "p2_naive_nested|fusion", "p2_sa_nested|fusion")
add3("P2 swap effect CI low", .113, lo_, 3); add3("P2 swap effect CI high", .760, hi_, 3)
# logistic regression
for k_, tp, tn, b in (("lr_naive_nested|fusion", 10, 10, .812), ("lr_naive_leaky|fusion", 13, 9, .856),
                      ("lr_sa_nested|fusion", 11, 8, .744), ("lr_sa_leaky|fusion", 11, 9, .794),
                      ("lr_naive_cole|abs", 15, 9, .919), ("lr_sa_cole|abs", 16, 7, .850), ("lr_sa_nested|mah", 16, 0, .500)):
    add3(f"T6 {k_} tumor", tp, RS[k_]["TP"], 0); add3(f"T6 {k_} healthy", tn, RS[k_]["TN"], 0); add3(f"T6 {k_} BA", b, RS[k_]["BA"], 3)
add3("LR referencing effect nested", .069, RS["lr_naive_nested|fusion"]["BA"] - RS["lr_sa_nested|fusion"]["BA"], 3)
add3("LR referencing effect leaky", .063, RS["lr_naive_leaky|fusion"]["BA"] - RS["lr_sa_leaky|fusion"]["BA"], 3)
add3("LR referencing effect Cole fusion", .138, RS["lr_naive_cole|fusion"]["BA"] - RS["lr_sa_cole|fusion"]["BA"], 3)
add3("LR Cole Mahalanobis SA", .438, RS["lr_sa_cole|mah"]["BA"], 3)
add3("LR Cole Mahalanobis SA healthy correct", 0, RS["lr_sa_cole|mah"]["TN"], 0)
# simulation
SIM_PRINTED = globals().get("SIM_PRINTED", [])
for item, printed, path, dec in SIM_PRINTED:
    v_ = SIMS
    for k in path:
        v_ = v_[k]
    add3(item, printed, v_, dec)

for item, printed, computed, dec in C3:
    ok_ = abs(float(computed) - float(printed)) <= 0.5 * 10 ** (-dec) + 1e-6
    CHECK.loc[len(CHECK)] = dict(Item=item, Manuscript=printed, Computed=round(float(computed), dec + 2), Status="PASS" if ok_ else "FAIL")
CHECK.to_csv(f"{OUT}/consistency_check.csv", index=False)
print(CHECK[CHECK.Status == "FAIL"].to_string() if (CHECK.Status == "FAIL").any() else "no failures")
print(f"after revision checks: {(CHECK.Status == 'PASS').sum()} / {len(CHECK)} values agree with the manuscript.")

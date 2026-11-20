"""Tables S1-S9 of Online Resource 1, written as LaTeX (compact layout for the 25-page limit)."""
import json
import os
import pandas as pd

REV = os.environ.get("REV_DIR", "outputs_ijs/revision")
OUTF = os.environ.get("SUPP_OUT", "outputs_ijs/supp_tables.tex")
L = []
R = json.load(open(f"{REV}/revision_results.json"))
S, CI = R["summary"], R["ci"]
RP = json.load(open(f"{REV}/revision_results_v2_outer_ranking.json"))
PATS = ["P1", "P2", "P3", "P4", "P5", "P6"]


def h3(v):
    """Three decimals, rounding halves up (as in the printed article); tiny float error is removed first."""
    from decimal import Decimal, ROUND_HALF_UP
    return str(Decimal(f"{float(v):.10f}").quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))


def s3(v):
    """Signed three-decimal value, halves rounded away from zero."""
    t = h3(abs(v))
    if t == "0.000":
        return t
    return ("-" if float(v) < 0 else "+") + t


def begin(caption, spec, header, size=r"\footnotesize", sep="4pt"):
    L.append(r"\begin{table}[H]" "\n" r"\centering" "\n" r"\caption{" + caption + "}\n" + size + "\n"
             r"\setlength{\tabcolsep}{" + sep + "}\n" r"\begin{tabular}{" + spec + "}\n" r"\toprule" "\n" + header + "\n" r"\midrule")


def end():
    L.append(r"\bottomrule" "\n" r"\end{tabular}" "\n" r"\end{table}" "\n")


# ---------------------------------------------------------------- S1 ICC and design effect
X = json.load(open(f"{REV}/reviewer_extras.json"))["design_effect"]
pn = {"ln R0": r"$\ln R_0$", "ln Rinf": r"$\ln R_{\infty}$", "log10 fc": r"$\log_{10} f_c$", "alpha": r"$\alpha$"}
begin(r"Intraclass correlation (ICC) of Cole parameters between tissue samples (after removal of patient means), design effect (DE), "
      r"and effective number of independent observations among the 767 spectra, without and with the Kish correction "
      r"(mean " + f"{X['m_bar']:.1f}" + r" spectra per sample, squared coefficient of variation " + f"{X['cv2']:.2f}" + r"). "
      r"ICC and DE are class-specific, but $N_{\mathrm{eff}}=767/\mathrm{DE}$ applies each DE to all 767 spectra: it is the effective size "
      r"of the whole dataset if every spectrum had that ICC, not the number of independent observations per class. The figures describe the "
      r"redundancy of repeats for one parameter at a time; for generalization, the six patients remain the top level of the hierarchy.",
      "llccccc", r"Class & Parameter & ICC & DE & $N_{\mathrm{eff}}$ & DE (Kish) & $N_{\mathrm{eff}}$ (Kish) \\")
for k, v in X["per_parameter"].items():
    cl, par = k.split("|")
    L.append(f"{cl.capitalize()} & {pn[par]} & {h3(v['ICC'])} & {v['DE']:.1f} & {v['N_eff']:.1f} & {v['DE_kish']:.1f} & {v['N_eff_kish']:.1f} \\\\")
end()

# ---------------------------------------------------------------- S2 per patient and influence
pp = pd.read_csv(f"{REV}/per_patient_confusion.csv")
inf = pd.read_csv(f"{REV}/leave_one_patient_out_influence.csv")
begin(r"Per-patient outcomes and influence. Left: correctly classified tumor (T) and healthy (H) samples of the pre-specified "
      r"classifier (nested frequencies, equal fusion, sample-wise referencing) and of the fully nested pipeline; healthy samples "
      r"of P2, P3 and P6 are not evaluable under sample-wise referencing. Right: pooled sample-level balanced accuracy recomputed "
      r"with the samples of the listed patient removed. Patient-level BA: sensitivity and specificity computed within each patient and "
      r"averaged over patients (specificity over P1, P4, and P5).",
      "lcccccccc",
      r" & \multicolumn{2}{c}{Pre-specified} & \multicolumn{2}{c}{Fully nested} & \multicolumn{4}{c}{BA without the patient} \\" "\n"
      r"Patient & T & H & T & H & \shortstack{Spectrum-wise,\\nested} & \shortstack{Pre-\\specified} & \shortstack{Published\\set} & \shortstack{Fully\\nested} \\",
      sep="3pt")


def th(cfg, p):
    r = pp[(pp.Config == cfg) & (pp.Patient == p)].iloc[0]
    t = f"{r.TP}/{r.TP + r.FN}" if r.TP + r.FN else "--"
    h = f"{r.TN}/{r.TN + r.FP}" if r.TN + r.FP else "--"
    return t, h


for p in PATS:
    a = th("Sample-wise, nested frequencies", p)
    b = th("Fully nested pipeline", p)
    vals = [inf[(inf.Config == c) & (inf.Removed == p)].BA.iloc[0] for c in
            ("Spectrum-wise, nested frequencies", "Sample-wise, nested frequencies", "Sample-wise, published set", "Fully nested pipeline")]
    L.append(f"{p} & {a[0]} & {a[1]} & {b[0]} & {b[1]} & " + " & ".join(f"{h3(v)}" for v in vals) + r" \\")


def patient_ba(cfg):
    d = pp[pp.Config == cfg]
    se = (d.TP / (d.TP + d.FN)).dropna().mean()
    sp = (d.TN / (d.TN + d.FP)).dropna().mean()
    return 0.5 * (se + sp)


L.append(r"\midrule")
L.append(r"\multicolumn{9}{l}{Patient-level BA: pre-specified " + h3(patient_ba("Sample-wise, nested frequencies")) +
         ", fully nested " + h3(patient_ba("Fully nested pipeline")) + r"} \\")
end()

# ---------------------------------------------------------------- S3 permutation null of fixed configurations
PR = json.load(open(f"{REV}/permutation_results.json"))
begin(r"Permutation null of the sample-level balanced accuracy for fixed configurations (" + f"{PR['n_perm']}" +
      r" within-patient permutations of tissue-sample labels). Null: mean and 2.5th--97.5th percentiles; $p$: permutation "
      r"$p$-value of the observed balanced accuracy.",
      "lllcccc", r"Features & Output & Referencing & Null mean & Null 95\% range & Observed & $p$ \\", sep="3.5pt")
fn = {"cole": "Cole", "published": "Published set"}
on = {"fusion": "Equal fusion", "abs": "Absolute", "mah": "Mahalanobis"}
for f in ("cole", "published"):
    for o in ("fusion", "abs", "mah"):
        for i, ref in enumerate(("naive", "sa")):
            v = PR[f"{f}|{ref}|{o}"]
            L.append(f"{fn[f] if (o == 'fusion' and i == 0) else ''} & {on[o] if i == 0 else ''} & "
                     f"{'Spectrum-wise' if ref == 'naive' else 'Sample-wise'} & {h3(v['null_mean'])} & "
                     f"{h3(v['null_q025'])}--{h3(v['null_q975'])} & {h3(v['observed'])} & {h3(v['p_value'])} \\\\")
end()

# ---------------------------------------------------------------- S4 nested pipeline inner scores
inner = pd.read_csv(f"{REV}/nested_pipeline_inner_scores.csv")
choice = R["nested_pipeline_choice"]
lab = {("freq", "fusion"): "Freq.\\ fusion", ("freq", "abs"): "Freq.\\ abs.", ("freq", "mah"): "Freq.\\ Mah.",
       ("cole", "fusion"): "Cole fusion", ("cole", "abs"): "Cole abs.", ("cole", "mah"): "Cole Mah."}
cols = list(lab)
begin(r"Fully nested pipeline selection (Algorithm~S1): inner leave-one-patient-out sample-level balanced accuracy of the six candidate "
      r"pipelines within each outer fold (five training patients, sample-wise referencing; frequencies re-ranked in every inner fold). "
      r"Freq.: five selected frequencies; Cole: Cole parameters; abs.: absolute-deviation branch; Mah.: Mahalanobis branch; fusion: equal "
      r"fusion of both. The candidate with the highest inner BA (in bold) was refitted and evaluated once on the held-out patient.",
      "l" + "c" * len(cols), "Held-out patient & " + " & ".join(lab[c] for c in cols) + r" \\")
for p in sorted(inner.Fold.unique()):
    d = inner[inner.Fold == p].set_index(["Representation", "Branch"]).InnerBA
    ch = (choice[p]["representation"], choice[p]["branch"])
    cells = [(r"\textbf{%s}" if c == ch else "%s") % h3(d[c]) for c in cols]
    L.append(f"{p} & " + " & ".join(cells) + r" \\")
end()

# ---------------------------------------------------------------- S5 fully nested selection: variants
NS = json.load(open(f"{REV}/nested_sensitivity.json"))
ab = {("cole", "abs"): "C-a", ("cole", "fusion"): "C-f", ("cole", "mah"): "C-m",
      ("freq", "abs"): "F-a", ("freq", "fusion"): "F-f", ("freq", "mah"): "F-m"}
main_ = dict(choice=R["nested_pipeline_choice"], **{k: S["nested_pipeline|selected"][k] for k in ("BA", "TP", "N1", "TN", "N0")},
             CI=CI["nested_pipeline|selected"])
part_ = dict(choice=RP["nested_pipeline_choice"], **{k: RP["summary"]["nested_pipeline|selected"][k] for k in ("BA", "TP", "N1", "TN", "N0")},
             CI=RP["ci"]["nested_pipeline|selected"])
variants = [("Primary (fallback: sample ranking)", main_), ("Fallback: fixed set", NS["fallback_generic"]),
            ("Fallback: outer ranking", NS["fallback_outer"]), ("Outer ranking, all inner folds", part_),
            ("Size-matched referencing", NS["matched"]), ("P2 excluded", NS["p2_excluded"]), ("P2 labels exchanged", NS["p2_exchanged"])]
begin(r"Fully nested pipeline selection under alternative treatments: pipeline chosen in each outer fold (C: Cole parameters, "
      r"F: frequencies; a: absolute branch, f: equal fusion, m: Mahalanobis branch), correctly classified tumor and healthy samples, "
      r"and sample-level balanced accuracy with patient-level cluster bootstrap 95\% CI. Fallback rows differ only in the six inner folds "
      r"in which patient-level frequency ranking was impossible (primary: leave-one-sample-out ranking within the inner-training patients; "
      r"fixed set: decade-spaced frequencies; outer ranking: frequencies of the outer fold, a deliberately leaky diagnostic).",
      "lccccccccc", r"Variant & P1 & P2 & P3 & P4 & P5 & P6 & Tumor & Healthy & BA (95\% CI) \\", sep="3pt")
for name, v in variants:
    ch = [ab[(v["choice"][p]["representation"], v["choice"][p]["branch"])] if p in v["choice"] else "--" for p in PATS]
    L.append(f"{name} & " + " & ".join(ch) + f" & {v['TP']}/{v['N1']} & {v['TN']}/{v['N0']} & {h3(v['BA'])} ({h3(v['CI'][0])}--{h3(v['CI'][1])}) \\\\")
end()

# ---------------------------------------------------------------- S6 additional real-data analyses
S["partial_pipeline|selected"] = RP["summary"]["nested_pipeline|selected"]
CI["partial_pipeline|selected"] = RP["ci"]["nested_pipeline|selected"]
rows = [("Pipeline selection", "nested_pipeline|selected", "Fully nested"),
        ("", "partial_pipeline|selected", "Outer-fold ranking reused"),
        ("Size-matched referencing", "matched_nested|fusion", "Nested frequencies, fusion"),
        ("", "matched_nested|abs", "Nested frequencies, absolute"),
        ("", "matched_nested|mah", "Nested frequencies, Mahalanobis"),
        ("", "matched_cole|fusion", "Cole, fusion"), ("", "matched_cole|abs", "Cole, absolute"),
        ("", "matched_cole|mah", "Cole, Mahalanobis"), ("", "matched_published|fusion", "Published set, fusion"),
        ("P2 labels exchanged", "p2_naive_leaky|fusion", "Spectrum-wise, leaky"),
        ("", "p2_naive_nested|fusion", "Spectrum-wise, nested"), ("", "p2_sa_leaky|fusion", "Sample-wise, leaky"),
        ("", "p2_sa_nested|fusion", "Sample-wise, nested"), ("", "p2_sa_cole|abs", "Sample-wise, Cole absolute"),
        ("", "p2_sa_published|fusion", "Sample-wise, published set"),
        ("Logistic regression", "lr_naive_leaky|fusion", "Spectrum-wise, leaky"),
        ("", "lr_naive_nested|fusion", "Spectrum-wise, nested"), ("", "lr_sa_leaky|fusion", "Sample-wise, leaky"),
        ("", "lr_sa_nested|fusion", "Sample-wise, nested"), ("", "lr_sa_nested|abs", "Sample-wise, nested, absolute"),
        ("", "lr_sa_nested|mah", "Sample-wise, nested, Mahalanobis"),
        ("", "lr_naive_cole|fusion", "Spectrum-wise, Cole fusion"), ("", "lr_naive_cole|abs", "Spectrum-wise, Cole absolute"),
        ("", "lr_sa_cole|fusion", "Sample-wise, Cole fusion"), ("", "lr_sa_cole|abs", "Sample-wise, Cole absolute"),
        ("", "lr_sa_cole|mah", "Sample-wise, Cole Mahalanobis"), ("", "lr_sa_published|fusion", "Sample-wise, published set")]
begin(r"Additional real-data analyses (sample level). Except for the P2 label exchange (15 tumor, 12 healthy samples), 16 tumor "
      r"and 10 healthy samples were evaluated. Spectrum-wise/sample-wise: referencing design; leaky/nested: frequency selection with or "
      r"without the test patient; fusion: equal fusion of the absolute and Mahalanobis branches. CI: patient-level cluster bootstrap 95\% interval.",
      "llccccc", r"Analysis & Configuration & Tumor & Healthy & BA & 95\% CI & AUC \\", size=r"\scriptsize", sep="3.5pt")
for a, k, c in rows:
    s = S[k]
    ci = CI.get(k)
    cis = f"{h3(ci[0])}--{h3(ci[1])}" if ci else "--"
    L.append(f"{a} & {c} & {s['TP']}/{s['N1']} & {s['TN']}/{s['N0']} & {h3(s['BA'])} & {cis} & {h3(s['AUC'])} \\\\")
end()

# ---------------------------------------------------------------- S7 simulation grid (logistic regression)
g = pd.read_csv(f"{REV}/sim_grid.csv")
for dn in ("naive", "sa", "matched"):
    g[f"{dn}_opt"] = g[f"{dn}_est"] - g[f"{dn}_true"]
c = g.groupby(["delta", "icc", "m"]).mean(numeric_only=True).reset_index()
begin(r"Simulation study with logistic regression (six patients with three healthy and three tumor samples, 30 replicates per "
      r"condition): mean leave-one-patient-out estimate of the sample-level balanced accuracy for spectrum-wise referencing (SW est.), "
      r"true deployment balanced accuracy of the size-matched pipeline (True, for reference only), and optimism for spectrum-wise (SW), "
      r"sample-wise (SA), and size-matched (SM) referencing. Each optimism is the estimate of a design minus the true deployment balanced "
      r"accuracy of the pipeline trained with that same design, not minus the True column. $\Delta=0$: no class difference.",
      "cc|cccc|ccccc",
      r" & & \multicolumn{4}{c|}{$\Delta=0$} & \multicolumn{5}{c}{$\Delta=4$} \\" "\n"
      r"ICC & $m$ & SW est. & SW opt. & SA opt. & SM opt. & SW est. & True & SW opt. & SA opt. & SM opt. \\",
      size=r"\scriptsize", sep="3.5pt")
for (icc, m), d in c.groupby(["icc", "m"]):
    r0 = d[d.delta == 0].iloc[0]
    r4 = d[d.delta == 4].iloc[0]
    L.append(f"{icc:g} & {int(m)} & {h3(r0.naive_est)} & {s3(r0.naive_opt)} & {s3(r0.sa_opt)} & {s3(r0.matched_opt)} & "
             f"{h3(r4.naive_est)} & {h3(r4.matched_true)} & {s3(r4.naive_opt)} & {s3(r4.sa_opt)} & {s3(r4.matched_opt)} \\\\")
end()

# ---------------------------------------------------------------- S8 simulation with XGBoost
x = pd.read_csv(f"{REV}/sim_xgb.csv")
for dn in ("naive", "sa", "matched"):
    x[f"{dn}_opt"] = x[f"{dn}_est"] - x[f"{dn}_true"]
c = x.groupby(["delta", "icc"]).mean(numeric_only=True).reset_index()
begin(r"Simulation study with XGBoost ($m=15$, 20 replicates per condition); SM est.: size-matched estimate; other abbreviations as in Table~S7.",
      "ccccccc", r"$\Delta$ & ICC & SW est. & SM est. & SW opt. & SA opt. & SM opt. \\", size=r"\scriptsize", sep="3pt")
for _, r in c.iterrows():
    L.append(f"{r.delta:g} & {r.icc:g} & {h3(r.naive_est)} & {h3(r.matched_est)} & {s3(r.naive_opt)} & {s3(r.sa_opt)} & {s3(r.matched_opt)} \\\\")
end()

# ---------------------------------------------------------------- S9 simulation scenarios
if os.path.exists(f"{REV}/sim_ext_cell_means.csv"):
    ce = pd.read_csv(f"{REV}/sim_ext_cell_means.csv")
    scn = {"base": "Base model", "correlated": "Correlated features", "heteroscedastic": "Heteroscedastic repeats",
           "heterogeneous": "Heterogeneous tumor effects"}
    begin(r"Simulation under departures from the base model (logistic regression, $m=15$, 30 replicates per condition): mean "
          r"optimism of spectrum-wise (SW), sample-wise (SA), and size-matched (SM) referencing.",
          "lc|ccc|ccc",
          r" & & \multicolumn{3}{c|}{$\Delta=0$} & \multicolumn{3}{c}{$\Delta=4$} \\" "\n"
          r"Scenario & ICC & SW & SA & SM & SW & SA & SM \\", size=r"\scriptsize", sep="3pt")
    for sc in ("base", "correlated", "heteroscedastic", "heterogeneous"):
        d = ce[ce.scenario == sc]
        for i, icc in enumerate(sorted(d.icc.unique())):
            r0 = d[(d.icc == icc) & (d.delta == 0)].iloc[0]
            r4 = d[(d.icc == icc) & (d.delta == 4)].iloc[0]
            L.append(f"{scn[sc] if i == 0 else ''} & {icc:g} & {s3(r0.naive_opt)} & {s3(r0.sa_opt)} & {s3(r0.matched_opt)} & "
                     f"{s3(r4.naive_opt)} & {s3(r4.sa_opt)} & {s3(r4.matched_opt)} \\\\")
    end()

# ---------------------------------------------------------------- compact layout: pair narrow tables side by side
text = "\n".join(L)
blocks = text.split(r"\begin{table}[H]")
head, blocks = blocks[0], [r"\begin{table}[H]" + b for b in blocks[1:]]


def as_minipage(b, w):
    b = b.replace(r"\begin{table}[H]", r"\begin{minipage}[t]{%s\textwidth}" % w, 1)
    return b.replace(r"\end{table}", r"\end{minipage}", 1).rstrip()


PAIRS = {(7, 8): (0.40, 0.56)}          # Tables S8 and S9 (0-based block indices)
out_blocks, skip = [], set()
for i, b in enumerate(blocks):
    if i in skip:
        continue
    pair = next((k for k in PAIRS if k[0] == i), None)
    if pair:
        w1, w2 = PAIRS[pair]
        out_blocks.append(r"\begin{table}[H]" "\n" + as_minipage(b, w1) + "\n" r"\hfill" "\n" +
                          as_minipage(blocks[pair[1]], w2) + "\n" r"\end{table}" "\n")
        skip.add(pair[1])
    else:
        out_blocks.append(b)
open(OUTF, "w").write(head + "".join(out_blocks))
print("written", OUTF)

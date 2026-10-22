"""Supplementary Material (Tables S1-S10) from the result CSVs. usage: python make_supplement.py <results folder> <paper folder>"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
R, P = Path(sys.argv[1]), Path(sys.argv[2]); T = R / "tables"
rd = lambda n: pd.read_csv(T / f"{n}.csv") if (T / f"{n}.csv").exists() else None
def esc(x):
    s = str(x)
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#"), ("$", r"\$"), ("{", r"\{"), ("}", r"\}"),
                 ("≥", r"$\geq$"), ("≤", r"$\leq$"), ("≠", r"$\neq$"), ("–", "--"), ("—", "---"), ("×", r"$\times$"), ("τ", r"$\tau$"), ("π", r"$\pi$"),
                 ("→", r"$\rightarrow$"), ("±", r"$\pm$"), (" > ", r" $>$ "), (" < ", r" $<$ "), ("~", r"\textasciitilde{}")):
        s = s.replace(a, b)
    return s
def num(x, d=3):
    if isinstance(x, (bool, np.bool_)): return "yes" if x else "no"
    if isinstance(x, (int, np.integer)): return str(x)
    if isinstance(x, (float, np.floating)): return "--" if pd.isna(x) else f"{x:.{d}f}"
    return esc(x)
def table(label, caption, df, spec, header=None, size=r"\scriptsize", d=3):
    hdr = " & ".join(esc(h) for h in (header or df.columns))
    body = "\n".join(" & ".join(num(v, d) for v in r) + r" \\" for r in df.itertuples(index=False))
    return (rf"\begin{{table}}[H]\centering{size}\setlength{{\tabcolsep}}{{3pt}}" + "\n" + rf"\caption{{{caption}}}\label{{{label}}}" + "\n"
            + rf"\begin{{tabularx}}{{\linewidth}}{{{spec}}}\toprule" + "\n" + hdr + r"\\\midrule" + "\n" + body + "\n" + r"\bottomrule\end{tabularx}\end{table}" + "\n\n")
X = r">{\raggedright\arraybackslash}X"; p = lambda w: rf">{{\raggedright\arraybackslash}}p{{{w}}}"
out = [r"""\documentclass[11pt]{article}
\usepackage[T1]{fontenc}\usepackage[margin=2.2cm]{geometry}\usepackage{booktabs,tabularx,float}\usepackage[hidelinks]{hyperref}
\renewcommand{\thetable}{S\arabic{table}}
\title{Supplementary Material\\[4pt]\large Scoring only where crime happened: evaluation, spatial scale and calibration of sparse urban crime forecasts}
\author{Sondip Poul Singh \and Susmoy Bless Singh}\date{}
\begin{document}\maketitle
All tables are generated from the result files of the repository (\url{https://github.com/SondipPoulSingh/crime-forecasting-audit}; Zenodo doi:10.5281/zenodo.23491403); the file name of each source table is given in its caption.
"""]
b1 = rd("b01_code_versions_and_patches")
out.append(table("tab:s1", r"Official code: repositories, pinned commits and the patches applied (\texttt{b01}).", b1[["repository", "commit", "patches"]],
                 p("2.2cm") + p("3.2cm") + X, ["Repository", "Commit", "Patches"]))
b3 = rd("b03_protocol_audit")
out.append(table("tab:s2", r"Automatic audit of the official evaluation code, run on the untouched clones (\texttt{b03}).", b3, p("1.6cm") + p("2.4cm") + X + p("1.4cm"),
                 ["Code", "Aspect", "What the official code does", "Verified in code"]))
b14 = rd("b14_simple_vs_deep_full_protocol")
b14["deep_model"] = b14["deep_model"].str.replace(" (official model code, re-trained)", "", regex=False).str.replace(" (re-run, epoch chosen on validation)", "", regex=False).str.replace(" (re-run)", "", regex=False)
out.append(table("tab:s3", r"Full protocol P3, literature setting: best and mean deviance of each deep model over 5 seeds against the worst and mean seed of our XGBoost-Poisson forecaster (\texttt{b14}).",
                 b14, "ll" + "r" * 6, ["City", "Deep model", "Seeds", "Deep, best seed", "Deep, mean", "Ours, worst seed", "Ours, mean", "Ours better in every seed"]))
b16 = rd("b16_stshn_epoch_budget")
g = b16.groupby(["city", "budget_epochs", "epoch_selected_by"])[["selected_epoch", "P1_MAE", "P1_MAPE", "P3_deviance_all"]].mean().reset_index()
out.append(table("tab:s4", r"ST-SHN: epoch budget (10 = official, 25 = extended) and epoch selection (validation vs.\ test MAPE, as the official code reports), mean over 5 seeds (\texttt{b16}).",
                 g, "lrlrrrr", ["City", "Budget", "Epoch selected by", "Epoch", "P1 MAE", "P1 MAPE", "P3 deviance"]))
b17 = rd("b17_v3_benchmark_intervals_with_zinb")
b17 = b17[b17.seed == 0][["city", "method", "coverage", "width", "interval_score", "cov_Q1", "cov_Q5", "worst_under_coverage"]]
out.append(table("tab:s5", r"Benchmark intervals (90\,\%; 30 validation days as calibration; seed 0), including the zero-inflated baselines (\texttt{b17}). Q1/Q5: quietest/busiest volume quintile of cells.",
                 b17, "ll" + "r" * 6, ["City", "Method", "Coverage", "Width", "Interval score", "Q1", "Q5", "Worst under"]))
z = rd("z01_hypothesis_verdicts")
if z is not None:
    out.append(table("tab:s6", r"Hypotheses fixed before the runs, with the automatic verdicts (\texttt{z01}); rows marked \emph{post hoc} were added after results were seen.",
                     z, p("1.6cm") + X + p("1.7cm") + X, ["Hypothesis", "Claim", "Verdict", "Evidence"], size=r"\tiny"))
s1 = rd("s01_bench_morans_i")
if s1 is not None:
    out.append(table("tab:s7", r"Moran's $I$ of the per-cell mean Pearson residual on the benchmark (queen contiguity among cells with test-period crime, 999 permutations), mean and sd over seeds, and the largest permutation $p$-value over the seeds (\texttt{s01}). Raw: the same for the unstandardised residual.",
                     s1, "ll" + "r" * 5, ["City", "Model", "Seeds", "Moran's I", "sd", "max p", "Moran's I (raw)"]))
c2 = rd("c02b_v3_panel_check_against_v1"); d6 = rd("d06_v3_consistency_with_v1")
out.append(table("tab:s8a", r"Consistency of the third stage with the first: weekly panels rebuilt months later (\texttt{c02b}).",
                 c2[["city", "weeks_used", "first_week", "last_week", "areas", "incidents_v1", "incidents_v3", "incident_change_pct"]], "l" + "r" * 7,
                 ["City", "Weeks", "First week", "Last week", "Areas", "Incidents (stage 1)", "Incidents (stage 3)", "Change (%)"]))
dd = d6.groupby("city")[["coverage_diff_pts", "worst_dev_diff_pts"]].agg(lambda x: np.abs(x).max()).reset_index()
out.append(table("tab:s8b", r"Consistency of the third stage with the first: largest absolute difference in overall coverage and in worst-quintile deviation over all methods and horizons, in points (\texttt{d06}).",
                 dd, "lrr", ["City", "Max |coverage difference|", "Max |worst-quintile deviation difference|"]))
e3 = rd("e03_boundary_sources")
if e3 is not None:
    out.append(table("tab:s9", r"Area boundaries used for the maps and the spatial weights (fourth stage, \texttt{e03}).",
                     e3[["city", "dataset_id", "features", "key", "areas_matched", "areas_expected"]], "lllrrr", ["City", "Portal dataset", "Features", "Area key", "Areas matched", "Areas in panel"]))
e1 = rd("e01_poisson_retrain_runs")
if e1 is not None:
    out.append(table("tab:s10", r"Graph WaveNet and MTGNN in the Poisson setting, per seed (\texttt{e01}).",
                     e1[["city", "model", "seed", "selected_epoch", "P1_MAE", "P3_deviance_all", "P4_hit_rate"]], "llrrrrr", ["City", "Model", "Seed", "Epoch", "P1 MAE", "P3 deviance", "P4 hit rate"]))
s5 = rd("s05_area_morans_i")
if s5 is not None:
    s5 = s5[["city", "method", "variable", "areas", "morans_I", "p_value", "min_coverage", "share_areas_below_85"]].copy(); s5["share_areas_below_85"] = 100 * s5["share_areas_below_85"]
    out.append(table("tab:s12", r"Weekly panels, test period, $H=1$: per-area coverage of the 90\,\% intervals and per-area mean Pearson residual of the forecaster. Moran's $I$ with queen contiguity of the area polygons (areas without a land neighbour get their two nearest neighbours; 999 permutations), lowest area coverage (\%) and share of areas below 85\,\% (\%) (\texttt{s05}).",
                     s5, "ll" + X + "rrrrr", ["City", "Method", "Variable", "Areas", "Moran's I", "p", "Lowest area", "Areas below 85"], d=3))
out.append(r"\end{document}" + "\n")
(P / "supplementary.tex").write_text("".join(out)); print("wrote supplementary.tex")

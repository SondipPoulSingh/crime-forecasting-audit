"""Build the LaTeX tables of the paper directly from the result CSVs (no hand-copied numbers).
usage: python make_tables.py <optionA_outputs folder> <paper folder>"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
R, P = Path(sys.argv[1]), Path(sys.argv[2]); T = R / "tables"; OUT = P / "tables"; OUT.mkdir(exist_ok=True)
rd = lambda n: pd.read_csv(T / f"{n}.csv")
def f(x, d=3): return "--" if pd.isna(x) else f"{x:.{d}f}"
def pts(x, d=1): return "--" if pd.isna(x) else f"{100 * x:.{d}f}"
def write(name, s): (OUT / f"{name}.tex").write_text(s); print("wrote", name)
CATS = {"NYC": ["Burglary", "Larceny", "Robbery", "Assault"], "CHI": ["Theft", "Battery", "Assault", "Damage"]}
CITY = {"NYC": "New York", "CHI": "Chicago", "SF": "San Francisco"}

# ---------------------------------------------------------------- T1 benchmark sparsity
b4 = rd("b04_benchmark_sparsity")
rows = []
for c in ("NYC", "CHI"):
    for _, r in b4[b4.city == c].iterrows():
        rows.append(f"{CITY[c]} & {r.category} & {f(r.mean_daily_count, 2)} & {pts(r.share_zero_cell_days)} & {pts(r.share_scored_P1)} & {pts(r.share_scored_P2)} \\\\")
write("t1_benchmark_sparsity", r"""\begin{table}[!htbp]\centering\small
\caption{The public NYC-/CHI-Crimes benchmark is sparse. Share of the 92 test days $\times$ cells that each protocol scores (P1: crime cell-days, the ST-HSL rule; P2: crime cell-days in cells active on 20--80\,\% of training days, the ST-SHN rule). Grids: $16\times16$ (NYC, 2014--2015) and $14\times12$ (CHI, 2016--2017) cells of about 3\,km; 4 crime categories; daily counts.}\label{tab:sparsity}
\begin{tabular}{llrrrr}\toprule
City & Category & Mean daily count & Zero cell-days (\%) & Scored by P1 (\%) & Scored by P2 (\%)\\\midrule
""" + "\n".join(rows) + r"""
\bottomrule\end{tabular}\end{table}
""")

# ---------------------------------------------------------------- T2 official protocol: published vs re-run vs trivial
b8 = rd("b08_official_protocol_with_published")
pick = [("ST-HSL", "published (ST-HSL paper, Table III)", "ST-HSL, published"), ("ST-SHN", "published (ST-HSL paper, Table III)", "ST-SHN, published"),
        ("GWN", "published (ST-HSL paper, Table III)", "Graph WaveNet, published"), ("MTGNN", "published (ST-HSL paper, Table III)", "MTGNN, published"),
        ("ST-HSL (re-run)", None, "ST-HSL, re-run (5 seeds)"), ("ST-SHN (re-run, epoch chosen on validation)", None, "ST-SHN, re-run (5 seeds)"),
        ("Graph WaveNet (official model code, re-trained)", None, "Graph WaveNet, re-run (5 seeds)"), ("MTGNN (official model code, re-trained)", None, "MTGNN, re-run (5 seeds)"),
        ("Ours: GBM-Poisson (XGBoost)", None, "GBM-Poisson (ours)"), ("Naive: 30-day mean", None, "30-day mean"),
        ("Trivial: constant 1", None, "Constant 1 (trivial)"), ("Ours – protocol-aware (zero-truncated median)", None, "Zero-truncated median (exploit)")]
rows = []
for c in ("NYC", "CHI"):
    rows.append(rf"\multicolumn{{10}}{{l}}{{\textit{{{CITY[c]}}}}}\\")
    for model, src, lab in pick:
        g = b8[(b8.city == c) & (b8.model == model)]
        if src: g = g[g.source == src]
        if g.empty: continue
        r = g.iloc[0]
        rows.append(f"{lab} & " + " & ".join(f"{f(r[f'MAE_{k}'])} & {f(r[f'MAPE_{k}'])}" for k in range(4)) + f" & {f(r.mean_MAE)} \\\\")
    rows.append(r"\midrule")
hdr = lambda c: " & ".join(rf"\multicolumn{{2}}{{c}}{{{n}}}" for n in CATS[c])
write("t2_official_protocol", r"""\begin{table}[!htbp]\centering\scriptsize\setlength{\tabcolsep}{3.5pt}
\caption{Official protocol P1 (only cell-days with at least one crime are scored): published numbers (ST-HSL paper, Table III), our re-runs of the official code (mean over 5 seeds), our forecaster, and two trivial predictors. Lower is better. Category columns follow the benchmark order (NYC: Burglary, Larceny, Robbery, Assault; CHI: Theft, Battery, Assault, Damage).}\label{tab:official}
\begin{tabular}{l*{8}{r}r}\toprule
 & \multicolumn{2}{c}{Category 1} & \multicolumn{2}{c}{Category 2} & \multicolumn{2}{c}{Category 3} & \multicolumn{2}{c}{Category 4} & \\
Model & MAE & MAPE & MAE & MAPE & MAE & MAPE & MAE & MAPE & Mean MAE\\\midrule
""" + "\n".join(rows[:-1]) + r"""
\bottomrule\end{tabular}\end{table}
""")

# ---------------------------------------------------------------- T3 rankings under protocols
b12 = rd("b12_rankings_by_protocol"); b13 = rd("b13_ranking_agreement")
show = ["Ours: GBM-Poisson (XGBoost)", "Ours: GBM-Poisson (LightGBM)", "MTGNN (official model code, re-trained)", "Graph WaveNet (official model code, re-trained)",
        "Naive: 30-day mean", "ST-HSL (re-run)", "ST-SHN (re-run, epoch chosen on validation)", "Ours – protocol-aware (zero-truncated median)", "Trivial: constant 1", "Naive: last day"]
lab = {"Ours: GBM-Poisson (XGBoost)": "GBM-Poisson, XGBoost (ours)", "Ours: GBM-Poisson (LightGBM)": "GBM-Poisson, LightGBM (ours)", "MTGNN (official model code, re-trained)": "MTGNN",
       "Graph WaveNet (official model code, re-trained)": "Graph WaveNet", "Naive: 30-day mean": "30-day mean", "ST-HSL (re-run)": "ST-HSL", "ST-SHN (re-run, epoch chosen on validation)": "ST-SHN",
       "Ours – protocol-aware (zero-truncated median)": "Zero-truncated median (exploit)", "Trivial: constant 1": "Constant 1", "Naive: last day": "Last day"}
rows = []
for m in show:
    cells = []
    for c in ("NYC", "CHI"):
        r = b12[(b12.city == c) & (b12.model == m)].iloc[0]
        cells += [f(r.P1_mean_MAE), f"{int(r['rank: P1 official ST-HSL (mean MAE, crime days)'])}", f(r.P3_deviance_all), f"{int(r['rank: P3 full (Poisson deviance, all cell-days)'])}"]
    rows.append(f"{lab[m]} & " + " & ".join(cells) + r" \\")
tau = {c: b13[(b13.city == c) & (b13.protocol_a == "P1_mean_MAE") & (b13.protocol_b == "P3_deviance_all")].iloc[0] for c in ("NYC", "CHI")}
write("t3_rankings", r"""\begin{table}[!htbp]\centering\small
\setlength{\tabcolsep}{4pt}\caption{The protocol decides the ranking. Mean MAE under the official protocol P1 and Poisson deviance under the full protocol P3 (all cell-days), with ranks among the 11 models (5 seeds for learned models). Kendall's $\tau$ between the two rankings: """
      + f"NYC $\\tau={tau['NYC'].kendall_tau:.2f}$ ($p={tau['NYC'].p_value:.2f}$), CHI $\\tau={tau['CHI'].kendall_tau:.2f}$ ($p={tau['CHI'].p_value:.2f}$)." + r"""}\label{tab:rankings}
\begin{tabular}{lrrrrrrrr}\toprule
 & \multicolumn{4}{c}{New York} & \multicolumn{4}{c}{Chicago}\\\cmidrule(lr){2-5}\cmidrule(lr){6-9}
Model & P1 MAE & rank & P3 dev. & rank & P1 MAE & rank & P3 dev. & rank\\\midrule
""" + "\n".join(rows) + r"""
\bottomrule\end{tabular}\end{table}
""")

# ---------------------------------------------------------------- T4 reproduction gap with t-intervals
g = rd("b09b_reproduction_gap_ci"); b6 = rd("b06_benchmark_scores_per_seed")
def official_gap(m, c, k, pub):
    if m != "ST-HSL (re-run)": return "--"
    v = b6[(b6.model == m) & (b6.city == c) & (b6.seed == 0)][f"P1_MAE_{k}"].iloc[0]; return f"{100 * (v - pub) / pub:+.1f}"
labm = {"ST-HSL (re-run)": "ST-HSL", "Graph WaveNet (official model code, re-trained)": "Graph WaveNet", "MTGNN (official model code, re-trained)": "MTGNN",
        "ST-SHN (re-run, epoch chosen on validation)": "ST-SHN"}
rows = []
for m in labm:
    for c in ("NYC", "CHI"):
        gg = g[(g.rerun == m) & (g.city == c)]
        for i, (_, r) in enumerate(gg.iterrows()):
            v = {"re-run worse": "worse", "re-run better": "better", "within seed noise": "noise"}[r.verdict]
            rows.append(f"{labm[m] if i == 0 and c == 'NYC' else ''} & {CITY[c] if i == 0 else ''} & {r.category} & {f(r.published_MAE, 4)} & {f(r.rerun_MAE, 4)} $\\pm$ {f(r.rerun_MAE_sd, 4)} & "
                        f"[{r.gap_ci_lo:+.3f}, {r.gap_ci_hi:+.3f}] & {r.gap_pct:+.1f} & {official_gap(m, c, CATS[c].index(r.category), r.published_MAE)} & {v} \\\\")
    rows.append(r"\midrule")
write("t4_reproduction_gap", r"""\begin{table}[!htbp]\centering\scriptsize\setlength{\tabcolsep}{3.5pt}
\caption{Reproduction gap under the official protocol P1: published MAE vs.\ re-run of the official code (mean $\pm$ sd over 5 seeds) and the 95\,\% $t$-interval of the gap; for ST-HSL also the gap of the official seed (523) alone. \emph{worse}/\emph{better}: the whole interval lies above/below the published number; \emph{noise}: it contains it.}\label{tab:repro}
\begin{tabular}{lllrrrrrl}\toprule
Model & City & Category & Published & Re-run & 95\,\% CI of gap & Gap (\%) & Official seed (\%) & Verdict\\\midrule
""" + "\n".join(rows[:-1]) + r"""
\bottomrule\end{tabular}\end{table}
""")

# ---------------------------------------------------------------- T5 weekly forecast deviance
a = rd("c05_forecast_accuracy_mean_sd"); a = a[a.split == "test"]
order = ["LightGBM-Poisson", "XGBoost-Poisson", "GRU (global)", "ST-GRU (graph diffusion input)", "Poisson GLM", "Historical average (52 w)", "Exp. smoothing (0.3)",
         "Moving average (4 w)", "Last week", "Seasonal naive (52 w)"]
rows = []
for m in order:
    cells = []
    for H in (1, 4):
        for c in ("CHI", "NYC", "SF"):
            r = a[(a.model == m) & (a.H == H) & (a.city == c)]
            if r.empty: cells.append("--"); continue
            r = r.iloc[0]; cells.append(f(r.deviance) + ("" if pd.isna(r.deviance_sd) else rf"\,{{\tiny$\pm${r.deviance_sd:.3f}}}"))
    rows.append(f"{m} & " + " & ".join(cells) + r" \\")
write("t5_weekly_forecasts", r"""\begin{table}[!htbp]\centering\scriptsize\setlength{\tabcolsep}{3.5pt}
\caption{Weekly area $\times$ crime-type forecasts, test period (July 2024 onwards): mean Poisson deviance (lower is better; $\pm$ sd over 3 seeds for learned models) at horizons of 1 and 4 weeks. The forecaster selected on validation deviance is LightGBM-Poisson.}\label{tab:weekly}
\begin{tabular}{lrrrrrr}\toprule
 & \multicolumn{3}{c}{$H=1$} & \multicolumn{3}{c}{$H=4$}\\\cmidrule(lr){2-4}\cmidrule(lr){5-7}
Model & Chicago & New York & San Francisco & Chicago & New York & San Francisco\\\midrule
""" + "\n".join(rows) + r"""
\bottomrule\end{tabular}\end{table}
""")

# ---------------------------------------------------------------- T6 intervals main split H = 1
d = rd("d01_interval_coverage_main_split"); d = d[d.seed == d.groupby(["H", "city"])["seed"].transform("min")]
qc = [f"cov_Q{q}" for q in range(1, 6)]; d["under"] = (0.9 - d[qc]).max(axis=1); d["over"] = (d[qc] - 0.9).max(axis=1)
meth = ["Poisson", "Neg. binomial", "Split conformal (CP)", "Locally weighted CP", "CQR", "ACI", "Mondrian CP", "Mondrian-ACI"]
rows = []
for c in ("CHI", "NYC", "SF"):
    rows.append(rf"\multicolumn{{10}}{{l}}{{\textit{{{CITY[c]}}}}}\\")
    for m in meth:
        r = d[(d.H == 1) & (d.city == c) & (d.method == m)].iloc[0]
        rows.append(f"{m} & {pts(r.coverage)} & {f(r.width, 2)} & {f(r.interval_score, 2)} & " + " & ".join(pts(r[q], 1) for q in qc) + f" & {pts(r.under)} \\\\")
    rows.append(r"\midrule")
write("t6_intervals_h1", r"""\begin{table}[!htbp]\centering\scriptsize\setlength{\tabcolsep}{3.5pt}
\caption{90\,\% intervals around the LightGBM-Poisson forecast, $H=1$, test period, seed 0. Coverage overall and per series-volume quintile (Q1 quietest, Q5 busiest), mean width, mean interval score \citep{gneiting2007strictly} (lower is better) and the worst under-coverage over the quintiles (points below 90\,\%; negative = no quintile under-covers).}\label{tab:intervals}
\begin{tabular}{lrrrrrrrrr}\toprule
Method & Coverage & Width & Int.\ score & Q1 & Q2 & Q3 & Q4 & Q5 & Worst under\\\midrule
""" + "\n".join(rows[:-1]) + r"""
\bottomrule\end{tabular}\end{table}
""")

# ---------------------------------------------------------------- T7 rolling origins
e = rd("e01_rolling_interval_coverage")
meth7 = ["Poisson", "Neg. binomial", "Split conformal (CP)", "CQR", "ACI", "Mondrian CP", "Mondrian-ACI"]
cols = [(c, y) for c in ("CHI", "NYC", "SF") for y in sorted(e[e.city == c].test_year.unique())]
rows = []
for m in meth7:
    cells = []
    for c, y in cols:
        r = e[(e.method == m) & (e.city == c) & (e.test_year == y)]
        cells.append(pts(r.iloc[0].max_abs_dev_from_90) if len(r) else "--")
    rows.append(f"{m} & " + " & ".join(cells) + r" \\")
spans = " & ".join(rf"\multicolumn{{{sum(1 for cc, _ in cols if cc == c)}}}{{c}}{{{CITY[c]}}}" for c in ("CHI", "NYC", "SF"))
write("t7_rolling", r"""\begin{table}[!htbp]\centering\scriptsize\setlength{\tabcolsep}{3.5pt}
\caption{Rolling origins: worst deviation from 90\,\% coverage over the five series-volume quintiles (points), per test year ($H=1$). San Francisco starts in 2018, so its first test year is 2021.}\label{tab:rolling}
\begin{tabular}{l*{""" + str(len(cols)) + r"""}{r}}\toprule
 & """ + spans + r"""\\
Method & """ + " & ".join(str(y) for _, y in cols) + r"""\\\midrule
""" + "\n".join(rows) + r"""
\bottomrule\end{tabular}\end{table}
""")

# ---------------------------------------------------------------- T8 equity
cv = rd("f02_coverage_by_group"); gp = rd("f06_coverage_gap_between_groups"); ae = rd("f05_area_coverage_vs_demographics")
meth8 = ["Poisson", "Neg. binomial", "Split conformal (CP)", "CQR", "Mondrian CP", "Mondrian-ACI"]
rows = []
for m in meth8:
    cells = []
    for c in ("CHI", "NYC", "SF"):
        hp = cv[(cv.city == c) & (cv.method == m) & (cv.group == "high poverty")].iloc[0].coverage
        gap = gp[(gp.city == c) & (gp.method == m) & (gp.grouping == "poverty_group")].iloc[0].gap
        rho = ae[(ae.city == c) & (ae.method == m)].iloc[0]
        cells += [pts(hp), f"{100 * gap:+.1f}", f"{rho.spearman_cov_minority:+.2f}"]
    rows.append(f"{m} & " + " & ".join(cells) + r" \\")
write("t8_equity", r"""\begin{table}[!htbp]\centering\scriptsize\setlength{\tabcolsep}{3.5pt}
\caption{Equity of interval coverage ($H=1$, test). Coverage in the poorest third of areas (ACS 2022 poverty rate), the coverage gap poorest minus richest third (points), and Spearman's $\rho$ between an area's coverage and its minority share (77, 76 and 40 areas).}\label{tab:equity}
\begin{tabular}{lrrrrrrrrr}\toprule
 & \multicolumn{3}{c}{Chicago} & \multicolumn{3}{c}{New York} & \multicolumn{3}{c}{San Francisco}\\\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}
Method & Cov.\ poor & Gap & $\rho$ minority & Cov.\ poor & Gap & $\rho$ minority & Cov.\ poor & Gap & $\rho$ minority\\\midrule
""" + "\n".join(rows) + r"""
\bottomrule\end{tabular}\end{table}
""")

# ---------------------------------------------------------------- T9 protocol census (v3)
if (T / "g02_v3_protocol_census.csv").exists():
    w = rd("g02_v3_protocol_census")
    rows = []
    for _, r in w.iterrows():
        get = lambda k: r[k] if k in r and isinstance(r[k], str) else "--"
        rows.append(f"{r.code} & {r.paper} & {r.task} & {get('scored cells (test metric)')} & {get('training loss')} & {get('epoch reported')} & "
                    f"{'yes' if r['test metric ignores zero cells'] else 'no'} & {'direct' if str(r.evidence).startswith('direct') else 'indirect'} \\\\")
    write("t9_census", r"""\begin{table}[!htbp]\centering\scriptsize
\caption{Protocol census of public crime-prediction code, audited automatically at pinned commits (Section~\ref{sec:census}). \emph{Indirect}: the crime pipeline was not released; the finding follows from the shared metric code and the shipped configuration.}\label{tab:census}
\setlength{\tabcolsep}{3pt}\begin{tabularx}{\linewidth}{@{}>{\raggedright\arraybackslash}p{1.45cm}>{\raggedright\arraybackslash}p{2.0cm}>{\raggedright\arraybackslash}p{1.25cm}>{\raggedright\arraybackslash}X>{\raggedright\arraybackslash}p{2.1cm}>{\raggedright\arraybackslash}p{1.8cm}>{\raggedright\arraybackslash}p{0.9cm}>{\raggedright\arraybackslash}p{1.05cm}@{}}\toprule
Code & Paper & Task & Cells scored by the test metric & Training loss & Epoch reported & Zeros ignored & Evidence\\\midrule
""" + "\n".join(rows).replace("_", r"\_").replace("≠", r"$\neq$").replace("–", "--").replace("×", r"$\times$").replace(" > ", " $>$ ").replace("%", r"\%") + r"""
\bottomrule\end{tabularx}\end{table}
""")
else:
    write("t9_census", r"\begin{table}[!htbp]\centering\caption{Protocol census (filled from the v3 run).}\label{tab:census}\fbox{\parbox{0.8\linewidth}{\centering [v3 results pending]}}\end{table}" + "\n")

# ---------------------------------------------------------------- T10 ZINB baselines (v3), H = 1, seed 0
if (T / "d03_v3_interval_coverage_with_zinb.csv").exists():
    z = rd("d03_v3_interval_coverage_with_zinb"); z = z[(z.seed == 0) & (z.H == 1)]; zp = rd("d05_v3_zinb_parameters")
    m10 = ["Poisson", "Neg. binomial", "ZINB (fitted)", "ZINB (fitted per volume bin)", "GRU-ZINB (learned)", "Split conformal (CP)", "CQR", "Mondrian CP", "Mondrian-ACI"]
    rows = []
    for c in ("CHI", "NYC", "SF"):
        pi = zp[(zp.H == 1) & (zp.seed == 0) & (zp.city == c) & (zp.variant == "global")].pi.iloc[0]
        rows.append(rf"\multicolumn{{7}}{{l}}{{\textit{{{CITY[c]}}} (fitted zero-inflation $\hat\pi={pi:.3f}$)}}\\")
        for m in m10:
            r = z[(z.city == c) & (z.method == m)].iloc[0]
            rows.append(f"{m} & {pts(r.coverage)} & {f(r.width, 2)} & {f(r.interval_score, 2)} & {pts(r.worst_under_coverage)} & {pts(r.worst_over_coverage)} & {pts(r.cov_Q5)} \\\\")
        rows.append(r"\midrule")
    write("t10_zinb", r"""\begin{table}[!htbp]\centering\scriptsize\setlength{\tabcolsep}{3.5pt}
\caption{Zero-inflated baselines (third run; $H=1$, seed 0). Overall coverage, mean width, mean interval score (lower is better), worst under- and over-coverage over the volume quintiles (points) and coverage of the busiest quintile Q5. $\hat\pi$: zero-inflation fitted on the calibration block.}\label{tab:zinb}
\begin{tabular}{lrrrrrr}\toprule
Method & Coverage & Width & Int.\ score & Worst under & Worst over & Q5\\\midrule
""" + "\n".join(rows[:-1]) + r"""
\bottomrule\end{tabular}\end{table}
""")
    e3 = rd("e03_v3_rolling_coverage_with_zinb"); e3 = e3[e3.test_year == 2020]
    rows = []
    for m in m10:
        cells = []
        for c in ("CHI", "NYC"):
            r = e3[(e3.city == c) & (e3.method == m)].iloc[0]
            cells += [pts(r.coverage), pts(r.worst_under_coverage), pts(r.max_abs_dev_from_90), f(r.interval_score, 2)]
        rows.append(f"{m} & " + " & ".join(cells) + r" \\")
    write("t11_shift2020", r"""\begin{table}[!htbp]\centering\scriptsize\setlength{\tabcolsep}{3.5pt}
\caption{The 2020 test year (rolling origin, $H=1$; third run). Overall coverage, worst under-coverage and worst absolute deviation over the volume quintiles (points), and mean interval score.}\label{tab:shift}
\begin{tabular}{lrrrrrrrr}\toprule
 & \multicolumn{4}{c}{Chicago} & \multicolumn{4}{c}{New York}\\\cmidrule(lr){2-5}\cmidrule(lr){6-9}
Method & Cov. & Under & Worst & IS & Cov. & Under & Worst & IS\\\midrule
""" + "\n".join(rows) + r"""
\bottomrule\end{tabular}\end{table}
""")

# ---------------------------------------------------------------- T12 decision protocol (top-10 % cells), T13 bootstrap
if (T / "h02_decision_protocol.csv").exists():
    h = rd("h02_decision_protocol"); ta = rd("h03_decision_rank_agreement")
    lab12 = {"Ours: GBM-Poisson (XGBoost)": "GBM-Poisson, XGBoost (ours)", "Ours: GBM-Poisson (LightGBM)": "GBM-Poisson, LightGBM (ours)", "MTGNN (official model code, re-trained)": "MTGNN",
             "Graph WaveNet (official model code, re-trained)": "Graph WaveNet", "Naive: 30-day mean": "30-day mean", "ST-HSL (re-run)": "ST-HSL", "ST-SHN (re-run, epoch chosen on validation)": "ST-SHN",
             "Ours – protocol-aware (zero-truncated median)": "Zero-truncated median", "Trivial: constant 1": "Constant 1", "Naive: last day": "Last day"}
    order = h[h.city == "CHI"].sort_values("hit_rate", ascending=False).model.tolist()
    rows = []
    for m in order:
        cells = []
        for c in ("NYC", "CHI"):
            r = h[(h.city == c) & (h.model == m)].iloc[0]
            cells += [pts(r.hit_rate), f(r.PAI, 2), f"{int(r.rank_P4_hit_rate)}"]
        rows.append(f"{lab12[m]} & " + " & ".join(cells) + r" \\")
    tt = {(r.city, r.protocol_b): r for r in ta.itertuples()}
    write("t12_decision", r"""\begin{table}[!htbp]\centering\small\setlength{\tabcolsep}{4pt}
\caption{Decision protocol P4 on the benchmark: share of each day's crimes that fall in the 10\,\% of cells with the highest forecast (hit rate), the Predictive Accuracy Index (PAI = hit rate / area share; 1 = no better than a random list), and the rank among the 10 models (mean over seeds). Kendall's $\tau$ of the P4 ranking with P3: """
          + f"NYC {tt[('NYC', 'P3_deviance_all')].kendall_tau:.2f}, CHI {tt[('CHI', 'P3_deviance_all')].kendall_tau:.2f}; with P1: NYC {tt[('NYC', 'P1_mean_MAE')].kendall_tau:.2f}, CHI {tt[('CHI', 'P1_mean_MAE')].kendall_tau:.2f}." + r"""}\label{tab:decision}
\begin{tabular}{lrrrrrr}\toprule
 & \multicolumn{3}{c}{New York} & \multicolumn{3}{c}{Chicago}\\\cmidrule(lr){2-4}\cmidrule(lr){5-7}
Model & Hit rate (\%) & PAI & Rank & Hit rate (\%) & PAI & Rank\\\midrule
""" + "\n".join(rows) + r"""
\bottomrule\end{tabular}\end{table}
""")
if (T / "h04_deviance_difference_bootstrap.csv").exists():
    bb = rd("h04_deviance_difference_bootstrap")
    lab13 = {"MTGNN (official model code, re-trained)": "MTGNN", "Graph WaveNet (official model code, re-trained)": "Graph WaveNet", "ST-HSL (re-run)": "ST-HSL",
             "ST-SHN (re-run, epoch chosen on validation)": "ST-SHN", "Naive: 30-day mean": "30-day mean"}
    rows = []
    for m, l in lab13.items():
        cells = []
        for c in ("NYC", "CHI"):
            r = bb[(bb.city == c) & (bb.model == m)].iloc[0]
            cells += [f(r.difference, 4), f"[{r.ci_lo:.4f}, {r.ci_hi:.4f}]", f"{int(r.days_ours_better)}/{int(r.days)}"]
        rows.append(f"{l} & " + " & ".join(cells) + r" \\")
    write("t13_bootstrap", r"""\begin{table}[!htbp]\centering\small\setlength{\tabcolsep}{3.5pt}
\caption{Full-protocol Poisson deviance of each model minus that of our XGBoost-Poisson forecaster (both averaged over seeds), with a 95\,\% interval from a bootstrap over the 92 test days (2{,}000 resamples), and the number of test days on which our forecaster has the lower deviance. Positive = our forecaster is better.}\label{tab:bootstrap}
\begin{tabular}{lrrrrrr}\toprule
 & \multicolumn{3}{c}{New York} & \multicolumn{3}{c}{Chicago}\\\cmidrule(lr){2-4}\cmidrule(lr){5-7}
Compared with & Difference & 95\,\% CI & Days & Difference & 95\,\% CI & Days\\\midrule
""" + "\n".join(rows) + r"""
\bottomrule\end{tabular}\end{table}
""")

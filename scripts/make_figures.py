"""Paper figures from the result CSVs. usage: python make_figures.py <optionA_outputs folder> <paper folder>"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
R, P = Path(sys.argv[1]), Path(sys.argv[2]); T = R / "tables"; OUT = P / "figures"; OUT.mkdir(exist_ok=True)
rd = lambda n: pd.read_csv(T / f"{n}.csv")
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
INK, INK2, GRID, GRAY = "#0b0b0b", "#52514e", "#e4e3df", "#9a9993"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "legend.frameon": False, "figure.dpi": 150, "savefig.bbox": "tight", "lines.linewidth": 1.6})
CITY = {"NYC": "New York", "CHI": "Chicago", "SF": "San Francisco"}
def save(fig, name): fig.savefig(OUT / f"{name}.pdf"); fig.savefig(OUT / f"{name}.png", dpi=200); plt.close(fig); print("wrote", name)

# ---------------------------------------------------------------- Fig 1: ranks under the official vs the full protocol
b12 = rd("b12_rankings_by_protocol")
lab = {"Ours: GBM-Poisson (XGBoost)": "GBM-Poisson (XGB)", "Ours: GBM-Poisson (LightGBM)": "GBM-Poisson (LGBM)", "MTGNN (official model code, re-trained)": "MTGNN",
       "Graph WaveNet (official model code, re-trained)": "Graph WaveNet", "Naive: 30-day mean": "30-day mean", "Naive: last day": "Last day", "ST-HSL (re-run)": "ST-HSL",
       "ST-SHN (re-run, epoch chosen on validation)": "ST-SHN", "Ours – protocol-aware (zero-truncated median)": "ZT median", "Trivial: constant 1": "Constant 1"}
role = lambda m: BLUE if m.startswith("Ours:") else (ORANGE if ("constant" in m or "protocol-aware" in m) else (INK2 if ("re-run" in m or "re-trained" in m) else GRAY))
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.2)); fig.subplots_adjust(wspace=1.6, left=0.17, right=0.83)
for ax, c in zip(axs, ("NYC", "CHI")):
    g = b12[(b12.city == c) & b12.model.isin(lab)].copy()
    g["r1"] = g["P1_mean_MAE"].rank(method="first"); g["r3"] = g["P3_deviance_all"].rank(method="first")
    for _, r in g.iterrows():
        col = role(r.model); lw = 2.2 if col in (BLUE, ORANGE) else 1.2
        ax.plot([0, 1], [r.r1, r.r3], color=col, lw=lw, marker="o", ms=4.5, mec="white", mew=1, zorder=3 if lw > 2 else 2)
        ax.text(1.07, r.r3, lab[r.model], va="center", fontsize=6.5, color=INK, clip_on=False)
        ax.text(-0.07, r.r1, lab[r.model], va="center", ha="right", fontsize=6.5, color=INK2, clip_on=False)
    ax.set_xlim(-0.08, 1.08); ax.set_ylim(len(g) + 0.6, 0.4); ax.set_xticks([0, 1]); ax.set_xticklabels(["Official (P1)\ncrime days only", "Full (P3)\nall cell-days"])
    ax.set_yticks([]); ax.grid(False); ax.spines["left"].set_visible(False); ax.set_title(CITY[c], fontsize=9, color=INK, loc="left")
save(fig, "fig1_rank_change")

# ---------------------------------------------------------------- Fig 2: reproduction gap with t-intervals
g = rd("b09b_reproduction_gap_ci")
mods = [("ST-HSL (re-run)", "ST-HSL", BLUE, "o"), ("Graph WaveNet (official model code, re-trained)", "Graph WaveNet", GRAY, "s"), ("MTGNN (official model code, re-trained)", "MTGNN", INK2, "D")]
fig, ax = plt.subplots(figsize=(7.2, 3.0))
cells = [(c, k) for c in ("NYC", "CHI") for k in g[(g.city == c) & (g.rerun == mods[0][0])].category]
for j, (m, l, col, mk) in enumerate(mods):
    for i, (c, k) in enumerate(cells):
        r = g[(g.rerun == m) & (g.city == c) & (g.category == k)].iloc[0]; pub = r.published_MAE
        lo, hi, mid = 100 * r.gap_ci_lo / pub, 100 * r.gap_ci_hi / pub, r.gap_pct; x = i + (j - 1) * 0.22
        ax.plot([x, x], [lo, hi], color=col, lw=1.4); ax.plot(x, mid, marker=mk, color=col, ms=5, mec="white", mew=0.8, ls="none", label=l if i == 0 else None)
b6 = rd("b06_benchmark_scores_per_seed"); CATS_ = {"NYC": ["Burglary", "Larceny", "Robbery", "Assault"], "CHI": ["Theft", "Battery", "Assault", "Damage"]}
for i, (c, k) in enumerate(cells):                       # official ST-HSL seed (523 = our seed 0), hollow marker
    r = g[(g.rerun == mods[0][0]) & (g.city == c) & (g.category == k)].iloc[0]
    v = b6[(b6.model == "ST-HSL (re-run)") & (b6.city == c) & (b6.seed == 0)][f"P1_MAE_{CATS_[c].index(k)}"].iloc[0]
    ax.plot(i - 0.22, 100 * (v - r.published_MAE) / r.published_MAE, marker="o", mfc="white", mec=BLUE, mew=1.4, ms=6, ls="none", label="ST-HSL, official seed" if i == 0 else None, zorder=4)
ax.axhline(0, color=INK, lw=0.8); ax.set_xticks(range(len(cells))); ax.set_xticklabels([f"{k}\n{'NYC' if c == 'NYC' else 'CHI'}" for c, k in cells], fontsize=7)
ax.set_ylabel("Re-run minus published MAE (%)\n(95 % t-interval, 5 seeds)"); ax.legend(loc="upper left", ncol=4, fontsize=7)
save(fig, "fig2_reproduction_gap")

# ---------------------------------------------------------------- Fig 3: coverage by series-volume quintile (H = 1), third run incl. ZINB
d = rd("d03_v3_interval_coverage_with_zinb") if (T / "d03_v3_interval_coverage_with_zinb.csv").exists() else rd("d01_interval_coverage_main_split")
d = d[(d.H == 1) & (d.seed == 0)]
meths = [("Poisson", GRAY, ":", "o"), ("Neg. binomial", GRAY, "--", "s"), ("ZINB (fitted)", INK2, "-.", "P"), ("GRU-ZINB (learned)", INK2, ":", "X"),
         ("Split conformal (CP)", ORANGE, "-", "o"), ("Mondrian CP", BLUE, "-", "o"), ("Mondrian-ACI", VIOLET, "--", "D")]
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.8), sharey=True)
for ax, c in zip(axs, ("CHI", "NYC", "SF")):
    for m, col, ls, mk in meths:
        r = d[(d.city == c) & (d.method == m)]
        if r.empty: continue
        r = r.iloc[0]
        ax.plot(range(1, 6), [100 * r[f"cov_Q{q}"] for q in range(1, 6)], color=col, ls=ls, marker=mk, ms=4, mec="white", mew=0.6, lw=2.2 if m == "Mondrian CP" else 1.3, label=m)
    ax.axhline(90, color=INK, lw=0.8, ls=(0, (2, 2))); ax.set_xticks(range(1, 6)); ax.set_xticklabels(["Q1\nquiet", "Q2", "Q3", "Q4", "Q5\nbusy"]); ax.set_title(CITY[c], fontsize=9, loc="left")
axs[0].set_ylabel("Coverage of the 90 % interval (%)"); axs[0].set_ylim(77, 101)
fig.legend(*axs[0].get_legend_handles_labels(), loc="lower center", ncol=4, fontsize=7, bbox_to_anchor=(0.5, -0.1)); fig.subplots_adjust(bottom=0.3)
save(fig, "fig3_coverage_by_quintile")

# ---------------------------------------------------------------- Fig 4: rolling origins, worst-quintile deviation
e = rd("e01_rolling_interval_coverage")
meths4 = [("Neg. binomial", GRAY, "--", "s"), ("Split conformal (CP)", ORANGE, "-", "o"), ("CQR", AQUA, "-.", "^"), ("ACI", ORANGE, ":", "v"),
          ("Mondrian CP", BLUE, "-", "o"), ("Mondrian-ACI", VIOLET, "--", "D")]
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.7), sharey=True)
for ax, c in zip(axs, ("CHI", "NYC", "SF")):
    for m, col, ls, mk in meths4:
        gg = e[(e.city == c) & (e.method == m)].sort_values("test_year")
        ax.plot(gg.test_year, 100 * gg.max_abs_dev_from_90, color=col, ls=ls, marker=mk, ms=4, mec="white", mew=0.6, lw=2.2 if m.startswith("Mondrian") else 1.3, label=m)
    ax.axhline(3, color=INK, lw=0.8, ls=(0, (2, 2))); ax.set_title(CITY[c], fontsize=9, loc="left"); ax.set_xticks(sorted(e.test_year.unique()))
axs[0].set_ylabel("Worst quintile deviation\nfrom 90 % (points)")
fig.legend(*axs[0].get_legend_handles_labels(), loc="lower center", ncol=6, fontsize=7, bbox_to_anchor=(0.5, -0.06)); fig.subplots_adjust(bottom=0.25)
save(fig, "fig4_rolling_origins")

# ---------------------------------------------------------------- Fig 5: equity, coverage in poorest vs richest third
cv = rd("f02_coverage_by_group"); cv = cv[cv.grouping == "poverty_group"]
meths5 = ["Poisson", "Neg. binomial", "Split conformal (CP)", "CQR", "Mondrian CP", "Mondrian-ACI"]
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.6), sharex=True)
for ax, c in zip(axs, ("CHI", "NYC", "SF")):
    for i, m in enumerate(meths5):
        hp = 100 * cv[(cv.city == c) & (cv.method == m) & (cv.group == "high poverty")].coverage.iloc[0]
        lp = 100 * cv[(cv.city == c) & (cv.method == m) & (cv.group == "low poverty")].coverage.iloc[0]
        ax.plot([lp, hp], [i, i], color=GRID, lw=3, solid_capstyle="round", zorder=1)
        ax.plot(lp, i, "o", color=GRAY, ms=5, zorder=2, label="richest third" if i == 0 else None)
        ax.plot(hp, i, "o", color=BLUE, ms=5, zorder=3, label="poorest third" if i == 0 else None)
    ax.axvline(90, color=INK, lw=0.8, ls=(0, (2, 2))); ax.set_yticks(range(len(meths5))); ax.set_yticklabels(meths5 if c == "CHI" else [""] * len(meths5))
    ax.set_title(CITY[c], fontsize=9, loc="left"); ax.set_xlabel("Coverage (%)"); ax.grid(axis="y", visible=False); ax.invert_yaxis()
axs[0].legend(loc="lower left", fontsize=7, bbox_to_anchor=(0, 1.08), ncol=2)
save(fig, "fig5_equity")

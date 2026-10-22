"""Spatial figures. usage: python make_figures_spatial.py <results folder> <paper folder>
fig6: benchmark grid - observed mean count and mean residual (observed - forecast) per cell, with Moran's I of the residuals.
fig7: (if boundaries are present) weekly panels - per-area coverage of the 90 % intervals for split CP and Mondrian CP."""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
R, P = Path(sys.argv[1]), Path(sys.argv[2]); T = R / "tables"; OUT = P / "figures"; OUT.mkdir(exist_ok=True)
BLUE, ORANGE = "#2a78d6", "#eb6834"; INK, INK2 = "#0b0b0b", "#52514e"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "figure.dpi": 150, "savefig.bbox": "tight"})
DIV = LinearSegmentedColormap.from_list("div", [BLUE, "#a9c8ee", "#f4f3ef", "#f5b896", ORANGE])
SEQ = LinearSegmentedColormap.from_list("seq", ["#f4f3ef", "#a9c8ee", BLUE, "#123d73"])
CITY = {"NYC": "New York", "CHI": "Chicago", "SF": "San Francisco"}
def save(fig, name): fig.savefig(OUT / f"{name}.pdf"); fig.savefig(OUT / f"{name}.png", dpi=200); plt.close(fig); print("wrote", name)

# ---------------------------------------------------------------- fig6
Z = np.load(R / "predictions" / "bench_residual_maps.npz"); keys = list(Z["keys"])
get = lambda k: Z[k.replace("|", "__").replace(" ", "_").replace(":", "")]
MI = pd.read_csv(T / "s01_bench_morans_i.csv")
cols = [("observed mean", "Observed mean count"), ("Ours: GBM-Poisson (XGBoost)", "Residual: GBM-Poisson"), ("MTGNN", "Residual: MTGNN"), ("ST-SHN", "Residual: ST-SHN")]
fig, axs = plt.subplots(2, 4, figsize=(7.2, 3.9), gridspec_kw=dict(wspace=0.32, hspace=0.28))
for r, c in enumerate(("NYC", "CHI")):
    obs = get(f"{c}|observed mean"); off = obs == 0                       # cells without any crime in the test period (water, parks, outside the city)
    lim = max(np.abs(get(f"{c}|{m}")[~off]).max() for m, _ in cols[1:3])
    for j, (m, title) in enumerate(cols):
        ax = axs[r, j]; A = np.ma.masked_where(off, get(f"{c}|{m}")); ax.set_facecolor("#e4e3df")
        if j == 0: im = ax.imshow(A, cmap=SEQ, vmin=0)
        else:
            l_ = lim if j < 3 else np.abs(A).max()
            im = ax.imshow(A, cmap=DIV, norm=TwoSlopeNorm(0, -l_, l_))
            g = MI[(MI.city == c) & (MI.model == m)].iloc[0]
            ax.text(0.03, 0.03, f"Moran's I = {g.morans_I:.2f}" + (" *" if g.max_p < 0.05 else ""), transform=ax.transAxes, fontsize=6.5, color=INK,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.2))
        ax.set_xticks([]); ax.set_yticks([])
        for s_ in ax.spines.values(): s_.set_color("#c9c8c3"); s_.set_linewidth(0.6)
        if r == 0: ax.set_title(title, fontsize=7.5, color=INK)
        if j == 0: ax.set_ylabel(CITY[c], fontsize=8.5, color=INK)
        if j != 1:
            cb = fig.colorbar(im, ax=axs[r, j] if j != 2 else axs[r, 1:3].tolist(), fraction=0.046, pad=0.02); cb.ax.tick_params(labelsize=6); cb.outline.set_visible(False)
save(fig, "fig6_bench_residual_maps")

# ---------------------------------------------------------------- fig7 (needs boundaries from v4)
BD = R / "boundaries"
if BD.exists() and (T / "s04_area_coverage.csv").exists():
    import geopandas as gpd
    AC = pd.read_csv(T / "s04_area_coverage.csv", dtype={"area": str}); SM = pd.read_csv(T / "s05_area_morans_i.csv")
    meths = [("Split conformal (CP)", "Split CP"), ("Mondrian CP", "Mondrian CP")]
    fig, axs = plt.subplots(2, 3, figsize=(7.2, 5.0), gridspec_kw=dict(wspace=0.05, hspace=0.12))
    norm = TwoSlopeNorm(90, 75, 100)
    for j, c in enumerate(("CHI", "NYC", "SF")):
        G = gpd.read_file(BD / f"{c}_areas.geojson"); key = pd.read_csv(T / "e03_boundary_sources.csv").set_index("city").loc[c, "key"]
        G["area_id"] = G[key].map(lambda v: str(int(float(v))) if str(v).replace(".", "").isdigit() else str(v))
        G = G[["area_id", "geometry"]].dissolve("area_id").reset_index().rename(columns={"area_id": "area_key"})
        for i, (m, lab) in enumerate(meths):
            ax = axs[i, j]; g = G.merge(AC[(AC.city == c) & (AC.method == m)], left_on="area_key", right_on="area", how="left")
            g.plot(column="coverage_pct", ax=ax, cmap=DIV.reversed(), norm=norm, edgecolor="white", linewidth=0.3, missing_kwds=dict(color="#dddddd"))
            mi = SM[(SM.city == c) & (SM.method == m)].iloc[0]
            ax.set_title(f"{CITY[c]} – {lab}\nworst area {g.coverage_pct.min():.0f} %, Moran's I {mi.morans_I:.2f}{' *' if mi.p_value < 0.05 else ''}", fontsize=7, color=INK)
            ax.set_axis_off()
    sm = plt.cm.ScalarMappable(norm=norm, cmap=DIV.reversed()); cb = fig.colorbar(sm, ax=axs.ravel().tolist(), fraction=0.02, pad=0.01)
    cb.set_label("coverage of 90 % intervals (%)", fontsize=7); cb.ax.tick_params(labelsize=6); cb.outline.set_visible(False)
    save(fig, "fig7_area_coverage_maps")

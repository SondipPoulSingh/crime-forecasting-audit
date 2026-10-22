"""Spatial analysis of the weekly city panels (needs the v4 boundaries).
Per area: coverage of each interval method and the mean Pearson residual of the LightGBM forecast (H = 1, test period, seed 0).
Moran's I with queen contiguity (islands get their 2 nearest neighbours), 999 permutations.
usage: python analysis_spatial_areas.py <results folder>  -> tables/s04_area_coverage.csv, s05_area_morans_i.csv"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, geopandas as gpd
from libpysal.weights import Queen, KNN, w_union
from esda.moran import Moran
warnings.filterwarnings("ignore")
R = Path(sys.argv[1]); T = R / "tables"; BD = R / "boundaries"
SRC = pd.read_csv(T / "e03_boundary_sources.csv").set_index("city")
METHODS = ["Poisson", "Neg. binomial", "Split conformal (CP)", "Mondrian CP", "Mondrian-ACI", "CQR", "Locally weighted CP"]
def norm(v):
    s = str(v).strip()
    try: return str(int(float(s)))
    except ValueError: return s
AC, MI = [], []
for c in ("CHI", "NYC", "SF"):
    f = [p for p in (R / "predictions").glob(f"intervals_test_H1_{c}.csv*") if p.is_file()][0]
    d = pd.read_csv(f, dtype={"area": str}); mu = np.maximum(d["lgbm_s0"], 1e-3)
    d["pearson"] = (d["y"] - mu) / np.sqrt(mu)
    G = gpd.read_file(BD / f"{c}_areas.geojson"); G["area_id"] = G[SRC.loc[c, "key"]].map(norm)            # ('area' is a GeoDataFrame property)
    G = G[["area_id", "geometry"]].dissolve("area_id").reset_index(); G = G[G["area_id"].isin(d.area.unique())].reset_index(drop=True)
    Wq = Queen.from_dataframe(G, use_index=False, silence_warnings=True)
    if Wq.islands:
        Wk = KNN.from_dataframe(G.to_crs(G.estimate_utm_crs()).set_geometry(G.to_crs(G.estimate_utm_crs()).centroid), k=2)
        Wq = w_union(Wq, Wk, silence_warnings=True)
    Wq.transform = "r"
    per = d.groupby("area").agg(residual=("pearson", "mean"), n=("y", "size")).reset_index()
    for m in METHODS:
        if f"lo|{m}" not in d: continue
        cov = ((d["y"] >= d[f"lo|{m}"]) & (d["y"] <= d[f"hi|{m}"])).groupby(d["area"]).mean().mul(100).rename("coverage_pct").reset_index()
        AC += cov.assign(city=c, method=m).to_dict("records")
        x = G[["area_id"]].rename(columns={"area_id": "area"}).merge(cov, on="area", how="left")["coverage_pct"].values
        mo = Moran(x, Wq, permutations=999)
        MI.append(dict(city=c, method=m, variable="area coverage", areas=len(x), morans_I=mo.I, expected_I=mo.EI, p_value=mo.p_sim,
                       min_coverage=np.nanmin(x), share_areas_below_85=np.mean(x < 85), islands_fixed=len(Queen.from_dataframe(G, use_index=False, silence_warnings=True).islands)))
    x = G[["area_id"]].rename(columns={"area_id": "area"}).merge(per, on="area", how="left")["residual"].values; mo = Moran(x, Wq, permutations=999)
    MI.append(dict(city=c, method="LightGBM-Poisson forecast", variable="mean Pearson residual", areas=len(x), morans_I=mo.I, expected_I=mo.EI, p_value=mo.p_sim))
    per.assign(city=c).to_csv(T / f"s06_area_residual_{c}.csv", index=False)
pd.DataFrame(AC).to_csv(T / "s04_area_coverage.csv", index=False)
M = pd.DataFrame(MI); M.to_csv(T / "s05_area_morans_i.csv", index=False); print(M.round(3).to_string())

"""Spatial analysis of the saved benchmark predictions (no re-training).
(1) Moran's I (queen contiguity among cells with at least one test-period crime, 999 permutations) of each model's per-cell mean Pearson residual.
(2) Spatial scale sensitivity (MAUP): the 3-km daily grid is aggregated to 2x2 blocks (about 6 km) and, in New York, 4x4 blocks (about 12 km); sparsity,
    the official protocol P1, the full protocol P3, the decision protocol P4 and the P1-P3 rank agreement are recomputed.
    Expected counts aggregate by summation; the constant-1 and zero-truncated-median predictors are re-derived at each scale.
    (Daily one-step forecasts cannot be summed into weekly forecasts, because they use information from within the week, so time is not aggregated.)
usage: python analysis_spatial_bench.py <results folder> <grid shapes json, e.g. '{"NYC":[16,16],"CHI":[14,12]}'>
writes tables/s01_bench_morans_i.csv, s02_bench_scale_sensitivity.csv, s03_bench_scale_rank_agreement.csv and arrays for the maps."""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import kendalltau, poisson
R = Path(sys.argv[1]); T = R / "tables"; PRED = R / "predictions"; SHAPES = json.loads(sys.argv[2])
RNG = np.random.default_rng(7); NPERM = 999
MODELS = {"Ours_GBM_Poisson_XGBoost_": "Ours: GBM-Poisson (XGBoost)", "Ours_GBM_Poisson_LightGBM_": "Ours: GBM-Poisson (LightGBM)",
          "MTGNN_official_model_code_re_trained_": "MTGNN", "Graph_WaveNet_official_model_code_re_trained_": "Graph WaveNet",
          "ST_HSL_re_run_": "ST-HSL", "ST_SHN_re_run_epoch_chosen_on_validation_": "ST-SHN", "Naive_30_day_mean": "Naive: 30-day mean",
          "Naive_last_day": "Naive: last day"}

def queen_w(Rr, Cc):
    W = np.zeros((Rr * Cc, Rr * Cc))
    for i in range(Rr):
        for j in range(Cc):
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if (di or dj) and 0 <= i + di < Rr and 0 <= j + dj < Cc: W[i * Cc + j, (i + di) * Cc + j + dj] = 1
    return W
def restrict(W, keep):
    W = W[np.ix_(keep, keep)].copy(); rs = W.sum(1, keepdims=True); keep2 = rs[:, 0] > 0
    return W, keep2
def morans_i(x, W):
    z = x - x.mean(); I = len(x) / W.sum() * (z @ W @ z) / (z @ z)
    perm = np.array([(lambda zz: len(x) / W.sum() * (zz @ W @ zz) / (zz @ zz))(RNG.permutation(z)) for _ in range(NPERM)])
    p = (1 + (np.abs(perm - perm.mean()) >= abs(I - perm.mean())).sum()) / (NPERM + 1)
    return I, p, perm.mean()
def pdev(y, mu):
    mu = np.maximum(mu, 0.01); return 2 * np.mean(np.where(y > 0, y * np.log(np.maximum(y, 1e-12) / mu), 0) - (y - mu))
def ztp_median(mu):
    mu = np.maximum(mu, 1e-3); p0 = np.exp(-mu); out = np.ones_like(mu)
    for k in range(1, 200): out = np.where((poisson.cdf(k, mu) - p0) / (1 - p0) < 0.5, k + 1, out)
    return out
def sel_w(p, k):
    v = np.sort(p)[::-1][k - 1]; above = p > v; tie = p == v; w = above.astype(float)
    if tie.any(): w[tie] = (k - above.sum()) / tie.sum()
    return w
def hit_rate(P, Y):                     # P, Y: (D, N, K)
    D, N, K = Y.shape; k = int(round(0.1 * N)); cap = tot = 0.0
    for kk in range(K):
        for d in range(D): cap += (sel_w(P[d, :, kk], k) * Y[d, :, kk]).sum(); tot += Y[d, :, kk].sum()
    return cap / tot
def aggregate(A, Rr, Cc, f):            # A: (D, N, K) -> (D, N', K), f x f blocks of cells (edge rows/columns that do not fill a block are dropped)
    D, N, K = A.shape; G = A.reshape(D, Rr, Cc, K)
    if f > 1:
        r2, c2 = Rr // f * f, Cc // f * f; G = G[:, :r2, :c2].reshape(D, r2 // f, f, c2 // f, f, K).sum((2, 4))
    return G.reshape(D, -1, K)

MOR, SCALE, TAU, MAPS = [], [], [], {}
for c, (Rr, Cc) in SHAPES.items():
    z = np.load(PRED / f"benchmark_{c}_test_predictions.npz"); y = z["y"].astype(float); K = int(z["cat"].max()) + 1; N = Rr * Cc; D = len(y) // (N * K)
    Y = y.reshape(D, N, K); active = Y.sum((0, 2)) > 0; W0, ok = restrict(queen_w(Rr, Cc), active)
    W = W0[np.ix_(ok, ok)]; W = W / W.sum(1, keepdims=True); sel = np.flatnonzero(active)[ok]
    for key, name in MODELS.items():
        Pm = np.atleast_2d(z[key]).astype(float)
        for s, p in enumerate(Pm):
            P = np.maximum(p.reshape(D, N, K), 1e-3); res = ((Y - P) / np.sqrt(P)).mean((0, 2))      # per-cell mean Pearson residual
            raw = (Y - P).mean((0, 2)); I, pv, EI = morans_i(res[sel], W); I2, pv2, _ = morans_i(raw[sel], W)
            MOR.append(dict(city=c, model=name, seed=s, cells=len(sel), morans_I_pearson=I, p_value=pv, expected_I_perm=EI, morans_I_raw_residual=I2, p_value_raw=pv2))
            if s == 0: MAPS[f"{c}|{name}"] = raw.reshape(Rr, Cc)
        if key == "Ours_GBM_Poisson_XGBoost_": MAPS[f"{c}|observed mean"] = Y.mean((0, 2)).reshape(Rr, Cc)
    # scale sensitivity
    for f_, label in ((1, "3 km (original)"), (2, "6 km"), (4, "12 km")):
        if Rr // f_ * (Cc // f_) < 16: continue
        Ya = aggregate(Y, Rr, Cc, f_); Da, Na, _ = Ya.shape
        rows = {}
        for key, name in list(MODELS.items()) + [("_const", "Trivial: constant 1"), ("_zt", "Ours – zero-truncated median")]:
            src = "Ours_GBM_Poisson_XGBoost_" if key in ("_const", "_zt") else key
            vals = []
            for p in np.atleast_2d(z[src]).astype(float):
                Pa = aggregate(p.reshape(D, N, K), Rr, Cc, f_)
                if key == "_const": Pa = np.ones_like(Pa)
                if key == "_zt": Pa = ztp_median(Pa)
                pos = Ya > 0
                p1 = np.mean([np.abs(Pa - Ya)[..., k][pos[..., k]].mean() for k in range(K)])
                vals.append(dict(P1_mean_MAE=p1, P3_deviance_all=pdev(Ya, Pa), P4_hit_rate=hit_rate(Pa, Ya)))
                if key in ("_const",): break
            m = pd.DataFrame(vals).mean(); rows[name] = m
            SCALE.append(dict(city=c, scale=label, units=Na, periods=Da, share_zero=(Ya == 0).mean(), model=name, seeds=len(vals), **m.to_dict()))
        g = pd.DataFrame(rows).T
        for a, b in (("P1_mean_MAE", "P3_deviance_all"), ("P4_hit_rate", "P3_deviance_all"), ("P4_hit_rate", "P1_mean_MAE")):
            xa = -g[a] if a == "P4_hit_rate" else g[a]; t, pv = kendalltau(xa, g[b])
            TAU.append(dict(city=c, scale=label, share_zero=(Ya == 0).mean(), protocol_a=a, protocol_b=b, kendall_tau=t, p_value=pv, models=len(g)))
M = pd.DataFrame(MOR); M.to_csv(T / "s01_bench_morans_i_per_seed.csv", index=False)
MS = M.groupby(["city", "model"]).agg(seeds=("seed", "size"), morans_I=("morans_I_pearson", "mean"), morans_I_sd=("morans_I_pearson", "std"),
                                      max_p=("p_value", "max"), morans_I_raw=("morans_I_raw_residual", "mean")).reset_index()
MS.to_csv(T / "s01_bench_morans_i.csv", index=False)
S = pd.DataFrame(SCALE); S.to_csv(T / "s02_bench_scale_sensitivity.csv", index=False)
TA = pd.DataFrame(TAU); TA.to_csv(T / "s03_bench_scale_rank_agreement.csv", index=False)
np.savez_compressed(T.parent / "predictions" / "bench_residual_maps.npz", **{k.replace("|", "__").replace(" ", "_").replace(":", ""): v for k, v in MAPS.items()},
                    keys=np.array(list(MAPS)))
print(MS.round(3).to_string()); print(TA.round(3).to_string())
print(S[S.model.isin(["Ours: GBM-Poisson (XGBoost)", "Trivial: constant 1", "Ours – zero-truncated median", "ST-SHN", "MTGNN"])].round(3).to_string())

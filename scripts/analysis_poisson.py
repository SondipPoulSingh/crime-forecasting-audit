"""Literature setting (squared error, crime-day selection; stage 2) vs. Poisson setting (Poisson NLL on all cell-days, validation deviance; stage 4)
for Graph WaveNet and MTGNN, against our XGBoost-Poisson forecaster. No re-training.
usage: python analysis_poisson.py <results folder>  (needs predictions/benchmark_{c}_poisson_predictions.npz from stage 4)
writes tables/s07_poisson_vs_literature.csv"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
R = Path(sys.argv[1]); T = R / "tables"; PRED = R / "predictions"; RNG = np.random.default_rng(11); B = 2000
SHAPES = {"NYC": (16, 16), "CHI": (14, 12)}
def load(f):
    if f.is_dir(): return {p.stem: np.load(p) for p in f.rglob("*.npy")}
    z = np.load(f); return {k: z[k] for k in z.files}
def pdev_cells(y, mu): mu = np.maximum(mu, 0.01); return 2 * (np.where(y > 0, y * np.log(np.maximum(y, 1e-12) / mu), 0) - (y - mu))
def sel_w(p, k):
    v = np.sort(p)[::-1][k - 1]; above = p > v; tie = p == v; w = above.astype(float)
    if tie.any(): w[tie] = (k - above.sum()) / tie.sum()
    return w
def hit_rate(P, Y):
    D, N, K = Y.shape; k = int(round(0.1 * N)); cap = tot = 0.0
    for kk in range(K):
        for d in range(D): cap += (sel_w(P[d, :, kk], k) * Y[d, :, kk]).sum(); tot += Y[d, :, kk].sum()
    return cap / tot
def queen(Rr, Cc):
    W = np.zeros((Rr * Cc, Rr * Cc))
    for i in range(Rr):
        for j in range(Cc):
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if (di or dj) and 0 <= i + di < Rr and 0 <= j + dj < Cc: W[i * Cc + j, (i + di) * Cc + j + dj] = 1
    return W
def moran(x, W, nperm=999):
    z = x - x.mean(); f = lambda zz: len(zz) / W.sum() * (zz @ W @ zz) / (zz @ zz); I = f(z)
    perm = np.array([f(RNG.permutation(z)) for _ in range(nperm)]); return I, (1 + (np.abs(perm - perm.mean()) >= abs(I - perm.mean())).sum()) / (nperm + 1)
ROWS = []
for c, (Rr, Cc) in SHAPES.items():
    A = load(PRED / f"benchmark_{c}_test_predictions.npz"); Bz = load(PRED / f"benchmark_{c}_poisson_predictions.npz")
    y = A["y"].astype(float); assert np.array_equal(y, Bz["y"].astype(float)), f"{c}: test sets differ between stage 2 and stage 4"
    cat = A["cat"]; K = int(cat.max()) + 1; N = Rr * Cc; D = len(y) // (N * K); Y = y.reshape(D, N, K)
    active = Y.sum((0, 2)) > 0; W = queen(Rr, Cc)[np.ix_(active, active)]; ok = W.sum(1) > 0; W = W[np.ix_(ok, ok)]; W = W / W.sum(1, keepdims=True)
    sel = np.flatnonzero(active)[ok]
    ours_days = np.mean([pdev_cells(y, p).reshape(D, -1).mean(1) for p in np.atleast_2d(A["Ours_GBM_Poisson_XGBoost_"]).astype(float)], 0)
    models = [("GBM-Poisson (ours)", "Poisson, all cell-days", A["Ours_GBM_Poisson_XGBoost_"]),
              ("Graph WaveNet", "squared error, crime-day selection", A["Graph_WaveNet_official_model_code_re_trained_"]),
              ("Graph WaveNet", "Poisson, all cell-days", Bz.get("Graph_WaveNet_Poisson_loss_all_cell_days_")),
              ("MTGNN", "squared error, crime-day selection", A["MTGNN_official_model_code_re_trained_"]),
              ("MTGNN", "Poisson, all cell-days", Bz.get("MTGNN_Poisson_loss_all_cell_days_"))]
    for name, setting, Pm in models:
        if Pm is None: continue
        Pm = np.atleast_2d(Pm).astype(float); per = []
        for p in Pm:
            pos = y > 0; P_ = p.reshape(D, N, K); res = ((Y - np.maximum(P_, 1e-3)) / np.sqrt(np.maximum(P_, 1e-3))).mean((0, 2))
            I, pI = moran(res[sel], W)
            per.append(dict(P1=np.mean([np.abs(p - y)[pos & (cat == k)].mean() for k in range(K)]), P3=pdev_cells(y, p).mean(), MAE_all=np.abs(p - y).mean(),
                            MAE_zero=np.abs(p - y)[~pos].mean(), P4=hit_rate(P_, Y), I=I, pI=pI))
        per = pd.DataFrame(per)
        days = np.mean([pdev_cells(y, p).reshape(D, -1).mean(1) for p in Pm], 0); diff = days - ours_days
        est = diff[RNG.integers(0, D, (B, D))].mean(1)
        ROWS.append(dict(city=c, model=name, setting=setting, seeds=len(per), P1_mean_MAE=per.P1.mean(), P3_deviance=per.P3.mean(), P3_deviance_sd=per.P3.std(),
                         P3_deviance_best_seed=per.P3.min(), MAE_all=per.MAE_all.mean(), MAE_zero_days=per.MAE_zero.mean(), P4_hit_rate=per.P4.mean(),
                         morans_I=per.I.mean(), morans_I_max_p=per.pI.max(), dev_minus_ours=diff.mean(), ci_lo=np.percentile(est, 2.5), ci_hi=np.percentile(est, 97.5),
                         days_ours_better=int((diff > 0).sum()) if name != "GBM-Poisson (ours)" else np.nan, days=D))
S = pd.DataFrame(ROWS); S.to_csv(T / "s07_poisson_vs_literature.csv", index=False); print(S.round(4).to_string())

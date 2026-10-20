"""Decision-oriented re-scoring of the saved benchmark predictions (no re-training).
P4: each test day and crime category, the k = 10 % of cells with the highest forecast are 'selected' (a patrol or hot-spot list);
hit rate = share of that day's crimes that fall in the selected cells; PAI = hit rate / share of area selected (Chainey et al., 2008).
Ties at the threshold are shared fractionally, so a constant forecast gets exactly the area share.
Also: a day-block bootstrap of the full-protocol deviance difference between our forecaster and each deep model.
usage: python analysis_decision.py <optionA_outputs folder>   -> writes tables/h01-h03"""
import sys, re
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import kendalltau
R = Path(sys.argv[1]); T = R / "tables"; PRED = R / "predictions"
K_SHARE, B, RNG = 0.10, 2000, np.random.default_rng(42)
NAMES = {"Ours_GBM_Poisson_XGBoost_": "Ours: GBM-Poisson (XGBoost)", "Ours_GBM_Poisson_LightGBM_": "Ours: GBM-Poisson (LightGBM)",
         "Ours_protocol_aware_zero_truncated_median_": "Ours – protocol-aware (zero-truncated median)", "Naive_last_day": "Naive: last day",
         "Naive_30_day_mean": "Naive: 30-day mean", "Trivial_constant_1": "Trivial: constant 1", "ST_HSL_re_run_": "ST-HSL (re-run)",
         "Graph_WaveNet_official_model_code_re_trained_": "Graph WaveNet (official model code, re-trained)", "MTGNN_official_model_code_re_trained_": "MTGNN (official model code, re-trained)",
         "ST_SHN_re_run_epoch_chosen_on_validation_": "ST-SHN (re-run, epoch chosen on validation)"}
def load(c):
    f = PRED / f"benchmark_{c}_test_predictions.npz"
    if f.is_dir(): return {p.stem: np.load(p) for p in f.rglob("*.npy")}
    z = np.load(f); return {k: z[k] for k in z.files}
def selected_weights(p, k):
    """Inclusion weight of every cell when the k largest forecasts are selected; ties at the threshold share the remaining slots."""
    order = np.sort(p)[::-1]; v = order[k - 1]; above = p > v; tie = p == v
    w = above.astype(float); free = k - above.sum()
    if tie.any(): w[tie] = free / tie.sum()
    return w
def pdev_cells(y, mu):
    mu = np.maximum(mu, 0.01); return 2 * (np.where(y > 0, y * np.log(np.maximum(y, 1e-12) / mu), 0) - (y - mu))

rows, boot = [], []
for c in ("NYC", "CHI"):
    Z = load(c); y = Z["y"]; K = int(Z["cat"].max()) + 1; N = len(y) // (K * 92); D = len(y) // (N * K)
    Y = y.reshape(D, N, K); k = int(round(K_SHARE * N))
    for key, name in NAMES.items():
        if key not in Z: continue
        Pm = np.atleast_2d(Z[key])
        for s, p in enumerate(Pm):
            P = p.reshape(D, N, K); cap = tot = 0.0; per_cat = []
            for kk in range(K):
                c_ = t_ = 0.0
                for d in range(D):
                    w = selected_weights(P[d, :, kk], k); c_ += (w * Y[d, :, kk]).sum(); t_ += Y[d, :, kk].sum()
                per_cat.append(c_ / t_); cap += c_; tot += t_
            hr = cap / tot
            rows.append(dict(city=c, model=name, seed=s, cells_selected=k, area_share=k / N, hit_rate=hr, PAI=hr / (k / N), **{f"hit_rate_cat{kk}": v for kk, v in enumerate(per_cat)}))
    # day-block bootstrap of deviance differences (seed-averaged per day): deep model minus ours
    ours = np.mean([pdev_cells(y, p).reshape(D, -1).mean(1) for p in np.atleast_2d(Z["Ours_GBM_Poisson_XGBoost_"])], 0)
    for key in ("MTGNN_official_model_code_re_trained_", "Graph_WaveNet_official_model_code_re_trained_", "ST_HSL_re_run_", "ST_SHN_re_run_epoch_chosen_on_validation_", "Naive_30_day_mean"):
        other = np.mean([pdev_cells(y, p).reshape(D, -1).mean(1) for p in np.atleast_2d(Z[key])], 0); diff = other - ours
        idx = RNG.integers(0, D, (B, D)); est = diff[idx].mean(1)
        boot.append(dict(city=c, model=NAMES[key], mean_deviance_model=other.mean(), mean_deviance_ours=ours.mean(), difference=diff.mean(),
                         ci_lo=np.percentile(est, 2.5), ci_hi=np.percentile(est, 97.5), days_ours_better=int((diff > 0).sum()), days=D))
H = pd.DataFrame(rows); H.to_csv(T / "h01_decision_protocol_per_seed.csv", index=False)
HM = H.groupby(["city", "model"]).agg(seeds=("seed", "size"), hit_rate=("hit_rate", "mean"), hit_rate_sd=("hit_rate", "std"), PAI=("PAI", "mean")).reset_index()
b12 = pd.read_csv(T / "b12_rankings_by_protocol.csv")
HM = HM.merge(b12[["city", "model", "P1_mean_MAE", "P3_deviance_all"]], on=["city", "model"], how="left")
HM["rank_P4_hit_rate"] = HM.groupby("city")["hit_rate"].rank(ascending=False, method="min")
HM.to_csv(T / "h02_decision_protocol.csv", index=False)
TAU = []
for c, g in HM.groupby("city"):
    for col in ("P1_mean_MAE", "P3_deviance_all"):
        t, pv = kendalltau(-g["hit_rate"], g[col]); TAU.append(dict(city=c, protocol_a="P4_hit_rate (higher better)", protocol_b=col, kendall_tau=t, p_value=pv, models=len(g)))
pd.DataFrame(TAU).to_csv(T / "h03_decision_rank_agreement.csv", index=False)
pd.DataFrame(boot).to_csv(T / "h04_deviance_difference_bootstrap.csv", index=False)
print(HM.sort_values(["city", "hit_rate"], ascending=[True, False]).round(4).to_string()); print(pd.DataFrame(TAU).round(3).to_string()); print(pd.DataFrame(boot).round(4).to_string())

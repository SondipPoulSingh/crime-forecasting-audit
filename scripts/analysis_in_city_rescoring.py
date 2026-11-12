"""Post hoc re-scoring of the saved benchmark test predictions (no re-training).
Full protocol P3 restricted to the grid cells that record at least one crime in the test period
(123 of 256 cells in New York, 92 of 168 in Chicago), next to P1 and P3 on all cells, with ranks and Kendall's tau.
Also recomputes the 10-model rank agreements reported in Section 5.2 and Table 3.
usage: python analysis_in_city_rescoring.py <results folder>   -> writes tables/s08_in_city_rescoring.csv (Supplementary Table S15)"""
import sys, numpy as np, pandas as pd
from scipy.stats import kendalltau
R=sys.argv[1]
b12=pd.read_csv(f"{R}/tables/b12_rankings_by_protocol.csv")
b12=b12[~b12.model.str.contains("best test epoch")]
NAMES = {"Ours_GBM_Poisson_XGBoost_": "Ours: GBM-Poisson (XGBoost)", "Ours_GBM_Poisson_LightGBM_": "Ours: GBM-Poisson (LightGBM)",
 "Ours_protocol_aware_zero_truncated_median_": "Ours – protocol-aware (zero-truncated median)", "Naive_last_day": "Naive: last day",
 "Naive_30_day_mean": "Naive: 30-day mean", "Trivial_constant_1": "Trivial: constant 1", "ST_HSL_re_run_": "ST-HSL (re-run)",
 "Graph_WaveNet_official_model_code_re_trained_": "Graph WaveNet (official model code, re-trained)", "MTGNN_official_model_code_re_trained_": "MTGNN (official model code, re-trained)",
 "ST_SHN_re_run_epoch_chosen_on_validation_": "ST-SHN (re-run, epoch chosen on validation)"}
def dev(y,mu):
    mu=np.maximum(mu,0.01); return 2*(np.where(y>0,y*np.log(np.maximum(y,1e-12)/mu),0)-(y-mu))
out=[]
for c in ("NYC","CHI"):
    d=b12[b12.city==c].set_index("model")
    for a,b in (("P1_mean_MAE","P3_deviance_all"),("P1_mean_MAE","P3_MAE_all"),("P2_mean_MAE","P3_deviance_all")):
        t=kendalltau(d[a],d[b]); print(c,a,b,"tau=%.3f p=%.3f n=%d"%(t[0],t[1],len(d)))
    d["r1"]=d.P1_mean_MAE.rank(); d["r3"]=d.P3_deviance_all.rank()
    z=np.load(f"{R}/predictions/benchmark_{c}_test_predictions.npz"); y=z["y"]; K=4; D=92; N=len(y)//(K*D)
    Y=y.reshape(D,N,K); active=Y.sum((0,2))>0; print(c,"cells",N,"active",active.sum(), "zero share all %.3f active %.3f"%((Y==0).mean(),(Y[:,active,:]==0).mean()))
    # check layout by reproducing the all-cell deviance
    res={}
    for k,n in NAMES.items():
        P=np.atleast_2d(z[k]); allv=np.mean([dev(y,p).mean() for p in P]); act=np.mean([dev(y,p).reshape(D,N,K)[:,active,:].mean() for p in P])
        p1=np.mean([np.mean([np.abs(y-p)[(y>0)&(z["cat"]==kk)].mean() for kk in range(K)]) for p in P])
        res[n]=(p1,allv,act)
    r=pd.DataFrame(res,index=["P1","P3_all","P3_active"]).T
    r["chk"]=d.P3_deviance_all; r["chkP1"]=d.P1_mean_MAE
    r["rank_P1"]=r.P1.rank().astype(int); r["rank_all"]=r.P3_all.rank().astype(int); r["rank_act"]=r.P3_active.rank().astype(int)
    print(r.round(4).to_string())
    for a,b in (("P1","P3_active"),("P3_all","P3_active"),("P1","P3_all")):
        t=kendalltau(r[a],r[b]); print("  ",c,a,b,"tau=%.3f p=%.3f"%(t[0],t[1]))
    r.insert(0,"city",c); out.append(r)
pd.concat(out).to_csv(f"{R}/tables/s08_in_city_rescoring.csv")

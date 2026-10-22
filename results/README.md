# Results

All tables are CSV files written by the notebooks (runs 1–4) and by the scripts in `scripts/` (h01–h04, s01–s07). Prefixes: **b** benchmark audit, **c** weekly forecasts, **d** intervals (main split), **e** rolling origins, **f** equity, **g** protocol census, **h** decision protocol and bootstrap, **s** spatial analysis and the deep baselines in the Poisson setting, **z** hypothesis verdicts. Run 4 adds `e00`, `e01_poisson_*`, `e02_poisson_*` and `e03_boundary_sources`, which are separate from the rolling-origin tables `e01`–`e03` of runs 1 and 3.

| File | Content |
|---|---|
| `b01_code_versions_and_patches.csv` | official code versions (commits) and the minimal patches |
| `b02_dataset_identity.csv` | benchmark files identical in the ST-HSL and ST-SHN repositories (MD5) |
| `b03_protocol_audit.csv` | protocol audit of the official evaluation code |
| `b04_benchmark_sparsity.csv` | benchmark sparsity and share of rows scored by each protocol |
| `b05_rerun_log.csv` | re-run log (epochs, selected epoch, minutes) |
| `b06_benchmark_scores_per_seed.csv` | benchmark scores per model and seed under P1/P2/P3 |
| `b07_benchmark_scores_mean_sd.csv` | benchmark scores, mean and sd over seeds |
| `b08_official_protocol_with_published.csv` | official protocol P1 with published numbers |
| `b09_reproduction_gap.csv` | reproduction gap (re-run vs published) |
| `b09b_reproduction_gap_ci.csv` | reproduction gap with 95 % t-intervals over seeds |
| `b10_stshn_test_selection_optimism.csv` | ST-SHN: test- vs validation-selected epoch |
| `b11_metric_exploit.csv` | metric exploit by trivial predictors |
| `b12_rankings_by_protocol.csv` | rankings under the protocols |
| `b13_ranking_agreement.csv` | rank agreement between protocols (Kendall tau) |
| `b14_simple_vs_deep_full_protocol.csv` | simple vs deep, seed by seed (full protocol) |
| `b15_benchmark_interval_coverage.csv` | benchmark intervals (run 1) |
| `b16_stshn_epoch_budget.csv` | ST-SHN: 10 vs 25 epochs |
| `b17_v3_benchmark_intervals_with_zinb.csv` | benchmark intervals with ZINB baselines (run 3) |
| `c01_offence_mapping.csv` | offence → crime-type mapping |
| `c01_offence_mapping_v3.csv` | offence → crime-type mapping |
| `c02_panel_summary.csv` | weekly panel summary |
| `c02_panel_summary_v3.csv` | weekly panel summary |
| `c02b_v3_panel_check_against_v1.csv` | panels rebuilt in run 3 vs run 1 |
| `c03_lightgbm_feature_importance_H1.csv` | LightGBM feature importance |
| `c04_forecast_accuracy_per_seed.csv` | weekly forecast accuracy per seed |
| `c04b_model_selection.csv` | model selection on validation |
| `c05_forecast_accuracy_mean_sd.csv` | weekly forecast accuracy, mean and sd |
| `c06_skill_vs_every_baseline_per_seed.csv` | forecast skill vs every baseline per seed |
| `c07_skill_summary.csv` | forecast skill summary |
| `c08_v3_forecast_deviance.csv` | run-3 forecast deviance (LightGBM, GRU-ZINB) |
| `d01_interval_coverage_main_split.csv` | interval coverage, main split (run 1) |
| `d02_interval_coverage_over_seeds.csv` | interval coverage over seeds (run 1) |
| `d03_v3_interval_coverage_with_zinb.csv` | interval coverage incl. ZINB baselines (run 3) |
| `d04_v3_interval_coverage_over_seeds.csv` | run-3 coverage over seeds |
| `d05_v3_zinb_parameters.csv` | fitted ZINB parameters |
| `d06_v3_consistency_with_v1.csv` | run 3 vs run 1 consistency |
| `e01_rolling_interval_coverage.csv` | rolling-origin interval coverage (run 1) |
| `e02_rolling_forecast_skill.csv` | rolling-origin forecast skill |
| `e03_v3_rolling_coverage_with_zinb.csv` | rolling-origin coverage incl. ZINB (run 3) |
| `f01_area_demographics.csv` | area demographics (ACS) |
| `f02_coverage_by_group.csv` | coverage by poverty / minority group |
| `f03_forecast_skill_by_group.csv` | forecast skill by group |
| `f04_risk_list_attention_by_poverty_group.csv` | risk-list attention by poverty group |
| `f05_area_coverage_vs_demographics.csv` | area coverage vs demographics (Spearman) |
| `f06_coverage_gap_between_groups.csv` | coverage gap between groups |
| `g01_v3_protocol_census_long.csv` | protocol census, long form |
| `g02_v3_protocol_census.csv` | protocol census, one row per codebase |
| `h01_decision_protocol_per_seed.csv` | decision protocol P4 per seed |
| `h02_decision_protocol.csv` | decision protocol P4 (hit rate, PAI, rank) |
| `h03_decision_rank_agreement.csv` | P4 rank agreement with P1 and P3 |
| `h04_deviance_difference_bootstrap.csv` | day-block bootstrap of deviance differences |
| `z01_hypothesis_verdicts.csv` | all hypotheses and verdicts |

`predictions/` holds the benchmark test predictions of every model and seed (`benchmark_<city>_test_predictions.npz`, arrays in day × cell × category order) and the weekly test predictions and intervals of run 1 (`*.csv.gz`). `census/` holds the ACS 2022 tract tables and Census gazetteer files used for the equity check. `logs/` holds the progress log, status and software environment of each run.

## Added in run 4 and by the spatial scripts

| File | Content |
|---|---|
| `e00_code_versions_v4.csv` | official code at the v1 commits (run 4) |
| `e01_poisson_retrain_runs.csv` | Graph WaveNet / MTGNN trained with Poisson loss on all cell-days, per seed (run 4) |
| `e02_poisson_retrain_summary.csv` | the same, mean / sd / min / max over seeds (run 4) |
| `e03_boundary_sources.csv` | portal datasets of the area boundaries (run 4) |
| `s01_bench_morans_i.csv`, `s01_bench_morans_i_per_seed.csv` | Moran's I of benchmark residuals |
| `s02_bench_scale_sensitivity.csv`, `s03_bench_scale_rank_agreement.csv` | benchmark aggregated to 6- and 12-km cells |
| `s04_area_coverage.csv` | per-area interval coverage, weekly panels |
| `s05_area_morans_i.csv` | Moran's I of area coverage and residuals |
| `s06_area_residual_{CHI,NYC,SF}.csv` | per-area mean Pearson residual |
| `s07_poisson_vs_literature.csv` | deep baselines in the literature vs. the Poisson setting |

`boundaries/` holds the GeoJSON area boundaries; `predictions/benchmark_{NYC,CHI}_poisson_predictions.npz` the run-4 test predictions.

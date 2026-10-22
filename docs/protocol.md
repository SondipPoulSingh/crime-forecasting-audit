# An evaluation protocol for sparse urban count forecasts

Derived from the results of this study (Section 6.3 of the paper). A checklist for authors and reviewers of crime- and other urban-event forecasting papers.

1. **Score all space-time units.**
   - Use a proper loss for counts: mean Poisson deviance, or a proper score of the predictive distribution.
   - Add the decision criterion the forecast serves — e.g. the hit rate and Predictive Accuracy Index of a top-k hot-spot list.
   - Errors on zero and non-zero units may be reported separately, but never only the latter.
   - Do not use MAPE on sparse counts.
2. **Report results at the unit of use, and check their sensitivity to the unit.**
   - Re-score on the spatial units on which decisions are made.
   - Also re-score on at least one coarser or finer unit. On the benchmark studied here, the agreement between the official and the full ranking moves from τ = −0.42 on 3-km cells to τ = 0.78 on 12-km cells.
3. **Include strong simple baselines and train deep baselines the same way.**
   - Use a recent mean, a seasonal mean and a gradient-boosted Poisson model as baselines.
   - Train deep baselines with the same loss and the same selection rule.
   - Select epochs and hyper-parameters on validation data only.
   - Report several random seeds with their spread. On the benchmark studied here, the seed alone moves MAE by up to 15 %.
4. **Map the errors, and evaluate uncertainty by series volume, by area and over time**, not only marginally.
   - Report Moran's I of the residuals. A small citywide difference can hide errors that cluster in a few districts.
   - Mondrian conformal calibration (bins of the forecast) with an adaptive update is a simple default when nominal coverage in every group is the goal.
   - A well-dispersed negative binomial is a strong alternative when conservative intervals are acceptable.
   - Compare intervals by a proper score (interval score) as well as by coverage, and report under- and over-coverage separately.
5. **Check error and calibration across communities**, and release data, code and the exact evaluation script.

## Quick self-test for an existing evaluation

- Does a constant forecast of 1 beat your model on your metric? If yes, the metric does not measure forecasting.
- Does the ranking of your models change when all units are scored? If yes, report the full-protocol ranking.
- Are your reported numbers from one seed? If yes, report at least five.
- Do the residuals of your model cluster in space (Moran's I)? If yes, show the map.

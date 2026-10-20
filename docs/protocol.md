# An evaluation protocol for sparse urban count forecasts

Derived from the results of this study (Section 6.2 of the paper). A checklist for authors and reviewers of crime- and other urban-event forecasting papers.

1. **Score all space-time units.**
   - Use a proper loss for counts: mean Poisson deviance, or a proper score of the predictive distribution.
   - Add the decision criterion the forecast serves — e.g. the hit rate and Predictive Accuracy Index of a top-k hot-spot list.
   - Errors on zero and non-zero units may be reported separately, but never only the latter.
   - Do not use MAPE on sparse counts.
2. **Include strong simple baselines** — a recent mean, a seasonal mean and a gradient-boosted Poisson model — alongside deep baselines.
3. **Select epochs and hyper-parameters on validation data only.** Report several random seeds with their spread; on the benchmark studied here, the seed alone moves MAE by up to 15 %.
4. **Evaluate uncertainty by series volume and over time**, not only marginally.
   - Mondrian conformal calibration (bins of the forecast) with an adaptive update is a simple default when nominal coverage in every group is the goal.
   - A well-dispersed negative binomial is a strong alternative when conservative intervals are acceptable.
   - Compare intervals by a proper score (interval score) as well as by coverage, and report under- and over-coverage separately.
5. **Check error and calibration across communities**, and release data, code and the exact evaluation script.

## Quick self-test for an existing evaluation

- Does a constant forecast of 1 beat your model on your metric? If yes, the metric does not measure forecasting.
- Does the ranking of your models change when all units are scored? If yes, report the full-protocol ranking.
- Are your reported numbers from one seed? If yes, report at least five.

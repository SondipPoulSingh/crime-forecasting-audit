# Internal review of draft 2 (written as a CEUS referee)

**Manuscript:** "Scoring only where crime happened: an audit of evaluation practice in deep spatio-temporal crime forecasting, and calibrated count forecasts for three U.S. cities"
**Recommendation at draft 2:** major revision. The core finding is important and well supported, but the paper does not yet speak to CEUS readers in their terms, several claims need firmer statistics, and some passages are too compressed for a reader outside machine learning.

## Summary of the submission
The authors show that the de-facto benchmark for deep crime prediction scores only cell-days with crime, that trivial predictors win under this metric, that rankings change on all cell-days, that a gradient-boosted Poisson model beats re-run deep models, and that the practice is common in public code. A second part studies calibrated weekly count forecasts for three cities, including conformal and zero-inflated intervals, a 2020 shift and an equity check.

## Strengths
1. A clear, consequential and verifiable finding; the trivial-predictor argument is mechanical and convincing.
2. Official code re-run with several seeds; patches documented; hypotheses fixed in advance.
3. The protocol census turns an anecdote into evidence about current practice.
4. Honest reporting of negative results (A6, A9b) and of the San Francisco exception.

## Major comments
**M1. Relevance to urban decision-making (fit to CEUS).** The paper evaluates forecasts with statistical losses only. Crime forecasts are used to select places (hot spots, patrol areas). Show whether the protocol also misleads on a *decision* metric that place-based policing uses, e.g. the hit rate and the Predictive Accuracy Index (PAI) of a top-k cell list (Chainey et al., 2008). Without it, a CEUS reader may conclude that P1 is merely "a different loss". → *Added: decision protocol P4 (top-10 % cells per day and category), Section 5.3, Table 4.*

**M2. Statistical support for "beats every deep model".** Differences to MTGNN are about 3 %. A seed-wise comparison is given, but not uncertainty over test days. → *Added: day-block bootstrap of the deviance difference and the number of days on which each model is better (Table 5).*

**M3. The reproduction gap needs a seed-level reading.** The official ST-HSL seed should be reported separately: does the official seed reproduce the published numbers? If so, the published figures may be a favourable draw. → *Added: per-seed gaps; in New York the official seed reproduces (−3.5 to +0.4 %) while the other seeds are 1.7–15.1 % worse; in Chicago no seed reaches the published value in any category. Text and Fig. 2 caption revised.*

**M4. "Pre-registered" overstates.** The hypotheses were fixed in the notebook before execution, but not registered with a third party. → *Changed to "pre-specified (fixed in the analysis notebook before the run; timestamps in the run logs)"; post-hoc additions labelled.*

**M5. Spatial units and the modifiable areal unit problem.** The benchmark uses ~3 km grid cells, the city panels administrative areas of very different size. Discuss why both, and that conclusions about sparsity depend on the unit and time step (daily vs weekly). → *Added to Data (3.2) and Threats to validity.*

**M6. Interval evaluation: what Mondrian guarantees and why the interval score disagrees.** Mondrian bins on the forecast; evaluation is by training-volume quintile. Explain the relation, and discuss discreteness and randomised intervals as an alternative. → *Expanded in Methods 4.5 and Results 5.8.*

**M7. Equity analysis is ecological and small-sample.** 40–77 areas per city; many comparisons. State this and avoid causal language. → *Revised in Results 5.10 and Limitations.*

**M8. The two parts of the paper need an explicit bridge.** Why does an audit of a 2014–2017 daily benchmark belong with weekly 2016–2026 city panels? → *Added a paragraph at the start of Section 3 and in the introduction: the benchmark shows what goes wrong; the city panels show the recommended protocol applied end-to-end on current data and at operational units.*

## Minor comments
- m1. Define MAPE, the deviance floor and the interval score formally. → *Equations added.*
- m2. Explain why P2 exists (ST-SHN's choice of "informative" cells) and why it is still outcome-selected. → *Added.*
- m3. Notation: use $c$, $t$, $k$ consistently; state the number of rows per protocol. → *Done.*
- m4. Table 4 (per-cell reproduction) is long; consider moving to an appendix. → *Moved to Appendix A; the figure stays in the main text.*
- m5. The term "exploit" may read as polemical; describe the zero-truncated median neutrally as a "protocol-aware" forecast. → *Done.*
- m6. Specify hardware, software versions and run times. → *Added to Methods 4.2 and the repository.*
- m7. Give the repository URL and a code/data availability statement in the journal's format. → *Done.*
- m8. Highlights should match the final results (zero-inflated intervals, decision metric). → *Updated.*

## Reference check
All 33 references were verified against Crossref, publisher pages, arXiv or the proceedings site. Corrections: ST-HHOL authors added (Du, Kang, Yang & Shao, ICLR 2026); EasyST pages and DOI added (CIKM 2024, 2220–2229, 10.1145/3627673.3679749); Richardson et al. page corrected to 192 (94 N.Y.U. L. Rev. Online 192); Lambert (1992) pages 1–14. Three references added for the revision (Weisburd, 2015; Osgood, 2000; Zheng et al., 2014) were verified the same way.

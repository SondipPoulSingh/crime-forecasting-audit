# Reproducing the study on Kaggle

The study ran in three stages. Each stage reads the outputs of the previous one, so they must run in order. All stages use one free Kaggle notebook with a T4 GPU.

## Common settings

- *Settings → Accelerator → GPU T4 x2*
- *Settings → Internet → On* (the notebooks download data and clone the official model repositories)
- *Save Version → Save & Run All (Commit)* — a commit keeps running when the browser is closed.
- Every part saves its tables as soon as it finishes and refreshes the output zip, so a late failure never loses earlier results. Errors are written to `logs/errors.log` and the notebook continues.

## Stage 1 — main run (`notebooks/01_main_run_v1.ipynb`, ≈3 h)

Benchmark code audit, city data and weekly panels, ten forecasters × four horizons × three seeds, all interval methods, rolling origins 2020–2024, and the benchmark re-runs (3 seeds; ST-SHN 2 seeds).

Output: `optionA_outputs.zip`.

## Stage 2 — patch run (`notebooks/02_patch_run_v2.ipynb`, ≈7.5 h)

1. Create a Kaggle dataset from `optionA_outputs.zip` (*Datasets → New Dataset*) and add it as input (*Add Input → Your Datasets*).
2. Get a free Census API key at <https://api.census.gov/data/key_signup.html> and activate it from the e-mail.
3. *Add-ons → Secrets → Add secret*, label `CENSUS_API_KEY`, and tick the box that attaches it to the notebook.

The notebook adds seeds 3–4 for ST-HSL, Graph WaveNet and MTGNN, re-runs ST-SHN with five seeds and a 25-epoch budget, runs the equity check, and recomputes all benchmark tables and verdicts. Finished re-runs are saved in `predictions/patch_runs/`; if the session stops, add the partial output as a second input and run again — finished runs are skipped.

Output: `optionA_outputs_v2.zip` (complete v1 + v2 result set).

## Stage 3 — census and ZINB run (`notebooks/03_zinb_and_census_v3.ipynb`, ≈45 min)

1. Add the stage-2 output as input — either as a dataset, or directly: *Add Input → Notebooks → Your Work* → the stage-2 notebook.

The notebook audits seven public crime-prediction repositories at pinned commits, rebuilds the weekly panels (cut at stage 1's last week), adds the zero-inflated interval baselines (fitted and learned GRU-ZINB) on the main split, the rolling origins and the benchmark, and checks consistency with stage 1.

Output: `optionA_outputs_v3.zip` (complete v1 + v2 + v3 result set; its `tables/` folder is this repository's `results/tables/`).

## Paper tables and figures (local)

```bash
pip install -r requirements.txt
python scripts/analysis_decision.py results     # decision protocol P4 and bootstrap → results/tables/h01–h04
python scripts/make_tables.py results paper     # paper/tables/*.tex
python scripts/make_figures.py results paper    # paper/figures/*.pdf and .png
```

Then compile `paper/main.tex` with pdflatex + bibtex (Overleaf works out of the box).

## Notes

- Kaggle unpacks `.csv.gz` and `.npz` files inside uploaded zip files into folders; the notebooks read both forms.
- Open-data portals revise old records; a rebuild months later can differ by a few hundredths of a per cent in incident counts (stage 3 reports the difference in `c02b_v3_panel_check_against_v1.csv`).
- GPU kernels are not fully deterministic, so re-runs of the deep models can differ slightly from the saved predictions.

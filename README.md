# The Per-Attribute ε Illusion: code, results and figures

Code and result files for the manuscript

> M. A. Sultan and I. Shaukat, *The Per-Attribute ε Illusion: Matched-Budget Local versus Central Differential Privacy for Tabular Credit-Scoring Data* (submitted).

The study compares central differential privacy (DP logistic regression, DP Gaussian Naive Bayes) with local DP (Laplace noise, randomized response, noise-aware Naive Bayes, logistic regression) on nine public tabular datasets at **matched total ε**, runs a linkage attack and a membership-inference attack, evaluates a public-sample-guided budget allocation, and verifies the closed-form results of the paper numerically.

## Layout

| Folder | Content |
|---|---|
| `notebooks/final_pipeline_v3.ipynb` | Main experiments (Colab): utility at matched total ε, linkage attack (pool 100/500/2,000), membership inference, tables and figures |
| `notebooks/step11_allocation.ipynb` | Public-sample-guided budget allocation (proportional and top-k) |
| `notebooks/step12_central_topk.ipynb` | Central DP-LR restricted to the same top-k attributes |
| `theory/` | Monte Carlo checks of the propositions (`theory_check.py`, `theory_check_exact_variance.py`) and their saved outputs |
| `results/` | Result CSV files (`util_*.csv`, `reid_*.csv`, `mia_*.csv`, `table_*.csv`) and rounded summaries of the allocation experiments |
| `scripts/make_figures.py` | Regenerates Fig. 1-7 of the paper from `results/` |
| `figures/` | Fig. 1-7 (PDF and PNG) |

## How to reproduce

1. **Main experiments.** Open `notebooks/final_pipeline_v3.ipynb` in Google Colab and run all cells. Results are saved per dataset to `dp_paper/final/` on Google Drive and the run can be resumed after a disconnect. It takes roughly one to two hours for all nine datasets. (OpenML occasionally returns HTTP 503; re-running the cell resumes the missing datasets.)
2. **Allocation experiment.** Run `notebooks/step11_allocation.ipynb` (it uses the same setup and Drive folder), then `notebooks/step12_central_topk.ipynb`.
3. **Mathematical checks.** `python theory/theory_check.py` and `python theory/theory_check_exact_variance.py` (NumPy only, under a minute each). They compare the formulas of Propositions 5, 6, 7, 9 and 10 and Corollary 6.1 with simulation.
4. **Figures.** `python scripts/make_figures.py` rebuilds the figures from `results/`.

## Data

All datasets are public and are **not** redistributed here. The notebooks download them from their sources: Kaggle loan-approval data (copy at `shrikant-temburwar/Loan-Prediction-Dataset`), UCI German credit, OpenML ids 42477 (Taiwan credit default), 29 (credit approval), 46468 (Give Me Some Credit), 1461 (bank marketing), 37 (Pima diabetes), OpenML "adult" (version 2), and the Lending Club 2007-2010 sample (`Amitabh1989/random_forest`). See Table 1 and the reference list of the paper for the original sources.

## Notes

* A random 10% of each dataset is held out as a *public reference sample*; all clipping bounds and Naive Bayes bin edges come from it. Models are trained and evaluated on the remaining 90%.
* `results/alloc_summary_rounded.csv` and `results/central_topk_summary_rounded.csv` contain the tables printed by the Step 11 and Step 12 notebooks, rounded to three decimals; the notebooks write the unrounded files to Google Drive.
* Random seeds are fixed (42 and the seeds in the notebooks); small differences from the paper can occur with other library versions. The central baselines use `diffprivlib` (tested with scikit-learn 1.5-1.6; newer scikit-learn releases may raise errors in `diffprivlib`).
* Parts of the code were written with the assistance of an AI tool (Claude, Anthropic); the authors reviewed the code and the results.

## Citation and license

See `CITATION.cff`. Code is released under the MIT license (see `LICENSE`). Contact: Muhammad Atif Sultan, atifbhatti865@gmail.com.

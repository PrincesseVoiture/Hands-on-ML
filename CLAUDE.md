# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Group project for the M1 AI "Hands-on Machine Learning" course (2026-27). `M1-HoML (2026-27).md` is the assignment; `HAR.ipynb` is the single deliverable: code + report (Markdown) in **English**, classifying 6 activities from the Kaggle/UCI *Human Activity Recognition with Smartphones* dataset (561 features, 7352 train / 2947 test rows). The user writes in French; the notebook and slides stay in English. `REVUE_HAR.md` (French) is the review of the notebook with an ordered fix checklist; tick items there as they get done.

Assignment constraints that affect edits:
- One notebook, < 10 MB (avoid Plotly / heavy embedded outputs), all member names in it.
- Data is never committed, only linked (`train.csv` / `test.csv` are gitignored).
- Report structure: Intro → Preprocessing → Training → Evaluation → Conclusion, plus a SOTA summary and arguments for why the approach is original.
- Grading: preprocessing /5, model /5, code readability /5, evaluation & analysis /5 (+ individual oral /20, 10 min + 10 min questions).

## Setup & commands

```bash
uv venv --python 3.14 .venv && uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -m ipykernel install --user --name hands-on-ml

# Data (public dataset, no Kaggle token needed); the notebook reads ./train.csv and ./test.csv
.venv/bin/python -c "import kagglehub,shutil;p=kagglehub.dataset_download('uciml/human-activity-recognition-with-smartphones');[shutil.copy(f'{p}/{f}','.') for f in ('train.csv','test.csv')]"

.venv/bin/jupyter lab

# Headless top-to-bottom run (~2 min). Write to a copy unless the user wants outputs refreshed in place.
.venv/bin/jupyter nbconvert --to notebook --execute --ExecutePreprocessor.kernel_name=hands-on-ml --output HAR_run.ipynb HAR.ipynb
```

There are no tests or linters. To check one experiment, run its pipeline in a scratch script instead of re-executing the whole notebook. Edit cells with NotebookEdit; to read the notebook, parse the JSON (it embeds base64 PNGs).

## Notebook architecture

Cells share global state and must run in order:
- `X_train`/`y_train`/`X_test`/`y_test` (all columns except `Activity` and `subject`, then the 21 duplicated columns dropped in section 2, so models see 540 features), `groups = train_df["subject"]`, `cv = GroupKFold(n_splits=5)`.
- All imports (stdlib, pandas, sklearn) live in the first code cell; no other cell imports anything.
- Every experiment in sections 3 and 4 is a `Pipeline` (StandardScaler / PCA / SelectKBest inside it) run through one `evaluate(name, steps, n_jobs=-1)` helper defined in §3.1, just before `lr_pipeline`: it calls `cross_validate(..., groups=groups, cv=cv, scoring=["accuracy", "f1_macro"], return_estimator=True)`, prints accuracy ± std (`ddof=1`), F1-macro and fit time, and stores `results[name] = {"scores": …, "dimensions": …}`. `dimensions` is read from the fitted pipelines: `named_steps["pca"].n_components_`, `named_steps["selector"].get_support().sum()` (SelectKBest has no `k_`), else 540 — so PCA rows carry the measured `"101-102"`, not a hard-coded value. RF experiments are called with `n_jobs=1` (the forest already parallelizes internally); bare `evaluate(...)` calls end in `;` so the returned dict is not displayed.
- Section 4 loops over `k_values = [50, 100, 200]` and builds the per-model `{lr,svm,rf}_fs_df` tables with a small `fs_row(scores, k)` helper.
- Section 5 builds `comparison_df` from `results` and `k_values`; its bar plot filters SelectKBest rows at `Dimensions == 200` (ints there, strings like `"101-102"` on PCA rows). A new experiment shows up in the summary table and plot by calling `evaluate` and being listed in that loop.
- Section 6 refits `lr_pipeline` on the full train set; it is the only place the test set is used.
- `X_train_scaled` and the full `pca` (sections 2.2 and 2.4) are exploration only (variance curve, 95 % threshold); no model consumes them, and before section 6 the test set only loses the duplicated columns found on train. Inside the CV pipelines, PCA(0.95) keeps 101–102 components depending on the fold, not the 103 found on the full train set.

Methodological invariants to keep: subject-grouped CV (never plain KFold, which leaks subjects and inflates scores), all fitting inside pipelines, test set used once for the final model.

## Gotchas

- Figures quoted in Markdown cells are typed by hand and drift from the outputs when cells are re-run; after any re-execution, re-check every number in the text against the outputs.
- The working notebook was just re-run top-to-bottom (execution counts 1–28); keep validating any later edit with a fresh top-to-bottom run.
- LogisticRegression (lbfgs, default `tol=1e-4`) scores move by about ±0.001 across machines, and even on one machine when the BLAS thread count changes (7/2947 test predictions flip); `tol=1e-8` makes it deterministic. SVC and RF (`random_state=42`) reproduce exactly.
- Dataset quirks: features are pre-normalized to [-1, 1]; 21 columns are exact duplicates of others (13 `tGravityAccMag-*` = `tBodyAccMag-*`, and `*Mag-sma()` = `*Mag-mean()` for the 8 other magnitude signals), dropped in section 2, so text quoting the model input says 540 while text describing the raw dataset says 561; 84 `bandsEnergy` names are duplicated in the source and get `.1`/`.2` suffixes from pandas; train (21 subjects) and test (9 subjects) are subject-disjoint.

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
- `X_train`/`y_train`/`X_test`/`y_test` (all columns except `Activity` and `subject`), `groups = train_df["subject"]`, `cv = GroupKFold(n_splits=5)`.
- Every experiment is a `Pipeline` (StandardScaler / PCA / SelectKBest inside it) scored with `cross_validate(..., groups=groups, cv=cv, scoring=["accuracy", "f1_macro"])`. Results go to `{lr,svm,rf}_scores` and `{lr,svm,rf}_pca_scores` (dicts) and `{lr,svm,rf}_fs_df` (DataFrames for k = 50/100/200).
- Section 5 aggregates them through the `model_results` dict into `comparison_df` (PCA dimension hard-coded as `"≈102"`); a new experiment must be added there to show up in the summary table and plot.
- Section 6 refits `lr_pipeline` on the full train set; it is the only place the test set is used.
- `X_train_scaled` and the full `pca` (sections 2.2 and 2.4) are exploration only (variance curve, 95 % threshold); no model consumes them, and the test set is not touched before section 6. Inside the CV pipelines, PCA(0.95) keeps 100–101 components depending on the fold, not the 102 found on the full train set.

Methodological invariants to keep: subject-grouped CV (never plain KFold, which leaks subjects and inflates scores), all fitting inside pipelines, test set used once for the final model.

## Gotchas

- Figures quoted in Markdown cells are typed by hand and drift from the outputs when cells are re-run; after any re-execution, re-check every number in the text against the outputs.
- The committed notebook has out-of-order execution counts; validate with a fresh top-to-bottom run.
- LogisticRegression (lbfgs, default `tol=1e-4`) scores move by about ±0.001 across machines, and even on one machine when the BLAS thread count changes (7/2947 test predictions flip); `tol=1e-8` makes it deterministic. SVC and RF (`random_state=42`) reproduce exactly.
- Dataset quirks: features are pre-normalized to [-1, 1]; 21 columns are exact duplicates of others (e.g. `tGravityAccMag-*` = `tBodyAccMag-*`); 84 `bandsEnergy` names are duplicated in the source and get `.1`/`.2` suffixes from pandas; train (21 subjects) and test (9 subjects) are subject-disjoint.

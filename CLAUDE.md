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

# Headless top-to-bottom run (~15-20 min: four grid searches, two SVM kernels, 500-tree RF, §3.9 nested CV; 17m18s measured with the nested cell). Write to a copy unless the user wants outputs refreshed in place.
.venv/bin/jupyter nbconvert --to notebook --execute --ExecutePreprocessor.kernel_name=hands-on-ml --output HAR_run.ipynb HAR.ipynb
```

There are no tests or linters. To check one experiment, run its pipeline in a scratch script instead of re-executing the whole notebook. Edit cells with NotebookEdit; to read the notebook, parse the JSON (it embeds base64 PNGs).

## Notebook architecture

Cells share global state and must run in order:
- `X_train`/`y_train`/`X_test`/`y_test` (all columns except `Activity` and `subject`, then the 21 duplicated columns dropped in section 2, so models see 540 features), `groups = train_df["subject"]`, `cv = GroupKFold(n_splits=5)`.
- All imports (stdlib, pandas, sklearn) live in the first code cell; no other cell imports anything.
- §3.1 = protocol Markdown, then the `evaluate` helper code cell, then a **tuning cell**: four `GridSearchCV(Pipeline([...steps named "classifier"...]), grid, cv=cv, scoring="f1_macro", n_jobs=-1)` runs (LR `classifier__C`; SVM linear `classifier__C`; SVM rbf `classifier__C` × `classifier__gamma`; RF `classifier__n_estimators` × `classifier__max_features`, forest `n_jobs=1` inside so the search parallelizes instead) fill `best_params`, keyed `"Logistic Regression" / "SVM linear" / "SVM rbf" / "Random Forest"` with the `classifier__` prefix stripped, and display a `tuning_rows` table. Every classifier in §3–§4 is then `Classifier(max_iter=2000, random_state=42, n_jobs=-1, **best_params[...])` as appropriate. `results` model names are `"SVM linear"` / `"SVM rbf"` everywhere (also `+ PCA`, `+ SelectKBest (k)`); §3.4/§3.5/§4.2 loop `for kernel in ["linear", "rbf"]`. Tuning selects on the same GroupKFold that produces the reported scores; §3.1 flags the resulting optimism and §3.9 (markdown + code cell) measures it with a nested CV written as an explicit outer loop — `cv.split(X_train, y_train, groups)` → inner `GridSearchCV(cv=GroupKFold(n_splits=4), n_jobs=-1, refit=True)` fitted on the outer-training part only (`search.fit(X_tr, y_tr, groups=g_tr)`, `predict` on the outer-validation subjects) → `nested_rows` table with tuned vs nested F1, an `Optimism` column, and std across outer folds; the inner grids drop `gamma=0.01` and `max_features=0.3` (documented in the §3.9 text).
- Every experiment in sections 3 and 4 is a `Pipeline` (StandardScaler / PCA / SelectKBest inside it) run through one `evaluate(name, steps, n_jobs=-1)` helper defined in §3.1 (`lr_pipeline` is built right after the tuning cell, in §3.2): it calls `cross_validate(..., groups=groups, cv=cv, scoring=["accuracy", "f1_macro"], return_estimator=True)`, prints accuracy ± std (`ddof=1`), F1-macro and fit time, and stores `results[name] = {"scores": …, "dimensions": …}`. `dimensions` is read from the fitted pipelines: `named_steps["pca"].n_components_`, `named_steps["selector"].get_support().sum()` (SelectKBest has no `k_`), else 540 — so PCA rows carry the measured `"101-102"`, not a hard-coded value. RF experiments are called with `n_jobs=1` (the forest already parallelizes internally); bare `evaluate(...)` calls end in `;` so the returned dict is not displayed.
- Section 4 loops over `k_values = [50, 100, 200]` and builds the per-model `{lr,svm,rf}_fs_df` tables with a small `fs_row(scores, k)` helper; `svm_fs_df` nests the kernel loop inside the k loop and reorders columns so `Kernel` comes first.
- Section 5 builds `comparison_df` from `results` and `k_values`, iterating `["Logistic Regression", "SVM linear", "SVM rbf", "Random Forest"]`; its bar plot filters SelectKBest rows at `Dimensions == 200` (ints there, strings like `"101-102"` on PCA rows). A new experiment shows up in the summary table and plot by calling `evaluate` and being listed in that loop.
- Section 6 refits `lr_pipeline` on the full train set; it is the only place the test set is used.
- `X_train_scaled` and the full `pca` (sections 2.2 and 2.4) are exploration only (variance curve, 95 % threshold); no model consumes them, and before section 6 the test set only loses the duplicated columns found on train. Inside the CV pipelines, PCA(0.95) keeps 101–102 components depending on the fold, not the 103 found on the full train set.

Methodological invariants to keep: subject-grouped CV (never plain KFold, which leaks subjects and inflates scores), all fitting inside pipelines, test set used once for the final model.

## Gotchas

- Figures quoted in Markdown cells are typed by hand and drift from the outputs when cells are re-run; after any re-execution, re-check every number in the text against the outputs.
- `HAR.ipynb` was last re-run top-to-bottom in place (17m18s, all code cells executed, no errors); every hand-typed number in the Markdown was re-checked against that run's outputs, including the §3.9 nested table. `HAR_run.ipynb` is stale scratch (it predates the §3.9 cells) — safe to delete. Keep validating any later edit with a fresh top-to-bottom run, then re-check all hand-typed numbers.
- sklearn 1.9.1: `cross_validate(GridSearchCV(...), ..., groups=groups)` fails with "The 'groups' parameter should not be None" (groups are not routed into a meta-estimator's inner splitter without metadata routing), and `GridSearchCV(refit=False)` has no `predict` attribute. That is why §3.1 passes `groups` directly to `search.fit(X, y, groups=groups)` and §3.9 uses an explicit outer loop instead of nesting a search inside `cross_validate`.
- LogisticRegression (lbfgs, default `tol=1e-4`) scores move by about ±0.001 across machines, and even on one machine when the BLAS thread count changes (7/2947 test predictions flip); `tol=1e-8` makes it deterministic. SVC and RF (`random_state=42`) reproduce exactly.
- Tuned hyperparameters (§3.1 grids, picked on the same GroupKFold as the reported scores): LR `C=1` (already the default, so all LR numbers are unchanged by tuning), linear SVM `C=0.1`, RBF SVM `C=10, gamma=0.001` (γ=0.01 collapses accuracy to ~85%; γ="scale" ≈ 1/540), RF 500 trees with `max_features="log2"` (beats default "sqrt" by ~1 pt; `max_features=0.3` is clearly worse). The finding to keep in the text: tuning adjusts but does not overturn the comparison.
- Dataset quirks: features are pre-normalized to [-1, 1]; 21 columns are exact duplicates of others (13 `tGravityAccMag-*` = `tBodyAccMag-*`, and `*Mag-sma()` = `*Mag-mean()` for the 8 other magnitude signals), dropped in section 2, so text quoting the model input says 540 while text describing the raw dataset says 561; 84 `bandsEnergy` names are duplicated in the source and get `.1`/`.2` suffixes from pandas; train (21 subjects) and test (9 subjects) are subject-disjoint.

# NB Category Classifier

A hierarchical multinomial naive Bayes model that predicts the four-level product category
(`Level 1 ID` … `Level 4 ID`) from item descriptions. It replaces `NB-Category-Classifier` and
`NB-Category-Classifier-Minimized`. It keeps the first repo's workflow but has a single model and
none of the threshold, cost or calibration machinery.

## Workflow

| notebook | purpose | writes |
|---|---|---|
| `notebooks/01_pull_and_clean_data.ipynb` | pull (SQL or file) → blacklist → clean, verbosely | `data/<ts>/` |
| `notebooks/02_evaluation.ipynb` | cross-validation, or a grid search of cross-validations, with checkpoints | `outputs/evaluation_<ts>/` |
| `notebooks/03_train_to_deploy.ipynb` | train on all labelled data and save the model | `models/model_<ts>/` |
| `notebooks/04_batch_predict.ipynb` | predict uncategorized items with a saved model | `outputs/predictions_<ts>/` |
| `notebooks/05_feedback_learning_curve.ipynb` | gain from training on prediction-derived labels, as a curve over their ratio | `outputs/feedback_learning_curve_<ts>/` |
| `notebooks/06_feedback_matched_ratio.ipynb` | the same gain at the production ratio, repeated at increasing scale | `outputs/feedback_matched_ratio_<ts>/` |
| `notebooks/07_feedback_noise_dose.ipynb` | noise of the derived labels vs teacher size, and the cost of that noise | `outputs/feedback_noise_dose_<ts>/` |

Every notebook starts with a settings cell. All hyperparameters are written out in the notebook, so
none are hidden as defaults. Each notebook has `USE_MLFLOW = True/False` at the top.

Notebooks 02 and 03 can either use a folder written by notebook 01 (`PREPARED_DATA = "<ts>"` or
`"latest"`) or pull and clean the data themselves in one quiet cell. That cell reports only the row
counts and the token/bigram counts.

### Running on Databricks

1. Clone the repo as a Git folder. The notebooks find the package themselves (no `pip install` needed).
2. SQL queries live in `data/queries/*.sql`. `SQL_QUERY = "labeled_items"` runs
   `data/queries/labeled_items.sql` through `spark.sql`.
3. NLTK and its `words` corpus are installed on first use by `nbclassifier/bootstrap.py`.
4. Run the tests once per cluster/runtime: `%pip install pytest` then `!python -m pytest -q ../tests`.

## Folder structure

```
cleaning/
  cleaning_method_20261015/   one folder per named cleaning method:
    method.yaml               settings (list files, step parameters)
    clean_steps.py            step functions      } byte-for-byte copies of cleaning_repo
    term_lists.py             list readers        }
    cleaning_lists/           units, colors, color abbreviations, sizes, good tokens
data/
  queries/               labeled_items.sql, unlabeled_items.sql
  blacklists/            one-column lists of items to exclude
  <ts>/                  output of notebook 01
models/model_<ts>/       trained models (see below)
nbclassifier/            the Python package used by the notebooks
notebooks/               01-04
outputs/
  evaluation_<ts>/       notebook 02 checkpoints, results, exported OOF predictions, plots
  predictions_<ts>/      notebook 04 predictions + metadata
tests/                   pytest: sklearn equivalence, cleaning parity with cleaning_repo, folds
```

Package modules: `paths`, `bootstrap`, `config` (hyperparameters, grid, CV settings), `cleaning`, `data`,
`vectorizers`, `model`, `folds`, `predict`, `crossval` (CV and grid search), `evaluation` (buckets, plots),
`artifacts` (model save/load), `tracking` (MLflow), `feedback` (notebooks 05-07). The tracking module isn't named `mlflow.py`
because that name would shadow the real `mlflow` package.

## Cleaning

There is one cleaning method, `cleaning_method_20261015`, selected with
`CLEANING_METHOD = "cleaning_method_20261015"`. It is the 13-step procedure from `cleaning_repo`
(`01_clean_data.ipynb`):

1. uppercase
2. new cleaned field
3. repair words split at the field boundary
4. remove record markers
5. remove the first word unless it is an English word or good token (with safeguards)
6. colors → `<COLOR>`
7. standardize units
8. glue units to numbers
9. sizes → `<SIZE>`, then drop stray digits
10. remove punctuation
11. remove rare words
12. collapse repeated placeholders
13. tidy spaces

The code and term lists are unchanged copies. The parameters from that notebook's settings cell live in
`method.yaml`. `tests/test_cleaning.py` checks the output against the original code on a sample of
`sample_spend_data.txt`.

To create another method, copy the folder under a new name, then edit `method.yaml`, the lists or the
code.

**Step 11 vocabulary.** Words kept by step 11 are learned from the training data when notebooks 01-03
clean it. They are then frozen into the cleaner, so notebook 04 applies the training vocabulary
(cleaning_repo's `BUILD_VOCABULARY = False` path). Steps 3 and 5 count words in whatever data is being
cleaned, exactly as in cleaning_repo.

Optional extras, set in the notebook:

- `APPEND_FIELDS = {"Global Vendno": "VEND", "Prodline": "PLINE"}` appends `VEND_<value>` tokens to the
  cleaned description. A field can also be split into several granularities:
  `{"UNSPSC": {"prefix": "UNSPSC", "levels": [2, 4, 6, 8]}}` turns `46181504` into
  `UNSPSC2_46 UNSPSC4_4618 UNSPSC6_461815 UNSPSC8_46181504`. With `APPEND_NULLS_AS_NONE = True`, nulls
  become `VEND_None`; with `False` the marker is skipped for that row.
- `BLACKLIST_FILE` / `BLACKLIST_COL`: items removed before anything else happens. The file's first column
  holds the identifiers (a header line is expected); `BLACKLIST_COL` (default `ID`) is the data column they
  are matched against.

The cleaner (`cleaner.pkl`) is saved with every dataset and every model. It holds the method's code,
lists and settings, the English word list, the learned vocabulary and the append fields. A readable copy
of the method folder and `vocabulary.txt` are saved alongside it. The step code runs from that frozen
copy, so editing `cleaning/` later never changes how an existing model cleans.

**Tokenizing.** Cleaned text is whitespace-separated words such as `<SIZE>IN` and `<COLOR>`. The
`token_pattern` hyperparameter controls how the vectorizer splits it:

- `"whitespace"` (default) keeps each cleaned word as one token.
- `"sklearn"` uses scikit-learn's default, which splits `<SIZE>IN` into `size` + `in` (and `in` is then
  removed as a stop word).

## Model

`SparseMultinomialNB` reproduces sklearn's `MultinomialNB` exactly, but keeps the class × feature counts
sparse. sklearn stores them densely, which at 3,600 classes × 200,000 features is 5.8 GB. The rewrite is

    log(fc + α) − log_norm = log α − log_norm + log1p(fc/α)

The last term is zero wherever the count is zero, so it stays sparse. Counts don't depend on α or
`fit_prior`, so the grid search fits each vectorizer once per fold and re-uses it for every α /
`fit_prior` / `prediction_mode` combination.

`prediction_mode` controls how the four levels are combined. Only category paths seen in training can be
predicted.

| mode | path score | sums to 1? |
|---|---|---|
| `independent` | product of the four levels' own probabilities | no (raw product) |
| `conditional` | chain rule: P(L1)·P(L2∣L1)·P(L3∣L2)·P(L4∣L3), i.e. one NB per parent node | yes |
| `flattened` | one NB over level-4 categories; levels 1-3 follow from the only valid path | yes |

The `conditional` mode is computed exactly from the per-level counts. A per-parent `MultinomialNB` has
the same feature counts for its children, and its prior differs only by a constant within the sibling
group (`tests/test_model.py` checks this against sklearn models fitted per parent).

Output per level `d`:

- **`Ld Confidence`**: the summed score of all valid paths sharing the predicted path's first `d`
  levels. These are raw scores; nothing is calibrated.
- **`Ld Margin`**: the confidence minus the best competing prefix at that depth. It can be negative.

## Output format (ensemble input)

Out-of-fold predictions (`outputs/evaluation_<ts>/best/oof_predictions.csv`) and batch predictions
(`outputs/predictions_<ts>/predictions_<ts>.csv`) share one comma-separated layout:

```
ID, Vallen ID, [fold], [Actual Level 1-4 ID],
Predicted Level 1-4 ID, [Predicted Level 1-4 Name],
L1-L4 Confidence, L1-L4 Margin, n_tokens, n_known, oov_rate, FullDesc, CleanDesc
```

The OOF file has a JSON sidecar (`oof_predictions.json`) with `n_folds`, `fold_seed`, `fold_key_col` and
`fold_method`. These are the names the Ensemble repo already reads. Folds use the same ID-hash method as
before (`sha1(f"{seed}|{ID}") % n_folds`, defaults 5 folds, seed 0, key `ID`), so they line up with the
neural network's folds.

## Evaluation, checkpoints and resuming

- `GRID_SEARCH = False`: every hyperparameter must be a scalar or a one-element list, and one
  cross-validation is run.
- `GRID_SEARCH = True`: lists are expanded into every combination. Combinations that would give
  identical models are dropped (e.g. `norm` with a count vectorizer).
- `CV_TYPE = "oof"` (default, k-fold out-of-fold) or `"repeated_holdout"`.
- `SAMPLE_FRAC = 0.2` (optional) evaluates on about 20% of items, chosen by a hash of `FOLD_KEY_COL`, so
  the NB and NN repos sample the same items. Use it to explore a grid quickly, then confirm on all rows.

Folds are assigned on the full dataset, so every candidate is evaluated on the same rows. The
rare-path threshold is applied to each fold's **training** part only, as it is in deployment. Every
held-out row is therefore predicted: the OOF file covers all rows, and paths unseen in training count
as errors.

Each (candidate, fold) prediction file is written to disk as soon as it is done. The dataset itself is
also copied into the evaluation folder. If the session dies, set `RESUME_FROM = "evaluation_<ts>"` and
re-run: finished work is skipped, and nothing is re-pulled. New grid values added to the notebook are
appended to the run.

Bucket tables, the accuracy-and-item-count plots (with a dashed green line at the coverage where
cumulative accuracy reaches `TARGET_ACCURACY`, default 95%) and violin plots (spread across folds) are all
computed from the saved predictions. `BUCKET_WIDTH` can therefore be changed and the cells re-run
without re-fitting.

## Saved model folder

`vectorizer.pkl`, `model_Level_<d>_ID.pkl` (one per level), `path_index.pkl`, `cleaner.pkl`,
`cleaning_method_20261015/` (copy), `vocabulary.txt`, `hierarchy_lookup.csv`, `hyperparameters.json`, `cleaning.json`,
`training_data.json` (source, query, blacklist, row counts) and `manifest.json` (fingerprint, sizes,
library versions). Notebook 03 reloads the saved model and checks that it predicts identically.

## MLflow

With `USE_MLFLOW = True` the notebooks log the following:

| notebook | logs |
|---|---|
| 01 | source, cleaning method, row/n-gram counts, metadata, vocabulary |
| 02 | CV settings; one nested run per candidate (parameters, mean accuracy by level, per-fold accuracy as steps); results table; plots |
| 03 | hyperparameters, data summary, the whole model folder |
| 04 | model used, row counts, and accuracy when the data has labels |

`MLFLOW_EXPERIMENT = None` uses each notebook's own experiment. Set a workspace path to collect all
runs in one place.

## Feedback-loop experiments (notebooks 05-07)

Items predicted by notebook 04 and written back as categories return in the labelled pull. Training on
them can look like a large gain, but scoring on them is circular, and the model that predicted them was
trained on all organic labels, so it has seen any organic holdout. Notebooks 05-07 estimate the real
effect without validating those items by hand.

**Terms.** *Derived* rows are labelled rows whose ID is listed in `DERIVED_IDS_FILE`: a one-column
.txt/.csv of the IDs of items already predicted for, in the same format as the blacklist files. *Organic*
rows are the rest. The *pool* the fold teachers re-label is the derived rows. Optionally,
`PREDICTION_FILES` (notebook-04 outputs, which need `INCLUDE_TEXT = True`) adds the items that were
predicted but not accepted, and the production confidences for the diagnostics. Without an ID file, a
row whose ID is in those files counts as derived, or `DERIVED_FLAG_COL` can mark them. The *ratio* is
derived rows / organic rows, and the production ratio is the one in the data.

**Cross-fitting.** Every experiment splits the organic rows into ID-hash folds. For each test fold, a
teacher is trained only on the other folds, re-labels the pool and re-applies the acceptance rule
(`ACCEPT_LEVEL`, `ACCEPT_MIN_CONFIDENCE`; leave it at None when the pool is the derived rows only). Students are scored on the test fold, which no model in
that fold has seen. The recorded labels of the derived rows are used for diagnostics only.
Differences between models are paired bootstraps over the same test rows.

| notebook | question | how |
|---|---|---|
| 05 | how does the gain change with the ratio? | students over a grid of organic fractions x ratios; a smooth surface fitted to the gains and read at the production ratio and size; leave-one-out check of the extrapolation |
| 06 | what is the gain at the production mix? | the production ratio at several organic scales, each teacher trained on that scale's rows; trend across scales extrapolated to production size |
| 07 | how noisy are the derived labels, and what does the noise cost? | teacher error vs teacher size on clean folds, transferred to the pool by calibration and extrapolated; then organic rows posing as the pool with a dialled share of the teacher's errors, read at the estimated noise |

Notebook 07's simulated pool comes from the organic population, so it tests the ratio and the noise,
not the shift between organic items and the items that needed predicting. The diagnostics table in each
notebook reports how often the recorded derived labels match the production predictions. Below 1.0,
some were corrected after prediction, and the experiments understate the real labels' quality.

Results are checkpointed like notebook 02: the data is copied into the output folder, each finished cell
is written at once, and `RESUME_FROM = "feedback_<kind>_<ts>"` continues a run without re-pulling.
`FOLDS_TO_RUN = [0]` gives a quick first look.

## What was removed from the old repos

Thresholds and cascades, cost models, trade-off sweeps, scoring simulation, calibration and confidence
references, drift checks, refusal reasons, `per_level` mode, and `abbrevs_to_expand.csv`. The earlier
`aggressive`/`light` rule sets and the find/replace normalization file were replaced by
`cleaning_method_20261015`.

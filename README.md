# Leakage-Aware Clinical ML

**Reproducible evaluation for patient-level clinical machine learning: how a common
validation mistake turns a mediocre model into a "0.98 AUROC" model.**

Clinical datasets often contain many records per patient (for example, repeated vital-sign
readings in an ICU). If records are split into train and test folds *row by row*, the same
patient appears on both sides. The model learns to recognise patients instead of disease, and
the reported performance is inflated.

This repo demonstrates the problem and the fix on **synthetic** pediatric vital-signs data
(no real patient data is included).

## Results (seed 42, 120 patients × 15 records, 5-fold CV)

| Setting | AUROC | AUPRC |
|---|---:|---:|
| Random forest, **row-level split (leaky)** | **0.983** | 0.963 |
| Random forest, patient-level split | 0.713 | 0.489 |
| Logistic regression, patient-level split | 0.856 | 0.741 |
| Random forest, nested patient-level CV (tuned) | 0.744 | 0.580 |

**Takeaways**
1. The leaky split overstates AUROC by about 0.27. The model is memorising patient baselines.
2. Under honest, grouped validation, the simpler logistic regression *beats* the random forest.
   On the leaky split, the flexible model wins only because it memorises better.
3. Tuning hyperparameters must happen inside an inner grouped loop (nested CV), or the tuning
   itself leaks information from the test fold.

## What this repo gets right

- **Grouped splits:** `StratifiedGroupKFold` keeps every patient in exactly one fold.
- **Preprocessing inside the pipeline:** the scaler is fitted on each training fold only.
- **Imbalance-aware metrics:** reports AUPRC as well as AUROC (about 30% positives).
- **Nested CV:** an unbiased performance estimate after hyperparameter search.
- **Reproducibility:** fixed seeds and pinned dependency ranges. Tests check that data and metrics are deterministic.
- **Tests that assert the leakage exists:** `test_row_split_inflates_performance` fails if the
  demonstration stops demonstrating.

## Run it

```bash
pip install -r requirements.txt
PYTHONPATH=src python -m lacml.run     # prints the table, writes results/metrics.json
python -m pytest -q                   # 4 tests
```

## Structure

```
src/lacml/data.py      synthetic vitals generator (patient baselines + condition effect)
src/lacml/evaluate.py  row vs grouped vs nested cross-validation
src/lacml/run.py       reproduces the results table
tests/                 reproducibility and leakage tests
```

## Context

This mirrors validation issues I handle in my own research on pediatric ICU vital-signs data
(Erasmus+ SAFE-RH project) and metabolomic biomarker models
([IEEE Access 2025](https://doi.org/10.1109/ACCESS.2024.3521116)).

**Author:** Dr. Maria Aslam · [Google Scholar](https://scholar.google.com/citations?user=wl7jHeEAAAAJ) · MIT License

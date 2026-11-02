# 🅿️ Campus Parking Occupancy Prediction & Intelligent Parking Assistant

CSE513 Machine Learning — Project 32 (individual).

A machine-learning system that forecasts how full each parking zone will be,
recommends where to park, and gives administrators a dashboard of demand
patterns, zone clusters and faulty-sensor alerts.

> **Data.** Public Birmingham (UK) car-park occupancy log, 4 Oct – 19 Dec 2016
> (UCI Machine Learning Repository), used as a stand-in for campus parking
> zones: 30 car parks, ~35.7k sensor readings. The pipeline works unchanged on
> any CSV with `zone, capacity, occupancy, timestamp` columns, so a real
> campus log can be dropped in later.

## What it does

| Module | File | What happens |
|---|---|---|
| Data collection & occupancy estimation | `src/data_prep.py` | Removes duplicates, clips impossible values, snaps readings to a 30-min grid, flags odd zone-days with an Isolation Forest |
| EDA, heatmaps, zone-level analysis | `src/eda.py` | Weekday×time heatmaps, K-means / Ward clustering of zones |
| Feature engineering | `src/features.py` | Leakage-safe calendar, lag and zone-profile features |
| Models | `src/models.py` | Baselines, Ridge, RBF-SVR, Random Forest, HistGB, XGBoost, quantile models, status classifier |
| Training & evaluation | `src/train.py` | Time-series CV tuning, stacking, bootstrap CIs, Wilcoxon & McNemar tests |
| Explainability | `src/explain.py` | SHAP (TreeExplainer) + permutation importance |
| Error analysis | `src/error_analysis.py` | Errors by time, weekday, zone, occupancy level, sensor anomaly |
| Deployment rule fix | `src/redeploy.py` | Documents the post-hoc change of the model-selection rule (see below) |
| Batch scoring | `src/export_forecasts.py` | Deployed models score the test period for the app |
| App | `app/streamlit_app.py` | Find parking · Zone forecast · Admin dashboard · How it works |

Two forecasting tasks: **1-hour-ahead** (uses today's latest readings) and
**day-ahead** (uses only data up to the previous evening).
Results are in `reports/results.json`, `reports/shap_results.json` and
`reports/error_analysis.json`; figures in `reports/figures/`.

## Run it on your computer

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements-full.txt   # everything (training, SHAP, tests, app)
streamlit run app/streamlit_app.py     # the app works straight away
bash run_all.sh                        # optional: rebuild everything (~25-30 min)
```

`requirements.txt` holds only what the web app needs (used by Streamlit Community
Cloud); `requirements-full.txt` holds the full pipeline.

**About the model files.** The deployed Random Forest bundles are 100–170 MB, over
GitHub's file limit, so they are not in the repository. The app reads
`data/processed/app_forecasts_*.csv`, which `src/export_forecasts.py` produces by
running those models on the whole test period (batch scoring); the forecasts are
identical to live model output. `bash run_all.sh` rebuilds the models.

## Project layout

```
app/streamlit_app.py     web app
src/                     pipeline code (one file per module)
data/raw/                original sensor log
data/processed/          cleaned grid, anomalies, clusters, test predictions
models/                  trained models (.joblib) used by the app
reports/                 results JSON files and figures
tests/                   sanity tests (python -m pytest -q)
```

## Evaluation protocol (why the numbers can be trusted)

* Chronological split — train 4 Oct–20 Nov, validation 21 Nov–4 Dec, test 5–19 Dec. No shuffling.
* Every feature uses only data available before the forecast moment; zone profiles are learned inside the pipeline on training folds only.
* Hyper-parameters tuned with 4-fold `TimeSeriesSplit`; the deployed model is never picked on test.
* **Disclosed change:** the first run chose the deployed model by validation MAE alone, which picked
  RBF-SVR. On test SVR did clearly worse than the tree ensembles; the 12-day validation window held
  only one weekend, where SVR is weakest. The rule is now "lowest average of CV MAE and validation
  MAE" (Random Forest for both tasks). This rule was adopted after seeing test scores, so the
  deployed model's test score is a post-hoc figure; all models' test scores are reported.
* The test set is used once. MAE confidence intervals use a day-block bootstrap.

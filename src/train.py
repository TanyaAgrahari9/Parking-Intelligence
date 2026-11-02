"""Module 5 - Training, hyper-parameter optimisation, ensembling and evaluation.

Protocol (identical for both forecasting tasks):
  1. Chronological split: train (Oct 4 - Nov 20), validation (Nov 21 - Dec 4),
     test (Dec 5 - Dec 19).  The test set is touched exactly once, at the end.
  2. Hyper-parameters tuned on TRAIN with TimeSeriesSplit (4 expanding folds),
     metric = MAE.  Grid search for small spaces, randomised search otherwise.
  3. Base models re-fitted on train -> predictions on validation -> a
     non-negative linear stacker is learned on those validation predictions.
  4. Everything re-fitted on train+validation, scored on test.
  5. Statistical comparison: day-block bootstrap CIs + Wilcoxon signed-rank
     test on absolute errors; McNemar test for the status classifiers.

Run:  python -m src.train          (about 15-25 minutes on a laptop)
"""
import json
import os
import time

import joblib
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, chi2
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score, f1_score,
                             accuracy_score, confusion_matrix, classification_report)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit

from src import config as C
from src.features import build_day_ahead, build_short_term, split_by_date, rate_to_status, ProfileEncoder
from src.models import (MODEL_NAMES, SEARCH_SPACES, make_pipeline_for, baseline_predictions,
                        make_quantile_model, make_status_classifier, feature_columns)

N_ITER = {"Random Forest": 12, "HistGB": 20, "XGBoost": 25}
SVR_TUNE_ROWS = 6000          # RBF-SVR is O(n^2): tune on a chronological subsample
SMOKE = bool(os.environ.get("SMOKE"))   # SMOKE=1 -> tiny run to test the code path quickly
if SMOKE:
    N_ITER = {k: 2 for k in N_ITER}
    SVR_TUNE_ROWS = 1500


def metrics(y, p, cap=None):
    m = {"MAE": float(mean_absolute_error(y, p)),
         "RMSE": float(np.sqrt(mean_squared_error(y, p))),
         "R2": float(r2_score(y, p))}
    if cap is not None:
        m["MAE_vehicles"] = float(np.mean(np.abs(y - p) * cap))
    ys, ps = rate_to_status(y), rate_to_status(p)
    m["status_acc"] = float(accuracy_score(ys, ps))
    m["status_macroF1"] = float(f1_score(ys, ps, average="macro"))
    return m


def day_block_bootstrap(abs_err, dates, n=1000, seed=0):
    """95% CI of MAE resampling whole days (errors within a day are correlated)."""
    rng = np.random.default_rng(seed)
    days = np.unique(dates)
    by_day = {d: abs_err[dates == d] for d in days}
    stats = []
    for _ in range(n):
        pick = rng.choice(days, size=len(days), replace=True)
        stats.append(np.concatenate([by_day[d] for d in pick]).mean())
    return [float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))]


def mcnemar(y, a, b):
    """McNemar's test (with continuity correction) for two classifiers on the same samples."""
    ca, cb = (a == y), (b == y)
    n01 = int(np.sum(ca & ~cb)); n10 = int(np.sum(~ca & cb))
    stat = (abs(n01 - n10) - 1) ** 2 / max(n01 + n10, 1)
    return {"a_right_b_wrong": n01, "a_wrong_b_right": n10, "chi2": float(stat),
            "p_value": float(chi2.sf(stat, 1))}


def tune(name, task, X, y):
    kind, space = SEARCH_SPACES[name]
    pipe = make_pipeline_for(name, task)
    cv = TimeSeriesSplit(n_splits=4)
    if name == "SVR (RBF)":
        X, y = X.iloc[-SVR_TUNE_ROWS:], y[-SVR_TUNE_ROWS:]
    if kind == "grid":
        search = GridSearchCV(pipe, space, cv=cv, scoring="neg_mean_absolute_error", n_jobs=1)
    else:
        search = RandomizedSearchCV(pipe, space, n_iter=N_ITER[name], cv=cv, random_state=C.RANDOM_STATE,
                                    scoring="neg_mean_absolute_error", n_jobs=1)
    t0 = time.time()
    search.fit(X, y)
    cvres = pd.DataFrame(search.cv_results_)
    return {"best_params": {k: (v.item() if hasattr(v, "item") else v) for k, v in search.best_params_.items()},
            "cv_mae": float(-search.best_score_),
            "cv_mae_std": float(cvres.loc[search.best_index_, "std_test_score"]),
            "n_candidates": int(len(cvres)), "tune_seconds": round(time.time() - t0, 1)}


def run_task(task: str, grid: pd.DataFrame) -> dict:
    print(f"\n================  TASK: {task}  ================")
    df = build_short_term(grid) if task == "short" else build_day_ahead(grid)
    df = df.sort_values(["slot_time", "zone"]).reset_index(drop=True)
    tr, va, te = split_by_date(df)
    if SMOKE:
        tr, va, te = tr.iloc[-3000:], va.iloc[:1500], te.iloc[:1500]
    num, cat = feature_columns(task)
    cols = num + cat
    res = {"n_rows": {"train": len(tr), "val": len(va), "test": len(te)}, "features": cols}

    # ---------- 1. tuning on train (time-series CV) ----------
    tuning = {}
    for name in MODEL_NAMES:
        tuning[name] = tune(name, task, tr[cols], tr.rate.values)
        print(f"  tuned {name:14s} CV-MAE={tuning[name]['cv_mae']:.4f}  ({tuning[name]['tune_seconds']}s)")
    res["tuning"] = tuning

    # ---------- 2. fit on train, predict validation (model selection + stacker) ----------
    val_pred, val_metrics = {}, {}
    for name in MODEL_NAMES:
        m = make_pipeline_for(name, task, **tuning[name]["best_params"]).fit(tr[cols], tr.rate.values)
        val_pred[name] = np.clip(m.predict(va[cols]), 0, 1)
        val_metrics[name] = metrics(va.rate.values, val_pred[name], va.capacity.values)
    stacker = LinearRegression(positive=True).fit(np.column_stack([val_pred[n] for n in MODEL_NAMES]), va.rate.values)
    weights = dict(zip(MODEL_NAMES, map(float, stacker.coef_)))
    res["stack_weights"] = weights; res["stack_intercept"] = float(stacker.intercept_)
    res["val_metrics"] = val_metrics

    # ---------- 3. refit on train+val, evaluate once on test ----------
    trva = pd.concat([tr, va])
    pe = ProfileEncoder().fit(trva[cols], trva.rate.values)
    test_pred = {k: np.clip(v, 0, 1) for k, v in baseline_predictions(te, task, pe).items()}
    final_models = {}
    for name in MODEL_NAMES:
        t0 = time.time()
        m = make_pipeline_for(name, task, **tuning[name]["best_params"]).fit(trva[cols], trva.rate.values)
        tuning[name]["final_fit_seconds"] = round(time.time() - t0, 1)
        final_models[name] = m
        test_pred[name] = np.clip(m.predict(te[cols]), 0, 1)
    test_pred["Stacked ensemble"] = np.clip(
        stacker.predict(np.column_stack([test_pred[n] for n in MODEL_NAMES])), 0, 1)

    y, cap, dates = te.rate.values, te.capacity.values, te.date.values
    test_metrics = {}
    for name, p in test_pred.items():
        mm = metrics(y, p, cap)
        mm["MAE_CI95"] = day_block_bootstrap(np.abs(y - p), dates)
        test_metrics[name] = mm
    res["test_metrics"] = test_metrics

    learned = [n for n in test_pred if n in MODEL_NAMES or n == "Stacked ensemble"]
    best = min(learned, key=lambda n: test_metrics[n]["MAE"])
    best_baseline = min([n for n in test_pred if n not in learned], key=lambda n: test_metrics[n]["MAE"])
    # Model chosen for deployment (never by test score).
    # Rule: lowest average of time-series-CV MAE and validation MAE.  Validation alone is
    # only 12 days with a single weekend, which proved too small to rank models reliably
    # (see src/redeploy.py and the report).  The validation-only pick is kept for reference.
    deploy = min(MODEL_NAMES, key=lambda n: (tuning[n]["cv_mae"] + val_metrics[n]["MAE"]) / 2)
    res.update(best_on_test=best, best_baseline=best_baseline, deployed_model=deploy,
               deployed_model_validation_only=min(MODEL_NAMES, key=lambda n: val_metrics[n]["MAE"]))
    ae = lambda n: np.abs(y - test_pred[n])
    res["wilcoxon"] = {}
    for other in [best_baseline] + [n for n in MODEL_NAMES if n != deploy]:
        stat, p = wilcoxon(ae(deploy), ae(other))
        res["wilcoxon"][f"{deploy} vs {other}"] = {"p_value": float(p),
                                                   "mean_diff": float(ae(deploy).mean() - ae(other).mean())}
    print(pd.DataFrame(test_metrics).T[["MAE", "RMSE", "R2", "MAE_vehicles", "status_macroF1"]].round(4))

    # ---------- 4. availability status: regression-threshold vs direct classifier ----------
    clf = make_status_classifier(task).fit(trva[cols], rate_to_status(trva.rate.values))
    s_true = rate_to_status(y)
    s_reg = rate_to_status(test_pred[deploy])
    s_clf = clf.predict(te[cols])
    res["status"] = {
        "regression_threshold": {"acc": float(accuracy_score(s_true, s_reg)),
                                 "macroF1": float(f1_score(s_true, s_reg, average="macro")),
                                 "cm": confusion_matrix(s_true, s_reg).tolist(),
                                 "report": classification_report(s_true, s_reg, target_names=C.STATUS_LABELS,
                                                                 output_dict=True)},
        "direct_classifier": {"acc": float(accuracy_score(s_true, s_clf)),
                              "macroF1": float(f1_score(s_true, s_clf, average="macro")),
                              "cm": confusion_matrix(s_true, s_clf).tolist(),
                              "report": classification_report(s_true, s_clf, target_names=C.STATUS_LABELS,
                                                              output_dict=True)},
        "mcnemar_reg_vs_clf": mcnemar(s_true, s_reg, s_clf),
        "test_class_share": np.bincount(s_true, minlength=3).tolist(),
    }

    # ---------- 5. prediction intervals (10% / 90% quantiles) ----------
    lo = make_quantile_model(task, 0.1).fit(trva[cols], trva.rate.values)
    hi = make_quantile_model(task, 0.9).fit(trva[cols], trva.rate.values)
    plo, phi = np.clip(lo.predict(te[cols]), 0, 1), np.clip(hi.predict(te[cols]), 0, 1)
    res["interval"] = {"nominal": 0.8, "coverage": float(np.mean((y >= plo) & (y <= phi))),
                       "mean_width": float(np.mean(phi - plo))}

    # ---------- 6. save artifacts ----------
    out = te[["zone", "date", "slot", "slot_time", "capacity", "rate"]].copy()
    for name, p in test_pred.items():
        out[f"pred_{name}"] = p
    out["pred_lo"], out["pred_hi"] = plo, phi
    out.to_csv(C.PROCESSED_DIR / f"test_predictions_{task}.csv", index=False)
    joblib.dump({"model": final_models[deploy], "name": deploy, "features": cols,
                 "q_lo": lo, "q_hi": hi, "status_clf": clf, "task": task},
                compress=3, filename=C.MODELS_DIR / f"{task}_model.joblib")
    joblib.dump(final_models["XGBoost"], C.MODELS_DIR / f"{task}_xgboost.joblib", compress=3)
    return res


def main():
    C.ensure_dirs()
    grid = pd.read_csv(C.GRID_PARQUET, parse_dates=["slot_time", "date"])
    t0 = time.time()
    results = {"short": run_task("short", grid), "day_ahead": run_task("day_ahead", grid)}
    results["total_seconds"] = round(time.time() - t0, 1)
    if SMOKE:
        print("smoke run OK"); return
    C.RESULTS_JSON.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nSaved {C.RESULTS_JSON}  ({results['total_seconds']}s)")


if __name__ == "__main__":
    main()

"""Module 6 - Explainable AI: SHAP values + permutation importance.

Question answered: *why* does the model predict a given occupancy?

* SHAP (SHapley Additive exPlanations) splits every single prediction into
  additive contributions, one per feature:
      prediction = base value + sum(SHAP value of each feature)
  TreeExplainer computes these exactly for tree ensembles such as XGBoost.
* Permutation importance is a model-agnostic cross-check: shuffle one feature
  on the test set and measure how much the MAE gets worse.

The 28 one-hot "zone_..." columns are summed into a single "zone" feature so
the plots are readable.

Run:  python -m src.explain        (after python -m src.train)
"""
import json

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.inspection import permutation_importance

from src import config as C
from src.features import build_day_ahead, build_short_term, split_by_date
from src.models import feature_columns
from src.plot_style import plt, SERIES, INK2

N_PLOT = 3000          # rows shown in the beeswarm (all test rows are used for the numbers)
TASK_TITLE = {"short": "1-hour-ahead", "day_ahead": "day-ahead"}


def _test_frame(task, grid):
    df = build_short_term(grid) if task == "short" else build_day_ahead(grid)
    df = df.sort_values(["slot_time", "zone"]).reset_index(drop=True)
    return split_by_date(df)[2]


def _group_zone(values: np.ndarray, names: list[str]):
    """Sum SHAP columns of the one-hot zone dummies into one 'zone' column."""
    zone_idx = [i for i, n in enumerate(names) if n.startswith("zone_") and n != "zone_profile"]
    keep = [i for i in range(len(names)) if i not in zone_idx]
    out = np.column_stack([values[:, keep], values[:, zone_idx].sum(axis=1)])
    return out, [names[i] for i in keep] + ["zone"]


def explain_task(task: str, grid: pd.DataFrame) -> dict:
    te = _test_frame(task, grid)
    num, cat = feature_columns(task)
    X = te[num + cat]

    pipe = joblib.load(C.MODELS_DIR / f"{task}_xgboost.joblib")
    Xt = pipe[:-1].transform(X)                                  # profile encoder + column transformer
    names = list(pipe.named_steps["prep"].get_feature_names_out())
    model = pipe.named_steps["model"]

    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(Xt)
    base = float(np.ravel(explainer.expected_value)[0])

    # sanity check: base + sum(SHAP) must equal the model output (additivity)
    raw_pred = model.predict(Xt)
    additivity_err = float(np.max(np.abs(base + sv.sum(axis=1) - raw_pred)))

    gsv, gnames = _group_zone(sv, names)
    Xg_num = pd.DataFrame(Xt, columns=names)[[n for n in gnames if n != "zone"]]
    # colour for the grouped zone column = the zone's average occupancy profile
    Xg = Xg_num.assign(zone=Xg_num["zone_profile"].values)

    mean_abs = pd.Series(np.abs(gsv).mean(axis=0), index=gnames).sort_values(ascending=False)

    # ---- figure 1: beeswarm (direction + spread of each feature's effect) ----
    rng = np.random.default_rng(C.RANDOM_STATE)
    pick = rng.choice(len(Xg), size=min(N_PLOT, len(Xg)), replace=False)
    expl = shap.Explanation(values=gsv[pick], base_values=np.full(len(pick), base),
                            data=Xg.values[pick], feature_names=gnames)
    plt.figure()
    shap.plots.beeswarm(expl, max_display=12, show=False, plot_size=(6.0, 4.2))
    plt.title(f"SHAP summary - XGBoost, {TASK_TITLE[task]} task")
    plt.savefig(C.FIG_DIR / f"shap_beeswarm_{task}.png"); plt.close("all")

    # ---- figure 2: mean |SHAP| bar chart ----
    top = mean_abs.head(12)[::-1]
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.barh(top.index, top.values, color=SERIES[0])
    ax.set_xlabel("mean |SHAP value|  (average change in predicted occupancy rate)")
    ax.set_title(f"Global feature importance (SHAP) - {TASK_TITLE[task]}")
    ax.grid(axis="y", visible=False)
    fig.savefig(C.FIG_DIR / f"shap_bar_{task}.png"); plt.close(fig)

    # ---- figure 3: dependence plot of the top feature ----
    f0 = mean_abs.index[0]
    j = gnames.index(f0)
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    ax.scatter(Xg[f0].values[pick], gsv[pick, j], s=4, alpha=0.4, color=SERIES[0], edgecolors="none")
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xlabel(f0); ax.set_ylabel(f"SHAP value of {f0}")
    ax.set_title(f"How {f0} pushes the prediction - {TASK_TITLE[task]}")
    fig.savefig(C.FIG_DIR / f"shap_dependence_{task}.png"); plt.close(fig)

    # ---- figure 4: waterfall for one real test prediction (the busiest moment) ----
    i_case = int(np.argmax(te.rate.values))
    case = te.iloc[i_case]
    one = shap.Explanation(values=gsv[i_case], base_values=base,
                           data=Xg.values[i_case], feature_names=gnames)
    plt.figure()
    shap.plots.waterfall(one, max_display=10, show=False)
    plt.title(f"{case.zone}  {pd.Timestamp(case.slot_time):%a %d %b %H:%M}  "
              f"(actual {case.rate:.2f})", fontsize=8)
    plt.savefig(C.FIG_DIR / f"shap_waterfall_{task}.png"); plt.close("all")

    # ---- permutation importance on the DEPLOYED model (model-agnostic check) ----
    dep = joblib.load(C.MODELS_DIR / f"{task}_model.joblib")
    sub = X.sample(n=min(4000, len(X)), random_state=C.RANDOM_STATE)
    pi = permutation_importance(dep["model"], sub, te.loc[sub.index, "rate"].values,
                                scoring="neg_mean_absolute_error", n_repeats=5,
                                random_state=C.RANDOM_STATE, n_jobs=1)
    perm = pd.Series(pi.importances_mean, index=sub.columns).sort_values(ascending=False)

    return {"base_value": base, "additivity_max_abs_error": additivity_err,
            "n_explained_rows": int(len(Xg)),
            "mean_abs_shap": {k: float(v) for k, v in mean_abs.items()},
            "waterfall_case": {"zone": case.zone, "time": str(case.slot_time),
                               "actual": float(case.rate),
                               "xgb_prediction": float(np.clip(raw_pred[i_case], 0, 1)),
                               "contributions": {k: float(v) for k, v in
                                                 pd.Series(gsv[i_case], index=gnames)
                                                 .sort_values(key=np.abs, ascending=False).head(6).items()}},
            "permutation_importance_deployed": {"model": dep["name"],
                                                "MAE_increase": {k: float(v) for k, v in perm.items()}}}


def main():
    C.ensure_dirs()
    grid = pd.read_csv(C.GRID_PARQUET, parse_dates=["slot_time", "date"])
    out = {t: explain_task(t, grid) for t in ("short", "day_ahead")}
    path = C.ROOT / "reports" / "shap_results.json"
    path.write_text(json.dumps(out, indent=2))
    for t, r in out.items():
        print(f"\n[{t}] additivity error = {r['additivity_max_abs_error']:.2e}")
        print(pd.Series(r["mean_abs_shap"]).head(8).round(4).to_string())
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()

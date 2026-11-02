"""Module 7 - Error analysis of the deployed models on the held-out test period.

A single MAE number hides *where* a model fails.  This script slices the test
errors by time of day, weekday, zone, zone cluster, occupancy level and
sensor-anomaly flag, checks bias (over- vs under-prediction), interval
coverage, and lists the worst individual predictions.

Run:  python -m src.error_analysis     (after python -m src.train)
"""
import json

import numpy as np
import pandas as pd

from src import config as C
from src.features import rate_to_status
from src.plot_style import plt, SERIES, INK2, SEQ_CMAP

DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
SLOT_LABELS = [f"{8 + i // 2:02d}:{30 * (i % 2):02d}" for i in range(C.N_SLOTS)]
TASK_TITLE = {"short": "1-hour-ahead", "day_ahead": "day-ahead"}


def _slice(df, by, col="abs_err"):
    g = df.groupby(by)
    return pd.DataFrame({"MAE": g[col].mean(), "bias": g["err"].mean(), "n": g.size()})


def analyse(task: str, res: dict, anomalies: pd.DataFrame, clusters: pd.DataFrame) -> dict:
    df = pd.read_csv(C.PROCESSED_DIR / f"test_predictions_{task}.csv", parse_dates=["date", "slot_time"])
    model = res["deployed_model"]
    base = res["best_baseline"]
    df["pred"] = df[f"pred_{model}"]
    df["err"] = df["pred"] - df["rate"]               # >0 = over-predicts occupancy
    df["abs_err"] = df["err"].abs()
    df["abs_err_base"] = (df[f"pred_{base}"] - df["rate"]).abs()
    df["dow"] = df.date.dt.dayofweek
    df = df.merge(clusters, on="zone", how="left")
    an = anomalies[["zone", "date", "anomaly"]].copy()
    an["date"] = pd.to_datetime(an["date"])
    df = df.merge(an, on=["zone", "date"], how="left").fillna({"anomaly": False})
    df["level"] = pd.cut(df.rate, [-0.01, 0.3, 0.5, 0.7, 0.9, 1.01],
                         labels=["<30%", "30-50%", "50-70%", "70-90%", ">=90%"])

    by_slot = _slice(df, "slot")
    by_slot["MAE_baseline"] = df.groupby("slot").abs_err_base.mean()
    by_dow = _slice(df, "dow")
    by_zone = _slice(df, "zone").sort_values("MAE", ascending=False)
    by_zone["MAE_vehicles"] = df.assign(v=df.abs_err * df.capacity).groupby("zone").v.mean()
    by_level = _slice(df, "level")
    by_cluster = _slice(df, "cluster")
    by_anom = _slice(df, "anomaly")
    df["covered"] = (df.rate >= df.pred_lo) & (df.rate <= df.pred_hi)
    cov_slot = df.groupby("slot").covered.mean()

    worst = df.nlargest(10, "abs_err")[["zone", "slot_time", "rate", "pred", "abs_err", "anomaly"]]

    # ------------------------------ figures ------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    ax = axes[0]
    ax.plot(by_slot.index, by_slot.MAE, marker="o", ms=3, label=model)
    ax.plot(by_slot.index, by_slot.MAE_baseline, marker="o", ms=3, ls="--", label=base)
    ax.set_xticks(range(0, C.N_SLOTS, 3), SLOT_LABELS[::3]); ax.set_ylabel("MAE (rate)")
    ax.set_title("Error by time of day"); ax.legend()
    ax = axes[1]
    ax.bar([DOW[i] for i in by_dow.index], by_dow.MAE, color=SERIES[0])
    ax.set_title("Error by weekday"); ax.grid(axis="x", visible=False)
    fig.suptitle(f"{TASK_TITLE[task]} - {model}", fontsize=9, y=1.04)
    fig.savefig(C.FIG_DIR / f"err_time_{task}.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    z = by_zone.sort_values("MAE")
    ax.barh(z.index, z.MAE, color=[SERIES[int(c)] if not np.isnan(c) else INK2
                                   for c in df.groupby("zone").cluster.first().reindex(z.index)])
    ax.set_xlabel("MAE (occupancy rate)"); ax.tick_params(axis="y", labelsize=6)
    ax.set_title(f"Error by zone (colour = zone cluster) - {TASK_TITLE[task]}")
    ax.grid(axis="y", visible=False)
    fig.savefig(C.FIG_DIR / f"err_zone_{task}.png"); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    ax = axes[0]
    ax.scatter(df.rate, df.pred, s=3, alpha=0.25, color=SERIES[0], edgecolors="none")
    ax.plot([0, 1], [0, 1], color=INK2, lw=0.8)
    for t in (C.LIMITED_THRESHOLD, C.FULL_THRESHOLD):
        ax.axvline(t, color=SERIES[1], lw=0.6, ls=":"); ax.axhline(t, color=SERIES[1], lw=0.6, ls=":")
    ax.set_xlabel("actual rate"); ax.set_ylabel("predicted rate"); ax.set_title("Predicted vs actual")
    ax = axes[1]
    ax.hist(df.err, bins=60, color=SERIES[0])
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_xlabel("residual (pred - actual)"); ax.set_title(f"Residuals, mean = {df.err.mean():+.3f}")
    fig.savefig(C.FIG_DIR / f"err_scatter_{task}.png"); plt.close(fig)

    cm = np.array(res["status"]["regression_threshold"]["cm"])
    cmn = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    ax.imshow(cmn, cmap=SEQ_CMAP, vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cm[i, j]}\n({cmn[i, j]:.0%})", ha="center", va="center", fontsize=7,
                    color="white" if cmn[i, j] > 0.5 else "black")
    ax.set_xticks(range(3), C.STATUS_LABELS); ax.set_yticks(range(3), C.STATUS_LABELS)
    ax.set_xlabel("predicted"); ax.set_ylabel("actual"); ax.grid(False)
    ax.set_title(f"Status confusion - {TASK_TITLE[task]}")
    fig.savefig(C.FIG_DIR / f"err_confusion_{task}.png"); plt.close(fig)

    tm = res["test_metrics"]
    names = sorted(tm, key=lambda n: tm[n]["MAE"], reverse=True)
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    mae = [tm[n]["MAE"] for n in names]
    lo = [tm[n]["MAE"] - tm[n]["MAE_CI95"][0] for n in names]
    hi = [tm[n]["MAE_CI95"][1] - tm[n]["MAE"] for n in names]
    ax.barh(names, mae, xerr=[lo, hi], color=[SERIES[1] if n == model else SERIES[0] for n in names],
            error_kw={"lw": 0.8, "capsize": 2})
    ax.set_xlabel("test MAE (occupancy rate), 95% day-block bootstrap CI")
    ax.set_title(f"Model comparison - {TASK_TITLE[task]}"); ax.grid(axis="y", visible=False)
    fig.savefig(C.FIG_DIR / f"model_comparison_{task}.png"); plt.close(fig)

    def rec(d):
        return {str(k): {kk: (float(vv) if kk != "n" else int(vv)) for kk, vv in row.items()}
                for k, row in d.to_dict(orient="index").items()}

    s_true, s_pred = rate_to_status(df.rate), rate_to_status(df.pred)
    full_missed = int(((s_true == 2) & (s_pred == 0)).sum())
    return {
        "model": model, "baseline": base, "n_test": int(len(df)),
        "overall_bias": float(df.err.mean()),
        "by_slot": rec(by_slot), "by_dow": rec(by_dow), "by_zone": rec(by_zone),
        "by_level": rec(by_level), "by_cluster": rec(by_cluster), "by_anomaly": rec(by_anom),
        "interval_coverage_by_slot": {int(k): float(v) for k, v in cov_slot.items()},
        "full_predicted_as_available": full_missed,
        "share_of_rows_beating_baseline": float((df.abs_err < df.abs_err_base).mean()),
        "worst10": worst.assign(slot_time=worst.slot_time.astype(str)).to_dict(orient="records"),
    }


def main():
    C.ensure_dirs()
    res = json.loads(C.RESULTS_JSON.read_text())
    anomalies = pd.read_csv(C.PROCESSED_DIR / "zone_day_anomalies.csv")
    clusters = pd.read_csv(C.PROCESSED_DIR / "zone_clusters.csv")
    out = {t: analyse(t, res[t], anomalies, clusters) for t in ("short", "day_ahead")}
    path = C.ROOT / "reports" / "error_analysis.json"
    path.write_text(json.dumps(out, indent=2, default=str))
    for t, r in out.items():
        print(f"\n[{t}] model={r['model']}  bias={r['overall_bias']:+.4f}")
        print(pd.DataFrame(r["by_level"]).T.round(4).to_string())
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()

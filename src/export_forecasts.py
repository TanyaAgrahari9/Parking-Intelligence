"""Module 9 - Batch scoring for the app.

The deployed Random Forest bundles are 100-170 MB each (deep trees), too large
for GitHub (100 MB file limit) and slow to load on a free cloud host.  The app
only serves the held-out test period, so the deployed models score every
(zone, date, slot) of that period once here, and the app reads the small CSVs.
Predictions are identical to running the model live.

Run:  python -m src.export_forecasts      (after python -m src.train)
"""
import joblib
import numpy as np

from src import config as C
from src.features import build_day_ahead, build_short_term
import pandas as pd

KEEP = ["zone", "date", "slot", "slot_time", "capacity", "rate", "pred", "lo", "hi"]


def export(task, grid):
    df = (build_short_term(grid, keep_missing_target=True) if task == "short"
          else build_day_ahead(grid, keep_missing_target=True))
    df = df[df.date > C.VAL_END].copy()
    b = joblib.load(C.MODELS_DIR / f"{task}_model.joblib")
    X = df[b["features"]]
    df["pred"] = np.clip(b["model"].predict(X), 0, 1)
    df["lo"] = np.minimum(np.clip(b["q_lo"].predict(X), 0, 1), df.pred)
    df["hi"] = np.maximum(np.clip(b["q_hi"].predict(X), 0, 1), df.pred)
    out = df[KEEP].round({"rate": 5, "pred": 5, "lo": 5, "hi": 5})
    path = C.PROCESSED_DIR / f"app_forecasts_{task}.csv"
    out.to_csv(path, index=False)
    print(f"{task}: {len(out)} rows, model = {b['name']} -> {path.name}")
    return b["name"]


def main():
    grid = pd.read_csv(C.GRID_PARQUET, parse_dates=["slot_time", "date"])
    names = {t: export(t, grid) for t in ("short", "day_ahead")}
    (C.PROCESSED_DIR / "app_models.txt").write_text("\n".join(f"{k}={v}" for k, v in names.items()))


if __name__ == "__main__":
    main()

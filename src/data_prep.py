"""Module 1 - Parking Data Collection & Occupancy Estimation.

Turns the raw sensor log (irregular timestamps, duplicates, impossible values)
into a clean, regular grid:  one row per (zone, date, 30-minute slot).

Run:  python -m src.data_prep
"""
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from src import config as C


def load_raw(path=C.RAW_CSV) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["LastUpdated"])
    df = df.rename(columns={"SystemCodeNumber": "zone", "Capacity": "capacity",
                            "Occupancy": "occupancy", "LastUpdated": "timestamp"})
    return df


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Apply the data-quality rules and return (clean_df, audit_log)."""
    audit = {"raw_rows": int(len(df)), "raw_zones": int(df.zone.nunique())}

    # 1) exact duplicate readings
    n = len(df)
    df = df.drop_duplicates()
    audit["duplicates_removed"] = int(n - len(df))

    # 2) zones with too little history cannot be modelled reliably
    days = df.groupby("zone").timestamp.apply(lambda s: s.dt.date.nunique())
    dropped = sorted(days[days < C.MIN_DAYS_PER_ZONE].index)
    audit["zones_dropped"] = {z: int(days[z]) for z in dropped}
    df = df[~df.zone.isin(dropped)].copy()

    # 3) physically impossible values
    audit["negative_occupancy"] = int((df.occupancy < 0).sum())
    df["occupancy"] = df.occupancy.clip(lower=0)
    df["rate"] = df.occupancy / df.capacity
    audit["rate_above_capacity"] = int((df.rate > C.MAX_RATE_CLIP).sum())
    df["rate"] = df.rate.clip(upper=C.MAX_RATE_CLIP)

    # 4) snap irregular timestamps to the nearest 30-minute slot
    df["slot_time"] = df.timestamp.dt.round(f"{C.SLOT_MINUTES}min")
    hhmm = df.slot_time.dt.strftime("%H:%M")
    outside = (hhmm < C.FIRST_SLOT) | (hhmm > C.LAST_SLOT)
    audit["readings_outside_window"] = int(outside.sum())
    df = df[~outside]

    # 5) two readings in the same slot -> average them
    before = len(df)
    df = (df.groupby(["zone", "slot_time"], as_index=False)
            .agg(capacity=("capacity", "first"), rate=("rate", "mean")))
    audit["same_slot_merged"] = int(before - len(df))
    audit["clean_rows"] = int(len(df))
    audit["clean_zones"] = int(df.zone.nunique())
    return df, audit


def to_grid(df: pd.DataFrame) -> pd.DataFrame:
    """Re-index onto the full zone x date x slot grid (missing -> NaN)."""
    dates = pd.date_range(df.slot_time.dt.normalize().min(),
                          df.slot_time.dt.normalize().max(), freq="D")
    slot_offsets = pd.to_timedelta(np.arange(C.N_SLOTS) * C.SLOT_MINUTES + 8 * 60, unit="min")
    times = (dates.values[:, None] + slot_offsets.values[None, :]).ravel()
    zones = sorted(df.zone.unique())
    idx = pd.MultiIndex.from_product([zones, pd.DatetimeIndex(times)], names=["zone", "slot_time"])
    grid = df.set_index(["zone", "slot_time"])[["rate"]].reindex(idx).reset_index()
    cap = df.groupby("zone").capacity.first()
    grid["capacity"] = grid.zone.map(cap).astype(int)
    grid["date"] = grid.slot_time.dt.normalize()
    grid["slot"] = ((grid.slot_time - grid.date).dt.total_seconds() // 60 - 8 * 60) // C.SLOT_MINUTES
    grid["slot"] = grid.slot.astype(int)
    grid["dow"] = grid.slot_time.dt.dayofweek
    return grid


def flag_anomalies(grid: pd.DataFrame) -> pd.DataFrame:
    """Flag suspicious zone-days with an Isolation Forest.

    Each zone-day is summarised by (mean, std, max, number of identical
    consecutive readings, deviation from that zone's typical weekday/weekend
    profile).  A stuck
    or broken sensor looks very different from a normal day on these features.
    Flagged days are reported to the admin, not deleted, because a genuinely
    unusual day (an event) is valuable information.
    """
    g = grid.dropna(subset=["rate"]).copy()
    g["weekend"] = (g.dow >= 5).astype(int)   # weekends have their own normal profile
    prof = g.groupby(["zone", "weekend", "slot"]).rate.median().rename("profile")
    g = g.join(prof, on=["zone", "weekend", "slot"])
    g["absdev"] = (g.rate - g.profile).abs()

    def stuck(s):
        return int((s.diff().abs() < 1e-9).sum())

    feats = g.groupby(["zone", "date"]).agg(
        mean=("rate", "mean"), std=("rate", "std"), max=("rate", "max"),
        n=("rate", "size"), stuck=("rate", stuck), dev=("absdev", "mean")).fillna(0)
    iso = IsolationForest(n_estimators=300, contamination=0.02, random_state=C.RANDOM_STATE)
    X = feats[["mean", "std", "max", "stuck", "dev"]].values
    feats["anomaly"] = iso.fit_predict(X) == -1
    feats["anomaly_score"] = -iso.score_samples(X)
    return feats.reset_index()


def main():
    C.ensure_dirs()
    raw = load_raw()
    df, audit = clean(raw)
    grid = to_grid(df)
    audit["grid_rows"] = int(len(grid))
    audit["grid_missing_pct"] = round(float(grid.rate.isna().mean() * 100), 2)
    anomalies = flag_anomalies(grid)
    audit["anomalous_zone_days"] = int(anomalies.anomaly.sum())
    grid.to_csv(C.GRID_PARQUET, index=False)
    anomalies.to_csv(C.PROCESSED_DIR / "zone_day_anomalies.csv", index=False)
    (C.PROCESSED_DIR / "data_audit.json").write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()

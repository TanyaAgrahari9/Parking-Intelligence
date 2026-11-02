"""Module 2 - Feature Engineering (leakage-safe).

Two forecasting tasks are built from the clean grid:

* short-term ("1 hour ahead"): at origin slot t on day d, predict the
  occupancy rate at slot t+2 of the same day.  Uses the latest readings.
* day-ahead ("plan tomorrow"): predict the rate at every slot of day d using
  only information available by the end of day d-1.

Rule that prevents leakage: every feature for a target at time T is computed
only from readings strictly before the forecast origin.  Statistics learned
from data (zone profiles) are learned inside the sklearn Pipeline, i.e. on the
training fold only - see ProfileEncoder.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src import config as C

CALENDAR_FEATURES = ["slot", "hour", "dow", "is_weekend", "days_to_christmas"]
DAY_LAG_FEATURES = ["lag_1d", "lag_7d", "roll7_same_slot", "prevday_mean"]
SHORT_LAG_FEATURES = ["rate_t", "rate_t_1", "rate_t_2", "delta_t", "today_mean_so_far"]


def _add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    df["hour"] = 8 + df["slot"] * C.SLOT_MINUTES / 60
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    # Seasonal shopping effect in the city-centre car parks (Oct-Dec data)
    df["days_to_christmas"] = (pd.Timestamp("2016-12-25") - df["date"]).dt.days
    return df


def _day_lags(grid: pd.DataFrame) -> pd.DataFrame:
    """Same-slot values from previous calendar days for every (zone, date, slot)."""
    full_dates = pd.date_range(grid.date.min(), grid.date.max(), freq="D")
    parts = []
    for z, gz in grid.groupby("zone"):
        # rows = dates, columns = slots
        w = gz.pivot_table(index="date", columns="slot", values="rate", dropna=False).reindex(full_dates)
        feats = {
            "lag_1d": w.shift(1),                                      # yesterday, same slot
            "lag_7d": w.shift(7),                                      # same weekday last week
            "roll7_same_slot": w.shift(1).rolling(7, min_periods=3).mean(),
        }
        long = pd.concat({k: v.stack(future_stack=True) for k, v in feats.items()}, axis=1)
        long.index.names = ["date", "slot"]
        long = long.reset_index()
        prev_mean = w.shift(1).mean(axis=1)                            # yesterday's average level
        long["prevday_mean"] = long["date"].map(prev_mean)
        long["zone"] = z
        parts.append(long)
    return pd.concat(parts, ignore_index=True)


def build_day_ahead(grid: pd.DataFrame, keep_missing_target: bool = False) -> pd.DataFrame:
    """One row per observed (zone, date, slot) target, features from d-1 and earlier.
    keep_missing_target=True keeps unobserved targets too (used by the app to forecast)."""
    lags = _day_lags(grid)
    df = grid.merge(lags, on=["zone", "date", "slot"], how="left")
    if not keep_missing_target:
        df = df.dropna(subset=["rate"])
    df = df.copy()
    df = _add_calendar(df)
    df["origin_time"] = df["date"] - pd.Timedelta(days=1)
    return df.reset_index(drop=True)


def build_short_term(grid: pd.DataFrame, horizon: int = C.SHORT_HORIZON_SLOTS,
                     keep_missing_target: bool = False) -> pd.DataFrame:
    """Origin slot t -> target slot t+horizon on the same day.
    keep_missing_target=True keeps origins whose target is unobserved (used by the app)."""
    g = grid.sort_values(["zone", "date", "slot"]).copy()
    by_day = g.groupby(["zone", "date"]).rate
    g["rate_t_1"] = by_day.shift(1)
    g["rate_t_2"] = by_day.shift(2)
    g["today_mean_so_far"] = by_day.transform(lambda s: s.expanding().mean())
    g["target"] = by_day.shift(-horizon)
    g = g.rename(columns={"rate": "rate_t"})
    g["delta_t"] = g["rate_t"] - g["rate_t_1"]
    # origin must be observed and target must exist (same day)
    g = g.dropna(subset=["rate_t"] if keep_missing_target else ["rate_t", "target"]).copy()
    if keep_missing_target:                               # target slot must still exist that day
        g = g[g["slot"] + horizon < C.N_SLOTS]
    g["origin_slot"] = g["slot"]
    g["slot"] = g["slot"] + horizon                     # calendar features describe the TARGET time
    lags = _day_lags(grid)
    g = g.merge(lags, on=["zone", "date", "slot"], how="left")
    g = _add_calendar(g)
    g = g.rename(columns={"target": "rate"})
    return g.reset_index(drop=True)


def split_by_date(df: pd.DataFrame):
    train = df[df.date <= C.TRAIN_END]
    val = df[(df.date > C.TRAIN_END) & (df.date <= C.VAL_END)]
    test = df[df.date > C.VAL_END]
    return train, val, test


class ProfileEncoder(BaseEstimator, TransformerMixin):
    """Learns typical occupancy per (zone, weekday/weekend, slot) on TRAINING data
    and adds it as a feature ("what does this zone usually look like now?").

    Because it is a fitted transformer inside the Pipeline, cross-validation
    re-learns it on each training fold -> no target leakage.
    Unseen combinations fall back to the zone mean, then the global mean.
    """

    def __init__(self, smoothing: float = 5.0):
        self.smoothing = smoothing

    def fit(self, X, y):
        d = X[["zone", "is_weekend", "slot"]].copy()
        d["y"] = np.asarray(y)
        self.global_ = float(d.y.mean())
        self.zone_ = d.groupby("zone").y.mean()
        grp = d.groupby(["zone", "is_weekend", "slot"]).y.agg(["mean", "count"])
        zone_prior = grp.index.get_level_values(0).map(self.zone_).values
        # shrink small groups towards the zone mean (empirical-Bayes smoothing)
        self.table_ = ((grp["mean"] * grp["count"] + zone_prior * self.smoothing)
                       / (grp["count"] + self.smoothing))
        return self

    def transform(self, X):
        X = X.copy()
        keys = pd.MultiIndex.from_frame(X[["zone", "is_weekend", "slot"]])
        prof = pd.Series(self.table_.reindex(keys).values, index=X.index)
        prof = prof.fillna(X["zone"].map(self.zone_)).fillna(self.global_)
        X["zone_profile"] = prof.values
        return X

    def get_feature_names_out(self, input_features=None):
        return np.array(list(input_features) + ["zone_profile"])


def rate_to_status(rate) -> np.ndarray:
    rate = np.asarray(rate)
    return np.where(rate >= C.FULL_THRESHOLD, 2, np.where(rate >= C.LIMITED_THRESHOLD, 1, 0))

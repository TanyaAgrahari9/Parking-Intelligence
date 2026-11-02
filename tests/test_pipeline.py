"""Quick checks that protect the most important guarantees of the project.

Run:  python -m pytest -q
"""
import numpy as np
import pandas as pd
import pytest

from src import config as C
from src.features import build_day_ahead, build_short_term, split_by_date, rate_to_status, ProfileEncoder


@pytest.fixture(scope="module")
def grid():
    return pd.read_csv(C.GRID_PARQUET, parse_dates=["slot_time", "date"])


def test_split_is_chronological(grid):
    tr, va, te = split_by_date(build_day_ahead(grid))
    assert tr.date.max() < va.date.min() and va.date.max() < te.date.min()


def test_day_ahead_lag_is_yesterday(grid):
    """lag_1d must equal the same slot on the previous calendar day (no peeking at today)."""
    df = build_day_ahead(grid)
    look = grid.set_index(["zone", "date", "slot"]).rate
    row = df.dropna(subset=["lag_1d"]).iloc[500]
    assert np.isclose(row.lag_1d, look[(row.zone, row.date - pd.Timedelta(days=1), row.slot)])


def test_short_term_uses_only_past(grid):
    """The latest reading used as a feature is 2 slots (1 hour) before the target."""
    df = build_short_term(grid)
    look = grid.set_index(["zone", "date", "slot"]).rate
    row = df.iloc[1000]
    assert np.isclose(row.rate_t, look[(row.zone, row.date, row.slot - C.SHORT_HORIZON_SLOTS)])
    assert np.isclose(row.rate, look[(row.zone, row.date, row.slot)])


def test_profile_encoder_falls_back_for_unseen_zone():
    X = pd.DataFrame({"zone": ["A", "A", "B"], "is_weekend": [0, 0, 1], "slot": [1, 1, 2]})
    pe = ProfileEncoder().fit(X, [0.2, 0.4, 0.9])
    out = pe.transform(pd.DataFrame({"zone": ["Z"], "is_weekend": [0], "slot": [1]}))
    assert np.isclose(out.zone_profile.iloc[0], 0.5)          # global mean


def test_status_thresholds():
    assert list(rate_to_status([0.1, 0.7, 0.89, 0.9, 1.0])) == [0, 1, 1, 2, 2]

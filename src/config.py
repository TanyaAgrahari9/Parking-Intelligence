"""Central configuration for the Parking Intelligence project.

Every script imports its paths and constants from here, so changing a value
(e.g. the train/test split dates) changes it everywhere at once.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "data" / "raw" / "birmingham_parking.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
GRID_PARQUET = PROCESSED_DIR / "occupancy_grid.csv"
MODELS_DIR = ROOT / "models"
FIG_DIR = ROOT / "reports" / "figures"
RESULTS_JSON = ROOT / "reports" / "results.json"

RANDOM_STATE = 42

# ---- Time grid -------------------------------------------------------------
# Sensors report roughly every 30 minutes between ~08:00 and ~16:30.
# We snap every reading to the nearest 30-minute slot.
SLOT_MINUTES = 30
FIRST_SLOT = "08:00"
LAST_SLOT = "16:30"
N_SLOTS = 18                      # 08:00, 08:30, ..., 16:30

# ---- Data quality rules ----------------------------------------------------
MIN_DAYS_PER_ZONE = 30            # zones with fewer observed days are dropped
MAX_RATE_CLIP = 1.0               # occupancy rate above capacity is clipped

# ---- Chronological split (no shuffling -> no look-ahead leakage) ----------
TRAIN_END = "2016-11-20"          # train:      2016-10-04 .. 2016-11-20
VAL_END = "2016-12-04"            # validation: 2016-11-21 .. 2016-12-04
                                  # test:       2016-12-05 .. 2016-12-19

# ---- Forecast horizons -----------------------------------------------------
SHORT_HORIZON_SLOTS = 2           # 2 slots = 1 hour ahead (same day)

# ---- Availability classes used by the recommender -------------------------
LIMITED_THRESHOLD = 0.70          # rate >= 0.70  -> "Limited"
FULL_THRESHOLD = 0.90             # rate >= 0.90  -> "Full"
STATUS_LABELS = ["Available", "Limited", "Full"]


def ensure_dirs():
    for d in (PROCESSED_DIR, MODELS_DIR, FIG_DIR):
        d.mkdir(parents=True, exist_ok=True)

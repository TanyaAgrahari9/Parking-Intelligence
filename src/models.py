"""Module 4 - Model zoo: baselines, sklearn Pipelines and search spaces.

Every learnable model is a single sklearn Pipeline:

    ProfileEncoder  ->  ColumnTransformer (impute / scale / one-hot)  ->  estimator

so the exact same object is used for cross-validation, final training,
saving to disk and serving predictions in the app (no training/serving skew).
"""
import numpy as np
from scipy.stats import loguniform, randint, uniform
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

from src import config as C
from src.features import CALENDAR_FEATURES, DAY_LAG_FEATURES, SHORT_LAG_FEATURES, ProfileEncoder


def feature_columns(task: str):
    num = CALENDAR_FEATURES + DAY_LAG_FEATURES + ["capacity"]
    if task == "short":
        num = num + SHORT_LAG_FEATURES
    return num, ["zone"]


def _prep(task: str, scale: bool):
    num, cat = feature_columns(task)
    num = num + ["zone_profile"]          # produced by ProfileEncoder
    if scale:   # linear / kernel models need imputation, missing flags and scaling
        num_pipe = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler())
    else:       # tree models handle NaN natively and are scale-invariant
        num_pipe = "passthrough"
    ct = ColumnTransformer([
        ("num", num_pipe, num),
        ("zone", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),
    ], verbose_feature_names_out=False)
    return ct


def make_pipeline_for(name: str, task: str, **params) -> Pipeline:
    rs = C.RANDOM_STATE
    est = {
        "Ridge": lambda: Ridge(alpha=1.0),
        "SVR (RBF)": lambda: SVR(kernel="rbf", C=1.0, epsilon=0.02, gamma="scale"),
        "Random Forest": lambda: RandomForestRegressor(n_estimators=300, min_samples_leaf=3,
                                                       max_features=0.5, n_jobs=-1, random_state=rs),
        "HistGB": lambda: HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, random_state=rs),
        "XGBoost": lambda: XGBRegressor(n_estimators=600, learning_rate=0.05, max_depth=6, subsample=0.8,
                                        colsample_bytree=0.8, tree_method="hist", n_jobs=2, random_state=rs),
    }[name]()
    scale = name in ("Ridge", "SVR (RBF)")
    pipe = Pipeline([("profile", ProfileEncoder()), ("prep", _prep(task, scale)), ("model", est)])
    if params:
        pipe.set_params(**params)
    return pipe


# Hyper-parameter search spaces (keys are Pipeline parameter names)
SEARCH_SPACES = {
    "Ridge": ("grid", {"model__alpha": [0.01, 0.1, 1, 10, 100]}),
    "SVR (RBF)": ("grid", {"model__C": [0.3, 1, 3], "model__epsilon": [0.01, 0.03],
                           "model__gamma": ["scale", 0.02]}),
    "Random Forest": ("random", {"model__n_estimators": randint(200, 500),
                                 "model__max_depth": [None, 12, 20],
                                 "model__min_samples_leaf": randint(1, 10),
                                 "model__max_features": uniform(0.3, 0.6)}),
    "HistGB": ("random", {"model__learning_rate": loguniform(0.02, 0.2),
                          "model__max_leaf_nodes": randint(15, 63),
                          "model__min_samples_leaf": randint(10, 60),
                          "model__l2_regularization": loguniform(1e-3, 10),
                          "model__max_iter": randint(200, 700)}),
    "XGBoost": ("random", {"model__n_estimators": randint(300, 900),
                           "model__learning_rate": loguniform(0.02, 0.15),
                           "model__max_depth": randint(3, 9),
                           "model__min_child_weight": loguniform(1, 20),
                           "model__subsample": uniform(0.6, 0.4),
                           "model__colsample_bytree": uniform(0.5, 0.5),
                           "model__reg_lambda": loguniform(0.1, 20)}),
}

MODEL_NAMES = list(SEARCH_SPACES)


def make_quantile_model(task: str, alpha: float, **hgb_params) -> Pipeline:
    """Gradient boosting with pinball loss -> lower/upper prediction bounds."""
    est = HistGradientBoostingRegressor(loss="quantile", quantile=alpha, max_iter=400,
                                        learning_rate=0.05, random_state=C.RANDOM_STATE, **hgb_params)
    return Pipeline([("profile", ProfileEncoder()), ("prep", _prep(task, False)), ("model", est)])


def make_status_classifier(task: str) -> Pipeline:
    """Direct 3-class classifier (Available / Limited / Full).
    class_weight='balanced' counters the imbalance (most readings are 'Available')."""
    est = RandomForestClassifier(n_estimators=400, min_samples_leaf=3, max_features=0.5,
                                 class_weight="balanced", n_jobs=-1, random_state=C.RANDOM_STATE)
    return Pipeline([("profile", ProfileEncoder()), ("prep", _prep(task, False)), ("model", est)])


# ---------------------------------------------------------------------------
# Non-learned baselines.  A model is only useful if it beats these.
# ---------------------------------------------------------------------------
def baseline_predictions(df, task: str, profile_encoder: ProfileEncoder):
    prof = profile_encoder.transform(df)["zone_profile"].values
    out = {}
    if task == "short":
        out["Persistence (last value)"] = df["rate_t"].values
    else:
        out["Persistence (yesterday)"] = np.where(df["lag_1d"].notna(), df["lag_1d"], prof)
    out["Seasonal naive (last week)"] = np.where(df["lag_7d"].notna(), df["lag_7d"], prof)
    out["Historical profile"] = prof
    return out

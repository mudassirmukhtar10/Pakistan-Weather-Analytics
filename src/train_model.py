"""
Pakistan Weather Analytics — Temperature Forecasting Model
==========================================================
Trains three models per city:
  1. Naive baseline (persistence: last known value)
  2. Ridge regression
  3. GradientBoostingRegressor (best in most cases)

Usage:
    python src/train_model.py

Outputs:
    models/temp_forecaster.joblib   — best model pipeline + metadata
    models/metrics.json             — MAE / RMSE / R² for all models
"""

import os
import sys
import json
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

import joblib
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths  (resolve relative to this file so script works from any cwd)
# ---------------------------------------------------------------------------
SRC_DIR    = Path(__file__).parent
ROOT_DIR   = SRC_DIR.parent
DATA_DIR   = ROOT_DIR / "Data"
MODELS_DIR = ROOT_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

MODEL_PATH   = MODELS_DIR / "temp_forecaster.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"

# ---------------------------------------------------------------------------
# DB connection — reuse exact URI from app.py / analysis.py
# ---------------------------------------------------------------------------
DB_URI = "mysql+pymysql://root:@localhost:3307/weather_db"

CITY_FILES = {
    "Islamabad": DATA_DIR / "islamabad.csv",
    "Karachi":   DATA_DIR / "karachi.csv",
    "Lahore":    DATA_DIR / "lahore.csv",
    "Peshawer":  DATA_DIR / "peshawer.csv",
}

TARGET_COL = "temperature_2m"

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _parse_city_csv(path: Path, city: str) -> pd.DataFrame:
    """
    Parse an Open-Meteo style CSV that has two metadata rows before headers.
    Layout (0-indexed rows):
        0  -> station metadata labels  (latitude, longitude, ...)
        1  -> station metadata values
        2  -> column headers           (time, temperature_2m, ...)
        3+ -> hourly data
    Column 5 is always NaN in data rows (timezone_abbreviation spillover).
    """
    df = pd.read_csv(
        path,
        skiprows=3,
        header=None,
        on_bad_lines="skip",
        names=["time", TARGET_COL, "wind_speed_10m",
               "relative_humidity_2m", "precipitation", "_drop"],
    )
    df.drop(columns=["_drop"], inplace=True, errors="ignore")
    df["city"] = city
    # Keep only pure hourly rows (containing 'T' in datetime string, e.g. 2025-01-01T00:00)
    df = df[df["time"].astype(str).str.contains("T", na=False)].copy()
    df["time"] = pd.to_datetime(df["time"], format="mixed")
    for col in [TARGET_COL, "wind_speed_10m", "relative_humidity_2m", "precipitation"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_data() -> pd.DataFrame:
    """
    Try MySQL first (port 3307); fall back to individual city CSVs.
    Returns a clean DataFrame with columns:
        time, temperature_2m, wind_speed_10m, relative_humidity_2m,
        precipitation, city
    """
    # --- MySQL attempt ---
    try:
        engine = create_engine(DB_URI, connect_args={"connect_timeout": 5})
        raw = pd.read_sql("SELECT * FROM weather_data", con=engine)
        # The DB schema uses garbled column names (metadata headers mixed in).
        # Detect by checking if the actual time column exists.
        if "time" not in raw.columns:
            raise ValueError("DB schema lacks 'time' column — using CSV fallback.")
        raw["time"] = pd.to_datetime(raw["time"], errors="coerce", format="mixed")
        raw = raw.dropna(subset=["time", TARGET_COL])
        for col in [TARGET_COL, "wind_speed_10m", "relative_humidity_2m", "precipitation"]:
            if col in raw.columns:
                raw[col] = pd.to_numeric(raw[col], errors="coerce")
        log.info("Loaded %d rows from MySQL.", len(raw))
        return raw
    except Exception as exc:
        log.warning("MySQL load failed (%s). Falling back to CSVs.", exc)

    # --- CSV fallback ---
    frames = []
    for city, path in CITY_FILES.items():
        if not path.exists():
            log.warning("CSV not found: %s", path)
            continue
        frames.append(_parse_city_csv(path, city))
        log.info("Loaded %s from CSV (%d rows).", city, len(frames[-1]))
    if not frames:
        raise FileNotFoundError("No data source available (MySQL failed and no CSVs found).")
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

CITY_ORDER = ["Islamabad", "Karachi", "Lahore", "Peshawer"]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build lag / rolling / calendar features.
    IMPORTANT: all lag/rolling features strictly use past data only (no leakage).
    The DataFrame must be sorted by city + time before calling this.
    """
    df = df.sort_values(["city", "time"]).reset_index(drop=True)

    # --- Calendar features ---
    df["hour"]        = df["time"].dt.hour
    df["month"]       = df["time"].dt.month
    df["day_of_year"] = df["time"].dt.dayofyear
    df["day_of_week"] = df["time"].dt.dayofweek

    # Cyclical encoding (avoids discontinuity at hour 23->0, Dec->Jan, etc.)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["doy_sin"]   = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["doy_cos"]   = np.cos(2 * np.pi * df["day_of_year"] / 365)
    df["dow_sin"]   = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"]   = np.cos(2 * np.pi * df["day_of_week"] / 7)

    # --- Lag features (per-city groupby keeps lags from bleeding across cities) ---
    grp = df.groupby("city")[TARGET_COL]
    for lag in [1, 2, 3, 6, 12, 24, 48, 168]:   # hours: 1h,2h,3h,6h,12h,1d,2d,7d
        df[f"temp_lag_{lag}h"] = grp.shift(lag)

    # --- Rolling statistics (past only: shift(1) before rolling) ---
    for window in [6, 24, 168]:                   # 6h, 1-day, 7-day
        shifted = df.groupby("city")[TARGET_COL].shift(1)
        df[f"temp_roll_mean_{window}h"] = shifted.transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )
        df[f"temp_roll_std_{window}h"] = shifted.transform(
            lambda x: x.rolling(window, min_periods=1).std().fillna(0)
        )

    # --- Other weather columns ---
    for col in ["wind_speed_10m", "relative_humidity_2m", "precipitation"]:
        if col in df.columns:
            lag1 = df.groupby("city")[col].shift(1)
            df[f"{col}_lag1"] = lag1
            df[f"{col}_roll24_mean"] = lag1.transform(
                lambda x: x.rolling(24, min_periods=1).mean()
            )

    # --- City encoding (label + one-hot) ---
    le = LabelEncoder()
    le.fit(CITY_ORDER)
    safe_city = df["city"].where(df["city"].isin(CITY_ORDER), other="Islamabad")
    df["city_enc"] = le.transform(safe_city)
    for c in CITY_ORDER:
        df[f"city_{c}"] = (df["city"] == c).astype(int)

    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    """Return ordered list of feature columns (excluding target & metadata)."""
    exclude = {"time", "city", TARGET_COL,
               "hour", "month", "day_of_year", "day_of_week"}  # keep only cyclical versions
    return [c for c in df.columns if c not in exclude and not c.startswith("_")]


# ---------------------------------------------------------------------------
# Metrics helper
# ---------------------------------------------------------------------------

def compute_metrics(y_true, y_pred, name: str) -> dict:
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2   = r2_score(y_true, y_pred)
    return {"model": name, "MAE": round(mae, 4), "RMSE": round(rmse, 4), "R2": round(r2, 4)}


# ---------------------------------------------------------------------------
# Naive baseline
# ---------------------------------------------------------------------------

def naive_persistence(y_train, y_test):
    """Persistence baseline: predict last training value for every test step."""
    last_val = float(y_train[-1])
    return np.full(len(y_test), last_val)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def chronological_train_test_split(df_model: pd.DataFrame, train_frac: float = 0.80):
    """
    Strict chronological train/test split (no random shuffle) to prevent temporal data leakage.
    Ensures all training records strictly precede testing records.
    """
    df_sorted = df_model.sort_values("time").reset_index(drop=True)
    split_idx = int(len(df_sorted) * train_frac)
    train_df  = df_sorted.iloc[:split_idx].copy()
    test_df   = df_sorted.iloc[split_idx:].copy()
    return train_df, test_df


def train(df: pd.DataFrame):
    """
    Main training routine.
    Returns: (best_pipeline, feature_cols, all_metrics, train_df, test_df, df_model)
    """
    df = engineer_features(df)
    feature_cols = get_feature_columns(df)

    # Drop rows with NaN in features or target (from lag warmup period)
    df_model = df[feature_cols + [TARGET_COL, "time", "city"]].dropna().copy()

    log.info("Dataset after feature engineering: %d rows, %d features.",
             len(df_model), len(feature_cols))

    # --- Chronological 80/20 split (NO shuffling — strict temporal separation) ---
    train_df, test_df = chronological_train_test_split(df_model, train_frac=0.80)

    log.info("Train: %d rows (up to %s) | Test: %d rows (from %s).",
             len(train_df), train_df["time"].max().date(),
             len(test_df), test_df["time"].min().date())

    X_train = train_df[feature_cols].values
    y_train = train_df[TARGET_COL].values
    X_test  = test_df[feature_cols].values
    y_test  = test_df[TARGET_COL].values

    all_metrics = []

    # 1. Naive baseline
    log.info("Evaluating naive persistence baseline ...")
    y_naive = naive_persistence(y_train, y_test)
    all_metrics.append(compute_metrics(y_test, y_naive, "Naive Persistence"))

    # 2. Ridge regression
    log.info("Training Ridge regression ...")
    ridge_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model",  Ridge(alpha=1.0)),
    ])
    ridge_pipe.fit(X_train, y_train)
    y_ridge = ridge_pipe.predict(X_test)
    all_metrics.append(compute_metrics(y_test, y_ridge, "Ridge Regression"))

    # 3. Gradient Boosting
    log.info("Training GradientBoostingRegressor (may take ~30-60s) ...")
    gb_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model",  GradientBoostingRegressor(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            min_samples_leaf=10,
            random_state=42,
        )),
    ])
    gb_pipe.fit(X_train, y_train)
    y_gb = gb_pipe.predict(X_test)
    all_metrics.append(compute_metrics(y_test, y_gb, "GradientBoosting"))

    # --- Print comparison table ---
    metrics_df = pd.DataFrame(all_metrics)
    log.info("\n\n%s\n", metrics_df.to_string(index=False))

    # --- Pick best model by RMSE ---
    best_row  = metrics_df.loc[metrics_df["RMSE"].idxmin()]
    best_name = best_row["model"]
    log.info("Best model: %s  (RMSE=%.4f, MAE=%.4f, R2=%.4f)",
             best_name, best_row["RMSE"], best_row["MAE"], best_row["R2"])

    model_map = {
        "Ridge Regression": ridge_pipe,
        "GradientBoosting": gb_pipe,
    }
    best_pipeline = model_map.get(best_name, gb_pipe)

    return best_pipeline, feature_cols, all_metrics, train_df, test_df, df_model


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def save_artifacts(pipeline, feature_cols: list, all_metrics: list,
                   df_model: pd.DataFrame):
    """Persist model, feature list, metrics, and last-known state per city."""

    def _last_known_for_city(g):
        idx = g["time"].idxmax()
        result = {"time": g.loc[idx, "time"].isoformat()}
        for c in [TARGET_COL, "wind_speed_10m", "relative_humidity_2m", "precipitation"]:
            if c in g.columns:
                result[c] = float(g.loc[idx, c])
        return result

    last_known = {
        city: _last_known_for_city(grp)
        for city, grp in df_model.groupby("city")
    }

    artifact = {
        "pipeline":     pipeline,
        "feature_cols": feature_cols,
        "city_order":   CITY_ORDER,
        "target_col":   TARGET_COL,
        "last_known":   last_known,
    }
    joblib.dump(artifact, MODEL_PATH)
    log.info("Model saved -> %s", MODEL_PATH)

    with open(METRICS_PATH, "w") as f:
        json.dump(all_metrics, f, indent=2)
    log.info("Metrics saved -> %s", METRICS_PATH)


# ---------------------------------------------------------------------------
# Inference: predict_next
# ---------------------------------------------------------------------------

def predict_next(city: str, horizon_days: int = 7) -> pd.DataFrame:
    """
    Generate an autoregressive hourly temperature forecast for `city` over `horizon_days`.

    Uses recursive multi-step forecasting with a historical buffer (from the
    most recent observations) so that dynamic lag features (1h, 2h, ..., 168h)
    and rolling statistics (6h, 24h, 168h) are seeded from true data and rolled
    forward with model predictions.

    Returns
    -------
    pd.DataFrame with columns: time, predicted_temperature
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No trained model at {MODEL_PATH}. Run: python src/train_model.py"
        )

    artifact     = joblib.load(MODEL_PATH)
    pipeline     = artifact["pipeline"]
    feature_cols = artifact["feature_cols"]

    city_path = CITY_FILES.get(city)
    if not city_path or not city_path.exists():
        raise FileNotFoundError(f"Historical data file for '{city}' not found.")

    # Load 200 hours of history to satisfy the 168h lag & rolling windows
    df_hist = _parse_city_csv(city_path, city).sort_values("time").tail(200)
    last_row = df_hist.iloc[-1]

    n_steps = horizon_days * 24
    start_time = df_hist["time"].max() + pd.Timedelta(hours=1)
    future_times = pd.date_range(start=start_time, periods=n_steps, freq="h")

    fut_df = pd.DataFrame([{
        "time": t,
        "city": city,
        TARGET_COL: np.nan,
        "wind_speed_10m": last_row["wind_speed_10m"],
        "relative_humidity_2m": last_row["relative_humidity_2m"],
        "precipitation": 0.0,
    } for t in future_times])

    combined = pd.concat([df_hist, fut_df], ignore_index=True)
    hist_len = len(df_hist)

    # Precalculate calendar features
    combined["hour"]        = combined["time"].dt.hour
    combined["month"]       = combined["time"].dt.month
    combined["day_of_year"] = combined["time"].dt.dayofyear
    combined["day_of_week"] = combined["time"].dt.dayofweek

    combined["hour_sin"]  = np.sin(2 * np.pi * combined["hour"] / 24)
    combined["hour_cos"]  = np.cos(2 * np.pi * combined["hour"] / 24)
    combined["month_sin"] = np.sin(2 * np.pi * combined["month"] / 12)
    combined["month_cos"] = np.cos(2 * np.pi * combined["month"] / 12)
    combined["doy_sin"]   = np.sin(2 * np.pi * combined["day_of_year"] / 365)
    combined["doy_cos"]   = np.cos(2 * np.pi * combined["day_of_year"] / 365)
    combined["dow_sin"]   = np.sin(2 * np.pi * combined["day_of_week"] / 7)
    combined["dow_cos"]   = np.cos(2 * np.pi * combined["day_of_week"] / 7)

    combined["wind_speed_10m_lag1"] = last_row["wind_speed_10m"]
    combined["wind_speed_10m_roll24_mean"] = last_row["wind_speed_10m"]
    combined["relative_humidity_2m_lag1"] = last_row["relative_humidity_2m"]
    combined["relative_humidity_2m_roll24_mean"] = last_row["relative_humidity_2m"]
    combined["precipitation_lag1"] = 0.0
    combined["precipitation_roll24_mean"] = 0.0

    combined["city_enc"] = CITY_ORDER.index(city) if city in CITY_ORDER else 0
    for c in CITY_ORDER:
        combined[f"city_{c}"] = 1.0 if c == city else 0.0

    temps = combined[TARGET_COL].values.copy()
    feat_indices = {col: i for i, col in enumerate(feature_cols)}

    X_fut = np.zeros((n_steps, len(feature_cols)), dtype=float)
    for col in feature_cols:
        if col in combined.columns:
            X_fut[:, feat_indices[col]] = combined.iloc[hist_len:][col].values

    # Autoregressive multi-step rollout
    preds = []
    for i in range(n_steps):
        curr_idx = hist_len + i
        for lag in [1, 2, 3, 6, 12, 24, 48, 168]:
            col_name = f"temp_lag_{lag}h"
            if col_name in feat_indices:
                X_fut[i, feat_indices[col_name]] = temps[curr_idx - lag]

        for w in [6, 24, 168]:
            w_slice = temps[curr_idx - w : curr_idx]
            mean_col = f"temp_roll_mean_{w}h"
            std_col  = f"temp_roll_std_{w}h"
            if mean_col in feat_indices:
                X_fut[i, feat_indices[mean_col]] = np.mean(w_slice)
            if std_col in feat_indices:
                X_fut[i, feat_indices[std_col]]  = np.std(w_slice)

        pred_val = float(pipeline.predict(X_fut[i:i+1])[0])
        temps[curr_idx] = pred_val
        preds.append(pred_val)

    return pd.DataFrame({
        "time":                  future_times,
        "predicted_temperature": np.round(preds, 2),
    })


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    log.info("=" * 60)
    log.info("Pakistan Weather -- Temperature Forecasting Training")
    log.info("=" * 60)

    log.info("Loading data ...")
    df = load_data()
    log.info("Total rows: %d | Cities: %s", len(df), df["city"].unique().tolist())

    best_pipeline, feature_cols, all_metrics, train_df, test_df, df_model = train(df)

    save_artifacts(best_pipeline, feature_cols, all_metrics, df_model)

    # Sanity check
    log.info("\n--- Forecast sanity check: Lahore, 3 days ---")
    try:
        fc = predict_next("Lahore", horizon_days=3)
        log.info("\n%s", fc.head(10).to_string(index=False))
    except Exception as exc:
        log.warning("Sanity check skipped: %s", exc)

    log.info("All done. Models saved to: %s", MODELS_DIR)


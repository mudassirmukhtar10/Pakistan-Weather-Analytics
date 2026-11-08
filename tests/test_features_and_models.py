"""
Tests for ML Feature Engineering, Temporal Split (No Data Leakage), and Model Persistence.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from src.train_model import (
    load_data,
    engineer_features,
    chronological_train_test_split,
    get_feature_columns,
    CITY_ORDER,
    TARGET_COL,
)


def test_data_loading_and_columns():
    """Verify that hourly dataset loads properly with required meteorological columns."""
    df = load_data()
    assert not df.empty, "Loaded dataframe should not be empty"
    assert "time" in df.columns, "'time' column must be present"
    assert "city" in df.columns, "'city' column must be present"
    assert TARGET_COL in df.columns, f"Target '{TARGET_COL}' must be present"
    assert set(df["city"].unique()).issubset(set(CITY_ORDER))


def test_no_data_leakage_in_lag_features():
    """
    CRITICAL: Verify that lag features use ONLY strictly past data.
    At timestamp t, lag_1 must equal the actual temperature at timestamp t-1.
    """
    df = load_data()
    # Test on a single city to verify sequential lags
    city_df = df[df["city"] == "Islamabad"].sort_values("time").copy()
    feat_df = engineer_features(city_df)

    # Check lag_1h
    for i in range(5, 20):
        t_current = feat_df.iloc[i]["time"]
        t_prev = feat_df.iloc[i - 1]["time"]
        assert (t_current - t_prev) == pd.Timedelta(hours=1), "Time steps must be consecutive 1 hour"
        assert feat_df.iloc[i]["temp_lag_1h"] == feat_df.iloc[i - 1][TARGET_COL], (
            f"temp_lag_1h at {t_current} must equal target at {t_prev}"
        )

    # Check lag_24h (exactly 24 hours prior)
    for i in range(30, 45):
        t_current = feat_df.iloc[i]["time"]
        t_24h_prior = feat_df.iloc[i - 24]["time"]
        assert (t_current - t_24h_prior) == pd.Timedelta(hours=24)
        assert feat_df.iloc[i]["temp_lag_24h"] == feat_df.iloc[i - 24][TARGET_COL]


def test_chronological_train_test_split():
    """
    CRITICAL: Verify train/test split is strictly chronological (no random shuffle).
    All train timestamps must precede all test timestamps.
    """
    df = load_data()
    feat_df = engineer_features(df)
    train_df, test_df = chronological_train_test_split(feat_df, train_frac=0.8)

    assert len(train_df) > 0 and len(test_df) > 0
    max_train_time = train_df["time"].max()
    min_test_time = test_df["time"].min()

    assert max_train_time <= min_test_time, (
        f"Data leakage detected! max_train_time ({max_train_time}) > min_test_time ({min_test_time})"
    )


def test_no_future_data_in_rolling_statistics():
    """Verify rolling statistics do not incorporate the current or future time steps."""
    df = load_data()
    city_df = df[df["city"] == "Karachi"].sort_values("time").copy()
    feat_df = engineer_features(city_df)

    # In train_model.py: roll_mean_6h is rolling(6).mean() of shift(1)
    # Thus roll_mean_6h at row i must equal the mean of temp at rows [i-6, ..., i-1]
    for i in range(20, 25):
        expected_mean = feat_df.iloc[i - 6 : i][TARGET_COL].mean()
        actual_roll_mean = feat_df.iloc[i]["temp_roll_mean_6h"]
        assert np.isclose(expected_mean, actual_roll_mean, atol=1e-4), (
            f"Rolling mean mismatch at row {i}: expected {expected_mean}, got {actual_roll_mean}"
        )

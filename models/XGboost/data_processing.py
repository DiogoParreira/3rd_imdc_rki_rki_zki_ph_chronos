import numpy as np
import pandas as pd
from data_catalog.data_factory import data_factory

TO_LAG = [
    "cases", "temp_min", "temp_med", "temp_max","precip_min", "precip_med", "precip_max",
    "pressure_min", "pressure_med", "pressure_max","rel_humid_min", "rel_humid_med", "rel_humid_max",
    "thermal_range", "rainy_days", "enso",]

ENGINEER_DROP = ["epiweek", "date", "year", "state", "year_month"]


def _lag_set(max_lag, short_lags):
    """Sparse lags: a few short ones + seasonal multiples of 26 up to max_lag."""
    seasonal = range(26, max_lag + 1, 26)
    return sorted({l for l in (*short_lags, *seasonal) if 1 <= l <= max_lag})


def _window_set(max_lag, short_windows):
    """Same idea as _lag_set: a few short windows + seasonal multiples of 26 up to max_lag."""
    seasonal = range(26, max_lag + 1, 26)
    return sorted({w for w in (*short_windows, *seasonal) if 1 <= w <= max_lag})


def _season_sincos(dates):
    doy = pd.DatetimeIndex(pd.to_datetime(dates)).dayofyear.to_numpy()
    ang = 2.0 * np.pi * (doy - 1) / 365.25
    return np.sin(ang), np.cos(ang)


def _build_features(dataset_id, max_lag, lag_cols, group_col, time_col, short_lags, short_windows):

    lags = _lag_set(max_lag, short_lags)
    windows = _window_set(max_lag, short_windows)

    X_raw, _ = data_factory(dataset_id)
    df = X_raw.copy().sort_values([group_col, time_col]).reset_index(drop=True)

    df["_date"] = pd.to_datetime(df["date"])
    df["_woy"] = df[time_col] % 100
    df["week_sin"], df["week_cos"] = _season_sincos(df["_date"])
    df["cases_per_10000"] = df["cases"] / df["population"] * 10_000


    g = df.groupby(group_col, sort=False)
    feats = {}
    for c in lag_cols:
        for L in lags:
            feats[f"{c}_lag{L}"] = g[c].shift(L)
        for w in windows:
            r = g[c].rolling(w, min_periods=w)
            feats[f"{c}_rollmean{w}"] = r.mean().reset_index(level=0, drop=True)
            feats[f"{c}_rollstd{w}"]  = r.std().reset_index(level=0, drop=True)

    engineered = pd.DataFrame(feats, index=df.index)
    feat_valid = ~engineered.isna().any(axis=1)

    df = pd.concat([df, engineered], axis=1)
    df = df.drop(columns=[c for c in ENGINEER_DROP if c in df.columns and c != time_col])
    feature_cols = [c for c in df.columns if c not in (time_col, "_woy", "_date", group_col)]
    return df, feature_cols, feat_valid


def make_dataset(
    dataset_id,
    max_lag=156,
    max_horizon=67,
    horizon_step=1,
    lag_cols=TO_LAG,
    group_col="uf_code",
    time_col="epiweek",
    short_lags=(1, 2, 3, 4, 8, 12, 18),
    short_windows=(4, 12),):

    df, feature_cols, feat_valid = _build_features(
        dataset_id, max_lag, lag_cols, group_col, time_col, short_lags, short_windows
)

    gh = df.groupby(group_col, sort=False)
    blocks = []
    for h in range(1, max_horizon + 1, horizon_step):
        y_h = gh["cases"].shift(-h)
        date_h = gh["_date"].shift(-h)
        valid = feat_valid & y_h.notna()
        if not valid.any():
            continue

        block = df.loc[valid, feature_cols].copy()
        block["h"] = h
        block["target_week_sin"], block["target_week_cos"] = _season_sincos(date_h[valid])
        block["_y"] = y_h[valid].to_numpy()
        blocks.append(block)

    full = pd.concat(blocks, ignore_index=True)
    y = full.pop("_y")
    return full, y


def make_forecast_frame(
    dataset_id,
    max_lag=156,
    max_horizon=67,
    horizon_step=1,
    lag_cols=TO_LAG,
    group_col="uf_code",
    time_col="epiweek",
    short_lags=(1, 2, 3, 4, 8, 12, 18),
    short_windows=(4, 12),):

    df, feature_cols, feat_valid = _build_features(
        dataset_id, max_lag, lag_cols, group_col, time_col, short_lags, short_windows)

    last_idx = df.loc[feat_valid].groupby(group_col, sort=False).tail(1).index
    origin = df.loc[last_idx, feature_cols].reset_index(drop=True)
    origin_uf = df.loc[last_idx, group_col].to_numpy()
    origin_date = pd.DatetimeIndex(df.loc[last_idx, "_date"].to_numpy())
    origin_week = df.loc[last_idx, time_col].to_numpy()

    blocks = []
    for h in range(1, max_horizon + 1, horizon_step):
        b = origin.copy()
        b[group_col] = origin_uf
        b["h"] = h
        target_date = origin_date + pd.to_timedelta(h * 7, unit="D")
        b["target_week_sin"], b["target_week_cos"] = _season_sincos(target_date)
        b["forecast_from_epiweek"] = origin_week
        blocks.append(b)

    return pd.concat(blocks, ignore_index=True)

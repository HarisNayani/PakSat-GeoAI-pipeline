"""Leakage-aware spatial evaluation and dual PM2.5 model training."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold, TimeSeriesSplit
from sklearn.pipeline import Pipeline

SATELLITE_FEATURES = (
    "AOD_047",
    "temperature",
    "relative_humidity",
    "wind_speed",
    "pblh",
    "atmospheric_stagnation_index",
    "thermal_confinement_ratio",
    "latitude",
    "longitude",
)
SENSOR_ASSISTED_FEATURES = (
    "AOD_047",
    "NO2_density",
    "temperature",
    "relative_humidity",
    "wind_speed",
    "pblh",
    "atmospheric_stagnation_index",
    "thermal_confinement_ratio",
    "photochemical_pm25_proxy",
    "hygroscopic_growth_factor",
    "lag_24h",
    "lag_48h",
)
MODEL_VARIANTS = {
    "sensor_assisted_nowcasting": SENSOR_ASSISTED_FEATURES,
    "unmonitored_satellite_downscaling": SATELLITE_FEATURES,
}
DEFAULT_PM25_EXCEEDANCE_THRESHOLD_UG_M3 = 55.0
PM25_EXCEEDANCE_RECALL_TARGET = 0.80


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").astype(float)


def _prior_station_lags(
    frame: pd.DataFrame,
    history: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Create past-only lags, optionally sourcing them only from prior train rows."""
    lags = pd.DataFrame(np.nan, index=frame.index, columns=("lag_24h", "lag_48h"))
    if not {"pm25", "timestamp", "city"}.issubset(frame.columns):
        return lags
    history_frame = frame if history is None else history
    if not {"pm25", "timestamp", "city"}.issubset(history_frame.columns):
        return lags

    query_events = pd.DataFrame(
        {
            "city": frame["city"].astype("string").to_numpy(),
            "timestamp": pd.to_datetime(frame["timestamp"], errors="coerce", utc=True).to_numpy(),
            "value": (
                pd.to_numeric(frame["pm25"], errors="coerce").to_numpy()
                if history is None else np.nan
            ),
            "position": np.arange(len(frame)),
            "is_query": True,
        }
    )
    if history is None:
        events = query_events
    else:
        history_events = pd.DataFrame(
            {
                "city": history_frame["city"].astype("string").to_numpy(),
                "timestamp": pd.to_datetime(history_frame["timestamp"], errors="coerce", utc=True).to_numpy(),
                "value": pd.to_numeric(history_frame["pm25"], errors="coerce").to_numpy(),
                "position": -1,
                "is_query": False,
            }
        )
        events = pd.concat([history_events, query_events], ignore_index=True)
    events = events.dropna(subset=["timestamp"])

    for _, city_rows in events.groupby("city", dropna=False, sort=False):
        timed = city_rows.sort_values(
            ["timestamp", "is_query", "position"], kind="mergesort"
        )
        values = pd.Series(
            timed["value"].to_numpy(), index=pd.DatetimeIndex(timed["timestamp"])
        )
        query_mask = timed["is_query"].to_numpy(dtype=bool)
        positions = timed.loc[query_mask, "position"].to_numpy(dtype=int)
        for hours, name in ((24, "lag_24h"), (48, "lag_48h")):
            # closed='left' excludes current/duplicate timestamps. Validation
            # queries have null targets and can only use the supplied train history.
            past = values.rolling(f"{hours}h", min_periods=1, closed="left").mean()
            lags.iloc[positions, lags.columns.get_loc(name)] = past.to_numpy()[query_mask]
    lags.index = frame.index
    return lags


def engineer_features(
    frame: pd.DataFrame,
    variant: str,
    *,
    history: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build deterministic per-row features for one already-isolated partition."""
    if variant not in MODEL_VARIANTS:
        raise ValueError(f"Unsupported model variant: {variant}")

    features = pd.DataFrame(index=frame.index)
    for column in ("AOD_047", "NO2_density", "temperature", "wind_speed", "pblh", "latitude", "longitude"):
        features[column] = _numeric(frame, column)

    humidity = _numeric(frame, "relative_humidity")
    humidity = humidity.where(humidity <= 1.0, humidity / 100.0)
    humidity = humidity.where(humidity.between(0.0, 1.0))
    features["relative_humidity"] = humidity

    wind = features["wind_speed"].where(features["wind_speed"] >= 0.0)
    pblh = features["pblh"].where(features["pblh"] > 0.0)
    features["atmospheric_stagnation_index"] = 1000.0 / (wind * pblh + 1.0)
    features["thermal_confinement_ratio"] = features["temperature"] / (pblh / 100.0)
    features["photochemical_pm25_proxy"] = (
        features["NO2_density"] * features["temperature"] * features["AOD_047"]
    )
    features["hygroscopic_growth_factor"] = 1.0 / (1.0 - humidity.clip(upper=0.99))

    if variant == "sensor_assisted_nowcasting":
        # Called independently on each CV partition, so held-out targets cannot
        # become training features and each row only sees earlier local readings.
        features[["lag_24h", "lag_48h"]] = _prior_station_lags(frame, history)
    return features.loc[:, MODEL_VARIANTS[variant]].replace([np.inf, -np.inf], np.nan)


def make_model_pipeline(estimator: Any | None = None) -> Pipeline:
    """Keep learned imputation inside the fold-fitted estimator pipeline."""
    if estimator is None:
        from lightgbm import LGBMRegressor

        estimator = LGBMRegressor(
            n_estimators=500,
            learning_rate=0.04,
            num_leaves=31,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42,
            verbosity=-1,
        )
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
            ("regressor", estimator),
        ]
    )


def _spatial_groups(frame: pd.DataFrame) -> pd.Series:
    if "city" in frame:
        return frame["city"].astype("string").fillna("unknown-city")
    if {"latitude", "longitude"}.issubset(frame.columns):
        latitude = pd.to_numeric(frame["latitude"], errors="coerce")
        longitude = pd.to_numeric(frame["longitude"], errors="coerce")
        # Fixed spatial blocks keep repeat observations at one location in one fold.
        return (
            latitude.div(0.25).floordiv(1).astype("string")
            + ":"
            + longitude.div(0.25).floordiv(1).astype("string")
        )
    raise ValueError("Spatial evaluation requires a `city` column or coordinates.")


def _temporal_splits(frame: pd.DataFrame, n_splits: int) -> list[tuple[np.ndarray, np.ndarray]]:
    if "timestamp" not in frame:
        raise ValueError("Sensor-assisted nowcasting evaluation requires timestamps.")
    timestamps = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True)
    unique_times = pd.DatetimeIndex(timestamps.dropna().unique()).sort_values()
    if len(unique_times) < 3:
        raise ValueError("At least three distinct timestamps are required for forward validation.")
    splitter = TimeSeriesSplit(n_splits=min(n_splits, len(unique_times) - 1))
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for train_time_index, test_time_index in splitter.split(unique_times):
        train_times = unique_times[train_time_index]
        test_times = unique_times[test_time_index]
        train_mask = timestamps.isin(train_times).to_numpy()
        test_mask = timestamps.isin(test_times).to_numpy()
        splits.append((np.flatnonzero(train_mask), np.flatnonzero(test_mask)))
    return splits


def _pm25_exceedance_recall(
    actual: pd.Series,
    predicted: np.ndarray,
    threshold_ug_m3: float,
) -> float:
    """Measure recall for row-level PM2.5 readings at a screening threshold."""
    actual_exceedances = actual.to_numpy(dtype=float) >= threshold_ug_m3
    if not actual_exceedances.any():
        return float("nan")
    return float((predicted[actual_exceedances] >= threshold_ug_m3).mean())


def train_dual_models(
    frame: pd.DataFrame,
    *,
    n_splits: int = 3,
    estimator_factory: Any | None = None,
    pm25_exceedance_threshold_ug_m3: float = DEFAULT_PM25_EXCEEDANCE_THRESHOLD_UG_M3,
) -> tuple[dict[str, Any], dict[str, dict[str, float]]]:
    """Evaluate each variant with its deployment-appropriate split, then refit.

    Feature generation is deterministic and partition-local. The only learned
    preprocessing step (median imputation) is fit by each training pipeline.
    """
    if "pm25" not in frame:
        raise ValueError("Input must contain `pm25`.")
    if not isinstance(n_splits, int) or n_splits < 2:
        raise ValueError("`n_splits` must be an integer of at least 2.")
    if not np.isfinite(pm25_exceedance_threshold_ug_m3) or pm25_exceedance_threshold_ug_m3 <= 0:
        raise ValueError("`pm25_exceedance_threshold_ug_m3` must be a positive finite number.")
    if not {"latitude", "longitude"}.issubset(frame.columns):
        raise ValueError("Both `latitude` and `longitude` are required for satellite downscaling.")
    target = pd.to_numeric(frame["pm25"], errors="coerce")
    coordinates = frame[["latitude", "longitude"]].apply(pd.to_numeric, errors="coerce")
    valid = target.notna() & coordinates.notna().all(axis=1)
    clean_frame = frame.loc[valid].copy()
    target = target.loc[valid].astype(float)
    if clean_frame.empty:
        raise ValueError("No training rows have both a valid PM2.5 target and coordinates.")
    groups = _spatial_groups(clean_frame)
    group_count = groups.nunique()
    if group_count < 2:
        raise ValueError("At least two distinct spatial groups are required.")
    spatial_splitter = GroupKFold(n_splits=min(n_splits, group_count))
    scores: dict[str, dict[str, float]] = {}
    fitted: dict[str, Any] = {}

    for variant in MODEL_VARIANTS:
        predictions = np.full(len(clean_frame), np.nan, dtype=float)
        lag_available = np.zeros(len(clean_frame), dtype=bool)
        fold_splits = (
            _temporal_splits(clean_frame, n_splits)
            if variant == "sensor_assisted_nowcasting"
            else list(spatial_splitter.split(clean_frame, target, groups))
        )
        for train_index, test_index in fold_splits:
            # Isolate raw rows before creating lags or fitting any transformer.
            train_rows = clean_frame.iloc[train_index]
            test_rows = clean_frame.iloc[test_index]
            train_x = engineer_features(train_rows, variant)
            test_x = engineer_features(
                test_rows,
                variant,
                history=train_rows if variant == "sensor_assisted_nowcasting" else None,
            )
            if variant == "sensor_assisted_nowcasting":
                lag_available[test_index] = test_x[["lag_24h", "lag_48h"]].notna().any(axis=1)
            model = make_model_pipeline(
                estimator_factory(variant) if estimator_factory else None
            )
            model.fit(train_x, target.iloc[train_index])
            predictions[test_index] = model.predict(test_x)

        from sklearn.metrics import mean_squared_error, r2_score

        evaluated = np.isfinite(predictions)
        scores[variant] = {
            "r2": float(r2_score(target.iloc[evaluated], predictions[evaluated])),
            "rmse_ug_m3": float(np.sqrt(mean_squared_error(target.iloc[evaluated], predictions[evaluated]))),
            "pm25_exceedance_recall": _pm25_exceedance_recall(
                target.iloc[evaluated], predictions[evaluated], pm25_exceedance_threshold_ug_m3
            ),
            "pm25_exceedance_threshold_ug_m3": float(pm25_exceedance_threshold_ug_m3),
            "rows": float(evaluated.sum()),
            "validation_groups": float(groups.iloc[evaluated].nunique()),
            "validation_folds": float(len(fold_splits)),
        }
        if variant == "sensor_assisted_nowcasting":
            scores[variant]["lag_coverage"] = float(lag_available[evaluated].mean())
        final_x = engineer_features(clean_frame, variant)
        final_model = make_model_pipeline(
            estimator_factory(variant) if estimator_factory else None
        )
        final_model.fit(final_x, target)
        fitted[variant] = final_model

    return fitted, scores


__all__ = [
    "DEFAULT_PM25_EXCEEDANCE_THRESHOLD_UG_M3",
    "MODEL_VARIANTS",
    "PM25_EXCEEDANCE_RECALL_TARGET",
    "SATELLITE_FEATURES",
    "SENSOR_ASSISTED_FEATURES",
    "engineer_features",
    "make_model_pipeline",
    "train_dual_models",
]
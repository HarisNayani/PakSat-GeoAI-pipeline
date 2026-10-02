"""Vectorized physics-informed feature store for PakSat telemetry."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

BASE_COLUMNS = ("wind_speed", "pblh", "temperature", "relative_humidity", "NO2_density", "AOD_047")
DERIVED_COLUMNS = ("atmospheric_stagnation_index", "thermal_confinement_ratio", "photochemical_pm25_proxy", "hygroscopic_growth_factor", "lag_24h", "lag_48h")
MODEL_COLUMNS = ("AOD_047", "NO2_density", "temperature", "relative_humidity", "wind_speed", "pblh", *DERIVED_COLUMNS)


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce")


def add_physics_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add bounded physics features without mutating the input frame."""
    missing = set(BASE_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required telemetry columns: {sorted(missing)}")
    result = frame.copy()
    wind = _numeric(result, "wind_speed").clip(lower=1e-6)
    pblh = _numeric(result, "pblh").clip(lower=1e-6)
    temperature = _numeric(result, "temperature")
    humidity = _numeric(result, "relative_humidity")
    humidity = humidity.where(humidity <= 1.0, humidity / 100.0).clip(0.0, 0.99)
    result["atmospheric_stagnation_index"] = 1000.0 / (wind * pblh + 1.0)
    result["thermal_confinement_ratio"] = temperature / (pblh / 100.0)
    result["photochemical_pm25_proxy"] = _numeric(result, "NO2_density") * temperature * _numeric(result, "AOD_047")
    result["hygroscopic_growth_factor"] = 1.0 / (1.0 - humidity)
    return result.replace([np.inf, -np.inf], np.nan)


def add_spatiotemporal_lags(frame: pd.DataFrame, target: str = "pm25") -> pd.DataFrame:
    """Compute prior-only 24/48-hour PM2.5 means independently per city."""
    if target not in frame:
        raise ValueError(f"Missing target column: {target}")
    original_index = frame.index
    result = frame.copy().reset_index(drop=True)
    timestamp_values = result["timestamp"] if "timestamp" in result else pd.Series(pd.NaT, index=result.index)
    result["timestamp"] = pd.to_datetime(timestamp_values, errors="coerce", utc=True)
    result["_target"] = pd.to_numeric(result[target], errors="coerce")
    result["_original_position"] = np.arange(len(result))
    group_columns = ["city"] if "city" in result else []
    sort_columns = group_columns + ["timestamp", "_original_position"] if group_columns else ["timestamp", "_original_position"]
    result = result.sort_values(sort_columns, kind="mergesort", na_position="last")
    groups = result.groupby(group_columns, sort=False, dropna=False) if group_columns else [(None, result)]

    for hours, name in ((24, "lag_24h"), (48, "lag_48h")):
        result[name] = np.nan
        for _, group in groups:
            timed_rows = group.loc[group["timestamp"].notna()]
            if not timed_rows.empty:
                history = timed_rows.set_index("timestamp")["_target"]
                prior_mean = history.rolling(f"{hours}h", min_periods=1, closed="left").mean()
                result.loc[timed_rows.index, name] = prior_mean.to_numpy()

            # Untimestamped rows have no defensible ordering; leave their lags missing.

    result = result.sort_values("_original_position", kind="mergesort")
    result = result.drop(columns=["_target", "_original_position"])
    result.index = original_index
    return result


def prepare_features(frame: pd.DataFrame, feature_columns: Iterable[str] | None = None) -> pd.DataFrame:
    """Return deterministic features without fitting or applying imputation."""
    enriched = add_spatiotemporal_lags(add_physics_features(frame)) if "pm25" in frame else add_physics_features(frame)
    columns = list(feature_columns or MODEL_COLUMNS)
    missing = set(columns).difference(enriched.columns)
    if missing:
        raise ValueError(f"Missing model feature columns: {sorted(missing)}")
    values = enriched[columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    # Keep missing telemetry as NaN. Learned imputation belongs inside the
    # fold-fitted estimator pipeline, never in this whole-frame helper.
    return values


__all__ = ["BASE_COLUMNS", "DERIVED_COLUMNS", "MODEL_COLUMNS", "add_physics_features", "add_spatiotemporal_lags", "prepare_features"]

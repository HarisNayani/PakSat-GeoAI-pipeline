"""Physics-informed feature engineering for PakSat PM2.5 estimation."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {
    "wind_speed",
    "pblh",
    "temperature",
    "relative_humidity",
    "NO2_density",
    "AOD_047",
}
DERIVED_COLUMNS = (
    "atmospheric_stagnation_index",
    "thermal_confinement_ratio",
    "photochemical_pm25_proxy",
    "hygroscopic_growth_factor",
)


def _positive(values: pd.Series, minimum: float = 1e-6) -> pd.Series:
    """Replace invalid or non-positive physical denominators."""
    return values.astype(float).clip(lower=minimum)


def add_physics_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with vectorized atmospheric and aerosol features added."""
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required telemetry columns: {sorted(missing)}")

    result = frame.copy()
    wind_speed = _positive(result["wind_speed"])
    pblh = _positive(result["pblh"])
    temperature = result["temperature"].astype(float)
    humidity = result["relative_humidity"].astype(float)
    humidity = humidity.where(humidity <= 0.99, humidity / 100.0).clip(0.0, 0.99)

    result["atmospheric_stagnation_index"] = 1000.0 / (wind_speed * pblh + 1.0)
    result["thermal_confinement_ratio"] = temperature / (pblh / 100.0)
    result["photochemical_pm25_proxy"] = (
        result["NO2_density"].astype(float)
        * temperature
        * result["AOD_047"].astype(float)
    )
    result["hygroscopic_growth_factor"] = 1.0 / (1.0 - humidity)
    return result.replace([np.inf, -np.inf], np.nan)


def prepare_features(
    frame: pd.DataFrame,
    feature_columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Engineer deterministic features without whole-frame interpolation."""
    result = add_physics_features(frame)
    columns = list(feature_columns) if feature_columns is not None else [
        "AOD_047",
        "NO2_density",
        "temperature",
        "relative_humidity",
        "wind_speed",
        "pblh",
        *DERIVED_COLUMNS,
    ]
    missing = set(columns).difference(result.columns)
    if missing:
        raise ValueError(f"Missing model feature columns: {sorted(missing)}")
    # Preserve missing values for fold-fitted imputation in the model pipeline.
    return result[columns].apply(pd.to_numeric, errors="coerce")


if __name__ == "__main__":
    print("Import add_physics_features() or prepare_features() to process telemetry.")

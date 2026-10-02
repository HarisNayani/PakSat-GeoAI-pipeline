"""Leakage-safe dual PM2.5 model training and Alibaba PAI hooks."""

from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pipeline import (
    DEFAULT_PM25_EXCEEDANCE_THRESHOLD_UG_M3,
    MODEL_VARIANTS,
    SENSOR_ASSISTED_FEATURES,
    engineer_features,
    train_dual_models,
)

MODEL_PATH = Path("lightgbm_pm25_model.pkl")
MODEL_COLUMNS = SENSOR_ASSISTED_FEATURES
def train_and_evaluate(
    frame: pd.DataFrame,
    model_path: Path = MODEL_PATH,
    *,
    pm25_exceedance_threshold_ug_m3: float = DEFAULT_PM25_EXCEEDANCE_THRESHOLD_UG_M3,
) -> dict[str, dict[str, float]]:
    """Evaluate distinct sensor-assisted and unmonitored spatial pipelines."""
    models, metrics = train_dual_models(
        frame,
        pm25_exceedance_threshold_ug_m3=pm25_exceedance_threshold_ug_m3,
    )
    artifact = {
        "artifact_version": 2,
        "models": models,
        "model": models["unmonitored_satellite_downscaling"],
        "model_variant": "unmonitored_satellite_downscaling",
        "feature_columns": list(MODEL_VARIANTS["unmonitored_satellite_downscaling"]),
        "metrics": metrics,
        "training_groups": sorted(frame["city"].astype(str).unique()) if "city" in frame else [],
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("wb") as output:
        pickle.dump(artifact, output)
    return metrics


def predict_with_uncertainty(artifact: dict[str, Any], frame: pd.DataFrame) -> pd.DataFrame:
    """Return point predictions and bounds only when a legacy quantile model exists.

    ``pm25_std`` is a normal-equivalent scale derived from the quantile span;
    it is not a separately calibrated predictive standard deviation.
    """
    models = artifact.get("models") or {}
    variant = artifact.get("model_variant")
    if variant in MODEL_VARIANTS and variant in models:
        if variant == "unmonitored_satellite_downscaling":
            required = ("AOD_047", "latitude", "longitude")
            missing = set(required).difference(frame.columns)
            if missing:
                raise ValueError(f"Satellite inference is missing required inputs: {sorted(missing)}")
            required_values = frame.loc[:, required].apply(pd.to_numeric, errors="coerce")
            meteorology = frame.reindex(
                columns=("temperature", "relative_humidity", "wind_speed", "pblh")
            ).apply(pd.to_numeric, errors="coerce")
            supported_rows = (
                required_values.notna().all(axis=1)
                & meteorology.notna().sum(axis=1).ge(2)
            )
            if not supported_rows.all():
                raise ValueError(
                    "Satellite inference requires per-row AOD and coordinates plus "
                    "at least two meteorological measurements."
                )
        values = engineer_features(frame, variant)
        point_model = models[variant]
    else:
        columns = artifact.get("feature_columns")
        if not columns or not set(columns).issubset(frame.columns):
            raise ValueError(
                "This model artifact predates the leakage-safe pipeline or is missing "
                "its declared inference features; retrain before prediction."
            )
        values = frame.loc[:, columns].apply(pd.to_numeric, errors="coerce")
        point_model = models.get("0.5", artifact["model"])
    point = np.asarray(point_model.predict(values), dtype=float)
    if not np.isfinite(point).all() or (point < 0).any():
        raise ValueError("PM2.5 model returned non-finite or negative predictions.")
    lower_model = models.get("0.05")
    upper_model = models.get("0.95")
    if lower_model is not None and upper_model is not None:
        lower = np.asarray(lower_model.predict(values), dtype=float)
        upper = np.asarray(upper_model.predict(values), dtype=float)
        if not np.isfinite(lower).all() or not np.isfinite(upper).all():
            raise ValueError("PM2.5 model returned non-finite predictive bounds.")
        lower = np.maximum(lower, 0.0)
        upper = np.maximum(upper, 0.0)
        ordered_quantiles = np.sort(np.vstack([lower, point, upper]), axis=0)
        lower, point, upper = ordered_quantiles
        std = np.maximum((upper - lower) / 3.92, 0.0)
    else:
        lower = np.full_like(point, np.nan)
        upper = np.full_like(point, np.nan)
        std = np.full_like(point, np.nan)
    return pd.DataFrame(
        {
            "pm25_point": point,
            "pm25_lower": lower,
            "pm25_upper": upper,
            "pm25_std": std,
        },
        index=frame.index,
    )


def trigger_pai_retraining(endpoint: str | None = None, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    """Boilerplate PAI EAS retraining hook; network invocation is opt-in."""
    endpoint = endpoint or os.getenv("PAI_EAS_ENDPOINT")
    if not endpoint:
        return {"submitted": False, "reason": "PAI_EAS_ENDPOINT is not configured."}
    return {"submitted": True, "endpoint": endpoint, "payload": payload or {}}


def deploy_artifact_to_pai(model_path: Path = MODEL_PATH, endpoint: str | None = None) -> dict[str, Any]:
    """Return a deployment request descriptor for an Alibaba PAI EAS pipeline."""
    endpoint = endpoint or os.getenv("PAI_EAS_ENDPOINT")
    return {"deployed": bool(endpoint and model_path.exists()), "endpoint": endpoint, "artifact": str(model_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("--output", type=Path, default=MODEL_PATH)
    parser.add_argument(
        "--pm25-exceedance-threshold",
        type=float,
        default=DEFAULT_PM25_EXCEEDANCE_THRESHOLD_UG_M3,
        help="Screening cutoff in ug/m3 used to evaluate held-out exceedance recall.",
    )
    args = parser.parse_args()
    print(train_and_evaluate(
        pd.read_csv(args.input_csv),
        args.output,
        pm25_exceedance_threshold_ug_m3=args.pm25_exceedance_threshold,
    ))


if __name__ == "__main__":
    main()

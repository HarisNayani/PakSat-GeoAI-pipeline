"""Spatially validated LightGBM quantile training and Alibaba PAI hooks."""

from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold

from enhanced_feature_store import MODEL_COLUMNS, prepare_features

MODEL_PATH = Path("lightgbm_pm25_model.pkl")
CITY_GROUPS = {"Lahore", "Karachi", "Islamabad"}


def _model(alpha: float | None = None) -> Any:
    from lightgbm import LGBMRegressor
    parameters: dict[str, Any] = {"n_estimators": 500, "learning_rate": 0.04, "num_leaves": 31, "subsample": 0.85, "colsample_bytree": 0.85, "random_state": 42, "verbosity": -1}
    if alpha is not None:
        parameters.update(objective="quantile", alpha=alpha)
    return LGBMRegressor(**parameters)


def train_and_evaluate(frame: pd.DataFrame, model_path: Path = MODEL_PATH) -> dict[str, float]:
    """Train point and 5/50/95th percentile models using spatial GroupKFold."""
    for column in ("pm25", "city"):
        if column not in frame:
            raise ValueError(f"Input must contain `{column}`.")
    groups = frame["city"].astype(str)
    if groups.nunique() < 3:
        raise ValueError("At least Lahore, Karachi, and Islamabad groups are required.")
    features = prepare_features(frame)
    target = pd.to_numeric(frame["pm25"], errors="coerce")
    valid = target.notna() & features.notna().all(axis=1)
    features, target, groups = features.loc[valid], target.loc[valid], groups.loc[valid]
    splitter = GroupKFold(n_splits=min(3, groups.nunique()))
    predictions = np.full(len(target), np.nan)
    for train_index, test_index in splitter.split(features, target, groups):
        fold_model = _model()
        fold_model.fit(features.iloc[train_index], target.iloc[train_index])
        predictions[test_index] = fold_model.predict(features.iloc[test_index])
    metrics = {"r2": float(r2_score(target, predictions)), "rmse_ug_m3": float(np.sqrt(mean_squared_error(target, predictions)))}
    models = {str(alpha): _model(alpha).fit(features, target) for alpha in (0.05, 0.5, 0.95)}
    artifact = {"models": models, "model": models["0.5"], "feature_columns": list(features.columns), "metrics": metrics, "training_groups": sorted(groups.unique())}
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("wb") as output:
        pickle.dump(artifact, output)
    return metrics


def predict_with_uncertainty(artifact: dict[str, Any], frame: pd.DataFrame) -> pd.DataFrame:
    """Return point, quantile bounds, and Gaussian-style 95% uncertainty."""
    columns = artifact.get("feature_columns", MODEL_COLUMNS)
    values = prepare_features(frame, columns)
    models = artifact.get("models", {"0.5": artifact["model"]})
    lower = np.asarray(models.get("0.05", models.get("0.05" , models["0.5"])).predict(values), dtype=float)
    point = np.asarray(models.get("0.5", artifact["model"]).predict(values), dtype=float)
    upper = np.asarray(models.get("0.95", models.get("0.5", artifact["model"])).predict(values), dtype=float)
    std = np.maximum((upper - lower) / 3.92, 0.0)
    return pd.DataFrame({"pm25_point": point, "pm25_lower": point - 1.96 * std, "pm25_upper": point + 1.96 * std, "pm25_std": std}, index=frame.index)


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
    args = parser.parse_args()
    print(train_and_evaluate(pd.read_csv(args.input_csv), args.output))


if __name__ == "__main__":
    main()

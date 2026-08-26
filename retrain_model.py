"""Train and evaluate the PakSat LightGBM PM2.5 regressor."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold

from generate_enhanced_features import prepare_features

MODEL_PATH = Path("lightgbm_pm25_model.pkl")
TARGET_COLUMN = "pm25"
GROUP_COLUMN = "city"


def train_and_evaluate(
    frame: pd.DataFrame,
    model_path: Path = MODEL_PATH,
) -> dict[str, float]:
    """Train with spatial GroupKFold and save a final model on all rows."""
    if TARGET_COLUMN not in frame or GROUP_COLUMN not in frame:
        raise ValueError(f"Input must contain `{TARGET_COLUMN}` and `{GROUP_COLUMN}` columns.")
    groups = frame[GROUP_COLUMN].astype(str)
    if groups.nunique() < 3:
        raise ValueError("At least three city groups are required for spatial GroupKFold.")

    features = prepare_features(frame)
    target = pd.to_numeric(frame[TARGET_COLUMN], errors="coerce")
    valid = target.notna() & features.notna().all(axis=1)
    features, target, groups = features.loc[valid], target.loc[valid], groups.loc[valid]
    splitter = GroupKFold(n_splits=min(3, groups.nunique()))
    predictions = pd.Series(index=target.index, dtype=float)

    from lightgbm import LGBMRegressor

    parameters: dict[str, Any] = {
        "n_estimators": 500,
        "learning_rate": 0.04,
        "num_leaves": 31,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "random_state": 42,
        "verbosity": -1,
    }
    for train_index, test_index in splitter.split(features, target, groups):
        fold_model = LGBMRegressor(**parameters)
        fold_model.fit(features.iloc[train_index], target.iloc[train_index])
        predictions.iloc[test_index] = fold_model.predict(features.iloc[test_index])

    metrics = {
        "r2": float(r2_score(target, predictions)),
        "rmse_ug_m3": float(mean_squared_error(target, predictions) ** 0.5),
    }
    final_model = LGBMRegressor(**parameters)
    final_model.fit(features, target)
    artifact = {
        "model": final_model,
        "feature_columns": list(features.columns),
        "metrics": metrics,
    }
    with model_path.open("wb") as output:
        pickle.dump(artifact, output)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("--output", type=Path, default=MODEL_PATH)
    args = parser.parse_args()
    metrics = train_and_evaluate(pd.read_csv(args.input_csv), args.output)
    print(f"Spatial CV R2: {metrics['r2']:.4f}")
    print(f"Spatial CV RMSE: {metrics['rmse_ug_m3']:.2f} ug/m3")


if __name__ == "__main__":
    main()

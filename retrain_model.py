"""Train and evaluate the PakSat LightGBM PM2.5 regressor."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pipeline import DEFAULT_PM25_EXCEEDANCE_THRESHOLD_UG_M3
from train_pai_model import train_and_evaluate as train_dual_pipeline

MODEL_PATH = Path("lightgbm_pm25_model.pkl")
TARGET_COLUMN = "pm25"
GROUP_COLUMN = "city"


def train_and_evaluate(
    frame: pd.DataFrame,
    model_path: Path = MODEL_PATH,
    *,
    pm25_exceedance_threshold_ug_m3: float = DEFAULT_PM25_EXCEEDANCE_THRESHOLD_UG_M3,
) -> dict[str, dict[str, float]]:
    """Delegate retraining to the shared fold-safe dual-model implementation."""
    if TARGET_COLUMN not in frame or GROUP_COLUMN not in frame:
        raise ValueError(f"Input must contain `{TARGET_COLUMN}` and `{GROUP_COLUMN}` columns.")
    return train_dual_pipeline(
        frame,
        model_path,
        pm25_exceedance_threshold_ug_m3=pm25_exceedance_threshold_ug_m3,
    )


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
    metrics = train_and_evaluate(
        pd.read_csv(args.input_csv),
        args.output,
        pm25_exceedance_threshold_ug_m3=args.pm25_exceedance_threshold,
    )
    print(metrics)


if __name__ == "__main__":
    main()

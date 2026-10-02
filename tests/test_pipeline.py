import unittest

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor

from pipeline import (
    DEFAULT_PM25_EXCEEDANCE_THRESHOLD_UG_M3,
    MODEL_VARIANTS,
    _pm25_exceedance_recall,
    engineer_features,
    make_model_pipeline,
    train_dual_models,
)
from train_pai_model import predict_with_uncertainty


class LeakageAwarePipelineTests(unittest.TestCase):
    def test_exceedance_recall_uses_only_observed_positive_rows(self):
        actual = pd.Series([40.0, 60.0, 80.0])
        predicted = np.array([90.0, 50.0, 70.0])

        recall = _pm25_exceedance_recall(
            actual, predicted, DEFAULT_PM25_EXCEEDANCE_THRESHOLD_UG_M3
        )

        self.assertEqual(recall, 0.5)
        no_events = _pm25_exceedance_recall(
            pd.Series([10.0, 20.0]), np.array([15.0, 25.0]), 55.0
        )
        self.assertTrue(np.isnan(no_events))

    def test_station_lags_are_prior_only_and_city_local(self):
        frame = pd.DataFrame(
            [
                {"city": "Lahore", "timestamp": "2026-01-01T00:00:00Z", "pm25": 10.0},
                {"city": "Karachi", "timestamp": "2026-01-01T00:00:00Z", "pm25": 100.0},
                {"city": "Lahore", "timestamp": "2026-01-01T12:00:00Z", "pm25": 30.0},
                {"city": "Karachi", "timestamp": "2026-01-01T12:00:00Z", "pm25": 300.0},
            ]
        )
        # Each partition is engineered independently: held-out targets cannot
        # seed lags from a separate train/test partition.
        validation = engineer_features(
            frame.iloc[[2, 3]],
            "sensor_assisted_nowcasting",
            history=frame.iloc[[0, 1]],
        )
        self.assertEqual(validation.loc[2, "lag_24h"], 10.0)
        self.assertEqual(validation.loc[3, "lag_24h"], 100.0)
        changed_targets = frame.iloc[[2, 3]].copy()
        changed_targets["pm25"] = [9000.0, 8000.0]
        changed_validation = engineer_features(
            changed_targets,
            "sensor_assisted_nowcasting",
            history=frame.iloc[[0, 1]],
        )
        pd.testing.assert_series_equal(
            validation["lag_24h"], changed_validation["lag_24h"]
        )
        combined = engineer_features(frame, "sensor_assisted_nowcasting")
        self.assertEqual(combined.loc[2, "lag_24h"], 10.0)
        self.assertEqual(combined.loc[3, "lag_24h"], 100.0)
        no_history = engineer_features(frame.iloc[[2, 3]], "sensor_assisted_nowcasting")
        self.assertTrue(no_history["lag_24h"].isna().all())

    def test_satellite_variant_excludes_ground_target_features(self):
        frame = pd.DataFrame(
            [{"AOD_047": 0.2, "temperature": 20, "relative_humidity": 50,
              "wind_speed": 2, "pblh": 300, "latitude": 31, "longitude": 74,
              "pm25": 100, "NO2_density": 0.5}]
        )
        features = engineer_features(frame, "unmonitored_satellite_downscaling")
        self.assertEqual(tuple(features.columns), MODEL_VARIANTS["unmonitored_satellite_downscaling"])
        self.assertNotIn("pm25", features.columns)
        self.assertNotIn("NO2_density", features.columns)

    def test_imputer_is_a_fitted_pipeline_step(self):
        model = make_model_pipeline(DummyRegressor(strategy="mean"))
        train = pd.DataFrame({"feature": [1.0, np.nan, 3.0]})
        model.fit(train, [1.0, 2.0, 3.0])
        self.assertEqual(model.named_steps["imputer"].statistics_[0], 2.0)
        self.assertEqual(model.predict(pd.DataFrame({"feature": [100.0]})).shape, (1,))

    def test_satellite_inference_rejects_missing_environmental_support(self):
        class FixedModel:
            def predict(self, values):
                return np.zeros(len(values))

        frame = pd.DataFrame(
            [{"AOD_047": np.nan, "latitude": 31.0, "longitude": 74.0,
              "temperature": np.nan, "relative_humidity": np.nan,
              "wind_speed": np.nan, "pblh": np.nan}]
        )
        artifact = {
            "model_variant": "unmonitored_satellite_downscaling",
            "models": {"unmonitored_satellite_downscaling": FixedModel()},
            "model": FixedModel(),
        }
        with self.assertRaisesRegex(ValueError, "per-row AOD"):
            predict_with_uncertainty(artifact, frame)

    def test_training_reports_both_spatial_evaluation_variants(self):
        rows = []
        for city_index, city in enumerate(("Lahore", "Karachi", "Islamabad")):
            for hour in range(4):
                rows.append(
                    {
                        "city": city,
                        "timestamp": f"2026-01-0{hour + 1}T00:00:00Z",
                        "pm25": float(20 + city_index * 10 + hour),
                        "AOD_047": 0.2 + hour / 100,
                        "NO2_density": 0.1,
                        "temperature": 20.0,
                        "relative_humidity": 50.0,
                        "wind_speed": 2.0,
                        "pblh": 300.0,
                        "latitude": 25.0 + city_index * 3,
                        "longitude": 67.0 + city_index * 3,
                    }
                )
        models, metrics = train_dual_models(
            pd.DataFrame(rows), estimator_factory=lambda _: DummyRegressor(strategy="mean")
        )
        self.assertEqual(set(models), set(MODEL_VARIANTS))
        self.assertEqual(set(metrics), set(MODEL_VARIANTS))
        for variant_metrics in metrics.values():
            self.assertIn("pm25_exceedance_recall", variant_metrics)
            self.assertIn("pm25_exceedance_threshold_ug_m3", variant_metrics)


if __name__ == "__main__":
    unittest.main()
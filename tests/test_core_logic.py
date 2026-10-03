import unittest

from app import demo_data
from clinical_api_engine import assess_patient_risk
from paksat.air_quality import (
    filter_observation_period,
    prepare_observation_upload,
    summarize_city_pm25,
)
from triage_engine import assess_clinical_risk
from train_pai_model import MODEL_COLUMNS, predict_with_uncertainty


class CoreLogicTests(unittest.TestCase):
    def test_uploaded_observations_normalize_timestamps_and_exclude_invalid_rows(self):
        import pandas as pd

        source = pd.DataFrame(
            {
                "city": [" Lahore ", "Karachi", "", "Islamabad", "Islamabad"],
                "timestamp": [
                    "2026-09-01T00:00:00Z",
                    "2026-09-01T01:00:00+00:00",
                    "2026-09-01T02:00:00Z",
                    "bad timestamp",
                    "2026-09-01T03:00:00Z",
                ],
                "pm25": [25.0, 40.0, 10.0, 30.0, -1.0],
            }
        )

        observations, excluded = prepare_observation_upload(source)

        self.assertEqual(len(observations), 2)
        self.assertEqual(excluded, 3)
        self.assertEqual(observations.loc[0, "city"], "Lahore")
        self.assertEqual(str(observations["timestamp"].dt.tz), "UTC")
        self.assertTrue(observations["record_source"].eq("Uploaded historical data").all())

    def test_uploaded_observations_reject_missing_schema(self):
        import pandas as pd

        with self.assertRaisesRegex(ValueError, "missing required columns"):
            prepare_observation_upload(pd.DataFrame({"city": ["Lahore"], "pm25": [20]}))

    def test_observation_date_filter_is_inclusive_and_utc_normalized(self):
        import pandas as pd

        frame = pd.DataFrame(
            {
                "timestamp": [
                    "2026-09-01T23:30:00-05:00",
                    "2026-09-02T12:00:00Z",
                    "2026-09-03T00:00:00Z",
                ],
                "pm25": [10.0, 20.0, 30.0],
            }
        )

        filtered = filter_observation_period(
            frame, pd.Timestamp("2026-09-02").date(), pd.Timestamp("2026-09-02").date()
        )

        self.assertEqual(filtered["pm25"].tolist(), [10.0, 20.0])
        self.assertTrue(filtered["timestamp"].dt.date.eq(pd.Timestamp("2026-09-02").date()).all())
        with self.assertRaisesRegex(ValueError, "start must be on or before"):
            filter_observation_period(
                frame, pd.Timestamp("2026-09-03").date(), pd.Timestamp("2026-09-02").date()
            )

    def test_urgent_symptoms_escalate_risk(self):
        result = assess_patient_risk(18.0, 24.0, ["Blue lips"], {"age": 22, "asthma": True})
        self.assertTrue(result["urgent"])
        self.assertEqual(result["severity"], "Environmental Acute Exacerbation")

    def test_triage_rejects_non_finite_or_negative_pm25(self):
        for point in (float("nan"), float("inf"), -1.0):
            with self.subTest(point=point), self.assertRaisesRegex(ValueError, "finite and non-negative"):
                assess_patient_risk(point, 10.0, [])

    def test_demo_data_includes_model_features(self):
        frame = demo_data()
        required = {"city", "latitude", "longitude", "pm25", "AOD_047", "NO2_density", "temperature", "relative_humidity", "wind_speed", "pblh", "atmospheric_stagnation_index", "thermal_confinement_ratio", "photochemical_pm25_proxy", "hygroscopic_growth_factor", "lag_24h", "lag_48h"}
        self.assertTrue(required.issubset(frame.columns))

    def test_demo_data_has_sufficient_24h_hour_coverage(self):
        frame = demo_data()
        lahore = summarize_city_pm25(frame, "Lahore")
        self.assertGreaterEqual(lahore.trailing_24h_hour_count, 18)
        self.assertIsNotNone(lahore.trailing_24h_mean)

    def test_exposure_summary_uses_latest_timestamp_and_coverage_checked_mean(self):
        import pandas as pd

        timestamps = pd.date_range("2026-10-01T00:00:00Z", periods=24, freq="h")
        frame = pd.DataFrame(
            {
                "city": ["Lahore"] * 25,
                "timestamp": [*timestamps, timestamps[-1]],
                "pm25": [10.0] * 23 + [50.0, 70.0],
                "sensor_id": list(range(24)) + [100],
            }
        )

        summary = summarize_city_pm25(frame, "Lahore")

        self.assertEqual(summary.latest_pm25, 60.0)
        self.assertEqual(summary.latest_timestamp, timestamps[-1])
        self.assertEqual(summary.trailing_24h_hour_count, 24)
        self.assertAlmostEqual(summary.trailing_24h_mean, (23 * 10 + 60) / 24)
        self.assertEqual(summary.latest_sensor_count, 2)

    def test_exposure_summary_withholds_mean_when_hourly_coverage_is_sparse(self):
        import pandas as pd

        frame = pd.DataFrame(
            {
                "city": ["Karachi", "Karachi"],
                "timestamp": ["2026-10-01T00:00:00Z", "2026-10-01T23:00:00Z"],
                "pm25": [30.0, 90.0],
            }
        )

        summary = summarize_city_pm25(frame, "Karachi")

        self.assertEqual(summary.latest_pm25, 90.0)
        self.assertEqual(summary.trailing_24h_hour_count, 2)
        self.assertIsNone(summary.trailing_24h_mean)

    def test_point_only_model_does_not_claim_predictive_interval(self):
        import numpy as np
        import pandas as pd

        class FixedModel:
            def predict(self, values):
                return np.full(len(values), 42.0)

        frame = pd.DataFrame([{column: 1.0 for column in MODEL_COLUMNS}])
        result = predict_with_uncertainty(
            {"model": FixedModel(), "feature_columns": list(MODEL_COLUMNS)}, frame
        )
        self.assertEqual(result.loc[0, "pm25_point"], 42.0)
        self.assertTrue(pd.isna(result.loc[0, "pm25_lower"]))
        self.assertTrue(pd.isna(result.loc[0, "pm25_upper"]))

    def test_model_inference_rejects_invalid_predictions(self):
        import numpy as np
        import pandas as pd

        class InvalidModel:
            def predict(self, values):
                return np.full(len(values), np.nan)

        frame = pd.DataFrame([{column: 1.0 for column in MODEL_COLUMNS}])
        with self.assertRaisesRegex(ValueError, "non-finite or negative"):
            predict_with_uncertainty(
                {"model": InvalidModel(), "feature_columns": list(MODEL_COLUMNS)}, frame
            )

    def test_model_inference_clamps_negative_quantile_bounds(self):
        import numpy as np
        import pandas as pd

        class FixedModel:
            def __init__(self, value):
                self.value = value

            def predict(self, values):
                return np.full(len(values), self.value)

        frame = pd.DataFrame([{column: 1.0 for column in MODEL_COLUMNS}])
        result = predict_with_uncertainty(
            {
                "model": FixedModel(5.0),
                "models": {"0.05": FixedModel(-4.0), "0.5": FixedModel(5.0), "0.95": FixedModel(9.0)},
                "feature_columns": list(MODEL_COLUMNS),
            },
            frame,
        )
        self.assertEqual(result.loc[0, "pm25_lower"], 0.0)
        self.assertEqual(result.loc[0, "pm25_point"], 5.0)

    def test_triage_engine_flags_urgent_symptoms(self):
        result = assess_clinical_risk(18.0, ["Severe breathing difficulty"])
        self.assertTrue(result["urgent"])
        self.assertEqual(result["severity"], "Environmental Acute Exacerbation")


if __name__ == "__main__":
    unittest.main()

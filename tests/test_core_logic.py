import unittest

from app import demo_data
from clinical_api_engine import assess_patient_risk
from triage_engine import assess_clinical_risk
from train_pai_model import MODEL_COLUMNS, predict_with_uncertainty


class CoreLogicTests(unittest.TestCase):
    def test_urgent_symptoms_escalate_risk(self):
        result = assess_patient_risk(18.0, 24.0, ["Blue lips"], {"age": 22, "asthma": True})
        self.assertTrue(result["urgent"])
        self.assertEqual(result["severity"], "Environmental Acute Exacerbation")

    def test_demo_data_includes_model_features(self):
        frame = demo_data()
        required = {"city", "latitude", "longitude", "pm25", "AOD_047", "NO2_density", "temperature", "relative_humidity", "wind_speed", "pblh", "atmospheric_stagnation_index", "thermal_confinement_ratio", "photochemical_pm25_proxy", "hygroscopic_growth_factor", "lag_24h", "lag_48h"}
        self.assertTrue(required.issubset(frame.columns))

    def test_point_only_model_does_not_claim_predictive_interval(self):
        import numpy as np
        import pandas as pd

        class FixedModel:
            def predict(self, values):
                return np.full(len(values), 42.0)

        frame = pd.DataFrame([{column: 1.0 for column in MODEL_COLUMNS}])
        result = predict_with_uncertainty({"model": FixedModel()}, frame)
        self.assertEqual(result.loc[0, "pm25_point"], 42.0)
        self.assertTrue(pd.isna(result.loc[0, "pm25_lower"]))
        self.assertTrue(pd.isna(result.loc[0, "pm25_upper"]))

    def test_triage_engine_flags_urgent_symptoms(self):
        result = assess_clinical_risk(18.0, ["Severe breathing difficulty"])
        self.assertTrue(result["urgent"])
        self.assertEqual(result["severity"], "Environmental Acute Exacerbation")


if __name__ == "__main__":
    unittest.main()

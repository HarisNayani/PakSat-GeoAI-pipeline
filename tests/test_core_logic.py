import unittest

from app import demo_data
from clinical_api_engine import assess_patient_risk
from triage_engine import assess_clinical_risk


class CoreLogicTests(unittest.TestCase):
    def test_urgent_symptoms_escalate_risk(self):
        result = assess_patient_risk(18.0, 24.0, ["Blue lips"], {"age": 22, "asthma": True})
        self.assertTrue(result["urgent"])
        self.assertEqual(result["severity"], "Environmental Acute Exacerbation")

    def test_demo_data_includes_model_features(self):
        frame = demo_data()
        required = {"city", "latitude", "longitude", "pm25", "AOD_047", "NO2_density", "temperature", "relative_humidity", "wind_speed", "pblh", "atmospheric_stagnation_index", "thermal_confinement_ratio", "photochemical_pm25_proxy", "hygroscopic_growth_factor", "lag_24h", "lag_48h"}
        self.assertTrue(required.issubset(frame.columns))

    def test_triage_engine_flags_urgent_symptoms(self):
        result = assess_clinical_risk(18.0, ["Severe breathing difficulty"])
        self.assertTrue(result["urgent"])
        self.assertEqual(result["severity"], "Environmental Acute Exacerbation")


if __name__ == "__main__":
    unittest.main()

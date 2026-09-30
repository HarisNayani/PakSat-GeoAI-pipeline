import math
import unittest

from paksat.telemetry_normalization import normalize_earth_engine_features


class TelemetryNormalizationTests(unittest.TestCase):
    def test_converts_satellite_weather_units_and_wind_components(self):
        normalized = normalize_earth_engine_features(
            {
                "AOD_047": 920.0,
                "temperature_kelvin": 303.15,
                "dewpoint_kelvin": 297.15,
                "wind_u_mps": 3.0,
                "wind_v_mps": 4.0,
                "pblh": 450.0,
                "NO2_density": 0.0001,
            }
        )
        self.assertAlmostEqual(normalized["AOD_047"], 0.92)
        self.assertAlmostEqual(normalized["temperature"], 30.0)
        self.assertAlmostEqual(normalized["wind_speed"], 5.0)
        self.assertAlmostEqual(normalized["wind_direction_degrees"], 216.86989765)
        self.assertGreater(normalized["relative_humidity"], 0.0)
        self.assertLess(normalized["relative_humidity"], 1.0)

    def test_missing_wind_inputs_do_not_create_fake_direction(self):
        normalized = normalize_earth_engine_features({"wind_u_mps": math.nan})
        self.assertIsNone(normalized["wind_speed"])
        self.assertIsNone(normalized["wind_direction_degrees"])


if __name__ == "__main__":
    unittest.main()
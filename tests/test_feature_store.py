import unittest

import pandas as pd

from enhanced_feature_store import add_spatiotemporal_lags, prepare_features


class FeatureStoreTests(unittest.TestCase):
    def test_pm25_lags_exclude_current_target_and_isolate_cities(self):
        frame = pd.DataFrame(
            [
                {"city": "Lahore", "timestamp": "2026-01-01T00:00:00Z", "pm25": 10.0},
                {"city": "Karachi", "timestamp": "2026-01-01T00:00:00Z", "pm25": 100.0},
                {"city": "Lahore", "timestamp": "2026-01-01T12:00:00Z", "pm25": 30.0},
                {"city": "Karachi", "timestamp": "2026-01-01T12:00:00Z", "pm25": 300.0},
            ]
        )
        result = add_spatiotemporal_lags(frame)
        self.assertTrue(pd.isna(result.loc[0, "lag_24h"]))
        self.assertTrue(pd.isna(result.loc[1, "lag_24h"]))
        self.assertEqual(result.loc[2, "lag_24h"], 10.0)
        self.assertEqual(result.loc[3, "lag_24h"], 100.0)

    def test_feature_preparation_does_not_fill_history_from_future_rows(self):
        rows = []
        for city, first, second in (("Lahore", 10.0, 30.0), ("Karachi", 100.0, 300.0)):
            for timestamp, pm25 in (
                ("2026-01-01T00:00:00Z", first),
                ("2026-01-01T12:00:00Z", second),
            ):
                rows.append(
                    {
                        "city": city,
                        "timestamp": timestamp,
                        "pm25": pm25,
                        "wind_speed": 2.0,
                        "pblh": 400.0,
                        "temperature": 25.0,
                        "relative_humidity": 0.5,
                        "NO2_density": 0.1,
                        "AOD_047": 0.2,
                    }
                )
        features = prepare_features(pd.DataFrame(rows))
        self.assertTrue(pd.isna(features.loc[0, "lag_24h"]))
        self.assertEqual(features.loc[1, "lag_24h"], 10.0)
        self.assertTrue(pd.isna(features.loc[2, "lag_24h"]))
        self.assertEqual(features.loc[3, "lag_24h"], 100.0)


if __name__ == "__main__":
    unittest.main()
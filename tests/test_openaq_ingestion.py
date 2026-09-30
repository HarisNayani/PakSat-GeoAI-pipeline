import unittest
from datetime import date
from unittest.mock import Mock, patch

from data_ingestion import CityRegion, fetch_openaq_measurements


class OpenAQIngestionTests(unittest.TestCase):
    def test_requires_api_key(self):
        with patch("data_ingestion.requests.get") as get:
            result = fetch_openaq_measurements(
                CityRegion("Lahore", 31.5, 74.3), date(2026, 9, 1), date(2026, 9, 2)
            )
        self.assertTrue(result.empty)
        get.assert_not_called()

    def test_discovers_pm25_sensors_and_fetches_hourly_rows(self):
        locations_response = Mock()
        locations_response.json.return_value = {
            "results": [
                {
                    "id": 10,
                    "coordinates": {"latitude": 31.6, "longitude": 74.4},
                    "sensors": [
                        {"id": 20, "parameter": {"name": "pm25"}},
                        {"id": 21, "parameter": {"name": "no2"}},
                    ],
                }
            ]
        }
        hourly_response = Mock()
        hourly_response.json.return_value = {
            "results": [
                {
                    "value": 48.5,
                    "period": {"datetimeTo": {"utc": "2026-09-02T10:00:00Z"}},
                    "coordinates": None,
                }
            ]
        }
        with patch("data_ingestion.requests.get", side_effect=[locations_response, hourly_response]) as get:
            result = fetch_openaq_measurements(
                CityRegion("Lahore", 31.5, 74.3, radius_m=50000),
                date(2026, 9, 1),
                date(2026, 9, 2),
                api_key="test-key",
            )

        self.assertEqual(len(result), 1)
        self.assertEqual(result.loc[0, "pm25"], 48.5)
        self.assertEqual(result.loc[0, "latitude"], 31.6)
        self.assertEqual(result.loc[0, "timestamp"], "2026-09-02T10:00:00Z")
        self.assertEqual(get.call_count, 2)
        self.assertEqual(get.call_args_list[0].kwargs["params"]["radius"], 25000)
        self.assertEqual(get.call_args_list[0].kwargs["params"]["parameters_id"], 2)
        self.assertIn("/sensors/20/hours", get.call_args_list[1].args[0])


if __name__ == "__main__":
    unittest.main()
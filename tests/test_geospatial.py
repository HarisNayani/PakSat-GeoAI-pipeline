import unittest

import pandas as pd

from paksat.geospatial import _wind_paths, prepare_pm25_map_data


class GeospatialTests(unittest.TestCase):
    def test_missing_live_wind_components_are_skipped(self):
        frame = pd.DataFrame(
            [{"latitude": 31.5, "longitude": 74.3, "wind_u_mps": None, "wind_v_mps": None}]
        )
        self.assertEqual(_wind_paths(frame), [])

    def test_valid_wind_components_create_a_vector(self):
        frame = pd.DataFrame(
            [{"latitude": 31.5, "longitude": 74.3, "wind_u_mps": 3.0, "wind_v_mps": 4.0}]
        )
        self.assertEqual(len(_wind_paths(frame)), 1)

    def test_map_data_filters_out_of_bounds_and_aggregates_to_a_cap(self):
        frame = pd.DataFrame(
            [
                {"latitude": 31.50, "longitude": 74.30, "pm25": 10.0},
                {"latitude": 31.51, "longitude": 74.31, "pm25": 30.0},
                {"latitude": 40.0, "longitude": 74.3, "pm25": 100.0},
            ]
        )
        result = prepare_pm25_map_data(frame, grid_degrees=0.1, max_points=1)
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result.loc[0, "pm25"], 20.0)




if __name__ == "__main__":
    unittest.main()
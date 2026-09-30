import unittest

import pandas as pd

from paksat.geospatial import _wind_paths


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


if __name__ == "__main__":
    unittest.main()
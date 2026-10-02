import unittest
from importlib.util import find_spec

import numpy as np
import pandas as pd

from paksat.causal_policy import (
    CausalPolicySimulator,
    PANEL_COLUMNS,
    build_policy_dag,
    simulate_policy_impact,
)
from paksat.resource_optimizer import optimize_hospital_resources


class EnterpriseModuleTests(unittest.TestCase):
    @staticmethod
    def make_policy_panel(rows=10):
        panel = pd.DataFrame({column: [0.5] * rows for column in PANEL_COLUMNS})
        panel.insert(0, "city", ["Lahore"] * rows)
        panel.insert(1, "timestamp", pd.date_range("2026-01-01", periods=rows, freq="D", tz="UTC"))
        return panel

    def test_policy_dag_contains_mediators_and_outcomes(self):
        graph = build_policy_dag("brick_kiln_shutdown_intensity")
        self.assertIn("brick_kiln_shutdown_intensity -> ground_emissions", graph)
        self.assertIn("stagnation_index -> pm25", graph)
        self.assertIn("pblh -> brick_kiln_shutdown_intensity", graph)
        self.assertIn("wind_speed -> pm25", graph)
        self.assertNotIn("brick_kiln_shutdown_intensity -> pblh", graph)
        self.assertIn("pm25 -> respiratory_admissions", graph)

    def test_policy_simulation_requires_configured_panel(self):
        with self.assertRaisesRegex(RuntimeError, "No policy panel is configured"):
            simulate_policy_impact("brick_kiln_shutdown", 0.5)

    def test_policy_simulation_rejects_invalid_policy_and_intensity(self):
        panel = self.make_policy_panel()
        panel["pm25"] = 30.0
        panel["respiratory_admissions"] = 5.0
        simulator = CausalPolicySimulator(panel)
        with self.assertRaisesRegex(ValueError, "Unsupported policy"):
            simulator.simulate_policy_impact("unknown", 0.5)
        with self.assertRaisesRegex(ValueError, "intensity"):
            simulator.simulate_policy_impact("brick_kiln_shutdown", 1.1)
        with self.assertRaisesRegex(ValueError, "Unsupported policy"):
            build_policy_dag("unknown")

    def test_policy_simulation_checks_positivity_support(self):
        panel = self.make_policy_panel()
        panel.loc[:, list(PANEL_COLUMNS)] = 0.0
        panel["brick_kiln_shutdown_intensity"] = np.linspace(0.2, 0.6, 10)
        panel["pm25"] = np.linspace(20.0, 50.0, 10)
        panel["respiratory_admissions"] = np.linspace(2.0, 8.0, 10)
        simulator = CausalPolicySimulator(panel)
        with self.assertRaisesRegex(ValueError, "outside observed treatment support"):
            simulator.simulate_policy_impact("brick_kiln_shutdown", 0.9)

    def test_policy_panel_requires_valid_unique_city_timestamp_keys(self):
        panel = self.make_policy_panel()
        with self.assertRaisesRegex(ValueError, "missing required columns"):
            CausalPolicySimulator(panel.drop(columns=["timestamp"]))

        panel.loc[1, "timestamp"] = panel.loc[0, "timestamp"]
        with self.assertRaisesRegex(ValueError, "one complete observation per city and timestamp"):
            CausalPolicySimulator(panel)

        panel = self.make_policy_panel()
        panel["timestamp"] = panel["timestamp"].astype("string")
        panel.loc[0, "timestamp"] = "not-a-timestamp"
        with self.assertRaisesRegex(ValueError, "keys must be present and valid"):
            CausalPolicySimulator(panel)

    def test_policy_panel_rejects_negative_health_outcomes(self):
        panel = self.make_policy_panel()
        panel["respiratory_admissions"] = 2.0
        panel.loc[0, "pm25"] = -1.0
        with self.assertRaisesRegex(ValueError, "outcomes must be non-negative"):
            CausalPolicySimulator(panel)

    @unittest.skipUnless(find_spec("dowhy"), "DoWhy is not installed in this interpreter.")
    def test_policy_estimate_surfaces_rank_deficiency_diagnostics(self):
        row_index = np.arange(40)
        panel = pd.DataFrame({column: np.zeros(40) for column in PANEL_COLUMNS})
        panel.insert(0, "city", ["Lahore", "Karachi"] * 20)
        panel.insert(1, "timestamp", pd.date_range("2026-01-01", periods=40, freq="D", tz="UTC"))
        panel["brick_kiln_shutdown_intensity"] = (row_index % 5) / 4
        panel["crop_burning_penalties_intensity"] = ((row_index * 2) % 5) / 4
        panel["heavy_vehicle_restriction_intensity"] = ((row_index * 3) % 5) / 4
        panel["pblh"] = 300 + row_index * 2
        panel["stagnation_index"] = 2 + row_index % 7
        panel["wind_speed"] = 1 + row_index % 4
        panel["ground_emissions"] = (
            50 - 10 * panel["brick_kiln_shutdown_intensity"]
            - 8 * panel["crop_burning_penalties_intensity"] + row_index % 3
        )
        panel["pm25"] = (
            80 - 20 * panel["brick_kiln_shutdown_intensity"]
            - 15 * panel["crop_burning_penalties_intensity"]
            - 8 * panel["heavy_vehicle_restriction_intensity"]
            + 2 * panel["stagnation_index"] + panel["wind_speed"]
        )
        panel["respiratory_admissions"] = 8 + 0.1 * panel["pm25"] + row_index % 3
        panel["policy_context"] = row_index % 2

        result = CausalPolicySimulator(panel).simulate_policy_impact(
            "brick_kiln_shutdown", 0.5
        )

        self.assertTrue(any("rank-deficient" in item for item in result.get("diagnostics", [])))

    def test_resource_optimizer_respects_supply_and_risk_priority(self):
        result = optimize_hospital_resources(
            {
                "Mayo Hospital Lahore": {
                    "risk_weight": 2.0,
                    "emergency_beds": 5,
                    "oxygen_cylinders": 0,
                    "nebulizer_stations": 0,
                },
                "JPMC Karachi": {
                    "risk_weight": 1.0,
                    "emergency_beds": 5,
                    "oxygen_cylinders": 0,
                    "nebulizer_stations": 0,
                },
            },
            {"emergency_beds": 5},
        )
        self.assertAlmostEqual(result["allocation"]["Mayo Hospital Lahore"]["emergency_beds"], 5)
        self.assertAlmostEqual(result["allocation"]["JPMC Karachi"]["emergency_beds"], 0)
        self.assertAlmostEqual(result["resource_totals_allocated"]["emergency_beds"], 5)
        self.assertIs(type(result["allocation"]["Mayo Hospital Lahore"]["emergency_beds"]), int)

    def test_resource_optimizer_rejects_fractional_counts(self):
        with self.assertRaisesRegex(ValueError, "integer count"):
            optimize_hospital_resources(
                {"Hospital": {"emergency_beds": 1.5}},
                {"emergency_beds": 1},
            )


if __name__ == "__main__":
    unittest.main()
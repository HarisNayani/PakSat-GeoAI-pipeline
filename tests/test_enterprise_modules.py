import unittest

import pandas as pd

from paksat.causal_policy import (
    CausalPolicySimulator,
    PANEL_COLUMNS,
    build_policy_dag,
    simulate_policy_impact,
)
from paksat.resource_optimizer import optimize_hospital_resources


class EnterpriseModuleTests(unittest.TestCase):
    def test_policy_dag_contains_mediators_and_outcomes(self):
        graph = build_policy_dag("brick_kiln_shutdown_intensity")
        self.assertIn("brick_kiln_shutdown_intensity -> ground_emissions", graph)
        self.assertIn("stagnation_index -> pm25", graph)
        self.assertIn("pm25 -> respiratory_admissions", graph)

    def test_policy_simulation_requires_configured_panel(self):
        with self.assertRaisesRegex(RuntimeError, "No policy panel is configured"):
            simulate_policy_impact("brick_kiln_shutdown", 0.5)

    def test_policy_simulation_rejects_invalid_policy_and_intensity(self):
        panel = pd.DataFrame({column: [0.5] * 10 for column in PANEL_COLUMNS})
        panel["pm25"] = 30.0
        panel["respiratory_admissions"] = 5.0
        simulator = CausalPolicySimulator(panel)
        with self.assertRaisesRegex(ValueError, "Unsupported policy"):
            simulator.simulate_policy_impact("unknown", 0.5)
        with self.assertRaisesRegex(ValueError, "intensity"):
            simulator.simulate_policy_impact("brick_kiln_shutdown", 1.1)
        with self.assertRaisesRegex(ValueError, "Unsupported policy"):
            build_policy_dag("unknown")

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


if __name__ == "__main__":
    unittest.main()
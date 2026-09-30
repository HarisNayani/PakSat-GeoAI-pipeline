import unittest

from paksat.economic_impact import EconomicAssumptions, estimate_health_economic_benefit


class EconomicImpactTests(unittest.TestCase):
    def test_estimates_per_observation_value_from_explicit_assumptions(self):
        assumptions = EconomicAssumptions(
            medical_cost_per_admission_pkr=12_000,
            workdays_lost_per_admission=3,
            value_per_workday_pkr=2_500,
            assumption_source="Example local hospital costing study",
            reporting_period="one city-day record",
        )
        result = estimate_health_economic_benefit(2.5, assumptions)
        self.assertEqual(result["direct_medical_savings_pkr"], 30_000)
        self.assertEqual(result["productivity_value_pkr"], 18_750)
        self.assertEqual(result["total_scenario_value_pkr"], 48_750)
        self.assertIn("not annualized", result["disclaimer"])

    def test_requires_sourced_assumptions_and_nonnegative_effects(self):
        with self.assertRaisesRegex(ValueError, "source"):
            EconomicAssumptions(0, 0, 0, " ", "one day")
        assumptions = EconomicAssumptions(1, 1, 1, "source", "one day")
        with self.assertRaisesRegex(ValueError, "non-negative"):
            estimate_health_economic_benefit(-1, assumptions)


if __name__ == "__main__":
    unittest.main()
"""Transparent translation of health effects into economic scenarios."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any


@dataclass(frozen=True)
class EconomicAssumptions:
    """Locally supplied valuation inputs; no national defaults are assumed."""

    medical_cost_per_admission_pkr: float
    workdays_lost_per_admission: float
    value_per_workday_pkr: float
    assumption_source: str
    reporting_period: str

    def __post_init__(self) -> None:
        values = {
            "medical_cost_per_admission_pkr": self.medical_cost_per_admission_pkr,
            "workdays_lost_per_admission": self.workdays_lost_per_admission,
            "value_per_workday_pkr": self.value_per_workday_pkr,
        }
        for name, raw_value in values.items():
            value = float(raw_value)
            if not isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative.")
        if not self.assumption_source.strip():
            raise ValueError("Provide a source for the economic assumptions.")
        if not self.reporting_period.strip():
            raise ValueError("Describe the outcome panel's reporting period.")


def estimate_health_economic_benefit(
    admissions_avoided_per_observation: float,
    assumptions: EconomicAssumptions,
) -> dict[str, Any]:
    """Estimate direct and productivity value per observational unit.

    This is an assumption-driven scenario, not a national loss estimate. The
    admissions effect must be expressed per row/unit in the uploaded causal
    panel. The function deliberately does not annualize or extrapolate it.
    """
    admissions_avoided = float(admissions_avoided_per_observation)
    if not isfinite(admissions_avoided) or admissions_avoided < 0:
        raise ValueError("Admissions avoided must be finite and non-negative.")

    direct_savings = admissions_avoided * assumptions.medical_cost_per_admission_pkr
    productivity_savings = (
        admissions_avoided
        * assumptions.workdays_lost_per_admission
        * assumptions.value_per_workday_pkr
    )
    return {
        "admissions_avoided_per_observation": admissions_avoided,
        "direct_medical_savings_pkr": direct_savings,
        "productivity_value_pkr": productivity_savings,
        "total_scenario_value_pkr": direct_savings + productivity_savings,
        "reporting_period": assumptions.reporting_period,
        "assumption_source": assumptions.assumption_source,
        "disclaimer": (
            "Per-observation scenario only; not annualized or nationally extrapolated. "
            "Check for overlap between medical costs and productivity assumptions."
        ),
    }


__all__ = ["EconomicAssumptions", "estimate_health_economic_benefit"]
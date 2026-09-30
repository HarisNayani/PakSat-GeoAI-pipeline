"""Linear-programming allocation of scarce respiratory-care resources."""

from __future__ import annotations

from collections.abc import Mapping
from math import isfinite
from typing import Any

RESOURCE_TYPES = ("emergency_beds", "oxygen_cylinders", "nebulizer_stations")
RESOURCE_RISK_WEIGHTS = {
    "emergency_beds": 1.0,
    "oxygen_cylinders": 1.5,
    "nebulizer_stations": 1.2,
}


def optimize_hospital_resources(
    predicted_surge_data: dict[str, dict[str, Any]],
    available_supplies: dict[str, float],
) -> dict[str, Any]:
    """Allocate supplies to minimize severity-weighted unmet resource demand.

    ``predicted_surge_data`` maps each hospital to resource demand counts and
    an optional positive ``risk_weight``. Demand may also be nested under
    ``resource_demand``. Supplies are system-wide totals keyed by resource.
    Returned counts can be fractional; round operational allocations only
    after applying local clinical and logistics rules.
    """
    if not predicted_surge_data:
        raise ValueError("At least one hospital demand record is required.")

    hospitals = list(predicted_surge_data)
    demands: dict[str, dict[str, float]] = {}
    risk_weights: dict[str, float] = {}
    for hospital in hospitals:
        record = predicted_surge_data[hospital]
        raw_demand = record.get("resource_demand", record)
        risk = float(record.get("risk_weight", 1.0))
        if not isfinite(risk) or risk <= 0:
            raise ValueError(f"Hospital risk_weight must be positive for {hospital}.")
        risk_weights[hospital] = risk
        demands[hospital] = {}
        for resource in RESOURCE_TYPES:
            amount = float(raw_demand.get(resource, 0.0))
            if not isfinite(amount) or amount < 0:
                raise ValueError(f"Demand for {resource} at {hospital} must be non-negative.")
            demands[hospital][resource] = amount

    supplies = {resource: float(available_supplies.get(resource, 0.0)) for resource in RESOURCE_TYPES}
    if any(not isfinite(amount) or amount < 0 for amount in supplies.values()):
        raise ValueError("Available supplies must be finite and non-negative.")

    try:
        from scipy.optimize import linprog
    except ImportError as error:
        raise RuntimeError("Install the `scipy` dependency to optimize hospital allocations.") from error

    variables = [(hospital, resource) for hospital in hospitals for resource in RESOURCE_TYPES]
    objective = [
        -risk_weights[hospital] * RESOURCE_RISK_WEIGHTS[resource]
        for hospital, resource in variables
    ]
    supply_constraints = []
    for resource in RESOURCE_TYPES:
        supply_constraints.append(
            [1.0 if variable_resource == resource else 0.0 for _, variable_resource in variables]
        )

    result = linprog(
        c=objective,
        A_ub=supply_constraints,
        b_ub=[supplies[resource] for resource in RESOURCE_TYPES],
        bounds=[(0.0, demands[hospital][resource]) for hospital, resource in variables],
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"Hospital resource optimization failed: {result.message}")

    allocation = {hospital: {} for hospital in hospitals}
    unmet_demand = {hospital: {} for hospital in hospitals}
    risk_score = 0.0
    for (hospital, resource), amount in zip(variables, result.x):
        allocated = float(amount)
        unmet = max(0.0, demands[hospital][resource] - allocated)
        allocation[hospital][resource] = allocated
        unmet_demand[hospital][resource] = unmet
        risk_score += unmet * risk_weights[hospital] * RESOURCE_RISK_WEIGHTS[resource]

    return {
        "status": "optimal",
        "allocation": allocation,
        "unmet_demand": unmet_demand,
        "total_unmet_critical_care_risk": risk_score,
        "resource_totals_allocated": {
            resource: sum(allocation[hospital][resource] for hospital in hospitals)
            for resource in RESOURCE_TYPES
        },
    }


__all__ = ["RESOURCE_TYPES", "optimize_hospital_resources"]
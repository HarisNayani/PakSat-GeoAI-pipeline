"""Linear-programming allocation of scarce respiratory-care resources."""

from __future__ import annotations

from collections.abc import Mapping
from math import isfinite
from typing import Any

import numpy as np

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
    Resource demands and supplies are integer counts and allocations are exact
    non-negative integers. This model does not replace local clinical rules.
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
            if not amount.is_integer():
                raise ValueError(f"Demand for {resource} at {hospital} must be an integer count.")
            demands[hospital][resource] = amount

    supplies = {resource: float(available_supplies.get(resource, 0.0)) for resource in RESOURCE_TYPES}
    if any(not isfinite(amount) or amount < 0 for amount in supplies.values()):
        raise ValueError("Available supplies must be finite and non-negative.")
    if any(not amount.is_integer() for amount in supplies.values()):
        raise ValueError("Available supplies must be integer counts.")

    try:
        from scipy.optimize import Bounds, LinearConstraint, milp
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

    demand_bounds = np.asarray(
        [demands[hospital][resource] for hospital, resource in variables], dtype=float
    )
    # Every allocation decision is integral; supply rows impose one cap per resource.
    result = milp(
        c=np.asarray(objective, dtype=float),
        integrality=np.ones(len(variables), dtype=int),
        bounds=Bounds(np.zeros(len(variables)), demand_bounds),
        constraints=LinearConstraint(
            np.asarray(supply_constraints, dtype=float),
            -np.inf,
            np.asarray([supplies[resource] for resource in RESOURCE_TYPES], dtype=float),
        ),
        options={"mip_rel_gap": 0.0},
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"Hospital resource optimization failed: {result.message}")

    allocation = {hospital: {} for hospital in hospitals}
    unmet_demand = {hospital: {} for hospital in hospitals}
    risk_score = 0.0
    for (hospital, resource), amount in zip(variables, result.x):
        allocated = int(round(float(amount)))
        unmet = int(demands[hospital][resource] - allocated)
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
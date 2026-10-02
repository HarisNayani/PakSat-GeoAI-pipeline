"""DoWhy-based policy effect estimation for environmental health outcomes.

The simulator requires an observational policy panel. Estimates are only as
credible as its treatment definitions, confounder coverage, and time alignment.
"""

from __future__ import annotations

import logging
from math import isfinite
from typing import Any
import warnings

import numpy as np
import pandas as pd

POLICY_TREATMENTS = {
    "brick_kiln_shutdown": "brick_kiln_shutdown_intensity",
    "crop_burning_penalties": "crop_burning_penalties_intensity",
    "heavy_vehicle_restriction": "heavy_vehicle_restriction_intensity",
}
PANEL_ALIGNMENT_COLUMNS = ("city", "timestamp")

PANEL_COLUMNS = {
    *POLICY_TREATMENTS.values(),
    "ground_emissions",
    "pblh",
    "stagnation_index",
    "wind_speed",
    "pm25",
    "respiratory_admissions",
    "policy_context",
}
LOGGER = logging.getLogger(__name__)
SCENARIO_LABEL = "Observational Counterfactual Scenario"
METEOROLOGICAL_CONFOUNDERS = ("pblh", "stagnation_index", "wind_speed")


def build_policy_dag(treatment_column: str) -> str:
    """Return a DOT DAG with meteorology as exogenous confounders."""
    if treatment_column not in POLICY_TREATMENTS.values():
        raise ValueError(f"Unsupported policy treatment column: {treatment_column}")
    policy_edges = "\n".join(
        f"        {policy} -> ground_emissions;"
        for policy in POLICY_TREATMENTS.values()
    )
    context_edges = "\n".join(
        f"        policy_context -> {policy};" for policy in POLICY_TREATMENTS.values()
    )
    meteorology_edges = "\n".join(
        f"        {weather} -> {policy};"
        for weather in METEOROLOGICAL_CONFOUNDERS
        for policy in POLICY_TREATMENTS.values()
    )
    meteorology_outcomes = "\n".join(
        f"        {weather} -> {outcome};"
        for weather in METEOROLOGICAL_CONFOUNDERS
        for outcome in ("pm25", "respiratory_admissions")
    )
    return f"""
    digraph {{
{policy_edges}
        ground_emissions -> pm25;
        pm25 -> respiratory_admissions;
{context_edges}
        policy_context -> pm25;
        policy_context -> respiratory_admissions;
{meteorology_edges}
{meteorology_outcomes}
    }}
    """


class CausalPolicySimulator:
    """Estimate policy dose effects from an aligned observational panel."""

    def __init__(self, panel: pd.DataFrame) -> None:
        required_columns = PANEL_COLUMNS.union(PANEL_ALIGNMENT_COLUMNS)
        missing = required_columns.difference(panel.columns)
        if missing:
            raise ValueError(f"Policy panel is missing required columns: {sorted(missing)}")
        cities = panel["city"].astype("string").str.strip()
        timestamps = pd.to_datetime(
            panel["timestamp"], format="mixed", errors="coerce", utc=True
        )
        if cities.isna().any() or cities.eq("").any() or timestamps.isna().any():
            raise ValueError("Policy panel city and timestamp keys must be present and valid.")
        numeric = panel.loc[:, sorted(PANEL_COLUMNS)].apply(pd.to_numeric, errors="coerce")
        numeric = numeric.replace([float("inf"), float("-inf")], pd.NA)
        complete_rows = numeric.notna().all(axis=1)
        alignment = pd.DataFrame(
            {"city": cities, "timestamp": timestamps}, index=panel.index
        )
        if alignment.loc[complete_rows].duplicated().any():
            raise ValueError("Policy panel must have at most one complete observation per city and timestamp.")
        numeric = numeric.loc[complete_rows]
        if len(numeric) < 10:
            raise ValueError("At least 10 complete policy-panel rows are required.")
        if (numeric[["pm25", "respiratory_admissions"]] < 0).any().any():
            raise ValueError("PM2.5 and respiratory admissions outcomes must be non-negative.")
        if (numeric[list(POLICY_TREATMENTS.values())] < 0).any().any() or (
            numeric[list(POLICY_TREATMENTS.values())] > 1
        ).any().any():
            raise ValueError("Policy intensity columns must be normalized to the range [0, 1].")
        self._panel = numeric

    def simulate_policy_impact(self, policy_type: str, intensity: float) -> dict[str, Any]:
        """Estimate observational counterfactual scenarios relative to zero dose.

        The outcome units match the supplied panel (PM2.5 concentration and
        admissions per observation). Do not interpret estimates as causal
        without validating consistency, positivity, and confounder coverage.
        """
        treatment_column = POLICY_TREATMENTS.get(policy_type)
        if treatment_column is None:
            raise ValueError(f"Unsupported policy type: {policy_type}")
        dose = float(intensity)
        if not isfinite(dose) or not 0.0 <= dose <= 1.0:
            raise ValueError("Policy intensity must be a finite value in [0, 1].")
        if dose == 0.0:
            result = self._result(policy_type, dose, 0.0, 0.0)
            LOGGER.info("%s: %s", SCENARIO_LABEL, result)
            return result

        self._check_treatment_support(treatment_column, dose)

        try:
            from dowhy import CausalModel
        except ImportError as error:
            raise RuntimeError("Install the `dowhy` dependency to estimate policy effects.") from error

        graph = build_policy_dag(treatment_column)
        changes: dict[str, float] = {}
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            for outcome in ("pm25", "respiratory_admissions"):
                model = CausalModel(
                    data=self._panel,
                    treatment=treatment_column,
                    outcome=outcome,
                    graph=graph,
                )
                estimand = model.identify_effect()
                estimate = model.estimate_effect(
                    estimand,
                    method_name="backdoor.linear_regression",
                    control_value=0.0,
                    treatment_value=dose,
                    target_units="ate",
                )
                changes[outcome] = float(estimate.value)

        diagnostics = list(dict.fromkeys(
            str(item.message)
            for item in caught_warnings
            if "rank-deficient" in str(item.message).lower()
            or "singular" in str(item.message).lower()
        ))

        result = self._result(
            policy_type,
            dose,
            changes["pm25"],
            changes["respiratory_admissions"],
        )
        if diagnostics:
            result["diagnostics"] = diagnostics
        LOGGER.info("%s: %s", SCENARIO_LABEL, result)
        return result

    def _check_treatment_support(self, treatment_column: str, dose: float) -> None:
        """Reject contrasts that extrapolate beyond observed treatment support."""
        values = self._panel[treatment_column].to_numpy(dtype=float)
        lower, upper = float(values.min()), float(values.max())
        tolerance = max(0.025, 0.1 * (upper - lower))
        if upper - lower <= 1e-9:
            raise ValueError(f"No treatment variation for `{treatment_column}`; positivity fails.")
        if dose < lower - tolerance or dose > upper + tolerance:
            raise ValueError(
                f"Requested intensity {dose:g} is outside observed treatment support "
                f"[{lower:g}, {upper:g}]."
            )
        if np.count_nonzero(np.abs(values) <= tolerance) < 2:
            raise ValueError("Treatment panel lacks at least two near-zero control observations.")
        if np.count_nonzero(np.abs(values - dose) <= tolerance) < 2:
            raise ValueError(
                f"Treatment panel lacks overlap near requested intensity {dose:g}."
            )

    @staticmethod
    def _result(
        policy_type: str,
        intensity: float,
        pm25_change: float,
        admissions_change: float,
    ) -> dict[str, Any]:
        return {
            "scenario_type": SCENARIO_LABEL,
            "policy_type": policy_type,
            "intensity": intensity,
            "estimated_pm25_change_ug_m3": pm25_change,
            "estimated_pm25_reduction_ug_m3": max(0.0, -pm25_change),
            "estimated_admissions_change": admissions_change,
            "projected_admissions_reduction": max(0.0, -admissions_change),
            "interpretation": (
                "Observational counterfactual scenario, not an identified causal effect. "
                "Validate confounding, time alignment, overlap, and model assumptions."
            ),
        }


_configured_simulator: CausalPolicySimulator | None = None


def configure_policy_simulator(panel: pd.DataFrame) -> CausalPolicySimulator:
    """Configure the module-level simulator for two-argument integrations."""
    global _configured_simulator
    _configured_simulator = CausalPolicySimulator(panel)
    return _configured_simulator


def simulate_policy_impact(
    policy_type: str,
    intensity: float,
    *,
    simulator: CausalPolicySimulator | None = None,
) -> dict[str, Any]:
    """Estimate PM2.5 and respiratory-admission changes for a policy dose.

    Pass a simulator directly, or call :func:`configure_policy_simulator`
    before using the two-argument form.
    """
    active_simulator = simulator or _configured_simulator
    if active_simulator is None:
        raise RuntimeError(
            "No policy panel is configured. Provide a CausalPolicySimulator or "
            "call configure_policy_simulator(panel)."
        )
    return active_simulator.simulate_policy_impact(policy_type, intensity)


__all__ = [
    "CausalPolicySimulator",
    "PANEL_ALIGNMENT_COLUMNS",
    "PANEL_COLUMNS",
    "POLICY_TREATMENTS",
    "build_policy_dag",
    "configure_policy_simulator",
    "simulate_policy_impact",
]
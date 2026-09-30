"""DoWhy-based policy effect estimation for environmental health outcomes.

The simulator requires an observational policy panel. Estimates are only as
credible as its treatment definitions, confounder coverage, and time alignment.
"""

from __future__ import annotations

from math import isfinite
from typing import Any

import pandas as pd

POLICY_TREATMENTS = {
    "brick_kiln_shutdown": "brick_kiln_shutdown_intensity",
    "crop_burning_penalties": "crop_burning_penalties_intensity",
    "heavy_vehicle_restriction": "heavy_vehicle_restriction_intensity",
}

PANEL_COLUMNS = {
    *POLICY_TREATMENTS.values(),
    "ground_emissions",
    "pblh_inversion",
    "stagnation_index",
    "pm25",
    "respiratory_admissions",
    "policy_context",
    "meteorology_index",
}


def build_policy_dag(treatment_column: str) -> str:
    """Return a DOT DAG with policy mediators and measured confounders."""
    if treatment_column not in POLICY_TREATMENTS.values():
        raise ValueError(f"Unsupported policy treatment column: {treatment_column}")
    policy_edges = "\n".join(
        f"        {policy} -> {intermediate};\n"
        f"        {policy} -> respiratory_admissions;"
        for policy in POLICY_TREATMENTS.values()
        for intermediate in ("ground_emissions", "pblh_inversion", "stagnation_index")
    )
    context_edges = "\n".join(
        f"        policy_context -> {policy};" for policy in POLICY_TREATMENTS.values()
    )
    return f"""
    digraph {{
{policy_edges}
        ground_emissions -> pm25;
        pblh_inversion -> pm25;
        stagnation_index -> pm25;
        pm25 -> respiratory_admissions;
{context_edges}
        policy_context -> pm25;
        policy_context -> respiratory_admissions;
        meteorology_index -> {treatment_column};
        meteorology_index -> pblh_inversion;
        meteorology_index -> stagnation_index;
        meteorology_index -> pm25;
        meteorology_index -> respiratory_admissions;
    }}
    """


class CausalPolicySimulator:
    """Estimate policy dose effects from an aligned observational panel."""

    def __init__(self, panel: pd.DataFrame) -> None:
        missing = PANEL_COLUMNS.difference(panel.columns)
        if missing:
            raise ValueError(f"Policy panel is missing required columns: {sorted(missing)}")
        numeric = panel.loc[:, sorted(PANEL_COLUMNS)].apply(pd.to_numeric, errors="coerce")
        numeric = numeric.replace([float("inf"), float("-inf")], pd.NA).dropna()
        if len(numeric) < 10:
            raise ValueError("At least 10 complete policy-panel rows are required.")
        if (numeric[list(POLICY_TREATMENTS.values())] < 0).any().any() or (
            numeric[list(POLICY_TREATMENTS.values())] > 1
        ).any().any():
            raise ValueError("Policy intensity columns must be normalized to the range [0, 1].")
        self._panel = numeric

    def simulate_policy_impact(self, policy_type: str, intensity: float) -> dict[str, Any]:
        """Estimate changes relative to zero intensity for one policy dose.

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
            return self._result(policy_type, dose, 0.0, 0.0)

        try:
            from dowhy import CausalModel
        except ImportError as error:
            raise RuntimeError("Install the `dowhy` dependency to estimate policy effects.") from error

        graph = build_policy_dag(treatment_column)
        changes: dict[str, float] = {}
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

        return self._result(
            policy_type,
            dose,
            changes["pm25"],
            changes["respiratory_admissions"],
        )

    @staticmethod
    def _result(
        policy_type: str,
        intensity: float,
        pm25_change: float,
        admissions_change: float,
    ) -> dict[str, Any]:
        return {
            "policy_type": policy_type,
            "intensity": intensity,
            "estimated_pm25_change_ug_m3": pm25_change,
            "estimated_pm25_reduction_ug_m3": max(0.0, -pm25_change),
            "estimated_admissions_change": admissions_change,
            "projected_admissions_reduction": max(0.0, -admissions_change),
            "interpretation": "Model-based estimate; validate identification assumptions before policy use.",
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
    "PANEL_COLUMNS",
    "POLICY_TREATMENTS",
    "build_policy_dag",
    "configure_policy_simulator",
    "simulate_policy_impact",
]
"""Reusable enterprise modules for the PakSat platform."""

from importlib import import_module
from typing import Any

_EXPORTS = {
    "CausalPolicySimulator": ("paksat.causal_policy", "CausalPolicySimulator"),
    "optimize_hospital_resources": (
        "paksat.resource_optimizer",
        "optimize_hospital_resources",
    ),
    "simulate_policy_impact": ("paksat.causal_policy", "simulate_policy_impact"),
    "EconomicAssumptions": ("paksat.economic_impact", "EconomicAssumptions"),
    "estimate_health_economic_benefit": (
        "paksat.economic_impact",
        "estimate_health_economic_benefit",
    ),
}


def __getattr__(name: str) -> Any:
    """Load optional module dependencies only when an export is requested."""
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = _EXPORTS[name]
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


__all__ = list(_EXPORTS)
"""Clinical decision-support contracts for environmental exposure and capacity."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from math import isfinite
from typing import Any

ACUTE_THRESHOLD = 150.0


def assess_patient_risk(pm25_point: float, pm25_upper_bound: float, symptoms: Iterable[str], medical_history: Mapping[str, Any] | None = None) -> dict[str, Any]:
    point, raw_upper = float(pm25_point), float(pm25_upper_bound)
    if not isfinite(point) or not isfinite(raw_upper) or point < 0 or raw_upper < 0:
        raise ValueError("PM2.5 values must be finite and non-negative.")
    upper = max(raw_upper, point)
    symptom_set = {str(value).strip().lower() for value in symptoms if value}
    history = {str(key).lower(): value for key, value in (medical_history or {}).items()}
    pediatric_asthma = bool(history.get("pediatric_asthma") or (history.get("age") is not None and float(history["age"]) < 18 and history.get("asthma")))
    elderly_copd = bool(history.get("elderly_copd") or (history.get("age") is not None and float(history["age"]) >= 65 and history.get("copd")))
    vulnerability = 1.0 + (0.25 if pediatric_asthma else 0.0) + (0.25 if elderly_copd else 0.0)
    acute = point > ACUTE_THRESHOLD or upper > ACUTE_THRESHOLD
    respiratory = {"wheezing", "shortness of breath", "chest tightness", "severe breathing difficulty", "blue lips", "cough"}
    urgent_symptoms = symptom_set.intersection({"severe breathing difficulty", "blue lips"})
    symptomatic = symptom_set.intersection(respiratory)

    if acute or urgent_symptoms:
        severity = "Environmental Acute Exacerbation"
    elif point > 55 or symptomatic:
        severity = "Elevated Environmental Risk"
    elif point > 35:
        severity = "Moderate Environmental Risk"
    else:
        severity = "Low Environmental Risk"

    advice = ["Use a well-fitted N95 respirator outdoors.", "Keep windows closed and use HEPA indoor filtration.", "Avoid strenuous outdoor activity while air quality is elevated."]
    if acute or urgent_symptoms:
        advice.append("Bronchodilator use must follow an existing prescription and clinician guidance; do not self-medicate.")
        advice.append("Seek emergency clinical assessment now for severe breathing symptoms.")
    elif symptomatic:
        advice.append("Follow the patient’s action plan and seek same-day clinical review if symptoms worsen.")
    return {"pm25_point": round(point, 2), "pm25_upper_bound": round(upper, 2), "severity": severity, "vulnerability_multiplier": round(vulnerability, 2), "symptoms": sorted(symptom_set), "advice": advice, "urgent": acute or bool(urgent_symptoms), "disclaimer": "Decision support only. Contact local emergency services for severe symptoms."}


__all__ = ["assess_patient_risk"]

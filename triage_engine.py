"""Environment-aware clinical triage and emergency surge estimation."""

from __future__ import annotations

from typing import Any, Iterable


ACUTE_THRESHOLD = 150.0


def assess_clinical_risk(pm25_level: float, patient_symptoms: Iterable[str]) -> dict[str, Any]:
    """Return explainable environmental risk guidance, not a medical diagnosis."""
    level = float(pm25_level)
    symptoms = {str(symptom).strip().lower() for symptom in patient_symptoms if symptom}
    respiratory = {"cough", "wheezing", "shortness of breath", "chest tightness"}

    if level > ACUTE_THRESHOLD:
        severity = "Environmental Acute Exacerbation"
    elif level > 55 or symptoms.intersection(respiratory):
        severity = "Elevated Environmental Risk"
    elif level > 35:
        severity = "Moderate Environmental Risk"
    else:
        severity = "Low Environmental Risk"

    recommendations = [
        "Use a well-fitted N95 respirator outdoors.",
        "Keep windows closed and use HEPA indoor filtration.",
        "Avoid strenuous outdoor activity while air quality is elevated.",
    ]
    if symptoms.intersection({"wheezing", "shortness of breath", "chest tightness"}):
        recommendations.append(
            "Follow the patient's existing action plan; seek urgent clinical assessment for worsening symptoms."
        )
    if level > ACUTE_THRESHOLD:
        recommendations.append(
            "Bronchodilator use should follow an existing prescription and clinician guidance; do not self-medicate."
        )

    return {
        "pm25_level": round(level, 2),
        "severity": severity,
        "symptoms": sorted(symptoms),
        "recommendations": recommendations,
        "urgent": level > ACUTE_THRESHOLD or bool(symptoms.intersection({"severe breathing difficulty", "blue lips"})),
        "disclaimer": "Decision support only. Contact local emergency services for severe symptoms.",
    }


def predict_hospital_surge(
    stagnation_index: float,
    photochemical_proxy: float,
) -> dict[str, Any]:
    """Estimate a 24-48 hour capacity warning from normalized pollution signals."""
    stagnation = max(float(stagnation_index), 0.0)
    photochemical = max(float(photochemical_proxy), 0.0)
    pressure = min(1.0, stagnation / 1000.0 + photochemical / 100.0)
    surge_percent = round(20.0 + 10.0 * pressure, 1)
    return {
        "horizon": "24-48 hours",
        "estimated_surge_percent": surge_percent,
        "capacity_warning": surge_percent >= 25.0,
        "message": f"Prepare for approximately {surge_percent:.1f}% additional emergency demand.",
    }

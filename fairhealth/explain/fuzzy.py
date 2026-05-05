"""
fairhealth.explain.fuzzy
=========================
Fuzzy logic risk system for maternal health.
Based on: Yesmin, F. et al. (2026) ICAIHE 2026, Waseda University.
The 6 fuzzy rules are derived from clinical guidelines and
Bangladesh maternal health literature.
"""

FUZZY_RULES = [
    {"id": 1, "condition": "High BP AND High Blood Sugar",     "outcome": "HIGH RISK"},
    {"id": 2, "condition": "Elevated BP AND Older Age (>40)",  "outcome": "HIGH RISK"},
    {"id": 3, "condition": "Normal BP AND Normal Blood Sugar",  "outcome": "LOW RISK"},
    {"id": 4, "condition": "High Blood Sugar alone",            "outcome": "MID RISK"},
    {"id": 5, "condition": "High Heart Rate AND High BP",       "outcome": "HIGH RISK"},
    {"id": 6, "condition": "Young Age (<=25) AND Normal BP",    "outcome": "LOW RISK"},
]

def get_fired_rules(age, sbp, bs, hr):
    """Return which fuzzy rules fire for a patient.
    Args:
        age : patient age in years
        sbp : systolic blood pressure mmHg
        bs  : blood sugar mmol/L
        hr  : heart rate bpm
    Returns:
        list of fired rule dicts
    """
    fired = []
    if sbp >= 130 and bs >= 11:
        fired.append(FUZZY_RULES[0])
    if sbp >= 100 and age >= 40:
        fired.append(FUZZY_RULES[1])
    if sbp < 110 and bs < 8:
        fired.append(FUZZY_RULES[2])
    if bs >= 7 and sbp < 110:
        fired.append(FUZZY_RULES[3])
    if hr >= 75 and sbp >= 130:
        fired.append(FUZZY_RULES[4])
    if age <= 25 and sbp < 110:
        fired.append(FUZZY_RULES[5])
    return fired

def score_to_label(score):
    """Convert fuzzy risk score (0-10) to risk label."""
    if score >= 6.5:   return "high risk"
    elif score >= 3.5: return "mid risk"
    else:              return "low risk"

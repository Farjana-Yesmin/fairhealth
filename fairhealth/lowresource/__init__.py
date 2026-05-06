"""
fairhealth.lowresource
=======================
Low-resource clinical decision support tools.

Based on:
    Yesmin, F. (2026). AI Chatbots for Dengue Symptom Triage
    in Bangladesh: A Decision Tree Classifier Approach.
    Accepted, DASGRI 2026, Springer LNNS, London.
    Preprint: https://www.researchgate.net/publication/385935162
    Status: Accepted — Springer proceedings forthcoming.

Results (from preprint):
    Accuracy   : 0.79  (non-leaky demographic features)
    F1-score   : 0.802
    AUC        : 0.851
    Top feature: Age (Gini importance = 0.686)
    Languages  : English + Bangla
    Pilot study: 75% satisfaction (n=50)
"""
from fairhealth.lowresource.triage import assess_dengue_risk, batch_triage
__all__ = ["assess_dengue_risk", "batch_triage"]

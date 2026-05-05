"""
fairhealth.lowresource
=======================
Low-resource clinical decision support tools.

Based on:
    Yesmin, F. (2026). "AI Chatbots for Dengue Symptom Triage
    in Bangladesh." DASGRI 2026, Springer LNNS, London.

Paper results:
    Decision Tree accuracy : 0.79
    F1-score               : 0.802
    AUC                    : 0.851
    Top feature            : Age (Gini importance = 0.686)
    Pilot satisfaction     : 75% (n=50)
    Languages              : English, Bangla
"""

from fairhealth.lowresource.triage import assess_dengue_risk, batch_triage

__all__ = ["assess_dengue_risk", "batch_triage"]

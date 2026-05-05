"""
fairhealth.explain
===================
Clinical explainability tools: SHAP, LIME, Fuzzy-XGBoost.

Based on:
    Yesmin, F., Shirmin, N., Bristy, S.S. (2026).
    Explainable AI for Maternal Health Risk Prediction in
    Bangladesh: A Hybrid Fuzzy-XGBoost Framework with
    Clinician Validation.
    Accepted, ICAIHE 2026, Waseda University, Tokyo.
    Preprint: https://www.researchsquare.com/article/rs-8584734/v1
    Status: Accepted — proceedings forthcoming.

Results (from preprint):
    Accuracy              : 88.67%
    ROC-AUC               : 0.9703
    Clinician preference  : 71.4% prefer hybrid (n=14)
    Trust in practice     : 54.8%
    Top SHAP feature      : Healthcare Access Score (1.49)
"""
from fairhealth.explain.fuzzy import get_fired_rules, score_to_label
from fairhealth.explain.shap_utils import get_shap_explainer, explain_patient
__all__ = [
    "get_fired_rules", "score_to_label",
    "get_shap_explainer", "explain_patient",
]

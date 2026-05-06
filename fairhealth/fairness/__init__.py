"""
fairhealth.fairness
====================
Fairness metrics for healthcare AI models.

Based on:
    Yesmin, F. (2026). Fairness-Aware Representation Learning
    for ECG-Based Disease Prediction in Wearable Systems.
    Accepted, MobiHealth 2026 (EAI), Heraklion, Greece.
    Preprint: https://www.researchgate.net/publication/396441645
    Status: Accepted — EAI proceedings forthcoming.

Results (from preprint):
    AUROC                : 0.8472
    Disparate Impact     : 0.23 to 0.71 (after debiasing)
    Dataset              : PTB-XL, 4,367 records (20% subsample)
    Sensitive attributes : sex, age group
"""
from fairhealth.fairness.metrics import (
    demographic_parity_diff,
    fairness_report,
)
__all__ = ["demographic_parity_diff", "fairness_report"]

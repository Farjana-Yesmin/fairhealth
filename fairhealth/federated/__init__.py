"""
fairhealth.federated
=====================
Privacy-preserving federated learning for healthcare AI.

Based on:
    Yesmin, F. (2026). MedHE: Communication-Efficient
    Privacy-Preserving Federated Learning for Healthcare.
    Under review, CIBB 2026.
    Preprint: https://arxiv.org/abs/2511.09043

Status: Under review — not yet peer reviewed.
Cite as: arXiv:2511.09043

Results (preprint):
    macro-F1              : 0.950 ± 0.005
    Accuracy              : 89.5% ± 0.8%
    Communication reduction: 97.5% (1,277 MB to 32 MB)
    MIA resistance        : 51.1% (ideal=50%)
    Privacy budget        : epsilon <= 1.0
"""
from fairhealth.federated import client, server, privacy
__all__ = ["client", "server", "privacy"]

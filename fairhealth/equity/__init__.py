"""
fairhealth.equity
==================
Fairness-aware AI for disaster response and resource allocation.

Based on:
    Yesmin, F., Akter, R. (2026). Toward Equitable Recovery:
    A Fairness-Aware AI Framework for Prioritizing Post-Flood
    Aid in Bangladesh. Accepted (oral), CCAI 2026 (IEEE), Nanjing.
    Preprint: https://arxiv.org/abs/2512.22210
    To be indexed: IEEE Xplore, Ei Compendex, Scopus.

Status: Accepted — proceedings forthcoming.
Cite preprint until IEEE Xplore indexed.

Results (from preprint):
    Fair model R2            : 0.784 (baseline: 0.811)
    SPD reduction            : 41.6%
    Regional fairness gap    : 43.2% reduction
    MAE std reduction        : 40.7%
    Upazilas reranked        : 70.6%
    Sunamganj ranking        : 14 to 6
"""
from fairhealth.equity.flood_aid import (
    compute_vulnerability_score,
    compute_priority_score,
    generate_priority_ranking,
    fairness_gap_analysis,
    PDNA_DISTRICT_DATA,
)
__all__ = [
    "compute_vulnerability_score",
    "compute_priority_score",
    "generate_priority_ranking",
    "fairness_gap_analysis",
    "PDNA_DISTRICT_DATA",
]

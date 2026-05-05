"""
fairhealth.equity
==================
Fairness-aware resource allocation for disaster response.

Based on:
    Yesmin, F., Akter, R. (2026). "Toward Equitable Recovery:
    A Fairness-Aware AI Framework for Prioritizing Post-Flood
    Aid in Bangladesh." CCAI 2026 (IEEE), Nanjing, China.

Paper results:
    Fair model R²         : 0.784 (vs 0.811 baseline)
    SPD reduction         : 41.6%
    Regional gap reduction: 43.2%
    MAE std reduction     : 40.7%
    Upazilas reranked     : 70.6%
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

# FairHealth

**Trustworthy Healthcare AI — built from peer-reviewed research.**

[![PyPI version](https://badge.fury.io/py/fairhealth.svg)](https://pypi.org/project/fairhealth/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

FairHealth is an open-source Python library for building **fair, explainable,
and privacy-preserving** machine learning models for healthcare.

Built by [Farjana Yesmin](https://farjana-yesmin.github.io/) from
peer-reviewed research on maternal health, biosignals, and federated learning.

---

## Install

```bash
pip install fairhealth
```

---

## What It Does

| Module | What | Paper |
|---|---|---|
| `fairhealth.fairness` | Demographic parity, equalized odds, disparate impact | MobiHealth 2026 |
| `fairhealth.explain` | SHAP wrappers + Fuzzy-XGBoost hybrid explainer | ICAIHE 2026 |
| `fairhealth.federated` | FedAvg + differential privacy + sparsification | MedHE, CIBB 2026 |
| `fairhealth.datasets` | Maternal health, dengue, flood PDNA — all public | Multiple |

---

## Quick Example

```python
import fairhealth as fh
import numpy as np

# ── Fairness audit ───────────────────────────────────────────────
from fairhealth.fairness.metrics import demographic_parity_diff

y_pred    = np.array([1, 0, 1, 0, 1, 0])
sensitive = np.array([0, 0, 0, 1, 1, 1])   # 0=young, 1=older

dpd = demographic_parity_diff(y_pred, sensitive)
print(f"Demographic Parity Difference: {dpd:.4f}")
# → 0.3333  (gap between age groups)

# ── Fuzzy risk explanation ───────────────────────────────────────
from fairhealth.explain.fuzzy import get_fired_rules, score_to_label

rules = get_fired_rules(age=42, sbp=145, bs=12, hr=88)
for rule in rules:
    print(f"Rule {rule['id']}: {rule['condition']} → {rule['outcome']}")
# → Rule 1: High BP AND High Blood Sugar → HIGH RISK
# → Rule 5: High Heart Rate AND High BP  → HIGH RISK

# ── Federated privacy ────────────────────────────────────────────
from fairhealth.federated.privacy import sparsify, add_gaussian_noise

weights       = np.array([0.4, 0.3, 0.15, 0.1, 0.03, 0.02])
sparse, rate  = sparsify(weights, sparsity=0.975)
print(f"Communication reduced by {rate:.1%}")
# → Communication reduced by 83.3%
```

---

## Real Results (from My Papers)

| Finding | Value |
|---|---|
| Maternal health model accuracy | 79.3% (Fuzzy-XGBoost hybrid) |
| Clinicians preferring hybrid explanation | 71% (14 clinicians, ICAIHE 2026) |
| Demographic parity difference (age groups) | 0.1011 |
| Federated vs central accuracy gap | 9.3% |
| Communication reduction (sparsification) | 83–97.5% |

---

## Datasets Used (All Public — No Hospital Access Needed)

| Dataset | Domain | Source |
|---|---|---|
| Maternal Health Risk | Risk prediction | UCI / Kaggle |
| Bangladesh Dengue | Symptom triage | DGHS Bangladesh |
| Bangladesh Flood PDNA 2022 | Disaster equity | Government open data |
| PTB-XL | ECG biosignals | PhysioNet (free) |

---

## Research Papers

If my library helps your work, please cite:

```bibtex
@software{fairhealth2026,
  author = {Yesmin, Farjana},
  title  = {FairHealth: Trustworthy Healthcare AI},
  year   = {2026},
  url    = {https://github.com/Farjana-Yesmin/fairhealth}
}
```

**Related papers:**
- Yesmin, F. (2026). *Fairness-Aware Representation Learning for ECG-Based Disease Prediction.* MobiHealth 2026.
- Yesmin, F. et al. (2026). *Explainable AI for Maternal Health Risk Prediction in Bangladesh.* ICAIHE 2026.
- Yesmin, F. (2026). *MedHE: Communication-Efficient Privacy-Preserving Federated Learning.* CIBB 2026.
- Yesmin, F. & Akter, R. (2026). *Toward Equitable Recovery.* CCAI 2026 (IEEE).

---

## Author

**Farjana Yesmin** — ML Researcher, Trustworthy AI for Healthcare  
Website: [farjana-yesmin.github.io](https://farjana-yesmin.github.io/)  
Email: farjanayesmin76@gmail.com

---

## License

MIT © Farjana Yesmin

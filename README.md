# FairHealth

**Trustworthy Healthcare AI — built from peer-reviewed research.**

[![PyPI version](https://badge.fury.io/py/fairhealth.svg)](https://pypi.org/project/fairhealth/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2605.08198-b31b1b.svg)](https://arxiv.org/abs/2605.08198)

FairHealth is an open-source Python library for building **fair, explainable,
and privacy-preserving** machine learning models for healthcare.

Built by [Farjana Yesmin](https://farjana-yesmin.github.io/) from 5 accepted research papers.

---

## Install

```bash
pip install fairhealth
```

---

## Modules and Results

| Module | What | Paper | Key Result |
|---|---|---|---|
| `fairhealth.fairness` | Demographic parity, equalized odds, disparate impact | MobiHealth 2026 | DI: 0.23 → 0.71 |
| `fairhealth.explain` | SHAP wrappers + Fuzzy-XGBoost hybrid | ICAIHE 2026, Waseda | 88.67% acc, 71.4% clinician preference |
| `fairhealth.federated` | FedAvg + CKKS HE + differential privacy | MedHE, CIBB 2026 | macro-F1=0.950, 97.5% comm reduction |
| `fairhealth.lowresource` | Dengue triage, multilingual, low-bandwidth | DASGRI 2026, Springer | F1=0.802, 75% satisfaction |
| `fairhealth.equity` | Fairness-aware flood aid allocation | CCAI 2026, IEEE | SPD↓41.6%, R²=0.784 |

---

## Quick Example

```python
import fairhealth as fh
import numpy as np

# Fairness audit
from fairhealth.fairness.metrics import demographic_parity_diff
dpd = demographic_parity_diff(
    y_pred    = np.array([1, 0, 1, 0, 1, 0]),
    sensitive = np.array([0, 0, 0, 1, 1, 1])
)
print(f"DPD: {dpd:.4f}")   # → 0.3333

# Dengue triage — English + Bangla
from fairhealth.lowresource.triage import assess_dengue_risk
result = assess_dengue_risk(age=8, gender="male",
                             area_type="urban", district="Dhaka",
                             language="bangla")
print(result["recommendation"])   # বাংলা output

# Flood aid equity
from fairhealth.equity.flood_aid import generate_priority_ranking
rankings = generate_priority_ranking(verbose=False)
print(f"Top priority: {rankings[0]['district']}")  # → Sunamganj

# Federated privacy
from fairhealth.federated.privacy import sparsify
_, rate = sparsify(np.random.randn(1000), sparsity=0.975)
print(f"Communication reduced: {rate:.1%}")  # → 97.5%
```

---

## Research Papers

All papers are accepted. Preprint links below; final proceedings forthcoming.

| Paper | Venue | Preprint | Status |
|---|---|---|---|
| ECG Fairness | MobiHealth 2026 (EAI) | [ResearchGate](https://www.researchgate.net/publication/396441645) | Accepted |
| Maternal Health XAI | ICAIHE 2026, Waseda | [ResearchSquare](https://www.researchsquare.com/article/rs-8584734/v1) | Accepted |
| MedHE Federated | CIBB 2026 | [arXiv:2511.09043](https://arxiv.org/abs/2511.09043) | Under review |
| Dengue Triage | DASGRI 2026, Springer LNNS | [ResearchGate](https://www.researchgate.net/publication/385935162) | Accepted |
| Flood Aid Equity | CCAI 2026 (IEEE), oral | [arXiv:2512.22210](https://arxiv.org/abs/2512.22210) | Accepted |

---

## Datasets (All Public — No Hospital Access Required)

| Dataset | Domain | Source |
|---|---|---|
| PTB-XL (4,367 records) | ECG biosignals | PhysioNet |
| Maternal Health Risk (1,014) | Risk prediction | UCI ML Repository |
| UCI Drug Reviews (215K) | NLP / drug effectiveness | UCI ML Repository |
| Bangladesh Dengue (4,700) | Symptom triage | Kaggle + DGHS |
| Bangladesh PDNA 2022 (87 upazilas) | Flood equity | Government open data |

---

## Cite

```bibtex
@article{yesmin2026fairhealth,
  author  = {Yesmin, Farjana},
  title   = {FairHealth: An Open-Source Python Library for
             Trustworthy Healthcare AI in Low-Resource Settings},
  journal = {arXiv preprint arXiv:2605.08198},
  year    = {2026},
  url     = {https://arxiv.org/abs/2605.08198}
}

@software{fairhealth2026,
  author = {Yesmin, Farjana},
  title  = {FairHealth: Trustworthy Healthcare AI},
  year   = {2026},
  url    = {https://github.com/Farjana-Yesmin/fairhealth}
}
```

**Author:** Farjana Yesmin · [farjana-yesmin.github.io](https://farjana-yesmin.github.io/) · MIT License

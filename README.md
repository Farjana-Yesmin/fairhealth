# FairHealth

**Trustworthy Healthcare AI — built from peer-reviewed research.**

[![PyPI version](https://badge.fury.io/py/fairhealth.svg)](https://pypi.org/project/fairhealth/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Documentation](https://readthedocs.org/projects/fairhealth/badge/?version=latest)](https://fairhealth.readthedocs.io)

FairHealth is an open-source Python library for building **fair, explainable,
and privacy-preserving** machine learning models for healthcare.

Built by [Farjana Yesmin](https://farjana-yesmin.github.io/) from 5 peer-reviewed papers.

---

## Install

```bash
pip install fairhealth
```

---

## Modules and Paper Results

| Module | What | Paper | Key Result |
|---|---|---|---|
| `fairhealth.fairness` | Demographic parity, equalized odds, disparate impact | MobiHealth 2026 | DI improved 0.23→0.71 |
| `fairhealth.explain` | SHAP wrappers + Fuzzy-XGBoost hybrid | ICAIHE 2026 | 88.67% acc, 71.4% clinician preference |
| `fairhealth.federated` | FedAvg + CKKS HE + differential privacy | MedHE, CIBB 2026 | macro-F1=0.950, 97.5% comm reduction |
| `fairhealth.lowresource` | Dengue triage, multilingual, low-bandwidth | DASGRI 2026 | F1=0.802, 75% user satisfaction |
| `fairhealth.equity` | Fairness-aware flood aid allocation | CCAI 2026 (IEEE) | SPD↓41.6%, Regional gap↓43.2% |

---

## Quick Example

```python
import fairhealth as fh
import numpy as np

# ── Fairness audit (MobiHealth 2026) ─────────────────────────────────────
from fairhealth.fairness.metrics import demographic_parity_diff
y_pred    = np.array([1, 0, 1, 0, 1, 0])
sensitive = np.array([0, 0, 0, 1, 1, 1])
dpd = demographic_parity_diff(y_pred, sensitive)
print(f"DPD: {dpd:.4f}")   # → 0.3333

# ── Fuzzy explanation (ICAIHE 2026) ──────────────────────────────────────
from fairhealth.explain.fuzzy import get_fired_rules
rules = get_fired_rules(age=42, sbp=145, bs=12, hr=88)
for r in rules:
    print(f"Rule {r['id']}: {r['condition']} → {r['outcome']}")

# ── Dengue triage (DASGRI 2026) ──────────────────────────────────────────
from fairhealth.lowresource.triage import assess_dengue_risk
result = assess_dengue_risk(age=8, gender="male",
                             area_type="urban", district="Dhaka")
print(result["recommendation"])

# ── Flood aid equity (CCAI 2026) ─────────────────────────────────────────
from fairhealth.equity.flood_aid import generate_priority_ranking
rankings = generate_priority_ranking(verbose=False)
print(f"Top priority district: {rankings[0]['district']}")

# ── Federated privacy (MedHE CIBB 2026) ──────────────────────────────────
from fairhealth.federated.privacy import sparsify
weights      = np.random.randn(1000)
sparse_w, r  = sparsify(weights, sparsity=0.975)
print(f"Communication reduced: {r:.1%}")  # → 97.5%
```

---

## Validated Results From My Papers

| Paper | Venue | Key Finding |
|---|---|---|
| ECG Fairness | MobiHealth 2026, EAI | Disparate Impact: 0.23 → 0.71 after debiasing |
| Maternal Health XAI | ICAIHE 2026, Waseda | 88.67% accuracy, ROC-AUC=0.9703, 71.4% clinician preference |
| MedHE Federated | CIBB 2026 | macro-F1=0.950, 97.5% comm reduction, MIA=51.1% |
| Dengue Triage | DASGRI 2026, Springer LNNS | F1=0.802, AUC=0.851, 75% user satisfaction |
| Flood Aid Equity | CCAI 2026, IEEE | SPD↓41.6%, Regional gap↓43.2%, R²=0.784 |

---

## Public Datasets Used (No Hospital Access Required)

| Dataset | Domain | Source |
|---|---|---|
| PTB-XL (4,367 records) | ECG biosignals | PhysioNet (free account) |
| Maternal Health Risk (1,014) | Risk prediction | UCI ML Repository |
| UCI Drug Reviews (215K) | NLP / drug effectiveness | UCI ML Repository |
| Bangladesh Dengue (4,700) | Symptom triage | Kaggle + DGHS Dashboard |
| Bangladesh PDNA 2022 (87 upazilas) | Flood disaster equity | Government open data |

---

## Cite

```bibtex
@software{fairhealth2026,
  author = {Yesmin, Farjana},
  title  = {FairHealth: Trustworthy Healthcare AI},
  year   = {2026},
  url    = {https://github.com/Farjana-Yesmin/fairhealth}
}
```

**Papers:**
- Yesmin, F. (2026). *Fairness-Aware ECG-Based Disease Prediction.* MobiHealth 2026.
- Yesmin, F. et al. (2026). *Explainable AI for Maternal Health Risk Prediction in Bangladesh.* ICAIHE 2026, Waseda.
- Yesmin, F. (2026). *MedHE: Privacy-Preserving Federated Learning for Healthcare.* CIBB 2026.
- Yesmin, F. (2026). *AI Chatbots for Dengue Symptom Triage in Bangladesh.* DASGRI 2026, Springer LNNS.
- Yesmin, F. & Akter, R. (2026). *Toward Equitable Recovery.* CCAI 2026 (IEEE), Nanjing.

---

**Author:** Farjana Yesmin · [farjana-yesmin.github.io](https://farjana-yesmin.github.io/) · MIT License

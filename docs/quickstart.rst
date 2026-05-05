Quick Start
===========

.. code-block:: python

   import fairhealth as fh
   import numpy as np

   # Fairness audit
   from fairhealth.fairness.metrics import demographic_parity_diff
   y_pred    = np.array([1, 0, 1, 0, 1, 0])
   sensitive = np.array([0, 0, 0, 1, 1, 1])
   dpd = demographic_parity_diff(y_pred, sensitive)
   print(f"DPD: {dpd:.4f}")

   # Fuzzy explanation
   from fairhealth.explain.fuzzy import get_fired_rules
   rules = get_fired_rules(age=42, sbp=145, bs=12, hr=88)
   for r in rules:
       print(f"Rule {r['id']}: {r['condition']} -> {r['outcome']}")

   # Dengue triage
   from fairhealth.lowresource.triage import assess_dengue_risk
   result = assess_dengue_risk(age=25, gender="female",
                               area_type="urban", district="Dhaka")
   print(result)

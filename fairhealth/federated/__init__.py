"""
fairhealth.federated
=====================
Privacy-preserving federated learning for healthcare AI.

Based on:
    Yesmin, F. (2026). "MedHE: Communication-Efficient
    Privacy-Preserving Federated Learning with Adaptive Gradient
    Sparsification for Healthcare." CIBB 2026.

Modules
-------
client.py   : local training on each hospital's data
server.py   : FedAvg aggregation
privacy.py  : differential privacy noise + gradient sparsification
"""

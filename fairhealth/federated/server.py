"""
fairhealth.federated.server
============================
FedAvg aggregation server.
Aggregates predictions from all clients without seeing patient data.
"""
import numpy as np

def fedavg(client_probas, client_weights):
    """Federated Averaging (FedAvg) — McMahan et al. 2017.

    Computes weighted average of client prediction probabilities.
    No raw patient data is ever shared with this function.

    Args:
        client_probas  : list of (n_test, n_classes) probability arrays
        client_weights : list of ints — number of patients per client

    Returns:
        averaged_proba : (n_test, n_classes) weighted average probabilities
        predictions    : (n_test,) final class predictions

    Example:
        >>> probas = [client.predict_proba(X_test) for client in clients]
        >>> weights = [client.n for client in clients]
        >>> avg_proba, preds = fedavg(probas, weights)
    """
    total   = sum(client_weights)
    avg     = np.zeros_like(client_probas[0])
    for proba, w in zip(client_probas, client_weights):
        avg += proba * (w / total)
    return avg, np.argmax(avg, axis=1)

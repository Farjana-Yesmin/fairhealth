"""
fairhealth.federated.client
============================
Local training loop for one federated client (hospital).
"""
import numpy as np

class FederatedClient:
    """One hospital in a federated learning setup.

    The client trains a model locally and returns
    prediction probabilities — never raw patient data.

    Args:
        client_id : name/ID of this hospital
        X         : local training features (n_patients, n_features)
        y         : local training labels (n_patients,)
        model     : any sklearn-compatible classifier

    Example:
        >>> client = FederatedClient("Hospital_1", X_local, y_local, model)
        >>> client.train()
        >>> proba = client.predict_proba(X_test)
    """

    def __init__(self, client_id, X, y, model):
        self.client_id = client_id
        self.X         = X
        self.y         = y
        self.model     = model
        self.n         = len(y)
        self.trained   = False

    def train(self):
        """Train model on local data only."""
        self.model.fit(self.X, self.y)
        self.trained = True
        return self

    def predict_proba(self, X_test):
        """Return class probabilities for test set.
        No patient data leaves this method.
        """
        if not self.trained:
            raise RuntimeError("Call .train() before .predict_proba()")
        return self.model.predict_proba(X_test)

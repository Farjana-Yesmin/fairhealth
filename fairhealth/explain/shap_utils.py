"""
fairhealth.explain.shap_utils
==============================
SHAP explainability wrappers for clinical healthcare models.
Based on: Yesmin, F. et al. (2026) ICAIHE 2026, Waseda University.
"""
import numpy as np
import shap

def get_shap_explainer(model, X_background=None):
    """Create a SHAP TreeExplainer for a tree-based model.
    Args:
        model        : trained XGBoost or GBM model
        X_background : optional background dataset for KernelExplainer
    Returns:
        shap.TreeExplainer
    """
    return shap.TreeExplainer(model)

def explain_patient(explainer, patient_features, feature_names):
    """Explain a single patient prediction using SHAP.
    Args:
        explainer       : shap.TreeExplainer
        patient_features: 1D array of feature values
        feature_names   : list of feature names
    Returns:
        dict — feature name to SHAP value, sorted by importance
    """
    shap_vals = explainer.shap_values(
        patient_features.reshape(1, -1)
    )
    if isinstance(shap_vals, list):
        vals = shap_vals[0][0]
    else:
        vals = shap_vals[0]
    return dict(sorted(
        zip(feature_names, vals),
        key=lambda x: abs(x[1]),
        reverse=True
    ))

def feature_importance_summary(shap_values, feature_names):
    """Compute mean absolute SHAP values per feature.
    Args:
        shap_values  : 2D array (n_patients, n_features)
        feature_names: list of feature names
    Returns:
        dict — feature name to mean absolute importance
    """
    mean_abs = np.abs(shap_values).mean(axis=0)
    return dict(sorted(
        zip(feature_names, mean_abs),
        key=lambda x: x[1],
        reverse=True
    ))

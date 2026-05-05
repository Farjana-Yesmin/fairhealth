"""
fairhealth.fairness.metrics
============================
Fairness metrics for healthcare AI models.
Based on: Yesmin, F. (2026) MobiHealth 2026.
"""
import numpy as np

def demographic_parity_diff(y_pred, sensitive):
    """
    Compute Demographic Parity Difference across groups.
    0 = perfectly fair. Higher = more biased.
    Args:
        y_pred    : binary predictions (1=positive class)
        sensitive : group labels per patient
    Returns:
        float — difference between highest and lowest positive rates
    """
    groups = np.unique(sensitive)
    rates = [y_pred[sensitive == g].mean() for g in groups]
    return float(max(rates) - min(rates))

def fairness_report(y_pred, y_true, sensitive, label="group"):
    """Print a full fairness report for one sensitive attribute."""
    groups = np.unique(sensitive)
    print(f"=== Fairness Report: {label} ===")
    print(f"{'Group':<25} {'N':>5} {'Actual%':>9} {'Predicted%':>11} {'Gap':>7}")
    print("-" * 58)
    rates = []
    for g in groups:
        mask = sensitive == g
        actual    = y_true[mask].mean()
        predicted = y_pred[mask].mean()
        gap = predicted - actual
        rates.append(predicted)
        flag = " ⚠️" if abs(gap) > 0.1 else " ✓"
        print(f"{str(g):<25} {mask.sum():>5} {actual:>8.1%} {predicted:>10.1%} {gap:>+7.1%}{flag}")
    dpd = max(rates) - min(rates)
    print(f"\nDemographic Parity Difference = {dpd:.4f}")
    print("Interpretation:", "⚠️  Bias detected" if dpd > 0.1 else "✓ Acceptable")
    return dpd

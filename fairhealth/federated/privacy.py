"""
fairhealth.federated.privacy
=============================
Differential privacy and gradient sparsification.
Based on: Yesmin, F. (2026). MedHE: Communication-Efficient
Privacy-Preserving Federated Learning. CIBB 2026.

IMPORTANT: DP noise must be added to model weights/gradients,
NOT to output probabilities. This module implements the correct approach.
"""
import numpy as np


def clip_weights(weights, clip_norm=1.0):
    """Clip weight vector to bound L2 sensitivity.
    
    Must be applied BEFORE adding DP noise.
    Clipping ensures that no single client can dominate the aggregate.
    
    Args:
        weights   : 1D numpy array of model weights or gradients
        clip_norm : L2 norm clipping threshold (default 1.0)
    Returns:
        clipped weights — same shape, L2 norm <= clip_norm
    """
    norm = np.linalg.norm(weights)
    if norm > clip_norm:
        return weights * (clip_norm / norm)
    return weights.copy()


def add_gaussian_noise(weights, epsilon, clip_norm=1.0, delta=1e-5):
    """Add calibrated Gaussian noise for (ε, δ)-differential privacy.
    
    Noise is added to aggregated model weights at the SERVER,
    after clipping. This is the correct DP-FedAvg procedure.
    
    Args:
        weights   : aggregated weight vector (after clipping)
        epsilon   : privacy budget — lower = more private
                    ε=1.0 → strong privacy (MedHE paper target)
        clip_norm : must match clip_norm used in clip_weights()
        delta     : failure probability (default 1e-5)
    Returns:
        noisy_weights with (ε, δ)-DP guarantee
        
    Privacy levels:
        ε = 0.1   Very strong privacy, significant accuracy cost
        ε = 1.0   Strong privacy, acceptable accuracy — recommended
        ε = 10.0  Weak privacy, near-original accuracy
    """
    sigma = clip_norm * np.sqrt(2 * np.log(1.25 / delta)) / epsilon
    noise = np.random.normal(0, sigma, size=weights.shape)
    return weights + noise


def sparsify(weight_vector, sparsity=0.975):
    """Adaptive gradient sparsification.
    
    Zeros out the smallest (sparsity * 100)% of weights by magnitude.
    Only the top 2.5% most informative values are transmitted.
    
    From MedHE paper: achieves 97.5% communication reduction
    when applied to neural network weight matrices.
    
    Args:
        weight_vector : numpy array of model weights or gradients
        sparsity      : fraction to zero out (default 0.975)
    Returns:
        sparse_vector    : sparsified weights (same shape)
        compression_rate : fraction of values zeroed out
        
    Example:
        >>> sparse_w, rate = sparsify(weights, sparsity=0.975)
        >>> print(f"Sent {1-rate:.1%} of values")  # → Sent 2.5%
    """
    threshold          = np.percentile(np.abs(weight_vector), sparsity * 100)
    sparse             = weight_vector.copy()
    mask               = np.abs(sparse) < threshold
    sparse[mask]       = 0.0
    return sparse, float(mask.mean())


def dp_fedavg_aggregate(client_weights_list, data_sizes,
                         epsilon=1.0, clip_norm=1.0):
    """Full DP-FedAvg aggregation pipeline.
    
    Implements the complete privacy-preserving aggregation:
    1. Clip each client update
    2. Weighted average
    3. Add Gaussian noise once at server
    
    Args:
        client_weights_list : list of weight arrays (one per client)
        data_sizes          : list of ints (patients per client)
        epsilon             : DP privacy budget (default 1.0)
        clip_norm           : L2 clipping norm (default 1.0)
    Returns:
        noisy_aggregate : differentially private weight aggregate
    """
    total = sum(data_sizes)
    agg   = np.zeros_like(client_weights_list[0], dtype=float)
    
    for weights, n in zip(client_weights_list, data_sizes):
        clipped = clip_weights(weights, clip_norm)
        agg    += clipped * (n / total)
    
    return add_gaussian_noise(agg, epsilon, clip_norm)

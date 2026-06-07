"""
Statistical utilities for RAMC assignment analyses.
=====================================================

Provides:
  - bootstrap_ci: bootstrap confidence intervals for paired differences
  - wilcoxon_signed_rank: wrapper for scipy Wilcoxon test with small-n handling
  - paired_summary: compute paired differences with effect sizes
  - utopia_point_select: normalised utopia-point λ selection (A3)

Used by: A3 (λ-selection), A4 (controller robustness), A1/A2 (reporting)
"""

import numpy as np
from typing import Tuple, Optional


def bootstrap_ci(
    data: np.ndarray,
    n_resamples: int = 10_000,
    ci_level: float = 0.95,
    statistic: str = "mean",
    rng_seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Bootstrap confidence interval for a 1-D array of paired differences.
    
    Parameters
    ----------
    data : array-like
        1-D array of values (typically paired differences Δ_i = RAMC_i - Fidelity_i).
    n_resamples : int
        Number of bootstrap resamples.
    ci_level : float
        Confidence level (default 0.95 -> 95% CI).
    statistic : str
        "mean" or "median".
    rng_seed : int
        For reproducibility.
        
    Returns
    -------
    (point_estimate, ci_lower, ci_upper)
    """
    data = np.asarray(data, dtype=float)
    rng = np.random.default_rng(rng_seed)
    
    stat_fn = np.mean if statistic == "mean" else np.median
    point = float(stat_fn(data))
    
    boot_stats = np.empty(n_resamples)
    n = len(data)
    for i in range(n_resamples):
        sample = data[rng.integers(0, n, size=n)]
        boot_stats[i] = stat_fn(sample)
    
    alpha = 1 - ci_level
    ci_lower = float(np.percentile(boot_stats, 100 * alpha / 2))
    ci_upper = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))
    
    return point, ci_lower, ci_upper


def wilcoxon_signed_rank(
    differences: np.ndarray,
) -> Tuple[float, float, int]:
    """
    Wilcoxon signed-rank test for paired differences.
    
    Parameters
    ----------
    differences : array-like
        Paired differences (RAMC - Fidelity). Negative = RAMC better for cost metrics.
    
    Returns
    -------
    (statistic, p_value, n_pairs)
    
    Notes
    -----
    With n=15 paired observations (3 scenarios × 5 seeds), this is a secondary
    check. Effect sizes and bootstrap CIs carry more weight at this sample size.
    Returns (nan, nan, n) if n < 6 (too few for meaningful test).
    """
    from scipy.stats import wilcoxon
    
    diffs = np.asarray(differences, dtype=float)
    # Remove exact zeros (ties at zero)
    diffs = diffs[diffs != 0.0]
    n = len(diffs)
    
    if n < 6:
        return float("nan"), float("nan"), n
    
    stat, pval = wilcoxon(diffs, alternative="two-sided")
    return float(stat), float(pval), n


def paired_summary(
    ramc_values: np.ndarray,
    baseline_values: np.ndarray,
    metric_name: str = "metric",
    n_bootstrap: int = 10_000,
) -> dict:
    """
    Compute a full paired-difference summary for one metric.
    
    Parameters
    ----------
    ramc_values, baseline_values : array-like
        Matched arrays (same scenario-seed pairing).
    metric_name : str
        Label for the metric (used in the returned dict).
    n_bootstrap : int
        Number of bootstrap resamples for CI.
    
    Returns
    -------
    dict with keys: metric, mean_delta, std_delta, ci_lower, ci_upper,
                    median_delta, wilcoxon_stat, wilcoxon_p, n_pairs,
                    sign_count_negative, sign_count_positive
    """
    ramc = np.asarray(ramc_values, dtype=float)
    base = np.asarray(baseline_values, dtype=float)
    assert len(ramc) == len(base), "Arrays must have same length"
    
    deltas = ramc - base  # negative = RAMC better for cost metrics
    
    mean_d, ci_lo, ci_hi = bootstrap_ci(deltas, n_resamples=n_bootstrap)
    med_d, _, _ = bootstrap_ci(deltas, n_resamples=n_bootstrap, statistic="median")
    w_stat, w_p, n_eff = wilcoxon_signed_rank(deltas)
    
    return {
        "metric": metric_name,
        "n_pairs": len(deltas),
        "mean_delta": mean_d,
        "std_delta": float(np.std(deltas, ddof=1)),
        "median_delta": float(np.median(deltas)),
        "ci_95_lower": ci_lo,
        "ci_95_upper": ci_hi,
        "ci_excludes_zero": not (ci_lo <= 0 <= ci_hi),
        "wilcoxon_stat": w_stat,
        "wilcoxon_p": w_p,
        "sign_negative": int(np.sum(deltas < 0)),  # RAMC better
        "sign_positive": int(np.sum(deltas > 0)),  # Fidelity better
        "sign_zero": int(np.sum(deltas == 0)),
    }


def utopia_point_select(
    lambdas: np.ndarray,
    fidelity_losses: np.ndarray,
    risk_scores: np.ndarray,
) -> Tuple[int, float, np.ndarray]:
    """
    Normalised utopia-point method for λ selection (Assignment 3).
    
    Both axes are normalised to [0, 1] using min-max scaling over the 
    observed λ-sweep. The utopia point is (0, 0). The selected λ has 
    the smallest Euclidean distance to utopia.
    
    Parameters
    ----------
    lambdas : array-like
        λ values (including 0 for fidelity baseline).
    fidelity_losses : array-like
        L_fid for each λ value.
    risk_scores : array-like
        R_total (or R_comfort) for each λ value.
    
    Returns
    -------
    (best_index, best_lambda, normalised_distances)
    """
    lam = np.asarray(lambdas, dtype=float)
    fid = np.asarray(fidelity_losses, dtype=float)
    risk = np.asarray(risk_scores, dtype=float)
    
    # Normalise to [0, 1]
    fid_min, fid_max = fid.min(), fid.max()
    risk_min, risk_max = risk.min(), risk.max()
    
    # Guard against zero range
    fid_range = fid_max - fid_min if fid_max > fid_min else 1.0
    risk_range = risk_max - risk_min if risk_max > risk_min else 1.0
    
    fid_norm = (fid - fid_min) / fid_range
    risk_norm = (risk - risk_min) / risk_range
    
    # Euclidean distance to utopia (0, 0)
    distances = np.sqrt(fid_norm**2 + risk_norm**2)
    
    best_idx = int(np.argmin(distances))
    best_lam = float(lam[best_idx])
    
    return best_idx, best_lam, distances


def gap_to_exact_model(
    metric_fidelity: float,
    metric_ramc: float,
    metric_exact: float,
) -> float:
    """
    Compute the fraction of the model-approximation gap closed by RAMC.
    
    Gap closed = (M_fidelity - M_ramc) / (M_fidelity - M_exact)
    
    Returns 0.0 if RAMC offers no improvement, 1.0 if RAMC matches
    the exact-model benchmark. Can exceed 1.0 or be negative.
    
    Used by: A2 (benchmark sufficiency)
    """
    denom = metric_fidelity - metric_exact
    if abs(denom) < 1e-12:
        return float("nan")  # Fidelity already matches exact model
    return (metric_fidelity - metric_ramc) / denom

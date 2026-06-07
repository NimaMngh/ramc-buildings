# -*- coding: utf-8 -*-
"""
RAMC loss functions and evaluation utilities (open-loop, one-step).

The RU-CVaR estimator works under torch.no_grad() via torch.enable_grad().
The occupancy split uses the full cost config, and the loss early-exits when
lambda_risk=0 to skip the CVaR computation.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple, Any, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# P1: Shared fidelity loss function (single source of truth)
# =============================================================================

def compute_fidelity_loss(
    preds: torch.Tensor,
    targets: torch.Tensor,
    output_std: Optional[torch.Tensor] = None,
    mse_normalize: bool = True,
    fidelity_weights: Optional[List[float]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    P1: Unified fidelity loss computation (normalized + weighted MSE).
    
    This is the single source of truth for fidelity loss across all trainers,
    ensuring apples-to-apples comparison between MSE baseline, FidelityOnly, and RAMC.
    
    Args:
        preds: [B, n_state] predicted next states
        targets: [B, n_state] ground truth next states
        output_std: [n_state] standard deviation for normalization (from model)
        mse_normalize: Whether to normalize by output_std
        fidelity_weights: [n_state] weights per state dimension
        
    Returns:
        fidelity_loss: scalar weighted (optionally normalized) MSE
        per_dim_mse: [n_state] per-dimension MSE (before weighting, after normalization)
    """
    if preds.ndim != 2 or targets.ndim != 2:
        raise ValueError(f"preds and targets must be [B, n_state], got {preds.shape}, {targets.shape}")
    
    err2 = (preds - targets) ** 2  # [B, n_state]
    
    if mse_normalize and output_std is not None:
        std = torch.as_tensor(output_std, device=preds.device, dtype=preds.dtype)
        if std.ndim == 0:
            std = std.unsqueeze(0)
        err2 = err2 / (std ** 2 + 1e-12)
    
    per_dim_mse = err2.mean(dim=0)  # [n_state]
    
    if fidelity_weights is None:
        weights = torch.ones_like(per_dim_mse)
    else:
        weights = torch.as_tensor(fidelity_weights, device=preds.device, dtype=preds.dtype)
    
    fidelity_loss = (weights * per_dim_mse).sum()
    
    return fidelity_loss, per_dim_mse


def compute_raw_mse(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    P1: Raw MSE (unweighted, unnormalized) for baseline comparison.
    """
    return torch.mean((preds - targets) ** 2)


# =============================================================================
# Rollout-aware fidelity loss (multi-step autoregressive)
# =============================================================================

def compute_rollout_fidelity_loss(
    model: nn.Module,
    x0_batch: torch.Tensor,
    controls_seq: torch.Tensor,
    disturbances_seq: torch.Tensor,
    targets_seq: torch.Tensor,
    output_std: Optional[torch.Tensor] = None,
    mse_normalize: bool = True,
    fidelity_weights: Optional[List[float]] = None,
    step_weight_mode: str = "linear",
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Multi-step autoregressive rollout fidelity loss.

    Rolls the model forward for H_r steps using ground-truth controls and
    disturbances (open-loop), feeding the predicted state back as the next
    input state.  The fidelity error at each step is measured using the same
    ``compute_fidelity_loss`` function used for one-step training, ensuring
    identical normalisation and weighting throughout.

    Args:
        model:              Neural network f_θ(x, u, d) -> x̂_next
        x0_batch:           [B_r, nx]       Initial states (ground truth)
        controls_seq:       [B_r, H_r, nu]  Ground-truth control sequences
        disturbances_seq:   [B_r, H_r, nd]  Ground-truth disturbance sequences
        targets_seq:        [B_r, H_r, nx]  Ground-truth next-state sequences
        output_std:         [nx]            Per-dimension std for normalisation
        mse_normalize:      Whether to normalise errors by output_std
        fidelity_weights:   [nx]            Per-dimension loss weights
        step_weight_mode:   "linear"  -> w_h = h/H_r  (emphasises later steps)
                            "uniform" -> w_h = 1/H_r

    Returns:
        rollout_loss: scalar — weighted average fidelity over H_r steps and B_r seqs
        info: dict with
            "per_step_mse"  : [H_r, nx] per-step per-dimension MSE
            "H_r"           : int rollout horizon used
    """
    if controls_seq.ndim != 3:
        raise ValueError(
            f"controls_seq must be [B_r, H_r, nu], got {tuple(controls_seq.shape)}"
        )
    if disturbances_seq.ndim != 3:
        raise ValueError(
            f"disturbances_seq must be [B_r, H_r, nd], got {tuple(disturbances_seq.shape)}"
        )
    if targets_seq.ndim != 3:
        raise ValueError(
            f"targets_seq must be [B_r, H_r, nx], got {tuple(targets_seq.shape)}"
        )

    B_r, H_r, _ = controls_seq.shape
    device = x0_batch.device
    dtype = x0_batch.dtype

    # Step weights
    if step_weight_mode == "linear":
        # w_h = h/H_r for h=1..H_r, then normalised to sum to 1
        w = torch.arange(1, H_r + 1, dtype=dtype, device=device)
        w = w / w.sum()
    elif step_weight_mode == "uniform":
        w = torch.ones(H_r, dtype=dtype, device=device) / float(H_r)
    else:
        raise ValueError(f"Unknown step_weight_mode: {step_weight_mode!r}")

    # Autoregressive rollout
    x_hat = x0_batch  # [B_r, nx]
    total_loss = torch.tensor(0.0, dtype=dtype, device=device)
    per_step_mse: List[torch.Tensor] = []

    for h in range(H_r):
        x_hat = model(x_hat, controls_seq[:, h], disturbances_seq[:, h])  # [B_r, nx]

        step_fid, step_per_dim = compute_fidelity_loss(
            x_hat,
            targets_seq[:, h],
            output_std=output_std,
            mse_normalize=mse_normalize,
            fidelity_weights=fidelity_weights,
        )

        total_loss = total_loss + w[h] * step_fid
        per_step_mse.append(step_per_dim.detach())

    info: Dict[str, Any] = {
        "per_step_mse": torch.stack(per_step_mse),  # [H_r, nx]
        "H_r": H_r,
    }

    return total_loss, info


# =============================================================================
# P4: Energy proxy computation
# =============================================================================

def compute_energy_Q_W(
    T_ret: torch.Tensor,
    controls: torch.Tensor,
    Cp_water: float = 4186.0,
) -> torch.Tensor:
    """
    P4: Compute energy proxy Q = mdot * Cp * max(T_supply - T_ret, 0).
    
    Args:
        T_ret: [B] return temperature
        controls: [B, 2] (T_supply, mdot)
        Cp_water: specific heat capacity (J/kgÂ·K)
        
    Returns:
        Q_W: [B] heat transfer rate in Watts
    """
    T_supply = controls[:, 0]
    mdot = torch.clamp(controls[:, 1], min=0.0)
    return mdot * float(Cp_water) * torch.relu(T_supply - T_ret)


def compute_energy_proxy_metrics(
    preds: torch.Tensor,
    targets: torch.Tensor,
    controls: torch.Tensor,
    t_ret_index: int = 5,
    Cp_water: float = 4186.0,
) -> Dict[str, float]:
    """
    P4: Compute energy proxy error metrics.
    
    Returns:
        Dict with Q_rmse, Q_mae, Q_bias, Q_abs_p95, Q_abs_p99
    """
    T_ret_pred = preds[:, t_ret_index]
    T_ret_true = targets[:, t_ret_index]
    
    Q_pred = compute_energy_Q_W(T_ret_pred, controls, Cp_water)
    Q_true = compute_energy_Q_W(T_ret_true, controls, Cp_water)
    
    diff = Q_pred - Q_true
    abs_diff = diff.abs()
    
    return {
        "Q_rmse": float(torch.sqrt(torch.mean(diff ** 2)).item()),
        "Q_mae": float(torch.mean(abs_diff).item()),
        "Q_bias": float(torch.mean(diff).item()),
        "Q_abs_p95": float(torch.quantile(abs_diff, 0.95).item()) if len(abs_diff) > 20 else float('nan'),
        "Q_abs_p99": float(torch.quantile(abs_diff, 0.99).item()) if len(abs_diff) > 100 else float('nan'),
    }


# =============================================================================
# Stage cost (comfort + energy proxy)
# =============================================================================

def stage_cost_ramc(
    next_states_pred: torch.Tensor,
    controls: torch.Tensor,
    Tmin: Optional[torch.Tensor] = None,
    Tmax: Optional[torch.Tensor] = None,
    comfort_bounds: Tuple[float, float] = (20.0, 22.0),
    comfort_hinge: str = "softplus",
    comfort_beta: float = 0.5,
    t_ret_index: int = 5,
    dt_minutes: float = 10.0,
    energy_cost_rate: float = 0.9,
    Cp_water: float = 4186.0,
    w_comfort: float = 63.0,
    w_energy: float = 1.0,
    cost_scale: float = 1.0,
    return_components: bool = False,
) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Differentiable one-step stage cost on model predictions.
    
    Args:
        next_states_pred: [B, 6] predicted next state
        controls:         [B, 2] (T_supply, mdot)
        Tmin/Tmax:        Optional [B] schedule-aware comfort bounds
        comfort_bounds:   Fallback (Tmin, Tmax) if None
        comfort_hinge:    "relu" or "softplus"
        t_ret_index:      Which state is T_ret (default 5)
        dt_minutes:       Timestep duration
        energy_cost_rate: District heating cost per kWh (SEK/kWh)
        w_comfort, w_energy, cost_scale: Weights

    Returns:
        cost_per_sample: [B]
        If return_components=True, also returns dict with comfort, energy_cost, Q_W
    """
    if next_states_pred.ndim != 2 or next_states_pred.size(1) < 6:
        raise ValueError(f"next_states_pred must be [B,6+], got {tuple(next_states_pred.shape)}")
    if controls.ndim != 2 or controls.size(1) != 2:
        raise ValueError(f"controls must be [B,2], got {tuple(controls.shape)}")

    T_air = next_states_pred[:, 0]
    T_ret = next_states_pred[:, t_ret_index]

    T_supply = controls[:, 0]
    mdot = controls[:, 1]

    # Comfort bounds
    if (Tmin is not None) and (Tmax is not None):
        T_min = Tmin.view(-1).to(device=T_air.device, dtype=T_air.dtype)
        T_max = Tmax.view(-1).to(device=T_air.device, dtype=T_air.dtype)
    else:
        Tmin_val, Tmax_val = comfort_bounds
        T_min = torch.full_like(T_air, float(Tmin_val))
        T_max = torch.full_like(T_air, float(Tmax_val))

    # Comfort penalty
    if comfort_hinge.lower() == "relu":
        comfort = torch.relu(T_min - T_air) + torch.relu(T_air - T_max)
    elif comfort_hinge.lower() == "softplus":
        beta = float(comfort_beta)
        comfort = beta * F.softplus((T_min - T_air) / beta) + beta * F.softplus((T_air - T_max) / beta)
    else:
        raise ValueError(f"Unknown comfort_hinge='{comfort_hinge}'. Use 'relu' or 'softplus'.")

    # Safety clamping for mdot
    mdot_eff = torch.clamp(mdot, min=0.0)
    
    # Energy proxy using return temperature
    Q_W = mdot_eff * Cp_water * torch.relu(T_supply - T_ret)

    dt_hours = float(dt_minutes) / 60.0
    energy_kWh = Q_W * dt_hours / 1000.0
    energy_cost = energy_kWh * float(energy_cost_rate)

    cost = float(cost_scale) * (float(w_comfort) * comfort + float(w_energy) * energy_cost)

    if return_components:
        return cost, {
            "comfort": comfort,
            "energy_cost": energy_cost,
            "Q_W": Q_W,
            "T_air": T_air,
            "T_ret": T_ret,
        }

    return cost


# =============================================================================
# P2: Configurable Gaussian perturbations with clamp bounds + P5A: Antithetic sampling
# =============================================================================

def sample_gaussian_perturbations(
    states: torch.Tensor,
    controls: torch.Tensor,
    disturbances: torch.Tensor,
    num_perturbations: int,
    sigma_state: float = 1.0,
    sigma_rad_scale: float = 0.5,
    sigma_T_supply: float = 0.5,
    sigma_mdot: float = 0.01,
    sigma_T_out: float = 1.0,
    sigma_Q_solar: float = 500.0,
    sigma_Q_internal: float = 200.0,
    clamp_physical: bool = True,
    clamp_bounds: Optional[Dict[str, Dict[str, Tuple[float, float]]]] = None,  # P2: Configurable
    p_occupancy_flip: float = 0.0,
    q_internal_nominal: float = 1000.0,
    use_antithetic: bool = True,  # P5A: Antithetic sampling
    return_clamp_stats: bool = False,  # P2: Clamp hit rate logging
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]]:
    """
    P2: Configurable clamp bounds + P5A: Antithetic sampling for variance reduction.
    
    Antithetic sampling: For even K, sample K/2 noise vectors and mirror them.
    This creates negatively correlated pairs, reducing variance without increasing K.
    
    Args:
        clamp_bounds: Dict with structure:
            {
                "state": {"T_air": (lo, hi), "T_env": (lo, hi), ...},
                "control": {"T_supply": (lo, hi), "mdot": (lo, hi)},
                "dist": {"T_out": (lo, hi), "Q_solar": (lo, hi), "Q_internal": (lo, hi)}
            }
        use_antithetic: If True and K is even, use antithetic sampling
        return_clamp_stats: If True, return clamp hit rates
        
    Returns:
        states_p:       [B, K, 6]
        controls_p:     [B, K, 2]
        disturbances_p: [B, K, 2 or 3]
        (optional) clamp_stats: Dict with clamp hit rates per variable
    """
    if num_perturbations < 1:
        raise ValueError("num_perturbations must be >= 1")

    B = states.size(0)
    K = int(num_perturbations)
    device = states.device
    dtype = states.dtype

    s = states.unsqueeze(1).expand(B, K, -1).contiguous()
    c = controls.unsqueeze(1).expand(B, K, -1).contiguous()
    d = disturbances.unsqueeze(1).expand(B, K, -1).contiguous()

    # P5A: Antithetic sampling for Gaussian noise
    if use_antithetic and K >= 2 and K % 2 == 0:
        K_half = K // 2
        
        # State noise (antithetic)
        s_eps = torch.randn(B, K_half, states.size(1), device=device, dtype=dtype)
        s_eps = torch.cat([s_eps, -s_eps], dim=1)  # [B, K, 6]
        s_noise = s_eps * float(sigma_state)
        s_noise[..., 3:6] *= float(sigma_rad_scale)
        
        # Control noise (antithetic)
        c_eps = torch.randn(B, K_half, 2, device=device, dtype=dtype)
        c_eps = torch.cat([c_eps, -c_eps], dim=1)  # [B, K, 2]
        c_noise = torch.zeros_like(c)
        c_noise[..., 0] = c_eps[..., 0] * float(sigma_T_supply)
        c_noise[..., 1] = c_eps[..., 1] * float(sigma_mdot)
        
        # Disturbance noise (antithetic)
        d_eps = torch.randn(B, K_half, disturbances.size(1), device=device, dtype=dtype)
        d_eps = torch.cat([d_eps, -d_eps], dim=1)  # [B, K, D]
        d_noise = torch.zeros_like(d)
        d_noise[..., 0] = d_eps[..., 0] * float(sigma_T_out)
        if d.size(-1) > 1:
            d_noise[..., 1] = d_eps[..., 1] * float(sigma_Q_solar)
        if d.size(-1) > 2:
            d_noise[..., 2] = d_eps[..., 2] * float(sigma_Q_internal)
    else:
        # Standard independent sampling
        s_noise = torch.randn_like(s) * float(sigma_state)
        s_noise[..., 3:6] *= float(sigma_rad_scale)
        
        c_noise = torch.zeros_like(c)
        c_noise[..., 0] = torch.randn_like(c[..., 0]) * float(sigma_T_supply)
        c_noise[..., 1] = torch.randn_like(c[..., 1]) * float(sigma_mdot)
        
        d_noise = torch.zeros_like(d)
        d_noise[..., 0] = torch.randn_like(d[..., 0]) * float(sigma_T_out)
        if d.size(-1) > 1:
            d_noise[..., 1] = torch.randn_like(d[..., 1]) * float(sigma_Q_solar)
        if d.size(-1) > 2:
            d_noise[..., 2] = torch.randn_like(d[..., 2]) * float(sigma_Q_internal)

    s_p = s + s_noise
    c_p = c + c_noise
    d_p = d + d_noise
    
    # Occupancy flip perturbations (regime uncertainty)
    if p_occupancy_flip > 0.0 and d_p.size(-1) > 2:
        flip_mask = (torch.rand(B, K, device=device) < float(p_occupancy_flip))
        q_internal = d_p[..., 2]
        q_flipped = torch.where(
            q_internal > 0.5 * float(q_internal_nominal),
            torch.zeros_like(q_internal),
            torch.full_like(q_internal, float(q_internal_nominal))
        )
        d_p[..., 2] = torch.where(flip_mask, q_flipped, q_internal)

    # P2: Configurable clamping with hit rate tracking
    clamp_stats = {}
    
    if clamp_physical:
        # Get bounds (use configurable or fallback to defaults)
        if clamp_bounds is not None:
            sb = clamp_bounds.get("state", {})
            cb = clamp_bounds.get("control", {})
            db = clamp_bounds.get("dist", {})
        else:
            # Fallback defaults
            sb = {
                "T_air": (10.0, 40.0), "T_env": (10.0, 40.0), "T_int": (10.0, 40.0),
                "T_rad1": (25.0, 70.0), "T_rad2": (25.0, 70.0), "T_ret": (25.0, 70.0)
            }
            cb = {"T_supply": (25.0, 70.0), "mdot": (0.0, 0.5)}
            db = {"T_out": (-30.0, 40.0), "Q_solar": (0.0, float('inf')), 
                  "Q_internal": (0.0, 1500.0)}
        
        state_names = ["T_air", "T_env", "T_int", "T_rad1", "T_rad2", "T_ret"]
        for i, name in enumerate(state_names):
            if name in sb:
                lo, hi = sb[name]
                before = s_p[..., i].clone()
                s_p[..., i] = s_p[..., i].clamp(min=lo, max=hi)
                if return_clamp_stats:
                    hit_rate = float(((before < lo) | (before > hi)).float().mean().item())
                    clamp_stats[f"state_{name}_clamp_rate"] = hit_rate
        
        control_names = ["T_supply", "mdot"]
        for i, name in enumerate(control_names):
            if name in cb:
                lo, hi = cb[name]
                before = c_p[..., i].clone()
                c_p[..., i] = c_p[..., i].clamp(min=lo, max=hi)
                if return_clamp_stats:
                    hit_rate = float(((before < lo) | (before > hi)).float().mean().item())
                    clamp_stats[f"control_{name}_clamp_rate"] = hit_rate
        
        dist_names = ["T_out", "Q_solar", "Q_internal"]
        for i, name in enumerate(dist_names):
            if i < d_p.size(-1) and name in db:
                lo, hi = db[name]
                before = d_p[..., i].clone()
                d_p[..., i] = d_p[..., i].clamp(min=lo, max=hi)
                if return_clamp_stats:
                    hit_rate = float(((before < lo) | (before > hi)).float().mean().item())
                    clamp_stats[f"dist_{name}_clamp_rate"] = hit_rate

    if return_clamp_stats:
        return s_p, c_p, d_p, clamp_stats
    return s_p, c_p, d_p


# =============================================================================
# P9: Fixed RU-CVaR to work under torch.no_grad()
# =============================================================================

def compute_cvar_ru(
    cost_samples: torch.Tensor,
    alpha: float = 0.9,
    n_eta_steps: int = 10,
    eta_lr: float = 0.2,
    eta_init: str = "median",
    detach_eta: bool = True,
) -> torch.Tensor:
    """
    Canonical empirical Rockafellar-Uryasev CVaR that works under torch.no_grad().
    
    Uses torch.enable_grad() internally for Î· optimization, allowing this to be called
    from evaluation functions that use torch.no_grad() context.
    
    CVaR_alpha(Z) = min_eta [ eta + 1/(1-alpha) * mean_k relu(Z_k - eta) ]

    Args:
        cost_samples: [B, K] cost samples under perturbations
        alpha: CVaR confidence level (e.g., 0.9 = worst 10%)
        n_eta_steps: Number of gradient steps for eta optimization
        eta_lr: Learning rate for eta updates
        eta_init: Initialization strategy ("mean", "median", or "quantile")
        detach_eta: Whether to detach eta_star before final CVaR computation
        
    Returns:
        cvar: [B] CVaR values (differentiable w.r.t. cost_samples when gradients enabled)
    """
    if cost_samples.ndim != 2:
        raise ValueError(f"cost_samples must be [B,K], got {tuple(cost_samples.shape)}")

    alpha = float(alpha)
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0,1)")

    B, K = cost_samples.shape
    device = cost_samples.device
    dtype = cost_samples.dtype

    # Detach cost_samples for Î· optimization (memory efficiency)
    cost_det = cost_samples.detach()

    # Initialize eta (detached)
    with torch.no_grad():
        if eta_init == "mean":
            eta = cost_det.mean(dim=1)
        elif eta_init == "median":
            eta = cost_det.median(dim=1).values
        elif eta_init == "quantile":
            eta = torch.quantile(cost_det, alpha, dim=1)
        else:
            raise ValueError("eta_init must be 'mean', 'median', or 'quantile'")

    # Inner optimization over eta using torch.enable_grad()
    # This allows CVaR to work even when called under torch.no_grad() context
    for _ in range(int(n_eta_steps)):
        with torch.enable_grad():
            eta = eta.detach().requires_grad_(True)
            excess = torch.relu(cost_det - eta.unsqueeze(1))
            obj = eta + excess.mean(dim=1) / (1.0 - alpha)
            grad_eta = torch.autograd.grad(obj.sum(), eta, retain_graph=False)[0]

        with torch.no_grad():
            eta = eta - float(eta_lr) * grad_eta

    eta_star = eta.detach() if detach_eta else eta

    # Final CVaR computation (uses original cost_samples with gradients if enabled)
    excess = torch.relu(cost_samples - eta_star.unsqueeze(1))
    cvar = eta_star + excess.mean(dim=1) / (1.0 - alpha)
    
    return cvar


def compute_var_empirical(
    cost_samples: torch.Tensor,
    alpha: float = 0.9,
    dim: int = 1,
) -> torch.Tensor:
    """
    Empirical (sorting-based) VaR along `dim`.
    Definition: k = ceil(alpha * m), VaR = x_(k).
    """
    if not (0.0 <= alpha < 1.0):
        raise ValueError(f"alpha must satisfy 0 <= alpha < 1, got {alpha}")
    x = cost_samples.movedim(dim, -1)
    m = x.shape[-1]
    if m < 1:
        raise ValueError("cost_samples must have at least 1 sample along `dim`")
    k = int(math.ceil(alpha * m))
    k = max(1, min(k, m))
    x_sorted, _ = torch.sort(x, dim=-1)
    var = x_sorted[..., k - 1]
    return var


def compute_cvar_empirical(
    cost_samples: torch.Tensor,
    alpha: float = 0.9,
    dim: int = 1,
) -> torch.Tensor:
    """
    Empirical (sorting-based) CVaR along `dim`, evaluation-only.
    Uses integral-of-quantile definition:
      CVaR_alpha = (1/(1-alpha)) * integral_{alpha}^{1} VaR_u du
    For equally weighted samples x_(1)<=...<=x_(m), k=ceil(alpha*m):
      integral = (k/m - alpha)*x_(k) + (1/m)*sum_{i=k+1..m} x_(i)
    """
    if not (0.0 <= alpha < 1.0):
        raise ValueError(f"alpha must satisfy 0 <= alpha < 1, got {alpha}")
    x = cost_samples.movedim(dim, -1)
    m = x.shape[-1]
    if m < 1:
        raise ValueError("cost_samples must have at least 1 sample along `dim`")
    k = int(math.ceil(alpha * m))
    k = max(1, min(k, m))
    x_sorted, _ = torch.sort(x, dim=-1)
    x_k = x_sorted[..., k - 1]
    tail_sum_exclusive = x_sorted[..., k:].sum(dim=-1)
    m_float = float(m)
    tail_integral = ((k / m_float) - alpha) * x_k + (tail_sum_exclusive / m_float)
    cvar = tail_integral / (1.0 - alpha)
    return cvar

# =============================================================================
# Risk operators - Added .strip() sanitization
# =============================================================================

def risk_from_cost_samples(
    cost_samples: torch.Tensor,
    risk_operator: str = "std",
    cvar_alpha: float = 0.9,
    cvar_method: str = "ru",
    cvar_n_steps: int = 10,
    cvar_eta_lr: float = 0.2,
    cvar_eta_init: str = "median",
) -> torch.Tensor:
    """
    Apply risk operator to cost samples.
    
    Supports:
    - "variance": Sample variance of costs
    - "mean"/"expectation": Expected cost over perturbations (A2 ablation)
    - "std": Standard deviation (default, more interpretable than variance)
    - "cvar"/"cvar_ru": Canonical Rockafellar-Uryasev CVaR
    - "cvar_quantile": Simple quantile-based approximation

    Args:
        cost_samples: [B, K] induced costs under K perturbations
        
    Returns:
        risk_per_sample: [B]
    """
    if cost_samples.ndim != 2:
        raise ValueError(f"cost_samples must be [B,K], got {tuple(cost_samples.shape)}")

    # Sanitize risk_operator string (handle whitespace and case)
    op = str(risk_operator).strip().lower()
    
    if op == "variance":
        return cost_samples.var(dim=1, unbiased=False)
    
    # A2 ablation: arithmetic mean over the K perturbation samples (no tail focus)
    if op in ("mean", "expectation", "expected"):
        return cost_samples.mean(dim=1)
    
    if op == "std":
        return torch.sqrt(cost_samples.var(dim=1, unbiased=False) + 1e-8)

    if op in ("cvar", "cvar_ru", "cvar_quantile"):
        alpha = float(cvar_alpha)
        if not (0.0 < alpha < 1.0):
            raise ValueError("cvar_alpha must be in (0, 1)")
        
        if op == "cvar_quantile":
            method = "quantile"
        else:
            method = cvar_method if op == "cvar" else "ru"
        
        if method == "ru":
            return compute_cvar_ru(
                cost_samples,
                alpha=alpha,
                n_eta_steps=cvar_n_steps,
                eta_lr=cvar_eta_lr,
                eta_init=cvar_eta_init,
                detach_eta=True,
            )
        
        elif method == "empirical":
            if cost_samples.requires_grad:
                raise ValueError(
                    "cvar_method='empirical' is evaluation-only (non-smooth). "
                    "Use 'ru' for training."
                )
            return compute_cvar_empirical(cost_samples, alpha=alpha, dim=1)
        
        elif method == "quantile":
            var_alpha = torch.quantile(cost_samples, alpha, dim=1, keepdim=True)
            excess = torch.relu(cost_samples - var_alpha)
            expected_shortfall = excess.mean(dim=1)
            cvar = var_alpha.squeeze(1) + expected_shortfall / (1.0 - alpha)
            return cvar
        
        else:
            raise ValueError(f"Unknown cvar_method='{method}'. Use 'ru', 'empirical', or 'quantile'.")

    raise ValueError(f"Unknown risk_operator='{risk_operator}' (sanitized: '{op}'). "
                     f"Use 'std', 'variance', 'mean', 'cvar', 'cvar_ru', 'cvar_quantile', or set cvar_method='empirical'.")
# =============================================================================
# RAMC loss - Early-exit when lambda_risk=0 to skip expensive CVaR
# =============================================================================

def calculate_ramc_loss(
    model: nn.Module,
    states: torch.Tensor,
    controls: torch.Tensor,
    disturbances: torch.Tensor,
    targets: torch.Tensor,
    *,
    lambda_risk: float = 1.0,
    risk_operator: str = "std",
    cvar_alpha: float = 0.9,
    cvar_method: str = "ru",
    cvar_n_steps: int = 10,
    cvar_eta_lr: float = 0.2,
    cvar_eta_init: str = "median",
    num_perturbations: int = 8,
    perturb_forward_chunk_size: Optional[int] = None,
    sigma_state: float = 1.0,
    sigma_rad_scale: float = 0.5,
    sigma_T_supply: float = 0.5,
    sigma_mdot: float = 0.01,
    sigma_T_out: float = 1.0,
    sigma_Q_solar: float = 500.0,
    sigma_Q_internal: float = 200.0,
    clamp_physical: bool = True,
    clamp_bounds: Optional[Dict] = None,  # P2: Configurable clamp bounds
    use_antithetic: bool = True,  # P5A: Antithetic sampling
    p_occupancy_flip: float = 0.0,
    q_internal_nominal: float = 1000.0,
    Tmin: Optional[torch.Tensor] = None,
    Tmax: Optional[torch.Tensor] = None,
    comfort_bounds: Tuple[float, float] = (20.0, 22.0),
    comfort_hinge: str = "softplus",
    comfort_beta: float = 0.5,
    dt_minutes: float = 10.0,
    energy_cost_rate: float = 0.9,
    w_comfort: float = 63.0,
    w_energy: float = 1.0,
    cost_scale: float = 1.0,
    t_ret_index: int = 5,
    mse_normalize: bool = True,
    fidelity_weights: Optional[list] = None,
    return_clamp_stats: bool = False,  # P2: Return clamp hit rates
    skip_risk_if_lambda_zero: bool = True,  # Skip expensive risk computation when lambda=0
    loss_mode: str = "ramc",   # "ramc" | "pert_only"
    perturbed_states:   Optional[torch.Tensor] = None,  # [B,K_label,state_dim]
    perturbed_controls: Optional[torch.Tensor] = None,  # [B,K_label,control_dim]
    perturbed_disturb:  Optional[torch.Tensor] = None,  # [B,K_label,disturbance_dim]
    perturbed_targets:  Optional[torch.Tensor] = None,  # [B,K_label,output_dim]
) -> Dict[str, Any]:
    """
    RAMC loss with P1-P10 improvements and performance fixes.
    
    When lambda_risk=0 and skip_risk_if_lambda_zero=True, skips the expensive
    perturbation sampling and risk computation entirely. This is critical for CVaR
    where K>=16 and the computation is expensive.
    
    Returns dict with:
    - fidelity_loss, per_dim_fid_mse (P1)
    - mse_raw (P1)
    - risk_loss, risk_comfort_loss, risk_energy_loss (P10)
    - expected_cost, expected_comfort, expected_energy_cost (P10)
    - total_loss
    - clamp_stats (P2, if return_clamp_stats=True)
    """
    # Sanitize risk_operator early
    risk_op_clean = str(risk_operator).strip().lower()
    loss_mode_clean = str(loss_mode).strip().lower()
    
    # Practical warning for CVaR with small K (only if we'll actually compute risk)
    if (float(lambda_risk) > 0.0 or not skip_risk_if_lambda_zero):
        if risk_op_clean.startswith("cvar") and cvar_alpha >= 0.9 and num_perturbations < 16:
            import warnings
            warnings.warn(
                f"CVaR with alpha={cvar_alpha} and K={num_perturbations} may be unstable. "
                f"Recommended: num_perturbations >= 16 for reliable tail estimation.",
                UserWarning
            )
    
    # Forward for nominal prediction (fidelity)
    preds = model(states, controls, disturbances)

    # P1: Use shared fidelity loss function
    output_std = getattr(model, "output_std", None) if mse_normalize else None
    fidelity_loss, per_dim_fid_mse = compute_fidelity_loss(
        preds, targets,
        output_std=output_std,
        mse_normalize=mse_normalize and getattr(model, "normalization_computed", False),
        fidelity_weights=fidelity_weights,
    )
    
    # P1: Also compute raw MSE for comparison
    mse_raw = compute_raw_mse(preds, targets)

    # P10: Stage cost on nominal predictions with components
    nominal_cost_per_sample, nominal_comp = stage_cost_ramc(
        preds, controls,
        Tmin=Tmin, Tmax=Tmax,
        comfort_bounds=comfort_bounds,
        comfort_hinge=comfort_hinge,
        comfort_beta=comfort_beta,
        t_ret_index=t_ret_index,
        dt_minutes=dt_minutes,
        energy_cost_rate=energy_cost_rate,
        w_comfort=w_comfort, w_energy=w_energy,
        cost_scale=cost_scale,
        return_components=True,
    )
    expected_cost = nominal_cost_per_sample.mean()
    expected_comfort = nominal_comp["comfort"].mean()
    expected_energy_cost = nominal_comp["energy_cost"].mean()

    # Early-exit when lambda_risk=0 to skip expensive risk computation
    if float(lambda_risk) == 0.0 and skip_risk_if_lambda_zero:
        # Return zeros for risk metrics - no perturbation sampling needed
        device = states.device
        dtype = states.dtype
        
        result = {
            # P1: Fidelity metrics
            "fidelity_loss": fidelity_loss,
            "per_dim_fid_mse": per_dim_fid_mse.detach(),
            "mse_raw": mse_raw,
            
            # P10: Decomposed expected cost
            "expected_cost": expected_cost,
            "expected_comfort": expected_comfort,
            "expected_energy_cost": expected_energy_cost,
            
            # P10: Decomposed risk (zeros since lambda=0)
            "risk_loss": torch.tensor(0.0, device=device, dtype=dtype),
            "risk_comfort_loss": torch.tensor(0.0, device=device, dtype=dtype),
            "risk_energy_loss": torch.tensor(0.0, device=device, dtype=dtype),
            
            # Total (just fidelity when lambda=0)
            "total_loss": fidelity_loss,
            
            # Config info
            "lambda_risk": 0.0,
            "risk_operator": risk_op_clean,
            "loss_mode": loss_mode_clean,
            "cvar_alpha": float(cvar_alpha),
            "num_perturbations": int(num_perturbations),
            "mse_normalized": bool(mse_normalize),
            "use_antithetic": bool(use_antithetic),
            "risk_skipped": True,  # Flag indicating risk was skipped
        }
        
        if return_clamp_stats:
            result["clamp_stats"] = {}
        
        return result

    # Risk term: compute induced costs under perturbations
    is_training = model.training
    model.eval()

    if loss_mode_clean == "ramc":
        # ─── Default RAMC path: sample perturbations, stage cost, risk op ───
        B = states.size(0)
        K = int(num_perturbations)

        perturb_result = sample_gaussian_perturbations(
            states, controls, disturbances,
            num_perturbations=K,
            sigma_state=sigma_state,
            sigma_rad_scale=sigma_rad_scale,
            sigma_T_supply=sigma_T_supply,
            sigma_mdot=sigma_mdot,
            sigma_T_out=sigma_T_out,
            sigma_Q_solar=sigma_Q_solar,
            sigma_Q_internal=sigma_Q_internal,
            clamp_physical=clamp_physical,
            clamp_bounds=clamp_bounds,
            p_occupancy_flip=p_occupancy_flip,
            q_internal_nominal=q_internal_nominal,
            use_antithetic=use_antithetic,
            return_clamp_stats=return_clamp_stats,
        )
        if return_clamp_stats:
            s_p, c_p, d_p, clamp_stats = perturb_result
        else:
            s_p, c_p, d_p = perturb_result
            clamp_stats = {}

        s_flat = s_p.reshape(B * K, -1)
        c_flat = c_p.reshape(B * K, -1)
        d_flat = d_p.reshape(B * K, -1)

        if (Tmin is not None) and (Tmax is not None):
            Tmin_flat = Tmin.view(B, 1).expand(B, K).reshape(B * K)
            Tmax_flat = Tmax.view(B, 1).expand(B, K).reshape(B * K)
        else:
            Tmin_flat = None
            Tmax_flat = None

        if perturb_forward_chunk_size is None:
            perturb_forward_chunk_size = B * K
        preds_pieces = []
        for start in range(0, B * K, int(perturb_forward_chunk_size)):
            end = min(B * K, start + int(perturb_forward_chunk_size))
            preds_pieces.append(model(s_flat[start:end], c_flat[start:end], d_flat[start:end]))
        preds_pert_flat = torch.cat(preds_pieces, dim=0)

        cost_pert_flat, comp_pert = stage_cost_ramc(
            preds_pert_flat, c_flat,
            Tmin=Tmin_flat, Tmax=Tmax_flat,
            comfort_bounds=comfort_bounds,
            comfort_hinge=comfort_hinge,
            comfort_beta=comfort_beta,
            t_ret_index=t_ret_index,
            dt_minutes=dt_minutes,
            energy_cost_rate=energy_cost_rate,
            w_comfort=w_comfort, w_energy=w_energy,
            cost_scale=cost_scale,
            return_components=True,
        )
        cost_samples    = cost_pert_flat.reshape(B, K)
        comfort_samples = comp_pert["comfort"].reshape(B, K)
        energy_samples  = comp_pert["energy_cost"].reshape(B, K)

        risk_kwargs = dict(
            risk_operator=risk_op_clean,
            cvar_alpha=cvar_alpha,
            cvar_method=cvar_method,
            cvar_n_steps=cvar_n_steps,
            cvar_eta_lr=cvar_eta_lr,
            cvar_eta_init=cvar_eta_init,
        )
        risk_loss         = risk_from_cost_samples(cost_samples,    **risk_kwargs).mean()
        risk_comfort_loss = risk_from_cost_samples(comfort_samples, **risk_kwargs).mean()
        risk_energy_loss  = risk_from_cost_samples(energy_samples,  **risk_kwargs).mean()

    elif loss_mode_clean == "pert_only":
        # ─── A1 ablation: pre-generated RC-plant labels for perturbed inputs ───
        if any(t is None for t in (perturbed_states, perturbed_controls,
                                    perturbed_disturb, perturbed_targets)):
            raise ValueError(
                "loss_mode='pert_only' requires perturbed_states, perturbed_controls, "
                "perturbed_disturb, and perturbed_targets to all be provided "
                "(usually from a PerturbedLabelDataset)."
            )

        B = perturbed_states.size(0)
        K = perturbed_states.size(1)
        s_flat = perturbed_states.reshape(B * K, -1)
        c_flat = perturbed_controls.reshape(B * K, -1)
        d_flat = perturbed_disturb.reshape(B * K, -1)
        y_flat = perturbed_targets.reshape(B * K, -1)

        if perturb_forward_chunk_size is None:
            perturb_forward_chunk_size = B * K
        preds_pieces = []
        for start in range(0, B * K, int(perturb_forward_chunk_size)):
            end = min(B * K, start + int(perturb_forward_chunk_size))
            preds_pieces.append(model(s_flat[start:end], c_flat[start:end], d_flat[start:end]))
        preds_pert_flat = torch.cat(preds_pieces, dim=0)

        # Weighted MSE against the RC-plant ground-truth labels, using the
        # same normalization and per-dim weights as the un-perturbed fidelity term.
        if mse_normalize and getattr(model, "normalization_computed", False):
            out_std = getattr(model, "output_std", None)
            if out_std is not None:
                scale = out_std.view(1, -1).to(preds_pert_flat.device)
                diff = (preds_pert_flat - y_flat) / (scale + 1e-8)
            else:
                diff = preds_pert_flat - y_flat
        else:
            diff = preds_pert_flat - y_flat

        if fidelity_weights is not None:
            w = torch.as_tensor(fidelity_weights,
                                device=diff.device, dtype=diff.dtype).view(1, -1)
            pert_mse_per_sample = ((diff * diff) * w).sum(dim=-1) / w.sum()
        else:
            pert_mse_per_sample = (diff * diff).mean(dim=-1)

        risk_loss = pert_mse_per_sample.mean()
        risk_comfort_loss = torch.zeros((), device=risk_loss.device, dtype=risk_loss.dtype)
        risk_energy_loss  = torch.zeros((), device=risk_loss.device, dtype=risk_loss.dtype)
        clamp_stats = {}

    else:
        raise ValueError(
            f"Unknown loss_mode={loss_mode!r}. Use 'ramc' (default) or 'pert_only'."
        )

    if is_training:
        model.train()

    total_loss = fidelity_loss + float(lambda_risk) * risk_loss

    result = {
        # P1: Fidelity metrics
        "fidelity_loss": fidelity_loss,
        "per_dim_fid_mse": per_dim_fid_mse.detach(),
        "mse_raw": mse_raw,
        
        # P10: Decomposed expected cost
        "expected_cost": expected_cost,
        "expected_comfort": expected_comfort,
        "expected_energy_cost": expected_energy_cost,
        
        # P10: Decomposed risk
        "risk_loss": risk_loss,
        "risk_comfort_loss": risk_comfort_loss,
        "risk_energy_loss": risk_energy_loss,
        
        # Total
        "total_loss": total_loss,
        
        # Config info
        "lambda_risk": float(lambda_risk),
        "risk_operator": risk_op_clean,
        "loss_mode": loss_mode_clean,
        "cvar_alpha": float(cvar_alpha),
        "num_perturbations": int(num_perturbations),
        "mse_normalized": bool(mse_normalize),
        "use_antithetic": bool(use_antithetic),
        "risk_skipped": False,
    }
    
    # P2: Clamp stats
    if return_clamp_stats:
        result["clamp_stats"] = clamp_stats

    return result


# =============================================================================
# Evaluation helpers - Occupancy split uses full cost config
# =============================================================================

def evaluate_model_predictions(model, states, controls, disturbances, targets):
    """Standard predictive metrics."""
    model.eval()
    with torch.no_grad():
        predictions = model(states, controls, disturbances)

        mse = torch.mean((predictions - targets) ** 2)
        rmse = torch.sqrt(mse)
        mae = torch.mean(torch.abs(predictions - targets))

        state_rmse = torch.sqrt(torch.mean((predictions - targets) ** 2, dim=0))

        ss_res = torch.sum((targets - predictions) ** 2, dim=0)
        ss_tot = torch.sum((targets - torch.mean(targets, dim=0, keepdim=True)) ** 2, dim=0)
        r2 = 1 - ss_res / (ss_tot + 1e-8)

        return {
            "mse": mse.item(),
            "rmse": rmse.item(),
            "mae": mae.item(),
            "state_rmse": state_rmse.cpu().numpy(),
            "r2": r2.cpu().numpy(),
            "mean_r2": torch.mean(r2).item(),
        }


def evaluate_on_loader(
    model, 
    loader, 
    loss_config: dict, 
    device: str = "cpu",
    compute_occupancy_split: bool = False,  # P11
) -> Dict[str, Any]:
    """
    RAMC evaluation on a data loader.
    
    P1: Reports both mse_raw and fidelity_loss
    P4: Reports energy proxy metrics (Q_rmse, Q_mae, Q_bias)
    P10: Reports decomposed expected cost and risk (comfort vs energy)
    P11: Optionally reports occupancy-conditional metrics
    
    Occupancy split now uses full cost config for consistency
    """
    model.eval()

    # Keys that belong to the training/evaluation wrapper but are NOT accepted
    # by calculate_ramc_loss.  Stripping them here prevents unexpected-keyword-
    # argument errors when **loss_config_clean is forwarded.
    #
    #   ignore_dataset_bounds — trainer-level flag, handled below
    #   alpha_rollout         — rollout fidelity weight  (RAMCTrainer only)
    #   rollout_horizon       — rollout steps            (RAMCTrainer only)
    #   rollout_batch_size    — rollout batch size       (RAMCTrainer only)
    #   rollout_step_weights  — rollout weight mode      (RAMCTrainer only)
    _NON_RAMC_KEYS = {
        "ignore_dataset_bounds",
        "alpha_rollout",
        "rollout_horizon",
        "rollout_batch_size",
        "rollout_step_weights",
    }

    ignore_bounds = loss_config.get("ignore_dataset_bounds", False)
    loss_config_clean = {
        k: v for k, v in loss_config.items()
        if k not in _NON_RAMC_KEYS
    }

    total_samples = 0
    
    # Accumulators
    sum_expected_cost = 0.0
    sum_expected_comfort = 0.0
    sum_expected_energy = 0.0
    sum_risk = 0.0
    sum_risk_comfort = 0.0
    sum_risk_energy = 0.0
    sum_mse_raw = 0.0
    sum_fidelity = 0.0
    sum_t_air_sq = 0.0
    
    # P4: Energy proxy accumulators
    all_Q_diffs = []
    
    # P11: Occupancy-conditional accumulators
    if compute_occupancy_split:
        occ_sum_cost = 0.0
        occ_sum_comfort = 0.0
        occ_sum_energy = 0.0
        occ_count = 0
        unocc_sum_cost = 0.0
        unocc_sum_comfort = 0.0
        unocc_sum_energy = 0.0
        unocc_count = 0
    
    # P2: Clamp stats aggregation
    clamp_stats_accum = {}
    clamp_count = 0
    
    # Extract cost config parameters for occupancy split consistency
    cost_config = {
        "comfort_bounds": loss_config_clean.get("comfort_bounds", (20.0, 22.0)),
        "comfort_hinge": loss_config_clean.get("comfort_hinge", "softplus"),
        "comfort_beta": loss_config_clean.get("comfort_beta", 0.5),
        "t_ret_index": loss_config_clean.get("t_ret_index", 5),
        "dt_minutes": loss_config_clean.get("dt_minutes", 10.0),
        "energy_cost_rate": loss_config_clean.get("energy_cost_rate", 0.9),
        "w_comfort": loss_config_clean.get("w_comfort", 63.0),
        "w_energy": loss_config_clean.get("w_energy", 1.0),
        "cost_scale": loss_config_clean.get("cost_scale", 1.0),
    }

    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (list, tuple)) and len(batch) >= 6:
                states, controls, disturbances, targets, Tmin, Tmax = batch[:6]
                
                # Check for perturbed labels from A1 ablation data loader
                perturbed_states = batch[6] if len(batch) > 6 else None
                perturbed_controls = batch[7] if len(batch) > 7 else None
                perturbed_disturb = batch[8] if len(batch) > 8 else None
                perturbed_targets = batch[9] if len(batch) > 9 else None
            else:
                states, controls, disturbances, targets = batch[:4]
                Tmin = None
                Tmax = None
                perturbed_states = None
                perturbed_controls = None
                perturbed_disturb = None
                perturbed_targets = None

            states = states.to(device)
            controls = controls.to(device)
            disturbances = disturbances.to(device)
            targets = targets.to(device)
            if Tmin is not None:
                Tmin = Tmin.to(device)
            if Tmax is not None:
                Tmax = Tmax.to(device)
                
            if perturbed_states is not None:
                perturbed_states = perturbed_states.to(device)
            if perturbed_controls is not None:
                perturbed_controls = perturbed_controls.to(device)
            if perturbed_disturb is not None:
                perturbed_disturb = perturbed_disturb.to(device)
            if perturbed_targets is not None:
                perturbed_targets = perturbed_targets.to(device)

            # ── FIX 1b: Honour ignore_dataset_bounds ─────────────────────
            # When True, discard per-sample bounds so stage_cost_ramc falls
            # back to comfort_bounds=(20, 22) from loss_config_clean,
            # matching Phase 3 evaluation exactly.
            if ignore_bounds:
                Tmin = None
                Tmax = None

            bsz = states.size(0)
            total_samples += bsz

            # RAMC loss with all components
            out = calculate_ramc_loss(
                model, states, controls, disturbances, targets,
                Tmin=Tmin, Tmax=Tmax,
                return_clamp_stats=True,
                skip_risk_if_lambda_zero=False,  # Always compute risk for evaluation
                perturbed_states=perturbed_states,
                perturbed_controls=perturbed_controls,
                perturbed_disturb=perturbed_disturb,
                perturbed_targets=perturbed_targets,
                **loss_config_clean,              # <- was **loss_config; clean dict prevents
                                                  #   ignore_dataset_bounds leaking as unexpected kwarg
            )

            # P1: Both metrics
            sum_mse_raw += float(out["mse_raw"].item()) * bsz
            sum_fidelity += float(out["fidelity_loss"].item()) * bsz
            
            # P10: Decomposed metrics
            sum_expected_cost += float(out["expected_cost"].item()) * bsz
            sum_expected_comfort += float(out["expected_comfort"].item()) * bsz
            sum_expected_energy += float(out["expected_energy_cost"].item()) * bsz
            sum_risk += float(out["risk_loss"].item()) * bsz
            sum_risk_comfort += float(out["risk_comfort_loss"].item()) * bsz
            sum_risk_energy += float(out["risk_energy_loss"].item()) * bsz

            # T_air RMSE
            preds = model(states, controls, disturbances)
            t_air_mse = torch.mean((preds[:, 0] - targets[:, 0]).pow(2))
            sum_t_air_sq += float(t_air_mse.item()) * bsz
            
            # P4: Energy proxy metrics
            t_ret_idx = cost_config["t_ret_index"]
            T_ret_pred = preds[:, t_ret_idx]
            T_ret_true = targets[:, t_ret_idx]
            Q_pred = compute_energy_Q_W(T_ret_pred, controls)
            Q_true = compute_energy_Q_W(T_ret_true, controls)
            all_Q_diffs.append((Q_pred - Q_true).cpu())
            
            # P2: Aggregate clamp stats
            if "clamp_stats" in out:
                for k, v in out["clamp_stats"].items():
                    clamp_stats_accum[k] = clamp_stats_accum.get(k, 0.0) + v
                clamp_count += 1
            
            # P11: Occupancy-conditional metrics - Use full cost config
            if compute_occupancy_split and disturbances.size(-1) > 2:
                q_int = disturbances[:, 2]
                q_nominal = loss_config.get("q_internal_nominal", 1000.0)
                occ_mask = q_int > 0.5 * q_nominal
                
                # Compute per-sample costs with full cost config
                nominal_costs, nominal_components = stage_cost_ramc(
                    preds, controls, 
                    Tmin=Tmin, 
                    Tmax=Tmax,
                    comfort_bounds=cost_config["comfort_bounds"],
                    comfort_hinge=cost_config["comfort_hinge"],
                    comfort_beta=cost_config["comfort_beta"],
                    t_ret_index=cost_config["t_ret_index"],
                    dt_minutes=cost_config["dt_minutes"],
                    energy_cost_rate=cost_config["energy_cost_rate"],
                    w_comfort=cost_config["w_comfort"],
                    w_energy=cost_config["w_energy"],
                    cost_scale=cost_config["cost_scale"],
                    return_components=True,
                )
                
                if occ_mask.any():
                    occ_sum_cost += float(nominal_costs[occ_mask].sum().item())
                    occ_sum_comfort += float(nominal_components["comfort"][occ_mask].sum().item())
                    occ_sum_energy += float(nominal_components["energy_cost"][occ_mask].sum().item())
                    occ_count += int(occ_mask.sum().item())
                if (~occ_mask).any():
                    unocc_sum_cost += float(nominal_costs[~occ_mask].sum().item())
                    unocc_sum_comfort += float(nominal_components["comfort"][~occ_mask].sum().item())
                    unocc_sum_energy += float(nominal_components["energy_cost"][~occ_mask].sum().item())
                    unocc_count += int((~occ_mask).sum().item())

    denom = max(1, total_samples)
    
    # P4: Compute Q metrics
    all_Q_diffs = torch.cat(all_Q_diffs, dim=0)
    Q_rmse = float(torch.sqrt(torch.mean(all_Q_diffs ** 2)).item())
    Q_mae = float(torch.mean(all_Q_diffs.abs()).item())
    Q_bias = float(torch.mean(all_Q_diffs).item())
    Q_abs_p95 = float(torch.quantile(all_Q_diffs.abs(), 0.95).item()) if len(all_Q_diffs) > 20 else float('nan')
    Q_abs_p99 = float(torch.quantile(all_Q_diffs.abs(), 0.99).item()) if len(all_Q_diffs) > 100 else float('nan')
    
    result = {
        # P1: Both MSE metrics
        "mse_raw": sum_mse_raw / denom,
        "fidelity_loss": sum_fidelity / denom,
        
        # P10: Decomposed expected
        "expected_cost": sum_expected_cost / denom,
        "expected_comfort": sum_expected_comfort / denom,
        "expected_energy_cost": sum_expected_energy / denom,
        
        # P10: Decomposed risk
        "risk_loss": sum_risk / denom,
        "risk_comfort_loss": sum_risk_comfort / denom,
        "risk_energy_loss": sum_risk_energy / denom,
        
        # Legacy compatibility
        "mse_loss": sum_mse_raw / denom,
        "t_air_rmse": float(np.sqrt(sum_t_air_sq / denom)),
        
        # P4: Energy proxy metrics
        "Q_rmse": Q_rmse,
        "Q_mae": Q_mae,
        "Q_bias": Q_bias,
        "Q_abs_p95": Q_abs_p95,
        "Q_abs_p99": Q_abs_p99,
    }
    
    # P2: Average clamp stats
    if clamp_count > 0:
        result["clamp_stats"] = {k: v / clamp_count for k, v in clamp_stats_accum.items()}
    
    # P11: Occupancy-conditional - Now includes comfort and energy breakdown
    if compute_occupancy_split:
        result["occ_expected_cost"] = occ_sum_cost / max(1, occ_count) if occ_count > 0 else float('nan')
        result["occ_expected_comfort"] = occ_sum_comfort / max(1, occ_count) if occ_count > 0 else float('nan')
        result["occ_expected_energy"] = occ_sum_energy / max(1, occ_count) if occ_count > 0 else float('nan')
        result["unocc_expected_cost"] = unocc_sum_cost / max(1, unocc_count) if unocc_count > 0 else float('nan')
        result["unocc_expected_comfort"] = unocc_sum_comfort / max(1, unocc_count) if unocc_count > 0 else float('nan')
        result["unocc_expected_energy"] = unocc_sum_energy / max(1, unocc_count) if unocc_count > 0 else float('nan')
        result["occ_sample_count"] = occ_count
        result["unocc_sample_count"] = unocc_count

    return result
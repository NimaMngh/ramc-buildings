"""
Gradient Alignment Analysis for Assignment 5.
===============================================

Computes the cosine similarity between NMPC cost gradients computed
using an NN planning model vs. the RC ground-truth plant.

The key question: does the NN give the optimiser gradient directions
that actually improve the TRUE plant cost?

Methodology:
  1. At a sampled operating point (x_k, u_k, d_{k:k+H}):
  2. Compute ∇_U J_NN(U)  via torch autograd through the NN rollout
  3. Compute ∇_U J_RC(U)  via finite differences through the RC rollout
  4. cos(θ) = (∇J_NN · ∇J_RC) / (‖∇J_NN‖ ‖∇J_RC‖)

  cos(θ) ≈ 1.0:  NN guides optimiser in the correct direction
  cos(θ) ≈ 0.0:  NN gradients are uninformative
  cos(θ) < 0.0:  NN actively misleads the optimiser

Cost function matches nmpc_direct_shooting.py exactly:
  J = Σ_k [ w_cold * smooth_pos(Tmin - T_air)²
          + w_energy * mdot * Cp * max(T_supply - T_ret, 0) * dt/3600/1000 ]
    + w_terminal * max(Tmin[-1] - T_air_final, 0)

  where smooth_pos(z) = softplus(beta * z) / beta, beta=20

Used by: A5 (primary)
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

# =============================================================================
# Constants — matched to nmpc_direct_shooting.py
# =============================================================================

CP_WATER = 4186.0          # J/(kg·K)
DT_SECONDS = 600           # 10 minutes
DT_HOURS = DT_SECONDS / 3600.0
SMOOTH_POS_BETA = 20.0     # softplus sharpness, matches nmpc_direct_shooting.py

IDX_T_AIR = 0
IDX_T_RET = 5


# =============================================================================
# Smooth penalty — matches nmpc_direct_shooting.py exactly
# =============================================================================

def smooth_pos_np(z: np.ndarray, beta: float = SMOOTH_POS_BETA) -> np.ndarray:
    """Numpy version of smooth_pos from nmpc_direct_shooting.py."""
    # softplus(beta * z) / beta = log(1 + exp(beta * z)) / beta
    # Numerically stable version:
    bz = beta * z
    return np.where(bz > 20, z, np.log1p(np.exp(np.clip(bz, -50, 50))) / beta)


def smooth_pos_torch(z: torch.Tensor, beta: float = SMOOTH_POS_BETA) -> torch.Tensor:
    """Torch version matching nmpc_direct_shooting.py smooth_pos."""
    return F.softplus(beta * z) / beta


# =============================================================================
# RC plant cost (numpy, for finite differences)
# =============================================================================

def nmpc_stage_cost_np(T_air, T_ret, T_supply, mdot, Tmin, Tmax,
                       w_cold=63.0, w_energy=0.9,
                       dt_hours=DT_HOURS) -> float:
    """
    Single-step NMPC stage cost matching nmpc_direct_shooting.py objective.

    The NMPC uses: w_cold * smooth_pos(Tmin - T_air)²
    NOT the un-squared softplus hinge from ramc_losses.py.
    """
    # Comfort: squared softplus penalty (matches NMPC objective)
    cold = smooth_pos_np(np.array(Tmin - T_air)) ** 2
    comfort = w_cold * float(cold)

    # Energy: mdot * Cp * max(T_supply - T_ret, 0) * dt_hours / 1000
    mdot_eff = max(float(mdot), 0.0)
    Q_heat_W = mdot_eff * CP_WATER * max(float(T_supply) - float(T_ret), 0.0)
    energy_kWh = Q_heat_W * dt_hours / 1000.0
    energy = w_energy * energy_kWh

    return float(comfort + energy)


def rollout_rc_cost(rc_plant, x0, U_blocked, D_horizon, Tmin_seq, Tmax_seq,
                    block_size=4, w_cold=63.0, w_energy=0.9,
                    w_terminal=20.0) -> float:
    """
    Roll out the RC plant for H steps with blocked controls and compute
    the NMPC objective (matching nmpc_direct_shooting.py cost).
    """
    n_blocks = len(U_blocked)
    H = n_blocks * block_size
    H = min(H, len(D_horizon), len(Tmin_seq))

    x = np.array(x0, dtype=float)
    total_cost = 0.0

    for h in range(H):
        block_idx = h // block_size
        block_idx = min(block_idx, n_blocks - 1)
        u = U_blocked[block_idx]  # [T_supply, mdot]
        d = D_horizon[h]

        x = rc_plant.step(x, u, d)

        cost_h = nmpc_stage_cost_np(
            x[IDX_T_AIR], x[IDX_T_RET], u[0], u[1],
            Tmin_seq[h], Tmax_seq[h],
            w_cold=w_cold, w_energy=w_energy,
        )
        total_cost += cost_h

    # Terminal cold penalty: w_terminal * smooth_pos(Tmin - T_air)²
    if H > 0:
        terminal_cold = smooth_pos_np(np.array(Tmin_seq[H - 1] - x[IDX_T_AIR]))
        total_cost += w_terminal * float(terminal_cold ** 2)

    return total_cost


def gradient_rc_finite_diff(rc_plant, x0, U_blocked, D_horizon,
                            Tmin_seq, Tmax_seq,
                            block_size=4, eps=0.1,
                            **cost_kwargs) -> np.ndarray:
    """
    Compute ∇_U J_RC via central finite differences.

    U_blocked: (n_blocks, 2) array of [T_supply, mdot] per block.
    Returns: gradient with same shape as U_blocked, flattened.
    """
    U_flat = U_blocked.flatten().astype(float)
    grad = np.zeros_like(U_flat)

    # Scale epsilon per control dimension
    # T_supply range ~28°C, mdot range ~4 kg/s
    eps_scale = np.tile([eps, eps * 0.05], len(U_blocked))

    for i in range(len(U_flat)):
        U_plus = U_flat.copy()
        U_minus = U_flat.copy()
        U_plus[i] += eps_scale[i]
        U_minus[i] -= eps_scale[i]

        cost_plus = rollout_rc_cost(
            rc_plant, x0,
            U_plus.reshape(-1, 2), D_horizon, Tmin_seq, Tmax_seq,
            block_size=block_size, **cost_kwargs,
        )
        cost_minus = rollout_rc_cost(
            rc_plant, x0,
            U_minus.reshape(-1, 2), D_horizon, Tmin_seq, Tmax_seq,
            block_size=block_size, **cost_kwargs,
        )
        grad[i] = (cost_plus - cost_minus) / (2 * eps_scale[i])

    return grad


# =============================================================================
# NN cost (torch autograd)
# =============================================================================

def gradient_nn_autograd(nn_model, x0_np, U_blocked_np, D_horizon_np,
                         Tmin_seq_np, Tmax_seq_np,
                         block_size=4, w_cold=63.0, w_energy=0.9,
                         w_terminal=20.0,
                         dtype=torch.float64) -> np.ndarray:
    """
    Compute ∇_U J_NN via torch autograd through the NN rollout.

    Calls nn_model(states, controls, disturbances) matching the
    ThermalDynamicsNet.forward() signature. The model handles residual
    addition internally (use_residual=True adds states + network output),
    so we do NOT add residuals here.

    Cost function matches nmpc_direct_shooting.py objective exactly:
      - Comfort: w_cold * smooth_pos(Tmin - T_air, beta=20)²
      - Energy:  w_energy * mdot * Cp * max(T_sup - T_ret, 0) * dt_h / 1000
      - Terminal: w_terminal * smooth_pos(Tmin[-1] - T_air_final, beta=20)²

    Returns: gradient with same shape as U_blocked, flattened.
    """
    device = next(nn_model.parameters()).device

    x = torch.tensor(x0_np, dtype=dtype, device=device)
    U = torch.tensor(U_blocked_np, dtype=dtype, device=device, requires_grad=True)
    D = torch.tensor(D_horizon_np, dtype=dtype, device=device)
    Tmin = torch.tensor(Tmin_seq_np, dtype=dtype, device=device)
    Tmax = torch.tensor(Tmax_seq_np, dtype=dtype, device=device)

    n_blocks = U.shape[0]
    H = min(n_blocks * block_size, len(D), len(Tmin))

    total_cost = torch.tensor(0.0, dtype=dtype, device=device)
    x_current = x.clone()

    for h in range(H):
        block_idx = min(h // block_size, n_blocks - 1)
        u_h = U[block_idx]

        # ── NN forward pass with separate (states, controls, disturbances) ──
        # ThermalDynamicsNet.forward() expects (batch, dim) tensors.
        # Residual addition is handled INSIDE the model (use_residual=True).
        states_in = x_current.unsqueeze(0)    # (1, 6)
        controls_in = u_h.unsqueeze(0)        # (1, 2)
        disturb_in = D[h].unsqueeze(0)        # (1, 3)

        x_next = nn_model(states_in, controls_in, disturb_in).squeeze(0)

        T_air = x_next[IDX_T_AIR]
        T_ret = x_next[IDX_T_RET]
        T_sup = u_h[0]
        mdot = torch.clamp(u_h[1], min=0.0)

        # Comfort cost: squared softplus (matches NMPC objective)
        cold_pen = smooth_pos_torch(Tmin[h] - T_air)
        comfort = w_cold * cold_pen ** 2

        # Energy cost (matches NMPC objective)
        Q_heat = mdot * CP_WATER * torch.clamp(T_sup - T_ret, min=0.0)
        energy_kWh = Q_heat * DT_HOURS / 1000.0
        energy = w_energy * energy_kWh

        total_cost = total_cost + comfort + energy
        x_current = x_next

    # Terminal: squared softplus (matches NMPC objective)
    terminal_cold = smooth_pos_torch(Tmin[H - 1] - x_current[IDX_T_AIR])
    total_cost = total_cost + w_terminal * terminal_cold ** 2

    total_cost.backward()
    grad = U.grad.detach().cpu().numpy().flatten()

    return grad


# =============================================================================
# Cosine similarity
# =============================================================================

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


# =============================================================================
# Single-point analysis
# =============================================================================

def analyse_single_point(
    nn_model,
    rc_plant,
    x0: np.ndarray,
    U_blocked: np.ndarray,
    D_horizon: np.ndarray,
    Tmin_seq: np.ndarray,
    Tmax_seq: np.ndarray,
    block_size: int = 4,
    **kwargs,
) -> Dict:
    """
    Full gradient alignment analysis at a single operating point.

    Returns dict with cos_similarity, grad norms, comfort_critical flag, etc.
    """
    T_air = x0[IDX_T_AIR]
    Tmin_current = Tmin_seq[0] if len(Tmin_seq) > 0 else 20.0

    # Is this point comfort-critical? (within 1°C of lower bound)
    comfort_critical = (T_air - Tmin_current) < 1.0

    # NN gradient (autograd)
    grad_nn = gradient_nn_autograd(
        nn_model, x0, U_blocked, D_horizon, Tmin_seq, Tmax_seq,
        block_size=block_size, **kwargs,
    )

    # RC gradient (finite differences)
    grad_rc = gradient_rc_finite_diff(
        rc_plant, x0, U_blocked, D_horizon, Tmin_seq, Tmax_seq,
        block_size=block_size,
        w_cold=kwargs.get("w_cold", 63.0),
        w_energy=kwargs.get("w_energy", 0.9),
        w_terminal=kwargs.get("w_terminal", 20.0),
    )

    cos_sim = cosine_similarity(grad_nn, grad_rc)

    return {
        "T_air": float(T_air),
        "Tmin": float(Tmin_current),
        "comfort_margin": float(T_air - Tmin_current),
        "comfort_critical": comfort_critical,
        "cos_similarity": cos_sim,
        "grad_nn_norm": float(np.linalg.norm(grad_nn)),
        "grad_rc_norm": float(np.linalg.norm(grad_rc)),
        "grad_nn": grad_nn.tolist(),
        "grad_rc": grad_rc.tolist(),
    }

#!/usr/bin/env python3
"""
Direct-Shooting Nonlinear MPC with Projected Adam
=============================================================

Replaces the linearize -> QP pipeline with:
    NN model -> exact rollout over horizon -> optimize control sequence directly

There is:
  - NO online Jacobian-based dynamics matrix
  - NO spectral stabilization
  - NO linear MPC QP
  - NO argument that RAMC got washed out by linearization

The ONLY thing that changes between controllers is the planning model f_θ.

Architecture:
  - Direct shooting: decision variables are blocked control sequences
  - Projected Adam optimizer with gradient clipping
  - Move blocking to reduce decision dimension
  - Soft comfort penalties (smoothed via softplus)
  - Receding-horizon warm start

Author: Implementation of expert procedure for RAMC Phase 3
Date: 2026-03-06
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, Tuple
import time


# =============================================================================
# Constants — MUST match closed_loop_simulator.py exactly
# =============================================================================

DT_SECONDS = 600
DT_HOURS = DT_SECONDS / 3600.0       # 1/6 hour
CP_WATER = 4186.0                      # J/(kg·K)

# Actuator bounds
T_SUPPLY_MIN = 32.0
T_SUPPLY_MAX = 60.0
MDOT_MIN = 0.0
MDOT_MAX = 4.05

# State indices (6-state RC model)
IDX_T_AIR = 0
IDX_T_ENV = 1
IDX_T_INT = 2
IDX_T_RAD1 = 3
IDX_T_RAD2 = 4
IDX_T_RET = 5


# =============================================================================
# Smooth penalty function
# =============================================================================

def smooth_pos(z: torch.Tensor, beta: float = 20.0) -> torch.Tensor:
    """
    Smooth approximation of max(z, 0) using softplus.

    For smooth optimization, this avoids zero derivatives at the boundary
    that hard relu would produce. At beta=20, this is virtually identical
    to relu for |z| > 0.2 but provides gradients everywhere.

    Args:
        z: Input tensor
        beta: Smoothing parameter (higher = sharper)

    Returns:
        Smooth positive part of z
    """
    return F.softplus(beta * z) / beta


# =============================================================================
# Move blocking utilities
# =============================================================================

def expand_blocks(U_blocks: torch.Tensor, H: int, block_size: int) -> torch.Tensor:
    """
    Expand blocked controls to step-by-step sequence.

    Args:
        U_blocks: (B, nu) blocked control values
        H: Total horizon length
        block_size: Steps per block

    Returns:
        U_seq: (H, nu) step-level controls
    """
    U_seq = U_blocks.repeat_interleave(block_size, dim=0)
    return U_seq[:H]


def shift_and_reblock(U_seq: torch.Tensor, block_size: int, H: int) -> torch.Tensor:
    """
    Warm-start helper: shift sequence left by 1, then re-block.

    After applying u_0, shift the rest and repeat the last value:
        [u_1, u_2, ..., u_{H-1}, u_{H-1}]
    Then re-block by averaging within each block.

    Args:
        U_seq: (H, nu) step-level control sequence
        block_size: Steps per block
        H: Horizon length

    Returns:
        U_blocks_next: (B, nu) re-blocked warm start for next solve
    """
    U_shift = torch.cat([U_seq[1:], U_seq[-1:]], dim=0)
    blocks = []
    for i in range(0, H, block_size):
        seg = U_shift[i:i + block_size]
        blocks.append(seg.mean(dim=0))
    return torch.stack(blocks, dim=0)


# =============================================================================
# Control projection
# =============================================================================

def project_blocks(
    U_blocks: torch.Tensor,
    u_prev: torch.Tensor,
    u_min: torch.Tensor,
    u_max: torch.Tensor,
    du_max: torch.Tensor,
) -> torch.Tensor:
    """
    Project blocked controls to satisfy bounds and rate limits.

    Sequentially clamps each block to:
      1. Absolute bounds [u_min, u_max]
      2. Rate limits relative to previous block: |u_b - u_{b-1}| <= du_max

    Uses list+stack instead of in-place indexing to preserve autograd graph.

    Args:
        U_blocks: (B, nu) candidate control blocks
        u_prev: (nu,) last applied control
        u_min: (nu,) lower bounds
        u_max: (nu,) upper bounds
        du_max: (nu,) max change per block

    Returns:
        (B, nu) projected controls
    """
    projected = []
    prev = u_prev

    for b in range(U_blocks.shape[0]):
        low = torch.maximum(u_min, prev - du_max)
        high = torch.minimum(u_max, prev + du_max)
        ub = torch.maximum(torch.minimum(U_blocks[b], high), low)
        projected.append(ub)
        prev = ub

    return torch.stack(projected, dim=0)


# =============================================================================
# Exact rollout
# =============================================================================

def rollout_exact(
    model: nn.Module,
    x0: torch.Tensor,
    U_seq: torch.Tensor,
    D_seq: torch.Tensor,
) -> torch.Tensor:
    """
    Roll out the exact NN dynamics over the horizon.

    This is the CORE difference from linearized MPC. The NN is evaluated
    exactly at every step — no Jacobians, no linearization, no approximation.

    Args:
        model: ThermalDynamicsNet in eval mode (normalization handled internally)
        x0: (nx,) initial state in physical units
        U_seq: (H, nu) control sequence in physical units
        D_seq: (H, nd) disturbance forecast in physical units

    Returns:
        X: (H+1, nx) state trajectory including x0
    """
    X = [x0]
    x = x0
    H = U_seq.shape[0]

    for k in range(H):
        # model.forward expects (batch, dim) -> add and remove batch dim
        x_next = model(
            x.unsqueeze(0),
            U_seq[k].unsqueeze(0),
            D_seq[k].unsqueeze(0),
        ).squeeze(0)
        X.append(x_next)
        x = x_next

    return torch.stack(X, dim=0)


# =============================================================================
# HVAC power model
# =============================================================================

def hvac_power(
    U_seq: torch.Tensor,
    X_seq: torch.Tensor,
) -> torch.Tensor:
    """
    Compute HVAC heat delivery power at each step.

    Q = mdot * Cp * max(T_supply - T_return, 0)

    Uses X_seq[:-1] (state at start of each interval) for T_return,
    matching the closed_loop_simulator energy calculation.

    Args:
        U_seq: (H, 2) controls [T_supply, mdot]
        X_seq: (H+1, 6) state trajectory

    Returns:
        power_W: (H,) heat power in Watts
    """
    T_supply = U_seq[:, 0]
    m_dot = U_seq[:, 1]
    T_ret = X_seq[:-1, IDX_T_RET]

    power = m_dot * CP_WATER * torch.clamp(T_supply - T_ret, min=0.0)
    return power


# =============================================================================
# Objective function
# =============================================================================

def objective(
    X_seq: torch.Tensor,
    U_seq: torch.Tensor,
    D_seq: torch.Tensor,
    occ_seq: torch.Tensor,
    T_low_seq: torch.Tensor,
    T_high_seq: torch.Tensor,
    u_prev: torch.Tensor,
    weights: Dict[str, float],
    u_scale: torch.Tensor,
    U_warm: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute the NMPC objective. IDENTICAL for both Fidelity and RAMC models.

    J = Σ_k [ w_E * E_k
            + w_cold * occ_k * φ(T_low_k - T_air_k)²
            + w_hot  * occ_k * φ(T_air_k - T_high_k)²
            + w_du   * ||S(u_k - u_{k-1})||²
            ]
        + w_terminal * occ_{H-1} * φ(T_low_{H-1} - T_air_H)²
        + w_trust * Σ_k ||u_k - u_k^warm||²    (optional)

    where φ(z) = smooth_pos(z) ≈ max(z, 0).

    Args:
        X_seq: (H+1, nx) state trajectory
        U_seq: (H, nu) control sequence
        D_seq: (H, nd) disturbance forecast
        occ_seq: (H,) occupancy weight (1.0 occupied, small value unoccupied)
        T_low_seq: (H,) lower comfort bound per step
        T_high_seq: (H,) upper comfort bound per step
        u_prev: (nu,) previous applied control
        weights: Dict with keys: energy, cold, hot, du, terminal, trust
        u_scale: (nu,) scaling for slew penalty normalization
        U_warm: (H, nu) optional warm-start trajectory for trust penalty

    Returns:
        J: scalar total cost
    """
    H = U_seq.shape[0]

    # ── Comfort penalty (evaluated at next-state, matching RAMC training) ──
    T_air = X_seq[1:, IDX_T_AIR]   # (H,) — state AFTER applying control

    cold = smooth_pos(T_low_seq - T_air)
    hot = smooth_pos(T_air - T_high_seq)

    comfort_cost = (
        weights["cold"] * occ_seq * cold ** 2
        + weights["hot"] * occ_seq * hot ** 2
    )

    # ── Energy cost ──
    power_W = hvac_power(U_seq, X_seq)
    energy_kWh = power_W * DT_HOURS / 1000.0
    energy_cost = weights["energy"] * energy_kWh

    # ── Slew rate penalty ──
    U_prev_full = torch.cat([u_prev.unsqueeze(0), U_seq[:-1]], dim=0)
    du = (U_seq - U_prev_full) / u_scale
    slew_cost = weights["du"] * (du ** 2).sum(dim=-1)

    # ── Terminal cold penalty ──
    terminal_T_air = X_seq[-1, IDX_T_AIR]
    terminal_cold = smooth_pos(T_low_seq[-1] - terminal_T_air)
    terminal_cost = weights["terminal"] * occ_seq[-1] * terminal_cold ** 2

    # ── Trust region penalty (optional) ──
    trust_cost = torch.tensor(0.0, dtype=X_seq.dtype)
    if U_warm is not None and weights.get("trust", 0.0) > 0:
        trust_diff = (U_seq - U_warm) / u_scale
        trust_cost = weights["trust"] * (trust_diff ** 2).sum(dim=-1).sum()

    total = comfort_cost.sum() + energy_cost.sum() + slew_cost.sum() + terminal_cost + trust_cost
    return total


# =============================================================================
# One-step NMPC solver
# =============================================================================

def solve_direct_shooting(
    model: nn.Module,
    x0: torch.Tensor,
    D_hat: torch.Tensor,
    occ_seq: torch.Tensor,
    T_low_seq: torch.Tensor,
    T_high_seq: torch.Tensor,
    u_prev: torch.Tensor,
    warm_blocks: torch.Tensor,
    u_min: torch.Tensor,
    u_max: torch.Tensor,
    du_max: torch.Tensor,
    weights: Dict[str, float],
    u_scale: torch.Tensor,
    H: int = 24,
    block_size: int = 4,
    n_iter: int = 25,
    lr: float = 0.05,
    grad_clip: float = 5.0,
) -> Tuple[torch.Tensor, torch.Tensor, float, Dict]:
    """
    Solve one NMPC step using projected Adam on blocked controls.

    Procedure:
      1. Initialize blocked control sequence (from warm start)
      2. For each iteration:
         a. Project to feasibility
         b. Expand blocks to step-level
         c. Rollout NN exactly over horizon
         d. Evaluate objective
         e. Backpropagate and update with Adam
         f. Project again after update
      3. Return best feasible candidate

    Args:
        model: NN dynamics model (frozen weights, eval mode)
        x0: (nx,) current plant state
        D_hat: (H, nd) disturbance forecast
        occ_seq: (H,) occupancy weights
        T_low_seq: (H,) lower comfort bounds
        T_high_seq: (H,) upper comfort bounds
        u_prev: (nu,) last applied control
        warm_blocks: (B, nu) warm-start blocked controls
        u_min: (nu,) lower control bounds
        u_max: (nu,) upper control bounds
        du_max: (nu,) max control change per block
        weights: Objective weight dict
        u_scale: (nu,) control scaling for penalties
        H: Horizon length
        block_size: Steps per control block
        n_iter: Adam iterations
        lr: Learning rate
        grad_clip: Max gradient norm

    Returns:
        U_best_seq: (H, nu) best step-level control sequence
        warm_next: (B, nu) warm start for next solve
        best_loss: float, best objective value
        info: dict with solver diagnostics
    """
    dtype = x0.dtype

    # Initialize as optimizable parameter
    U_blocks = warm_blocks.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([U_blocks], lr=lr)

    best_loss = float("inf")
    best_blocks = project_blocks(
        warm_blocks.clone().detach(), u_prev, u_min, u_max, du_max
    )

    loss_history = []
    n_valid = 0

    for it in range(n_iter):
        opt.zero_grad(set_to_none=True)

        # Project (differentiable-friendly: clone inside)
        U_proj = project_blocks(U_blocks, u_prev, u_min, u_max, du_max)
        U_seq = expand_blocks(U_proj, H, block_size)

        # Expand warm start for optional trust penalty
        U_warm_seq = expand_blocks(
            project_blocks(warm_blocks.clone().detach(), u_prev, u_min, u_max, du_max),
            H, block_size
        ).detach()

        # Exact rollout
        X_seq = rollout_exact(model, x0, U_seq, D_hat)

        # Objective
        loss = objective(
            X_seq, U_seq, D_hat,
            occ_seq, T_low_seq, T_high_seq,
            u_prev, weights, u_scale,
            U_warm=U_warm_seq,
        )

        if not torch.isfinite(loss):
            loss_history.append(float("nan"))
            break

        loss_val = loss.item()
        loss_history.append(loss_val)
        n_valid += 1

        # Backward
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_([U_blocks], grad_clip)

        # Optimizer step
        opt.step()

        # Re-project after optimizer step (with no_grad to avoid graph issues)
        # Use .data.copy_(.data) to avoid incrementing the version counter,
        # which would cause "in-place operation" errors on the next backward pass.
        with torch.no_grad():
            projected = project_blocks(U_blocks, u_prev, u_min, u_max, du_max)
            U_blocks.data.copy_(projected.data)

        # Track best
        if loss_val < best_loss:
            best_loss = loss_val
            best_blocks = U_blocks.clone().detach()

    # Final projection of best
    U_best_blocks = project_blocks(
        best_blocks, u_prev, u_min, u_max, du_max
    )
    U_best_seq = expand_blocks(U_best_blocks, H, block_size)

    # Warm start for next MPC step
    warm_next = shift_and_reblock(U_best_seq, block_size, H)

    info = {
        "n_iter": n_valid,
        "loss_history": loss_history,
        "loss_initial": loss_history[0] if loss_history else float("nan"),
        "loss_final": loss_history[-1] if loss_history else float("nan"),
        "loss_reduction": (
            (loss_history[0] - best_loss) / (abs(loss_history[0]) + 1e-10)
            if loss_history else 0.0
        ),
    }

    return U_best_seq.detach(), warm_next.detach(), best_loss, info


# =============================================================================
# NMPC Controller class — drop-in replacement for MPCControllerHydronic
# =============================================================================

class NMPCDirectShooting:
    """
    Nonlinear MPC controller using direct shooting with projected Adam.

    Drop-in replacement for the linearized QP controller. Uses the EXACT
    neural network rollout — no linearization, no approximation.

    The planning model and the evaluation plant are SEPARATE.
    This controller only does planning; the closed-loop simulator
    handles plant stepping.
    """

    def __init__(
        self,
        model: nn.Module,
        horizon: int = 24,
        block_size: int = 4,
        n_iter: int = 25,
        lr: float = 0.05,
        grad_clip: float = 5.0,
        # Control bounds
        u_min: Optional[np.ndarray] = None,
        u_max: Optional[np.ndarray] = None,
        du_max: Optional[np.ndarray] = None,
        # Objective weights
        w_energy: float = 0.9,
        w_cold: float = 63.0,
        w_hot: float = 30.0,
        w_du: float = 1e-3,
        w_terminal: float = 20.0,
        w_trust: float = 0.0,
        # Energy
        energy_cost_rate: float = 0.9,
        # Config
        dtype: torch.dtype = torch.float64,
        verbose: bool = True,
    ):
        """
        Initialize the direct-shooting NMPC controller.

        Args:
            model: ThermalDynamicsNet (frozen, eval mode)
            horizon: Planning horizon in steps (default 24 = 4 hours)
            block_size: Move blocking size (default 4 = 40 min blocks)
            n_iter: Adam iterations per MPC solve
            lr: Adam learning rate
            grad_clip: Gradient norm clipping
            u_min: Control lower bounds [T_supply_min, mdot_min]
            u_max: Control upper bounds [T_supply_max, mdot_max]
            du_max: Rate limits per block [dT_supply, dmdot]
            w_energy: Energy cost weight (= energy_cost_rate for consistency)
            w_cold: Cold comfort penalty weight (matched to Phase 2 training)
            w_hot: Hot comfort penalty weight
            w_du: Slew rate penalty weight
            w_terminal: Terminal cold penalty weight
            w_trust: Trust region penalty weight (0 = disabled)
            energy_cost_rate: SEK/kWh (matched to Phase 2)
            dtype: Tensor dtype (float64 recommended for numerical stability)
            verbose: Print initialization info
        """
        self.dtype = dtype
        self.H = horizon
        self.block_size = block_size
        self.n_blocks = (horizon + block_size - 1) // block_size
        self.n_iter = n_iter
        self.lr = lr
        self.grad_clip = grad_clip

        # Prepare model
        self.model = model.to(dtype=dtype)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        # Enable normalization if loaded from checkpoint
        if hasattr(self.model, 'enable_normalization_if_stats_present'):
            self.model.enable_normalization_if_stats_present()

        # Control bounds
        self.u_min = torch.tensor(
            u_min if u_min is not None else [T_SUPPLY_MIN, MDOT_MIN],
            dtype=dtype,
        )
        self.u_max = torch.tensor(
            u_max if u_max is not None else [T_SUPPLY_MAX, MDOT_MAX],
            dtype=dtype,
        )
        self.du_max = torch.tensor(
            du_max if du_max is not None else [2.0, 0.3],
            dtype=dtype,
        )

        # Objective weights — SAME for all models (non-negotiable for fair comparison)
        self.weights = {
            "energy": float(w_energy),
            "cold": float(w_cold),
            "hot": float(w_hot),
            "du": float(w_du),
            "terminal": float(w_terminal),
            "trust": float(w_trust),
        }

        # Control scaling for slew penalty normalization
        self.u_scale = torch.tensor(
            [T_SUPPLY_MAX - T_SUPPLY_MIN, MDOT_MAX - MDOT_MIN],
            dtype=dtype,
        )

        # State dimensions
        self.nx = 6
        self.nu = 2
        self.nd = 3

        # Warm start storage
        self._warm_blocks = None

        if verbose:
            print(f"\n{'='*70}")
            print("NMPC DIRECT SHOOTING CONTROLLER")
            print(f"{'='*70}")
            print(f"  Horizon: {self.H} steps ({self.H * DT_SECONDS / 3600:.1f}h)")
            print(f"  Block size: {self.block_size} ({self.block_size * DT_SECONDS / 60:.0f} min)")
            print(f"  Decision vars: {self.n_blocks} blocks × {self.nu} inputs = {self.n_blocks * self.nu}")
            print(f"  Optimizer: Adam (lr={self.lr}, grad_clip={self.grad_clip}, iters={self.n_iter})")
            print(f"  dtype: {self.dtype}")
            print(f"  Control bounds:")
            print(f"    T_supply: [{self.u_min[0]:.1f}, {self.u_max[0]:.1f}] °C")
            print(f"    mdot:     [{self.u_min[1]:.2f}, {self.u_max[1]:.2f}] kg/s")
            print(f"    du_max:   [{self.du_max[0]:.1f}, {self.du_max[1]:.2f}] per block")
            print(f"  Weights: {self.weights}")
            print(f"{'='*70}\n")

    def _default_warm_blocks(self) -> torch.Tensor:
        """Create default warm start: mid-range controls."""
        u_mid = (self.u_min + self.u_max) / 2.0
        return u_mid.unsqueeze(0).expand(self.n_blocks, -1).clone()

    def reset_warm_start(self):
        """Reset warm start (call between seeds/experiments)."""
        self._warm_blocks = None

    def solve(
        self,
        x0: np.ndarray,
        d_forecast: np.ndarray,
        u_prev: np.ndarray,
        Tmin_seq: np.ndarray,
        Tmax_seq: np.ndarray,
        occ_seq: Optional[np.ndarray] = None,
        verbose: bool = False,
    ) -> Dict:
        """
        Solve one MPC step.

        Interface matches MPCControllerHydronic.solve() for drop-in use.

        Args:
            x0: (nx,) current plant state
            d_forecast: (H_avail, nd) disturbance forecast
            u_prev: (nu,) last applied control
            Tmin_seq: (H_avail,) lower comfort bounds over horizon
            Tmax_seq: (H_avail,) upper comfort bounds over horizon
            occ_seq: (H_avail,) optional occupancy weights.
                     If None, inferred from Tmin_seq (occupied if Tmin >= 19.0)
            verbose: Print solver progress

        Returns:
            Dict with keys matching MPCControllerHydronic output:
              u_opt, u_opt_traj, status_solver, solve_time_ms, etc.
        """
        start_time = time.time()
        dtype = self.dtype

        # Convert inputs to tensors
        x0_t = torch.as_tensor(x0, dtype=dtype)
        u_prev_t = torch.as_tensor(u_prev, dtype=dtype)

        # Robust pad-or-trim helpers — each input is handled independently.
        # This prevents double-padding when the caller has already extended or
        # truncated one sequence but not another (e.g. the tail-of-episode bug
        # where d_forecast is length H_avail < H but Tmin/Tmax were pre-padded
        # to H, causing the old shared-length logic to over-extend them).
        H = self.H

        def _pad_or_trim_1d(arr: np.ndarray) -> np.ndarray:
            arr = np.asarray(arr, dtype=float).reshape(-1)
            if arr.shape[0] == 0:
                raise ValueError("Empty 1-D sequence passed to NMPCDirectShooting.solve().")
            if arr.shape[0] >= H:
                return arr[:H]
            return np.pad(arr, (0, H - arr.shape[0]), mode='edge')

        def _pad_or_trim_2d(arr: np.ndarray) -> np.ndarray:
            arr = np.asarray(arr, dtype=float)
            if arr.shape[0] == 0:
                raise ValueError("Empty 2-D sequence passed to NMPCDirectShooting.solve().")
            if arr.shape[0] >= H:
                return arr[:H]
            pad_rows = H - arr.shape[0]
            return np.concatenate([arr, np.repeat(arr[-1:], pad_rows, axis=0)], axis=0)

        D_hat = torch.as_tensor(_pad_or_trim_2d(d_forecast), dtype=dtype)
        T_low  = torch.as_tensor(_pad_or_trim_1d(Tmin_seq),  dtype=dtype)
        T_high = torch.as_tensor(_pad_or_trim_1d(Tmax_seq),  dtype=dtype)

        # Occupancy weights: 1.0 if occupied (Tmin >= 19.0), small otherwise
        if occ_seq is not None:
            occ_t = torch.as_tensor(_pad_or_trim_1d(occ_seq), dtype=dtype)
        else:
            # Infer from comfort bounds: occupied if Tmin >= 19.0
            occ_t = (T_low >= 19.0).to(dtype=dtype)

        # Sanity-check all sequences are exactly H steps
        assert D_hat.shape[0] == H, f"D_hat length {D_hat.shape[0]} != H {H}"
        assert T_low.shape[0]  == H, f"T_low length {T_low.shape[0]} != H {H}"
        assert T_high.shape[0] == H, f"T_high length {T_high.shape[0]} != H {H}"
        assert occ_t.shape[0]  == H, f"occ_t length {occ_t.shape[0]} != H {H}"

        # Initialize warm start
        if self._warm_blocks is None:
            warm_blocks = self._default_warm_blocks()
        else:
            warm_blocks = self._warm_blocks.clone()

        # Solve
        U_best_seq, warm_next, best_loss, info = solve_direct_shooting(
            model=self.model,
            x0=x0_t,
            D_hat=D_hat,
            occ_seq=occ_t,
            T_low_seq=T_low,
            T_high_seq=T_high,
            u_prev=u_prev_t,
            warm_blocks=warm_blocks,
            u_min=self.u_min,
            u_max=self.u_max,
            du_max=self.du_max,
            weights=self.weights,
            u_scale=self.u_scale,
            H=H,
            block_size=self.block_size,
            n_iter=self.n_iter,
            lr=self.lr,
            grad_clip=self.grad_clip,
        )

        # Store warm start for next call
        self._warm_blocks = warm_next

        solve_time_ms = (time.time() - start_time) * 1000.0

        # Convert to numpy for compatibility
        u_opt = U_best_seq[0].numpy()
        u_opt_traj = U_best_seq.numpy().T  # (nu, H) to match QP convention

        # Check solution quality
        is_finite = np.isfinite(best_loss)
        status = "nmpc_optimal" if is_finite else "nmpc_failed"

        if verbose:
            print(f"  NMPC solve: loss={best_loss:.4f}, iters={info['n_iter']}, "
                  f"Δloss={info['loss_reduction']:.3f}, time={solve_time_ms:.1f}ms")

        return {
            "status_solver": status,
            "status_mode": "normal" if is_finite else "fallback",
            "solve_time_ms": solve_time_ms,
            "fallback_used": not is_finite,
            "user_limit_used": False,
            "u_opt": u_opt,
            "u_opt_traj": u_opt_traj,
            "qp_cost": best_loss,
            # Compatibility fields
            "max_slack": -1.0,
            "mean_slack": -1.0,
            "slack_nonzero_frac": -1.0,
            "max_cold_slack": -1.0,
            "max_warm_slack": -1.0,
            "mean_cold_slack": -1.0,
            "mean_warm_slack": -1.0,
            "cold_slack_0": -1.0,
            "warm_slack_0": -1.0,
            "argmax_cold_slack": -1,
            "argmax_warm_slack": -1,
            "iterations": info["n_iter"],
            "osqp_iters_last": -1,
            "osqp_iters_total": -1,
            "rho_A_before": None,
            "rho_A_after": None,
            # NMPC-specific diagnostics
            "nmpc_loss_initial": info["loss_initial"],
            "nmpc_loss_final": info["loss_final"],
            "nmpc_loss_reduction": info["loss_reduction"],
            "nmpc_loss_history": info["loss_history"],
        }

    def get_config(self) -> Dict:
        """Return controller config for logging."""
        return {
            "controller_type": "nmpc_direct_shooting",
            "horizon": self.H,
            "block_size": self.block_size,
            "n_blocks": self.n_blocks,
            "n_iter": self.n_iter,
            "lr": self.lr,
            "grad_clip": self.grad_clip,
            "weights": self.weights,
            "u_min": self.u_min.numpy().tolist(),
            "u_max": self.u_max.numpy().tolist(),
            "du_max": self.du_max.numpy().tolist(),
            "dtype": str(self.dtype),
        }


# =============================================================================
# Validation utilities (Checklist A–E from expert procedure)
# =============================================================================

def validate_one_step_consistency(
    model: nn.Module,
    x: np.ndarray,
    u: np.ndarray,
    d: np.ndarray,
    dtype: torch.dtype = torch.float64,
    atol: float = 1e-8,
) -> Dict:
    """
    Checklist A: Verify exact wrapper gives same output as direct model call.

    Args:
        model: ThermalDynamicsNet
        x, u, d: Test inputs in physical units

    Returns:
        Dict with match status and max absolute error
    """
    model.eval()
    model.to(dtype=dtype)

    x_t = torch.as_tensor(x, dtype=dtype).unsqueeze(0)
    u_t = torch.as_tensor(u, dtype=dtype).unsqueeze(0)
    d_t = torch.as_tensor(d, dtype=dtype).unsqueeze(0)

    with torch.no_grad():
        y_direct = model(x_t, u_t, d_t).squeeze(0).numpy()
        y_phys = model.forward_phys(
            torch.as_tensor(x, dtype=dtype),
            torch.as_tensor(u, dtype=dtype),
            torch.as_tensor(d, dtype=dtype),
        ).numpy()

    max_err = float(np.max(np.abs(y_direct - y_phys)))
    match = max_err < atol

    return {
        "match": match,
        "max_abs_error": max_err,
        "y_direct": y_direct,
        "y_phys": y_phys,
    }


def validate_rollout_consistency(
    model: nn.Module,
    x0: np.ndarray,
    U_seq_np: np.ndarray,
    D_seq_np: np.ndarray,
    dtype: torch.dtype = torch.float64,
    atol: float = 1e-8,
) -> Dict:
    """
    Checklist B: Verify rollout function matches manual step-by-step.

    Args:
        model: ThermalDynamicsNet
        x0: (nx,) initial state
        U_seq_np: (H, nu) controls
        D_seq_np: (H, nd) disturbances

    Returns:
        Dict with match status and max absolute error
    """
    model.eval()
    model.to(dtype=dtype)

    H = U_seq_np.shape[0]
    x0_t = torch.as_tensor(x0, dtype=dtype)
    U_t = torch.as_tensor(U_seq_np, dtype=dtype)
    D_t = torch.as_tensor(D_seq_np, dtype=dtype)

    # Method 1: rollout function
    with torch.no_grad():
        X_rollout = rollout_exact(model, x0_t, U_t, D_t).numpy()

    # Method 2: manual step-by-step
    X_manual = np.zeros((H + 1, x0.shape[0]))
    X_manual[0] = x0
    with torch.no_grad():
        x = x0_t.clone()
        for k in range(H):
            x = model(
                x.unsqueeze(0), U_t[k].unsqueeze(0), D_t[k].unsqueeze(0)
            ).squeeze(0)
            X_manual[k + 1] = x.numpy()

    max_err = float(np.max(np.abs(X_rollout - X_manual)))
    match = max_err < atol

    return {
        "match": match,
        "max_abs_error": max_err,
        "X_rollout_shape": X_rollout.shape,
    }


def validate_gradient_sanity(
    model: nn.Module,
    x0: np.ndarray,
    U_seq_np: np.ndarray,
    D_seq_np: np.ndarray,
    occ_seq_np: np.ndarray,
    T_low_np: np.ndarray,
    T_high_np: np.ndarray,
    u_prev_np: np.ndarray,
    weights: Dict[str, float],
    u_scale_np: np.ndarray,
    var_idx: int = 0,
    eps: float = 1e-5,
    dtype: torch.dtype = torch.float64,
) -> Dict:
    """
    Checklist C: Compare autograd gradient vs finite differences.

    Tests gradient of total loss w.r.t. one control variable.

    Args:
        var_idx: Which element of U_seq to perturb (flattened index)
        eps: Finite difference step size

    Returns:
        Dict with autograd gradient, FD gradient, and relative error
    """
    model.eval()
    model.to(dtype=dtype)

    x0_t = torch.as_tensor(x0, dtype=dtype)
    D_t = torch.as_tensor(D_seq_np, dtype=dtype)
    occ_t = torch.as_tensor(occ_seq_np, dtype=dtype)
    T_low_t = torch.as_tensor(T_low_np, dtype=dtype)
    T_high_t = torch.as_tensor(T_high_np, dtype=dtype)
    u_prev_t = torch.as_tensor(u_prev_np, dtype=dtype)
    u_scale_t = torch.as_tensor(u_scale_np, dtype=dtype)

    # Autograd
    U_t = torch.as_tensor(U_seq_np, dtype=dtype).requires_grad_(True)
    X = rollout_exact(model, x0_t, U_t, D_t)
    loss = objective(X, U_t, D_t, occ_t, T_low_t, T_high_t,
                     u_prev_t, weights, u_scale_t)
    loss.backward()
    grad_auto = U_t.grad.flatten()[var_idx].item()

    # Finite difference
    U_plus = U_seq_np.copy().flatten()
    U_plus[var_idx] += eps
    U_minus = U_seq_np.copy().flatten()
    U_minus[var_idx] -= eps

    with torch.no_grad():
        U_p = torch.as_tensor(U_plus.reshape(U_seq_np.shape), dtype=dtype)
        X_p = rollout_exact(model, x0_t, U_p, D_t)
        loss_p = objective(X_p, U_p, D_t, occ_t, T_low_t, T_high_t,
                           u_prev_t, weights, u_scale_t).item()

        U_m = torch.as_tensor(U_minus.reshape(U_seq_np.shape), dtype=dtype)
        X_m = rollout_exact(model, x0_t, U_m, D_t)
        loss_m = objective(X_m, U_m, D_t, occ_t, T_low_t, T_high_t,
                           u_prev_t, weights, u_scale_t).item()

    grad_fd = (loss_p - loss_m) / (2 * eps)

    rel_err = abs(grad_auto - grad_fd) / (abs(grad_fd) + 1e-10)

    return {
        "grad_autograd": grad_auto,
        "grad_fd": grad_fd,
        "relative_error": rel_err,
        "directionally_consistent": (grad_auto * grad_fd > 0) if abs(grad_fd) > 1e-12 else True,
    }


def validate_optimization_sanity(
    model: nn.Module,
    x0: np.ndarray,
    D_seq_np: np.ndarray,
    occ_seq_np: np.ndarray,
    T_low_np: np.ndarray,
    T_high_np: np.ndarray,
    u_prev_np: np.ndarray,
    weights: Dict[str, float],
    dtype: torch.dtype = torch.float64,
    H: int = 24,
    block_size: int = 4,
    n_iter: int = 20,
) -> Dict:
    """
    Checklist D: Verify loss decreases from constant-control initialization.

    Args:
        Uses same arguments as solve_direct_shooting

    Returns:
        Dict with loss trajectory and whether it decreased
    """
    model.eval()
    model.to(dtype=dtype)

    u_min = torch.tensor([T_SUPPLY_MIN, MDOT_MIN], dtype=dtype)
    u_max = torch.tensor([T_SUPPLY_MAX, MDOT_MAX], dtype=dtype)
    du_max = torch.tensor([2.0, 0.3], dtype=dtype)
    u_scale = torch.tensor([T_SUPPLY_MAX - T_SUPPLY_MIN, MDOT_MAX - MDOT_MIN], dtype=dtype)

    # Initialize with mid-range constant controls
    u_mid = (u_min + u_max) / 2.0
    n_blocks = (H + block_size - 1) // block_size
    warm_blocks = u_mid.unsqueeze(0).expand(n_blocks, -1).clone()

    x0_t = torch.as_tensor(x0, dtype=dtype)
    D_t = torch.as_tensor(D_seq_np[:H], dtype=dtype)
    occ_t = torch.as_tensor(occ_seq_np[:H], dtype=dtype)
    T_low_t = torch.as_tensor(T_low_np[:H], dtype=dtype)
    T_high_t = torch.as_tensor(T_high_np[:H], dtype=dtype)
    u_prev_t = torch.as_tensor(u_prev_np, dtype=dtype)

    _, _, best_loss, info = solve_direct_shooting(
        model=model,
        x0=x0_t,
        D_hat=D_t,
        occ_seq=occ_t,
        T_low_seq=T_low_t,
        T_high_seq=T_high_t,
        u_prev=u_prev_t,
        warm_blocks=warm_blocks,
        u_min=u_min,
        u_max=u_max,
        du_max=du_max,
        weights=weights,
        u_scale=u_scale,
        H=H,
        block_size=block_size,
        n_iter=n_iter,
    )

    history = info["loss_history"]
    decreased = len(history) > 1 and history[-1] < history[0]

    return {
        "loss_decreased": decreased,
        "loss_initial": history[0] if history else float("nan"),
        "loss_final": history[-1] if history else float("nan"),
        "best_loss": best_loss,
        "loss_history": history,
        "n_valid_iters": info["n_iter"],
    }


# =============================================================================
# Main (self-test)
# =============================================================================

if __name__ == "__main__":
    print("NMPC Direct Shooting module loaded successfully.")
    print(f"Constants: DT={DT_SECONDS}s, Cp={CP_WATER}, "
          f"T_supply=[{T_SUPPLY_MIN},{T_SUPPLY_MAX}], "
          f"mdot=[{MDOT_MIN},{MDOT_MAX}]")

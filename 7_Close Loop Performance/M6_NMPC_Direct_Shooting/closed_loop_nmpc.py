#!/usr/bin/env python3
"""
Closed-Loop NMPC Simulator
=======================================

Replaces the linearize -> QP closed-loop pipeline with direct-shooting NMPC.

Key design:
  - The planning model and the plant are SEPARATE.
  - Controller A uses Fidelity_Baseline as planning model.
  - Controller B uses RAMC_lambda_0.0015 as planning model.
  - BOTH control the SAME reference plant (RC ground truth).
  - The ONLY thing that changes is f_θ. Everything else is identical.

Comparison protocol:
  - Same plant, same scenario, same seed, same optimizer settings.
  - Compare REALIZED plant metrics (not predicted objectives).

Author: Implementation of expert procedure for RAMC Phase 3
Date: 2026-03-06
"""

import numpy as np
import pandas as pd
import time
import json
import torch
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Union, List

from nmpc_direct_shooting import (
    NMPCDirectShooting,
    validate_one_step_consistency,
    validate_rollout_consistency,
    validate_gradient_sanity,
    validate_optimization_sanity,
    DT_SECONDS, DT_HOURS, CP_WATER,
    T_SUPPLY_MIN, T_SUPPLY_MAX, MDOT_MIN, MDOT_MAX,
    IDX_T_AIR, IDX_T_RET,
)

# Import from existing codebase
import sys
from pathlib import Path

_this_dir = Path(__file__).resolve().parent
_project_dir = _this_dir.parent  # 7_Close Loop Performance

sys.path.insert(0, str(_this_dir))                          # M6 (finds nmpc_direct_shooting)
sys.path.insert(0, str(_project_dir / "M4_Closed_Loop_Simulator"))

# Also need the NN architecture module
NN_ARCH_PATH = Path(
    r"C:\Users\nmi03\OneDrive - Mälardalens universitet\Studying Folder"
    r"\New Project for Risk Aware Model then Control\6_Neural Network Architecture"
)
sys.path.insert(0, str(NN_ARCH_PATH))

from rc_ground_truth import RCGroundTruthModel
from load_ramc_model import load_ramc_model
from thermal_dynamics_net import ThermalDynamicsNet

# Re-use scenario/occupancy logic from existing simulator
from shared_constants import (
    get_occupancy_status,
    get_comfort_bounds,
    stage_cost_ramc_np,
    OCC_TARGET_C, DEADBAND_C, UNOCC_TARGET_C,
    ENERGY_COST_RATE,
    Q_INTERNAL_OCCUPIED_W, Q_INTERNAL_UNOCCUPIED_W,
)



# =============================================================================
# RC Plant Parameters (must match run_experimental_matrix.py)
# =============================================================================

RC_PARAMS = {
    "C_air": 84426246.51832934,
    "C_env": 661376048.5634619,
    "C_int": 6002555765.428176,
    "C_rad": 597376.9622532368,
    "R_ex": 0.0003079967462046204,
    "R_ae": 0.00016180515492072352,
    "R_ai": 5.818704417041902e-05,
    "K_rad": 283.97226311004107,
    "a_rad": 0.2795410071110868,
    "A_sol": 0.5598500539656959,
}


# =============================================================================
# Closed-Loop NMPC Simulator
# =============================================================================

class ClosedLoopNMPCSimulator:
    """
    Closed-loop simulator with direct-shooting NMPC controller.

    Mirrors ClosedLoopSimulator from closed_loop_simulator.py but uses
    NMPCDirectShooting instead of linearized QP. Produces compatible
    output format so existing analysis code works.
    """

    def __init__(
        self,
        nn_model_path: Union[str, Path],
        weather_truth_path: Union[str, Path],
        weather_forecast_path: Optional[Union[str, Path]] = None,
        # NMPC settings
        nmpc_horizon: int = 24,
        nmpc_block_size: int = 4,
        nmpc_n_iter: int = 25,
        nmpc_lr: float = 0.05,
        nmpc_grad_clip: float = 5.0,
        # Objective weights (SAME for all models)
        w_energy: float = 0.9,
        w_cold: float = 63.0,
        w_hot: float = 30.0,
        w_du: float = 1e-3,
        w_terminal: float = 20.0,
        w_trust: float = 0.0,
        # Control bounds
        du_max: Optional[np.ndarray] = None,
        # Config
        energy_cost_rate: float = ENERGY_COST_RATE,
        dtype: torch.dtype = torch.float64,
        comfort_margin_C: float = 0.0,
        verbose_init: bool = True,
    ):
        """
        Initialize NMPC closed-loop simulator.

        Args:
            nn_model_path: Path to NN checkpoint (planning model)
            weather_truth_path: Path to ground truth weather CSV
            weather_forecast_path: Path to forecast weather CSV (None = use truth)
            nmpc_horizon: Planning horizon in steps
            nmpc_block_size: Move blocking size
            nmpc_n_iter: Adam iterations per MPC solve
            nmpc_lr: Adam learning rate
            nmpc_grad_clip: Gradient norm clipping
            w_energy..w_trust: Objective weights (SAME for all models)
            du_max: Actuator rate limits per block
            energy_cost_rate: SEK/kWh
            dtype: Tensor dtype
            comfort_margin_C: Upper shift for occupied lower comfort bound
            verbose_init: Print init info
        """
        if verbose_init:
            print(f"\n{'='*70}")
            print("PRIORITY 3: CLOSED-LOOP NMPC SIMULATOR")
            print(f"{'='*70}")

        self.dt_seconds = DT_SECONDS
        self.dt_minutes = DT_SECONDS / 60.0
        self.energy_cost_rate = energy_cost_rate
        self.dtype = dtype
        # Planning-only comfort margin (R1.4 Pareto study). Shifts the OCCUPIED
        # lower comfort bound seen by the NMPC planner upward by this many °C.
        # Default 0.0 means no change, so every existing caller (the ablation
        # matrix, the main-paper matrix) is bitwise-identical to before.
        self.comfort_margin_C = float(comfort_margin_C)
        self.nx = 6
        self.nu = 2
        self.nd = 3

        # ── Load planning model (NN) ──
        if verbose_init:
            print(f"\nLoading planning model...")

        self.model = load_ramc_model(
            nn_model_path,
            device='cpu',
            dtype=dtype,
            verbose=verbose_init,
        )

        # ── Create plant (RC ground truth) ──
        if verbose_init:
            print(f"\nCreating ground truth plant...")

        self.ground_truth = RCGroundTruthModel(
            params=RC_PARAMS,
            dt_seconds=self.dt_seconds,
        )

        if verbose_init:
            print(f"  {self.ground_truth}")

        # ── Load weather ──
        if verbose_init:
            print(f"\nLoading weather data...")

        self.weather_truth = self._load_weather(weather_truth_path, "truth")

        if weather_forecast_path is not None:
            self.weather_forecast = self._load_weather(weather_forecast_path, "forecast")
        else:
            self.weather_forecast = self.weather_truth.copy()
            if verbose_init:
                print(f"  Using truth as forecast (nominal scenario)")

        self._validate_weather_alignment(verbose_init)

        # ── Create NMPC controller ──
        du_max_arr = du_max if du_max is not None else np.array([2.0, 0.3])

        self.nmpc = NMPCDirectShooting(
            model=self.model,
            horizon=nmpc_horizon,
            block_size=nmpc_block_size,
            n_iter=nmpc_n_iter,
            lr=nmpc_lr,
            grad_clip=nmpc_grad_clip,
            du_max=du_max_arr,
            w_energy=w_energy,
            w_cold=w_cold,
            w_hot=w_hot,
            w_du=w_du,
            w_terminal=w_terminal,
            w_trust=w_trust,
            energy_cost_rate=energy_cost_rate,
            dtype=dtype,
            verbose=verbose_init,
        )

        if verbose_init:
            print(f"\n{'='*70}")
            print("NMPC Simulator initialized successfully!")
            print(f"{'='*70}\n")

    def _load_weather(self, path: Union[str, Path], name: str) -> pd.DataFrame:
        """Load weather CSV."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Weather file not found: {path}")

        df = pd.read_csv(path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        required_cols = ['timestamp', 'T_out_C', 'Q_solar_W']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            df = df.rename(columns={'T_outdoor_C': 'T_out_C'})
            missing = [c for c in required_cols if c not in df.columns]
            if missing:
                raise ValueError(f"Missing columns in {name} weather: {missing}")

        print(f"  Loaded {name}: {len(df)} steps, "
              f"T_out: [{df['T_out_C'].min():.1f}, {df['T_out_C'].max():.1f}]°C")
        return df

    def _validate_weather_alignment(self, verbose: bool = True):
        """Validate forecast/truth alignment."""
        n_check = min(len(self.weather_truth), len(self.weather_forecast))
        truth_ts = self.weather_truth['timestamp'].iloc[:n_check].reset_index(drop=True)
        forecast_ts = self.weather_forecast['timestamp'].iloc[:n_check].reset_index(drop=True)

        if not truth_ts.equals(forecast_ts):
            raise ValueError("Forecast and truth timestamps do not align.")

        if verbose:
            print(f"  Weather alignment verified")

    def _build_disturbance_truth(self, weather_idx: int, occupied: bool) -> np.ndarray:
        """Build disturbance from truth weather."""
        row = self.weather_truth.iloc[weather_idx]
        T_out = float(row['T_out_C'])
        Q_solar = float(row['Q_solar_W'])
        if 'Q_internal_W' in self.weather_truth.columns:
            Q_internal = float(row['Q_internal_W'])
        else:
            Q_internal = Q_INTERNAL_OCCUPIED_W if occupied else Q_INTERNAL_UNOCCUPIED_W
        return np.array([T_out, Q_solar, Q_internal], dtype=float)

    def _build_disturbance_forecast(
        self, start_idx: int, horizon: int, start_timestamp: pd.Timestamp
    ) -> np.ndarray:
        """Build forecast disturbance matrix."""
        d_forecast = np.zeros((horizon, self.nd), dtype=float)
        has_q_internal = 'Q_internal_W' in self.weather_forecast.columns

        for h in range(horizon):
            idx = start_idx + h
            if idx < len(self.weather_forecast):
                row = self.weather_forecast.iloc[idx]
            else:
                row = self.weather_forecast.iloc[-1]

            T_out = float(row['T_out_C'])
            Q_solar = float(row['Q_solar_W'])

            if has_q_internal:
                Q_internal = float(row['Q_internal_W'])
            else:
                ts_h = start_timestamp + timedelta(seconds=h * self.dt_seconds)
                occ_h = get_occupancy_status(ts_h)
                Q_internal = Q_INTERNAL_OCCUPIED_W if occ_h else Q_INTERNAL_UNOCCUPIED_W

            d_forecast[h] = [T_out, Q_solar, Q_internal]

        return d_forecast

    def _build_comfort_bounds_sequence(
        self, start_timestamp: pd.Timestamp, horizon: int
    ) -> tuple:
        """Build horizon-scheduled comfort bounds (matches existing simulator)."""
        Tmin_seq = np.zeros(horizon, dtype=float)
        Tmax_seq = np.zeros(horizon, dtype=float)

        for h in range(horizon):
            ts_h = start_timestamp + timedelta(seconds=(h + 1) * self.dt_seconds)
            occ_h = get_occupancy_status(ts_h)
            tmin_h, tmax_h = get_comfort_bounds(occ_h)
            # Planning-only margin: tighten the OCCUPIED lower bound only.
            # With the default comfort_margin_C=0.0 this adds 0.0 (an IEEE-754
            # identity), so ablation/main-paper paths are unchanged.
            if occ_h:
                tmin_h = tmin_h + self.comfort_margin_C
            Tmin_seq[h] = tmin_h
            Tmax_seq[h] = tmax_h

        return Tmin_seq, Tmax_seq

    def reset_for_new_seed(self):
        """Reset for new seed."""
        self.nmpc.reset_warm_start()

    def simulate_episode(
        self,
        initial_state: np.ndarray,
        simulation_steps: Optional[int] = None,
        start_idx: int = 0,
        verbose: bool = True,
        log_interval: int = 100,
    ) -> Dict[str, Any]:
        """
        Run closed-loop simulation with NMPC controller.

        Produces output compatible with ClosedLoopSimulator.simulate_episode().

        Args:
            initial_state: (nx,) initial plant state
            simulation_steps: Number of steps (None = full weather data)
            start_idx: Starting weather index
            verbose: Print progress
            log_interval: Steps between progress prints

        Returns:
            Dict with all metrics matching existing simulator output
        """
        max_steps = len(self.weather_truth) - start_idx
        if simulation_steps is None:
            simulation_steps = max_steps
        else:
            simulation_steps = min(simulation_steps, max_steps)

        nmpc_horizon = self.nmpc.H

        if verbose:
            print(f"\n{'='*70}")
            print("STARTING NMPC SIMULATION")
            print(f"{'='*70}")
            print(f"  Steps: {simulation_steps} ({simulation_steps * self.dt_seconds / 3600:.1f}h)")
            print(f"  Initial T_air: {initial_state[0]:.2f}°C")
            print(f"  Planning horizon: {nmpc_horizon} steps ({nmpc_horizon * self.dt_seconds / 3600:.1f}h)")
            print(f"  Comfort (occ): [{OCC_TARGET_C - DEADBAND_C}, {OCC_TARGET_C + DEADBAND_C}]°C")

        # Storage
        states = np.zeros((simulation_steps + 1, self.nx))
        controls = np.zeros((simulation_steps, self.nu))
        Tmin_series = np.zeros(simulation_steps)
        Tmax_series = np.zeros(simulation_steps)
        occupancy_series = np.zeros(simulation_steps, dtype=bool)

        energy_kWh_step = []
        energy_cost_step = []
        Q_heat_W_step = []
        solver_time_ms = []
        solver_status = []
        fallback_used = []
        user_limit_used = []

        # Compatibility fields
        cold_slack_0_step = []
        warm_slack_0_step = []
        argmax_cold_slack_step = []
        max_slack_step = []
        mean_slack_step = []
        slack_nonzero_frac_step = []
        osqp_iters_last_step = []
        osqp_iters_total_step = []
        rho_A_before_step = []
        rho_A_after_step = []
        max_cold_slack_step = []
        max_warm_slack_step = []
        residual_Tair_step = []

        # NMPC-specific diagnostics
        nmpc_loss_initial_step = []
        nmpc_loss_final_step = []
        nmpc_loss_reduction_step = []

        # RAMC-style cost
        stage_cost_ramc_step = []
        comfort_cost_step = []
        energy_cost_ramc_step = []

        states[0] = initial_state
        start_time = time.time()

        for k in range(simulation_steps):
            if verbose and k % log_interval == 0:
                elapsed = time.time() - start_time
                rate = k / elapsed if elapsed > 0 else 0
                eta = (simulation_steps - k) / rate if rate > 0 else 0
                print(f"  Step {k}/{simulation_steps} ({k/simulation_steps*100:.1f}%) - "
                      f"Rate: {rate:.1f}/s, ETA: {eta:.0f}s")

            x_current = states[k]
            weather_idx = start_idx + k

            timestamp_k = self.weather_truth.iloc[weather_idx]['timestamp']
            timestamp_kp1 = timestamp_k + timedelta(seconds=self.dt_seconds)

            occupied_interval = get_occupancy_status(timestamp_k)
            occupied_next = get_occupancy_status(timestamp_kp1)

            occupancy_series[k] = occupied_next
            T_min_eval, T_max_eval = get_comfort_bounds(occupied_next)
            Tmin_series[k] = T_min_eval
            Tmax_series[k] = T_max_eval

            d_truth = self._build_disturbance_truth(weather_idx, occupied_interval)

            horizon_steps = min(nmpc_horizon, len(self.weather_truth) - weather_idx)
            d_forecast = self._build_disturbance_forecast(
                weather_idx, horizon_steps, timestamp_k
            )

            # Always build a full planning-horizon comfort sequence.
            # NMPCDirectShooting.solve() handles padding/truncation internally,
            # so we must NOT pre-pad here — doing so would cause a double-pad bug
            # at the tail of the episode (last <nmpc_horizon steps).
            Tmin_seq, Tmax_seq = self._build_comfort_bounds_sequence(
                timestamp_k, nmpc_horizon
            )

            u_prev = controls[k - 1] if k > 0 else np.array([45.0, 0.2])

            try:
                mpc_result = self.nmpc.solve(
                    x0=x_current,
                    d_forecast=d_forecast,
                    u_prev=u_prev,
                    Tmin_seq=Tmin_seq,
                    Tmax_seq=Tmax_seq,
                    verbose=False,
                )

                u_opt = mpc_result['u_opt']
                solver_time_ms.append(mpc_result['solve_time_ms'])
                solver_status.append(mpc_result['status_solver'])
                fallback_used.append(mpc_result['fallback_used'])
                user_limit_used.append(mpc_result.get('user_limit_used', False))

                # Compatibility fields
                max_slack_step.append(mpc_result.get('max_slack', -1.0))
                mean_slack_step.append(mpc_result.get('mean_slack', -1.0))
                slack_nonzero_frac_step.append(mpc_result.get('slack_nonzero_frac', -1.0))
                osqp_iters_last_step.append(-1)
                osqp_iters_total_step.append(-1)
                rho_A_before_step.append(None)
                rho_A_after_step.append(None)
                max_cold_slack_step.append(-1.0)
                max_warm_slack_step.append(-1.0)
                cold_slack_0_step.append(-1.0)
                warm_slack_0_step.append(-1.0)
                argmax_cold_slack_step.append(-1)
                residual_Tair_step.append(None)

                # NMPC diagnostics
                nmpc_loss_initial_step.append(mpc_result.get('nmpc_loss_initial', float('nan')))
                nmpc_loss_final_step.append(mpc_result.get('nmpc_loss_final', float('nan')))
                nmpc_loss_reduction_step.append(mpc_result.get('nmpc_loss_reduction', 0.0))

            except Exception as e:
                if verbose:
                    print(f"  NMPC failed at step {k}: {e}")
                # Rate-limited fallback
                u_des = np.array([55.0, 2.0])
                du_max_np = self.nmpc.du_max.numpy()
                du = np.clip(u_des - u_prev, -du_max_np, du_max_np)
                u_opt = np.clip(
                    u_prev + du,
                    [T_SUPPLY_MIN, MDOT_MIN],
                    [T_SUPPLY_MAX, MDOT_MAX],
                )
                solver_time_ms.append(0.0)
                solver_status.append(f"exception: {e}")
                fallback_used.append(True)
                user_limit_used.append(False)
                max_slack_step.append(-1.0)
                mean_slack_step.append(-1.0)
                slack_nonzero_frac_step.append(-1.0)
                osqp_iters_last_step.append(-1)
                osqp_iters_total_step.append(-1)
                rho_A_before_step.append(None)
                rho_A_after_step.append(None)
                max_cold_slack_step.append(-1.0)
                max_warm_slack_step.append(-1.0)
                cold_slack_0_step.append(-1.0)
                warm_slack_0_step.append(-1.0)
                argmax_cold_slack_step.append(-1)
                residual_Tair_step.append(None)
                nmpc_loss_initial_step.append(float('nan'))
                nmpc_loss_final_step.append(float('nan'))
                nmpc_loss_reduction_step.append(0.0)

            # Apply control to plant
            T_supply = float(np.clip(u_opt[0], T_SUPPLY_MIN, T_SUPPLY_MAX))
            mdot = float(np.clip(max(u_opt[1], 0.0), MDOT_MIN, MDOT_MAX))
            controls[k] = [T_supply, mdot]

            # Energy computation (matches existing simulator exactly)
            T_return = x_current[5]
            Q_heat_W = mdot * CP_WATER * max(T_supply - T_return, 0.0)
            energy_kWh = Q_heat_W * self.dt_seconds / 3600.0 / 1000.0
            energy_cost = energy_kWh * self.energy_cost_rate

            energy_kWh_step.append(energy_kWh)
            energy_cost_step.append(energy_cost)
            Q_heat_W_step.append(Q_heat_W)

            # Step plant with TRUE disturbance
            u = np.array([T_supply, mdot])
            x_next = self.ground_truth.step(x_current, u, d_truth)
            states[k + 1] = x_next

            # RAMC-style stage cost (same form as Phase 2 training)
            total_c, c_comfort, c_energy, _ = stage_cost_ramc_np(
                T_air=np.array([x_next[0]]),
                T_ret=np.array([x_next[5]]),
                T_supply=np.array([T_supply]),
                mdot=np.array([mdot]),
                Tmin=np.array([Tmin_series[k]]),
                Tmax=np.array([Tmax_series[k]]),
                comfort_beta=0.5,
                dt_minutes=self.dt_minutes,
                energy_cost_rate=self.energy_cost_rate,
                w_comfort=63.0,
                w_energy=1.0,
            )
            stage_cost_ramc_step.append(float(total_c))
            comfort_cost_step.append(float(c_comfort))
            energy_cost_ramc_step.append(float(c_energy))

        # ── Compute metrics (matching existing simulator format) ──
        results = self._compute_metrics(
            states=states,
            controls=controls,
            Tmin_series=Tmin_series,
            Tmax_series=Tmax_series,
            occupancy_series=occupancy_series,
            energy_kWh_step=energy_kWh_step,
            energy_cost_step=energy_cost_step,
            Q_heat_W_step=Q_heat_W_step,
            solver_time_ms=solver_time_ms,
            solver_status=solver_status,
            fallback_used=fallback_used,
            user_limit_used=user_limit_used,
            stage_cost_ramc_step=stage_cost_ramc_step,
            comfort_cost_step=comfort_cost_step,
            energy_cost_ramc_step=energy_cost_ramc_step,
            simulation_steps=simulation_steps,
            # Compatibility
            osqp_iters_last_step=osqp_iters_last_step,
            osqp_iters_total_step=osqp_iters_total_step,
            rho_A_before_step=rho_A_before_step,
            rho_A_after_step=rho_A_after_step,
            max_cold_slack_step=max_cold_slack_step,
            max_warm_slack_step=max_warm_slack_step,
            cold_slack_0_step=cold_slack_0_step,
            warm_slack_0_step=warm_slack_0_step,
            argmax_cold_slack_step=argmax_cold_slack_step,
            max_slack_step=max_slack_step,
            mean_slack_step=mean_slack_step,
            slack_nonzero_frac_step=slack_nonzero_frac_step,
            residual_Tair_step=residual_Tair_step,
            # NMPC-specific
            nmpc_loss_initial_step=nmpc_loss_initial_step,
            nmpc_loss_final_step=nmpc_loss_final_step,
            nmpc_loss_reduction_step=nmpc_loss_reduction_step,
        )

        if verbose:
            self._print_summary(results)

        return results

    def _compute_metrics(self, **kwargs) -> Dict[str, Any]:
        """Compute all metrics matching existing simulator output format."""
        states = kwargs['states']
        controls = kwargs['controls']
        Tmin_series = kwargs['Tmin_series']
        Tmax_series = kwargs['Tmax_series']
        occupancy = kwargs['occupancy_series']
        N = kwargs['simulation_steps']

        T_air = states[1:N+1, IDX_T_AIR]

        # Occupied mask
        occ_mask = occupancy[:N].astype(bool)
        n_occupied = int(np.sum(occ_mask))
        n_unoccupied = N - n_occupied

        # ── Cold violations (PRIMARY) ──
        cold_violations = np.maximum(Tmin_series[:N] - T_air, 0.0)
        cold_occ = cold_violations[occ_mask] if n_occupied > 0 else np.zeros(1)

        deg_hours_cold_occ = float(np.sum(cold_occ) * self.dt_seconds / 3600.0)
        hours_cold_outside_occ = float(np.sum(cold_occ > 0) * self.dt_seconds / 3600.0)
        hours_cold_outside_occ_025 = float(np.sum(cold_occ > 0.25) * self.dt_seconds / 3600.0)
        peak_cold_occ = float(np.max(cold_occ)) if n_occupied > 0 else 0.0
        n_cold_violations_occ = int(np.sum(cold_occ > 0))

        var90_cold, cvar90_cold = self._compute_var_cvar(cold_occ, 0.9)
        var95_cold, cvar95_cold = self._compute_var_cvar(cold_occ, 0.95)

        # ── Warm violations (SECONDARY) ──
        warm_violations = np.maximum(T_air - Tmax_series[:N], 0.0)
        warm_occ = warm_violations[occ_mask] if n_occupied > 0 else np.zeros(1)

        deg_hours_warm_occ = float(np.sum(warm_occ) * self.dt_seconds / 3600.0)
        hours_warm_outside_occ = float(np.sum(warm_occ > 0) * self.dt_seconds / 3600.0)
        peak_warm_occ = float(np.max(warm_occ)) if n_occupied > 0 else 0.0
        n_warm_violations_occ = int(np.sum(warm_occ > 0))

        var90_warm, cvar90_warm = self._compute_var_cvar(warm_occ, 0.9)
        var95_warm, cvar95_warm = self._compute_var_cvar(warm_occ, 0.95)

        # ── Band violations (combined) ──
        band_violations = cold_violations + warm_violations
        band_occ = band_violations[occ_mask] if n_occupied > 0 else np.zeros(1)
        deg_hours_band_occ = deg_hours_cold_occ + deg_hours_warm_occ
        hours_band_outside_occ = float(np.sum(band_occ > 0) * self.dt_seconds / 3600.0)
        peak_band_occ = float(np.max(band_occ)) if n_occupied > 0 else 0.0
        n_band_violations_occ = int(np.sum(band_occ > 0))

        var90_band, cvar90_band = self._compute_var_cvar(band_occ, 0.9)
        var95_band, cvar95_band = self._compute_var_cvar(band_occ, 0.95)

        # ── T_air statistics ──
        T_air_occ = T_air[occ_mask] if n_occupied > 0 else np.array([0.0])
        T_air_occ_mean = float(np.mean(T_air_occ))
        T_air_occ_min = float(np.min(T_air_occ))
        T_air_occ_max = float(np.max(T_air_occ))

        # ── Energy ──
        total_energy_kWh = float(np.sum(kwargs['energy_kWh_step']))
        total_energy_cost = float(np.sum(kwargs['energy_cost_step']))

        # ── Full-simulation stats ──
        deg_hours_full = float(np.sum(band_violations) * self.dt_seconds / 3600.0)
        deg_hours_below_full = float(np.sum(cold_violations) * self.dt_seconds / 3600.0)
        deg_hours_above_full = float(np.sum(warm_violations) * self.dt_seconds / 3600.0)

        # ── RAMC-style cost ──
        stage_costs = np.array(kwargs['stage_cost_ramc_step'])
        stage_costs_occ = stage_costs[occ_mask] if n_occupied > 0 else np.zeros(1)

        var90_cost_all, cvar90_cost_all = self._compute_var_cvar(stage_costs, 0.9)
        var95_cost_all, cvar95_cost_all = self._compute_var_cvar(stage_costs, 0.95)
        var90_cost_occ, cvar90_cost_occ = self._compute_var_cvar(stage_costs_occ, 0.9)
        var95_cost_occ, cvar95_cost_occ = self._compute_var_cvar(stage_costs_occ, 0.95)

        return {
            # Controller type identifier
            'controller_type': 'nmpc_direct_shooting',

            # Energy
            'total_energy_kWh': total_energy_kWh,
            'total_energy_cost': total_energy_cost,

            # PRIMARY: Cold violations
            'peak_cold_violation_occ_C': peak_cold_occ,
            'deg_hours_cold_occ': deg_hours_cold_occ,
            'hours_cold_outside_occ': hours_cold_outside_occ,
            'hours_cold_outside_occ_025': hours_cold_outside_occ_025,
            'n_cold_violations_occ': n_cold_violations_occ,
            'var90_cold_occ_C': var90_cold,
            'cvar90_cold_occ_C': cvar90_cold,
            'var95_cold_occ_C': var95_cold,
            'cvar95_cold_occ_C': cvar95_cold,

            # SECONDARY: Warm violations
            'peak_warm_violation_occ_C': peak_warm_occ,
            'deg_hours_warm_occ': deg_hours_warm_occ,
            'hours_warm_outside_occ': hours_warm_outside_occ,
            'n_warm_violations_occ': n_warm_violations_occ,
            'var90_warm_occ_C': var90_warm,
            'cvar90_warm_occ_C': cvar90_warm,
            'var95_warm_occ_C': var95_warm,
            'cvar95_warm_occ_C': cvar95_warm,

            # Band (combined)
            'peak_band_violation_occ_C': peak_band_occ,
            'deg_hours_band_occ': deg_hours_band_occ,
            'hours_band_outside_occ': hours_band_outside_occ,
            'n_band_violations_occ': n_band_violations_occ,
            'var90_band_occ_C': var90_band,
            'cvar90_band_occ_C': cvar90_band,
            'var95_band_occ_C': var95_band,
            'cvar95_band_occ_C': cvar95_band,

            # Legacy (backward compat)
            'deg_hours_occ': deg_hours_band_occ,
            'deg_hours_below_occ': deg_hours_cold_occ,
            'deg_hours_above_occ': deg_hours_warm_occ,
            'hours_outside_occ': hours_band_outside_occ,
            'peak_violation_occ_C': peak_band_occ,
            'n_violations_occ': n_band_violations_occ,
            'var90_occ_C': var90_band,
            'cvar90_occ_C': cvar90_band,
            'var95_occ_C': var95_band,
            'cvar95_occ_C': cvar95_band,

            # T_air
            'T_air_occ_mean_C': T_air_occ_mean,
            'T_air_occ_min_C': T_air_occ_min,
            'T_air_occ_max_C': T_air_occ_max,
            'T_air_mean_C': float(np.mean(T_air)),
            'T_air_min_C': float(np.min(T_air)),
            'T_air_max_C': float(np.max(T_air)),

            # Full simulation
            'deg_hours_full': deg_hours_full,
            'deg_hours_below_full': deg_hours_below_full,
            'deg_hours_above_full': deg_hours_above_full,

            'n_occupied_steps': n_occupied,
            'n_unoccupied_steps': n_unoccupied,
            'occupied_hours': float(n_occupied * self.dt_seconds / 3600.0),

            # RAMC-style cost
            'total_stage_cost_ramc': float(np.sum(stage_costs)),
            'total_stage_cost_ramc_occ': float(np.sum(stage_costs_occ)),
            'total_comfort_cost_ramc': float(np.sum(kwargs['comfort_cost_step'])),
            'total_energy_cost_ramc': float(np.sum(kwargs['energy_cost_ramc_step'])),
            'var90_stage_cost_all': var90_cost_all,
            'cvar90_stage_cost_all': cvar90_cost_all,
            'var95_stage_cost_all': var95_cost_all,
            'cvar95_stage_cost_all': cvar95_cost_all,
            'var90_stage_cost_occ': var90_cost_occ,
            'cvar90_stage_cost_occ': cvar90_cost_occ,
            'var95_stage_cost_occ': var95_cost_occ,
            'cvar95_stage_cost_occ': cvar95_cost_occ,

            # Solver
            'solver_time_ms': kwargs['solver_time_ms'],
            'solver_status_step': kwargs['solver_status'],
            'fallback_used': kwargs['fallback_used'],
            'user_limit_used': kwargs['user_limit_used'],

            # Per-step data
            'states': states,
            'controls': controls,
            'Tmin_series': Tmin_series,
            'Tmax_series': Tmax_series,
            'occupancy_series': occupancy,
            'energy_kWh_step': kwargs['energy_kWh_step'],
            'energy_cost_step': kwargs['energy_cost_step'],
            'Q_heat_W_step': kwargs['Q_heat_W_step'],

            # Compatibility fields (QP-specific, set to -1/None)
            'osqp_iters_last_step': kwargs['osqp_iters_last_step'],
            'osqp_iters_total_step': kwargs['osqp_iters_total_step'],
            'rho_A_before_step': kwargs['rho_A_before_step'],
            'rho_A_after_step': kwargs['rho_A_after_step'],
            'max_cold_slack_step': kwargs['max_cold_slack_step'],
            'max_warm_slack_step': kwargs['max_warm_slack_step'],
            'cold_slack_0_step': kwargs['cold_slack_0_step'],
            'warm_slack_0_step': kwargs['warm_slack_0_step'],
            'argmax_cold_slack_step': kwargs['argmax_cold_slack_step'],
            'max_slack_step': kwargs['max_slack_step'],
            'mean_slack_step': kwargs['mean_slack_step'],
            'slack_nonzero_frac_step': kwargs['slack_nonzero_frac_step'],
            'residual_Tair_step': kwargs['residual_Tair_step'],

            # NMPC-specific
            'nmpc_loss_initial_step': kwargs['nmpc_loss_initial_step'],
            'nmpc_loss_final_step': kwargs['nmpc_loss_final_step'],
            'nmpc_loss_reduction_step': kwargs['nmpc_loss_reduction_step'],
        }

    def _compute_var_cvar(self, arr: np.ndarray, alpha: float = 0.9) -> tuple:
        """Compute empirical VaR and CVaR (matches existing simulator)."""
        arr = np.asarray(arr, dtype=float)
        arr = arr[~np.isnan(arr)]
        if len(arr) == 0:
            return 0.0, 0.0

        var = float(np.quantile(arr, alpha))
        n_tail = max(1, int(np.ceil(len(arr) * (1.0 - alpha))))
        sorted_desc = np.sort(arr)[::-1]
        tail = sorted_desc[:n_tail]
        cvar = float(np.mean(tail))

        return var, cvar

    def _print_summary(self, results: Dict[str, Any]):
        """Print summary matching existing simulator format."""
        print(f"\n{'='*70}")
        print("NMPC SIMULATION METRICS")
        print(f"{'='*70}")

        print(f"\nENERGY:")
        print(f"  Total: {results['total_energy_kWh']:.1f} kWh")
        print(f"  Cost: {results['total_energy_cost']:.2f} SEK")

        print(f"\nCOMFORT - COLD (PRIMARY, Occupied: {results['occupied_hours']:.1f}h):")
        print(f"  Comfort band: [{OCC_TARGET_C - DEADBAND_C}, {OCC_TARGET_C + DEADBAND_C}]°C")
        print(f"  Degree-hours (cold): {results['deg_hours_cold_occ']:.3f} °C·h")
        print(f"  Hours below Tmin: {results['hours_cold_outside_occ']:.2f} h")
        print(f"  Hours below Tmin (>0.25°C): {results['hours_cold_outside_occ_025']:.2f} h")
        print(f"  Peak cold violation: {results['peak_cold_violation_occ_C']:.2f}°C")
        print(f"  CVaR90 (cold): {results['cvar90_cold_occ_C']:.3f}°C")
        print(f"  CVaR95 (cold): {results['cvar95_cold_occ_C']:.3f}°C")

        print(f"\nCOMFORT - WARM (SECONDARY):")
        print(f"  Degree-hours (warm): {results['deg_hours_warm_occ']:.3f} °C·h")
        print(f"  Peak warm violation: {results['peak_warm_violation_occ_C']:.2f}°C")

        print(f"\nT_air (occupied):")
        print(f"  Mean: {results['T_air_occ_mean_C']:.2f}°C")
        print(f"  Range: [{results['T_air_occ_min_C']:.2f}, {results['T_air_occ_max_C']:.2f}]°C")

        print(f"\nRAMC-STYLE COST:")
        print(f"  Total: {results['total_stage_cost_ramc']:.2f}")
        print(f"  CVaR90 (occ): {results['cvar90_stage_cost_occ']:.4f}")

        fallbacks = np.array(results['fallback_used'])
        print(f"\nNMPC SOLVER:")
        print(f"  Fallback: {np.mean(fallbacks)*100:.1f}%")
        print(f"  Median time: {np.median(results['solver_time_ms']):.1f}ms")
        print(f"  P95 time: {np.percentile(results['solver_time_ms'], 95):.1f}ms")

        # NMPC-specific
        valid_reductions = [r for r in results['nmpc_loss_reduction_step']
                           if np.isfinite(r)]
        if valid_reductions:
            print(f"  Median loss reduction: {np.median(valid_reductions):.3f}")
            print(f"  Mean loss reduction: {np.mean(valid_reductions):.3f}")

        print(f"{'='*70}\n")


# =============================================================================
# Pairwise comparison runner
# =============================================================================

def run_pairwise_comparison(
    model_a_path: Union[str, Path],
    model_b_path: Union[str, Path],
    weather_truth_path: Union[str, Path],
    weather_forecast_path: Union[str, Path],
    initial_state: np.ndarray,
    simulation_steps: Optional[int] = None,
    seed: int = 42,
    label_a: str = "Fidelity_Baseline",
    label_b: str = "RAMC_lambda_0.0015",
    verbose: bool = True,
    **nmpc_kwargs,
) -> Dict[str, Any]:
    """
    Run a fair pairwise comparison between two models.

    EVERYTHING is identical except the planning model f_θ.

    Args:
        model_a_path: Path to first model checkpoint
        model_b_path: Path to second model checkpoint
        weather_truth_path: Path to ground truth weather
        weather_forecast_path: Path to forecast weather
        initial_state: (nx,) initial plant state
        simulation_steps: Number of steps
        seed: Random seed
        label_a: Name of model A
        label_b: Name of model B
        verbose: Print progress
        **nmpc_kwargs: Additional NMPC settings

    Returns:
        Dict with results for both models and comparison metrics
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    if verbose:
        print(f"\n{'#'*70}")
        print(f"PAIRWISE COMPARISON: {label_a} vs {label_b}")
        print(f"Seed: {seed}")
        print(f"{'#'*70}")

    # ── Run A ──
    if verbose:
        print(f"\n--- Running {label_a} ---")
    torch.manual_seed(seed)
    sim_a = ClosedLoopNMPCSimulator(
        nn_model_path=model_a_path,
        weather_truth_path=weather_truth_path,
        weather_forecast_path=weather_forecast_path,
        verbose_init=verbose,
        **nmpc_kwargs,
    )
    results_a = sim_a.simulate_episode(
        initial_state=initial_state.copy(),
        simulation_steps=simulation_steps,
        verbose=verbose,
    )

    # ── Run B ──
    if verbose:
        print(f"\n--- Running {label_b} ---")
    torch.manual_seed(seed)
    sim_b = ClosedLoopNMPCSimulator(
        nn_model_path=model_b_path,
        weather_truth_path=weather_truth_path,
        weather_forecast_path=weather_forecast_path,
        verbose_init=verbose,
        **nmpc_kwargs,
    )
    results_b = sim_b.simulate_episode(
        initial_state=initial_state.copy(),
        simulation_steps=simulation_steps,
        verbose=verbose,
    )

    # ── Comparison ──
    comparison_metrics = [
        'cvar90_cold_occ_C', 'cvar95_cold_occ_C',
        'peak_cold_violation_occ_C',
        'deg_hours_cold_occ',
        'hours_cold_outside_occ', 'hours_cold_outside_occ_025',
        'total_energy_kWh', 'total_energy_cost',
        'total_stage_cost_ramc',
        'cvar90_stage_cost_occ',
    ]

    if verbose:
        print(f"\n{'='*70}")
        print(f"COMPARISON: {label_a} vs {label_b}")
        print(f"{'='*70}")
        print(f"{'Metric':<35s} {'Model A':>12s} {'Model B':>12s} {'Δ(B-A)':>12s}")
        print(f"{'-'*71}")

        for m in comparison_metrics:
            va = results_a.get(m, float('nan'))
            vb = results_b.get(m, float('nan'))
            delta = vb - va
            print(f"  {m:<33s} {va:>12.4f} {vb:>12.4f} {delta:>+12.4f}")

        print(f"{'='*70}\n")

    return {
        'label_a': label_a,
        'label_b': label_b,
        'seed': seed,
        'results_a': results_a,
        'results_b': results_b,
        'comparison_metrics': {
            m: {
                'a': results_a.get(m, float('nan')),
                'b': results_b.get(m, float('nan')),
                'delta': results_b.get(m, float('nan')) - results_a.get(m, float('nan')),
            }
            for m in comparison_metrics
        },
    }


# =============================================================================
# Validation runner (Checklists A-E)
# =============================================================================

def run_validation_checklist(
    model_path: Union[str, Path],
    weather_truth_path: Union[str, Path],
    weather_forecast_path: Optional[Union[str, Path]] = None,
    dtype: torch.dtype = torch.float64,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run the full validation checklist (A–D) before trusting results.

    Checklist A: One-step consistency
    Checklist B: Multi-step rollout consistency
    Checklist C: Gradient sanity (autograd vs finite diff)
    Checklist D: Optimization sanity (loss decreases)

    Args:
        model_path: Path to NN checkpoint
        weather_truth_path: Path to weather truth CSV for test data
        dtype: Tensor dtype
        verbose: Print results

    Returns:
        Dict with pass/fail for each check
    """
    if verbose:
        print(f"\n{'='*70}")
        print("VALIDATION CHECKLIST (A–D)")
        print(f"{'='*70}")

    model = load_ramc_model(model_path, device='cpu', dtype=dtype, verbose=verbose)
    model.eval()

    # Test state/control/disturbance
    x0 = np.array([21.0, 19.0, 20.5, 45.0, 42.0, 40.0])
    u0 = np.array([50.0, 0.5])
    d0 = np.array([-5.0, 5000.0, 30000.0])

    results = {}

    # ── Checklist A ──
    if verbose:
        print(f"\n--- Checklist A: One-step consistency ---")
    check_a = validate_one_step_consistency(model, x0, u0, d0, dtype=dtype)
    results['A_one_step'] = check_a
    if verbose:
        print(f"  Match: {check_a['match']} (max error: {check_a['max_abs_error']:.2e})")

    # ── Checklist B ──
    if verbose:
        print(f"\n--- Checklist B: Multi-step rollout consistency ---")
    H_test = 12
    U_test = np.tile(u0, (H_test, 1))
    D_test = np.tile(d0, (H_test, 1))
    check_b = validate_rollout_consistency(model, x0, U_test, D_test, dtype=dtype)
    results['B_rollout'] = check_b
    if verbose:
        print(f"  Match: {check_b['match']} (max error: {check_b['max_abs_error']:.2e})")

    # ── Checklist C ──
    if verbose:
        print(f"\n--- Checklist C: Gradient sanity ---")
    weights = {"energy": 0.9, "cold": 63.0, "hot": 30.0, "du": 1e-3,
               "terminal": 20.0, "trust": 0.0}
    u_scale = np.array([T_SUPPLY_MAX - T_SUPPLY_MIN, MDOT_MAX - MDOT_MIN])
    occ_test = np.ones(H_test)
    T_low_test = np.full(H_test, 20.0)
    T_high_test = np.full(H_test, 22.0)

    check_c = validate_gradient_sanity(
        model, x0, U_test, D_test, occ_test, T_low_test, T_high_test,
        u0, weights, u_scale, var_idx=0, dtype=dtype,
    )
    results['C_gradient'] = check_c
    if verbose:
        print(f"  Autograd: {check_c['grad_autograd']:.6f}")
        print(f"  Finite diff: {check_c['grad_fd']:.6f}")
        print(f"  Rel error: {check_c['relative_error']:.4f}")
        print(f"  Directionally consistent: {check_c['directionally_consistent']}")

    # ── Checklist D ──
    if verbose:
        print(f"\n--- Checklist D: Optimization sanity ---")
    H_opt = 24
    U_opt = np.tile(u0, (H_opt, 1))
    D_opt = np.tile(d0, (H_opt, 1))
    occ_opt = np.ones(H_opt)
    T_low_opt = np.full(H_opt, 20.0)
    T_high_opt = np.full(H_opt, 22.0)

    check_d = validate_optimization_sanity(
        model, x0, D_opt, occ_opt, T_low_opt, T_high_opt, u0,
        weights, dtype=dtype, H=H_opt, block_size=4, n_iter=20,
    )
    results['D_optimization'] = check_d
    if verbose:
        print(f"  Loss decreased: {check_d['loss_decreased']}")
        print(f"  Initial: {check_d['loss_initial']:.4f}")
        print(f"  Final: {check_d['loss_final']:.4f}")
        print(f"  Best: {check_d['best_loss']:.4f}")

    # ── Summary ──
    all_pass = (
        check_a['match']
        and check_b['match']
        and check_c['directionally_consistent']
        and check_d['loss_decreased']
    )

    if verbose:
        print(f"\n{'='*70}")
        status = "ALL PASSED" if all_pass else "SOME FAILED"
        print(f"VALIDATION RESULT: {status}")
        print(f"{'='*70}\n")

    results['all_pass'] = all_pass
    return results


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("Closed-Loop NMPC Simulator")
    print(f"Plant: RC ground truth (6-state, adaptive substeps)")
    print(f"Controller: Direct-shooting NMPC with projected Adam")
    print(f"Comfort: [{OCC_TARGET_C - DEADBAND_C}, {OCC_TARGET_C + DEADBAND_C}]°C")
    print(f"Energy rate: {ENERGY_COST_RATE} SEK/kWh")


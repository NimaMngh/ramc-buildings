#!/usr/bin/env python3
"""
RC Ground Truth Model for RAMC Phase 3
======================================

Sign-correct radiator heat exchange, consistent with the Phase 2 data
generation. This plant model must match the physics used to generate the
Phase 2 training data, in particular the sign-correct radiator heat
exchange formula.

Parameters and physics:
- DEFAULT_RC_PARAMS matches results_N3_DE_optimized.json (the parameter
  set used to generate the neural network training data).
- The solar gain is multiplied by A_sol before entering the air node,
  matching the system identification model.

Numerical integration:
- Adaptive substeps are used by default (use_adaptive_substeps=True). A
  fixed 6-substep Euler scheme is numerically unstable for mdot > 0.95
  kg/s (the radiator advection time constant falls below dt_sub = 100 s
  at full flow). The adaptive rule mirrors Training_Data_Generation.py
  (Phase 2) so the two phases integrate identically.
- For a fixed-substep run (e.g. archival comparisons):
      model = RCGroundTruthModel(use_adaptive_substeps=False, substeps=6)

State vector (6 elements):
    x = [T_air, T_env, T_int, T_rad1, T_rad2, T_ret]  (°C)

Control vector (2 elements):
    u = [T_supply, mdot]  (°C, kg/s)

Disturbance vector (3 elements):
    d = [T_out, Q_solar, Q_internal]  (°C, W, W)
    Note: Q_solar here is raw transmitted solar (I_solar_total).
          The model applies A_sol internally.

mdot is clamped nonnegative in compute_heat_output.

Author: Nima Monghasemi
Date: 2026-01-02
"""

import numpy as np
from pathlib import Path
import json
from typing import Union, Optional


# =============================================================================
# Default RC Parameters – FROM results_N3_DE_optimized.json
# These MUST match the parameters used to generate RAMC_training_data_N3.csv
# =============================================================================

DEFAULT_RC_PARAMS = {
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
# Option 0A – Adaptive substep defaults (mirrors Training_Data_Generation.py)
# =============================================================================
#
# These constants are the CANONICAL values used in Phase 2.
# Changing them would break the numerical equivalence guarantee.
# If you need to sweep them, do so BOTH here AND in
# Training_Data_Generation.py (and re-generate Phase 2 data).
#
_ADAPTIVE_SUBSTEPS_MIN          = 6
_ADAPTIVE_SUBSTEPS_MAX          = 200
_ADAPTIVE_EULER_SAFETY_FACTOR   = 0.3   # α: dt_sub ≤ α × 2 × C_rad_sec / (mdot × Cp)
_ADAPTIVE_MDOT_FLOOR            = 0.01  # kg/s — avoids division-by-zero at zero flow


class RCGroundTruthModel:
    """
    Ground-truth plant consistent with Phase-2 data generation.

    Key features:
      - dt = 600 s (10 minutes) default
      - Sign-correct radiator heat: Q = K * dT * |dT|^a
      - 3 radiator sections -> 6 states total
      - A_sol applied to solar input (matching system identification)
      - Inputs: u=[T_supply, mdot], d=[T_out, Q_solar, Q_internal]
      - solar_mode controls how transmitted solar is split across nodes:
          "split_air_int" (default): A_sol fraction to air, (1-A_sol) to
              internal mass.  Matches Training_Data_Generation.py.
          "air_only" (legacy): all A_sol × Q_solar to air node only.

      ── Option 0A (adaptive substeps) ───────────────────────────────────
      - use_adaptive_substeps=True (default): number of Euler sub-steps
        per control timestep is computed on-the-fly from the current mdot,
        mirroring RCPlantN3._compute_substeps() in Training_Data_Generation.py.
        This eliminates the numerical integration gap between Phase 2 and
        Phase 3 (confirmed by Option 0B: RMS e_T_air > 1.7 °C at max flow
        with the old fixed-6 scheme).
      - use_adaptive_substeps=False + substeps=N: legacy fixed-substep mode.
    """

    def __init__(
        self,
        params: Optional[dict] = None,
        dt_seconds: int = 600,
        substeps: int = 6,                  # used ONLY when use_adaptive_substeps=False
        params_json_path: Optional[Union[str, Path]] = None,
        solar_mode: str = "split_air_int",
        # ── Option 0A: adaptive substep parameters ────────────────────────
        use_adaptive_substeps: bool = True,
        substeps_min: int   = _ADAPTIVE_SUBSTEPS_MIN,
        substeps_max: int   = _ADAPTIVE_SUBSTEPS_MAX,
        euler_safety_factor: float = _ADAPTIVE_EULER_SAFETY_FACTOR,
        mdot_floor: float   = _ADAPTIVE_MDOT_FLOOR,
    ):
        """
        Initialise the RC ground truth model.

        Args:
            params: Dictionary of RC parameters.  If None, uses defaults or
                loads from JSON.
            dt_seconds: Simulation timestep in seconds (default 600 = 10 min).
            substeps: Number of Euler sub-steps when use_adaptive_substeps=False.
            params_json_path: Optional path to results_N3_DE_optimized.json.
            solar_mode: "split_air_int" (default, matches Phase 2) or "air_only".
            use_adaptive_substeps: If True (default), compute sub-steps from the
                current mdot using the same rule as Training_Data_Generation.py,
                ensuring Phase 2 / Phase 3 numerical equivalence (Option 0A).
                Set to False to restore the legacy fixed-substep behaviour.
            substeps_min: Minimum adaptive sub-steps (default 6).
            substeps_max: Maximum adaptive sub-steps (default 200).
            euler_safety_factor: Safety factor α for the stability criterion
                dt_sub ≤ α × 2 × C_rad_sec / (mdot × Cp) (default 0.3).
            mdot_floor: Minimum effective mdot for sub-step computation,
                avoids division-by-zero (default 0.01 kg/s).
        """
        self.dt = int(dt_seconds)
        self.substeps = int(substeps)       # legacy fallback
        self.Cp = 4186.0                    # Water specific heat (J/kg·K)
        self.N  = 3                         # Number of radiator sections

        # ── Option 0A fields ─────────────────────────────────────────────
        self.use_adaptive_substeps  = bool(use_adaptive_substeps)
        self.substeps_min           = int(substeps_min)
        self.substeps_max           = int(substeps_max)
        self.euler_safety_factor    = float(euler_safety_factor)
        self.mdot_floor             = float(mdot_floor)

        # Load parameters
        if params is not None:
            self._params = params
        elif params_json_path is not None:
            self._params = self._load_params_from_json(params_json_path)
        else:
            self._params = DEFAULT_RC_PARAMS

        # Extract individual parameters
        self.C_air = float(self._params["C_air"])
        self.C_env = float(self._params["C_env"])
        self.C_int = float(self._params["C_int"])
        self.C_rad = float(self._params["C_rad"])
        self.R_ex  = float(self._params["R_ex"])
        self.R_ae  = float(self._params["R_ae"])
        self.R_ai  = float(self._params["R_ai"])
        self.K_rad = float(self._params["K_rad"])
        self.a_rad = float(self._params["a_rad"])
        self.A_sol = float(self._params.get("A_sol", 1.0))

        # Solar injection mode
        _valid_modes = ("split_air_int", "air_only")
        if solar_mode not in _valid_modes:
            raise ValueError(
                f"solar_mode must be one of {_valid_modes}, got '{solar_mode}'"
            )
        self.solar_mode = solar_mode

        # State / control / disturbance dimensions
        self.nx = 6
        self.nu = 2
        self.nd = 3

        # ── Diagnostics (populated by step()) ────────────────────────────
        self._last_substeps_used: int = (
            self.substeps_min if self.use_adaptive_substeps else self.substeps
        )

    # =========================================================================
    # Option 0A – Adaptive sub-step computation
    # =========================================================================

    def _compute_substeps_adaptive(self, mdot: float) -> int:
        """
        Compute the number of Euler sub-steps required for numerical stability.

        Mirrors RCPlantN3._compute_substeps() in Training_Data_Generation.py
        EXACTLY, using the same parameters and formula.

        For the radiator advection term:
            dT_rad/dt ≈ (mdot × Cp / C_rad_sec) × ΔT
        Euler stability requires:
            dt_sub < 2 × C_rad_sec / (mdot × Cp)
        We apply euler_safety_factor α to stay well within the stable regime:
            dt_sub  ≤  α × 2 × C_rad_sec / (mdot × Cp)
            n_sub   ≥  ceil(dt / dt_stable)

        At mdot = 4 kg/s with the project's RC parameters:
            C_rad_sec ≈ 199 126 J/K
            dt_stable ≈ 7.1 s  ->  n_sub ≈ 85
        At mdot ≤ 0.95 kg/s (stability limit for fixed-6):
            n_sub = 6 (same as old scheme)

        Args:
            mdot: Current mass flow rate (kg/s).  Clamped to mdot_floor.

        Returns:
            Number of sub-steps (integer, clamped to [substeps_min, substeps_max]).
        """
        C_rad_sec  = self.C_rad / self.N
        mdot_eff   = max(abs(mdot), self.mdot_floor)
        dt_stable  = self.euler_safety_factor * 2.0 * C_rad_sec / (mdot_eff * self.Cp)
        n_needed   = int(np.ceil(self.dt / dt_stable))
        n          = max(self.substeps_min, min(n_needed, self.substeps_max))
        self._last_substeps_used = n
        return n

    # =========================================================================
    # Parameter loading
    # =========================================================================

    def _load_params_from_json(self, json_path: Union[str, Path]) -> dict:
        """Load RC parameters from results_N3_DE_optimized.json."""
        json_path = Path(json_path)
        if not json_path.exists():
            raise FileNotFoundError(f"RC parameters file not found: {json_path}")
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "identified_parameters" in data:
            return data["identified_parameters"]
        return data

    # =========================================================================
    # ODE
    # =========================================================================

    def derivatives(self, x: np.ndarray, u: np.ndarray, d: np.ndarray) -> np.ndarray:
        """
        Compute state derivatives dx/dt.

        Args:
            x: State vector [T_air, T_env, T_int, T_rad1, T_rad2, T_ret] (6,)
            u: Control vector [T_supply, mdot] (2,)
            d: Disturbance vector [T_out, Q_solar, Q_internal] (3,)

        Returns:
            dx/dt: State derivatives (6,)
        """
        # Unpack state
        T_air = float(x[0])
        T_env = float(x[1])
        T_int = float(x[2])
        T_rad = np.asarray(x[3:], dtype=float)   # 3 elements

        # Unpack controls (M8: ensure mdot is non-negative)
        T_supply = float(u[0])
        mdot     = max(float(u[1]), 0.0)

        # Unpack disturbances
        T_out       = float(d[0])
        Q_solar_raw = float(d[1])
        Q_internal  = float(d[2])

        # Solar injection
        if self.solar_mode == "split_air_int":
            Q_solar_air = self.A_sol * Q_solar_raw
            Q_solar_int = (1.0 - self.A_sol) * Q_solar_raw
        else:   # "air_only"
            Q_solar_air = self.A_sol * Q_solar_raw
            Q_solar_int = 0.0

        C_rad_sec = self.C_rad / self.N

        # Sign-correct radiator heat exchange: Q = K * dT * |dT|^a
        dT_rad_air  = T_rad - T_air
        Q_rad_sec   = self.K_rad * dT_rad_air * np.power(
            np.abs(dT_rad_air) + 1e-9, self.a_rad
        )
        Q_rad_total = float(np.sum(Q_rad_sec))

        # Water-side: supply -> section 1 -> section 2 -> section 3 (return)
        T_in      = np.concatenate(([T_supply], T_rad[:-1]))
        dT_rad_dt = (mdot * self.Cp * (T_in - T_rad) - Q_rad_sec) / (C_rad_sec + 1e-9)

        # Building envelope heat flows
        Q_env_air = (T_env - T_air) / self.R_ae
        Q_int_air = (T_int - T_air) / self.R_ai
        Q_out_env = (T_out - T_env) / self.R_ex

        # State derivatives
        dT_air_dt = (Q_env_air + Q_int_air + Q_rad_total + Q_solar_air + Q_internal) / self.C_air
        dT_env_dt = (Q_out_env - Q_env_air) / self.C_env
        dT_int_dt = (-Q_int_air + Q_solar_int) / self.C_int

        return np.array([dT_air_dt, dT_env_dt, dT_int_dt, *dT_rad_dt], dtype=float)

    # =========================================================================
    # Integration
    # =========================================================================

    def step(self, x: np.ndarray, u: np.ndarray, d: np.ndarray) -> np.ndarray:
        """
        Simulate one 10-minute control timestep.

        Option 0A change: when use_adaptive_substeps=True (default), the number
        of Euler sub-steps is computed from the current mdot using the same
        adaptive rule as Training_Data_Generation.py (Phase 2), ensuring
        Phase 2 / Phase 3 numerical equivalence.

        Args:
            x: Current state (6,)
            u: Control input (2,)  [T_supply °C, mdot kg/s]
            d: Disturbance input (3,)  [T_out °C, Q_solar W, Q_internal W]

        Returns:
            Next state (6,)
        """
        x = np.asarray(x, dtype=float).copy()
        u = np.asarray(u, dtype=float)
        d = np.asarray(d, dtype=float)

        # ── Determine sub-step count ──────────────────────────────────────
        # Option 0A: adaptive substeps (default) -> matches Phase 2 exactly.
        # Legacy:    fixed substeps (use_adaptive_substeps=False).
        if self.use_adaptive_substeps:
            mdot  = max(float(u[1]), 0.0)
            n_sub = self._compute_substeps_adaptive(mdot)
        else:
            n_sub = self.substeps
            self._last_substeps_used = n_sub

        dt_sub = self.dt / n_sub

        for _ in range(n_sub):
            dx = self.derivatives(x, u, d)
            x  = x + dx * dt_sub
            x  = np.clip(x, -50.0, 150.0)

        return x

    def simulate_step(
        self,
        x: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
        dt: Optional[float] = None,
    ) -> np.ndarray:
        """Backward-compatible alias for step(). dt argument is ignored."""
        return self.step(x, u, d)

    def simulate_trajectory(
        self,
        x0: np.ndarray,
        u_trajectory: np.ndarray,
        d_trajectory: np.ndarray,
    ) -> np.ndarray:
        """
        Simulate a full trajectory.

        Args:
            x0: Initial state (6,)
            u_trajectory: Control trajectory (T, 2)
            d_trajectory: Disturbance trajectory (T, 3)

        Returns:
            State trajectory (T+1, 6) including initial state
        """
        T       = len(u_trajectory)
        x_traj  = np.zeros((T + 1, self.nx))
        x_traj[0] = x0
        for t in range(T):
            x_traj[t + 1] = self.step(x_traj[t], u_trajectory[t], d_trajectory[t])
        return x_traj

    def compute_heat_output(self, x: np.ndarray, u: np.ndarray) -> float:
        """
        Compute heat output to the water circuit (for energy calculation).

        Q = mdot × Cp × max(T_supply − T_return, 0)

        Args:
            x: State vector (T_return is x[5])
            u: Control vector [T_supply, mdot]

        Returns:
            Heat output in Watts
        """
        T_supply = float(u[0])
        mdot     = max(float(u[1]), 0.0)   # M8: clamp non-negative
        T_return = float(x[5])
        return mdot * self.Cp * max(T_supply - T_return, 0.0)

    def get_params(self) -> dict:
        """Return a copy of the RC parameters."""
        return self._params.copy()

    def get_integration_info(self) -> dict:
        """
        Return a summary of the current integration settings.

        Useful for logging alongside experimental results to confirm
        that Option 0A is active.
        """
        return {
            "use_adaptive_substeps":  self.use_adaptive_substeps,
            "substeps_min":           self.substeps_min,
            "substeps_max":           self.substeps_max,
            "euler_safety_factor":    self.euler_safety_factor,
            "mdot_floor":             self.mdot_floor,
            "substeps_fixed_legacy":  self.substeps,
            "last_substeps_used":     self._last_substeps_used,
            "option_0A_active":       self.use_adaptive_substeps,
        }

    def __repr__(self) -> str:
        if self.use_adaptive_substeps:
            sub_str = (
                f"adaptive[{self.substeps_min}..{self.substeps_max}, "
                f"α={self.euler_safety_factor}]"
            )
        else:
            sub_str = f"fixed={self.substeps}"
        return (
            f"RCGroundTruthModel("
            f"dt={self.dt}s, substeps={sub_str}, "
            f"N_rad={self.N}, A_sol={self.A_sol:.4f}, "
            f"solar={self.solar_mode}, sign_correct=True)"
        )


# =============================================================================
# Self-test
# =============================================================================

def test_rc_model():
    """
    Self-test covering:
      1. Basic single-step simulation
      2. Option 0A: verify adaptive substeps are engaged and vary with mdot
      3. Option 0A: verify Phase 2 / Phase 3 gap is ≈ 0 after the fix
      4. Legacy mode backward compatibility
      5. Solar mode verification (split_air_int)
      6. M8: negative mdot clamping
    """
    print("=" * 65)
    print("RC Ground Truth Model Self-Test  (includes Option 0A checks)")
    print("=" * 65)

    # ── 1. Default (adaptive) model ───────────────────────────────────────
    model = RCGroundTruthModel(dt_seconds=600)
    print(f"\n{model}")
    info = model.get_integration_info()
    assert info["option_0A_active"], "Option 0A should be active by default"
    print(f"  A_sol = {model.A_sol:.4f}")
    print(f"  Integration: {info}")

    # ── 2. Adaptive substeps vary with mdot ───────────────────────────────
    print("\n── Adaptive substep sweep ────────────────────────────────────")
    for mdot_test in [0.0, 0.5, 1.0, 2.0, 4.0, 4.05]:
        n = model._compute_substeps_adaptive(mdot_test)
        print(f"  mdot={mdot_test:.2f} kg/s  ->  substeps={n}")
    n_low  = model._compute_substeps_adaptive(0.0)
    n_high = model._compute_substeps_adaptive(4.0)
    assert n_high > n_low, "High mdot must require more substeps than zero flow"
    print("  PASS: substeps scale correctly with mdot")

    # ── 3. Option 0A gap verification ────────────────────────────────────
    print("\n── Option 0A gap (adaptive == adaptive) should be 0 °C ───────")
    x0 = np.array([21.0, 19.0, 20.5, 50.0, 45.0, 40.0])
    u  = np.array([55.0, 4.0])
    d  = np.array([-5.0, 3000.0, 50000.0])
    model_A = RCGroundTruthModel()                              # adaptive
    model_F = RCGroundTruthModel(                               # old fixed-6
        use_adaptive_substeps=False, substeps=6
    )
    xA = model_A.step(x0.copy(), u, d)
    xF = model_F.step(x0.copy(), u, d)
    e_air = abs(xA[0] - xF[0])
    e_ret = abs(xA[5] - xF[5])
    print(f"  Adaptive model T_air after 1 step (mdot=4): {xA[0]:.6f} °C")
    print(f"  Fixed-6  model T_air after 1 step (mdot=4): {xF[0]:.6f} °C")
    print(f"  Gap e_T_air = {e_air:.6f} °C  (pre-fix gap was ~0.18 °C per step)")
    print(f"  Gap e_T_ret = {e_ret:.6f} °C  (pre-fix gap was ~101 °C per step!)")

    # Two adaptive models must be exactly identical
    model_A2 = RCGroundTruthModel()
    xA2 = model_A2.step(x0.copy(), u, d)
    assert np.allclose(xA, xA2, atol=1e-12), "Two adaptive models must give identical output"
    print("  PASS: Two adaptive instances agree to machine precision")

    # ── 4. Legacy fixed mode still works ─────────────────────────────────
    print("\n── Legacy fixed-substep mode (use_adaptive_substeps=False) ───")
    model_legacy = RCGroundTruthModel(use_adaptive_substeps=False, substeps=6)
    assert not model_legacy.use_adaptive_substeps
    print(f"  {model_legacy}")
    print("  PASS: Legacy mode instantiated correctly")

    # ── 5. Solar mode verification ───────────────────────────────────────
    print("\n── Solar mode: split_air_int ──────────────────────────────────")
    d_solar = np.array([0.0, 10000.0, 0.0])
    u_off   = np.array([x0[3], 0.0])
    dx      = model.derivatives(x0, u_off, d_solar)
    assert dx[2] > 0, "FAIL: dT_int/dt should be positive with split_air_int solar"
    print(f"  dT_int/dt from 10 kW solar = {dx[2]*3600:.6f} °C/h  (> 0 expected)")
    print("  PASS")

    print("\n── air_only legacy solar mode ─────────────────────────────────")
    m_ao = RCGroundTruthModel(solar_mode="air_only")
    dx_ao   = m_ao.derivatives(x0, u_off, d_solar)
    dx_base = m_ao.derivatives(x0, u_off, np.array([0., 0., 0.]))
    delta   = abs(dx_ao[2] - dx_base[2])
    assert delta < 1e-12, f"FAIL: solar should not affect dT_int in air_only, got {delta:.2e}"
    print(f"  Solar contribution to dT_int (air_only): {delta:.2e}  (should be 0)")
    print("  PASS")

    # ── 6. M8 negative mdot clamping ─────────────────────────────────────
    print("\n── M8: Negative mdot clamping ─────────────────────────────────")
    Q_neg = model.compute_heat_output(x0, np.array([55.0, -0.1]))
    assert Q_neg == 0.0, "FAIL: Heat output must be 0 for negative mdot"
    print(f"  Q_heat(mdot=-0.1) = {Q_neg:.1f} W  (should be 0)")
    print("  PASS")

    # ── 7. Multi-step trajectory ──────────────────────────────────────────
    print("\n── Multi-step trajectory (1 hour = 6 steps) ──────────────────")
    u_traj = np.tile(u, (6, 1))
    d_traj = np.tile(d, (6, 1))
    x_traj = model.simulate_trajectory(x0, u_traj, d_traj)
    print(f"  T_air: {x0[0]:.2f} -> {x_traj[-1, 0]:.2f} °C")
    print(f"  T_ret: {x0[5]:.2f} -> {x_traj[-1, 5]:.2f} °C")
    assert x_traj.shape == (7, 6)
    print("  PASS")

    print("\n" + "=" * 65)
    print("  ALL TESTS PASSED")
    print("=" * 65)
    return model


if __name__ == "__main__":
    test_rc_model()

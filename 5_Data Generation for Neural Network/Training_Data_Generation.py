# -*- coding: utf-8 -*-
"""
Training_Data_Generation.py (N=3)

Key features:
1) Adaptive substeps based on mass flow rate to ensure Euler stability
2) Solar semantics: I_solar_total treated as Q_solar_transmitted_W [W]
3) Split-solar model: Q_solar_air = A_sol * Q_trans, Q_solar_int = (1-A_sol) * Q_trans
4) Control bounds aligned to the dataset statistics

Numerical stability:
- At high mdot (~4 kg/s), the radiator advection term has time constant ~12s
- A fixed 6 substeps (dt_sub=100s) gives a ~22% clip rate (unstable)
- Adaptive substeps compute the required substeps per step based on mdot
- Target: clip rate < 0.1%
"""

import json
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

# =============================================================================
# Configuration
# =============================================================================

# ---- Paths ----
RC_PARAMS_PATH = "results_N3_DE_optimized.json"
PROCESSED_CSV_PATH = "ASHRAE901_RetailStandalone_STD2022GBaseline_Vasteras_processed.csv"

OUTPUT_CSV_PATH = "RAMC_training_data_N3.csv"
OUTPUT_META_PATH = "RAMC_training_data_N3_meta.json"

# ---- Time ----
DT_SECONDS = 600  # 10 minutes
STEPS_PER_DAY = int(24 * 3600 / DT_SECONDS)

# ---- Reproducibility ----
RANDOM_SEED = 42

# ---- Dataset size ----
EPISODE_DAYS = 7
N_EPISODES = 300  # 300 * 7 * 144 = 302,400 samples

# If the processed CSV is already a winter slice, keep False.
FILTER_MONTHS = False
MONTHS_KEEP = {10, 11, 12, 1, 2, 3, 4}

# ---- Datetime parsing ----
DATETIME_DAYFIRST = False  # set True if the DateTime strings are day-first

# ---- Columns expected in processed CSV ----
COL_TIME = "DateTime"
COL_T_OUT = "T_outdoor"
COL_Q_INT = "Q_internal_no_solar"

# IMPORTANT: this is actually a heat rate [W] from E+:
# "Enclosure Windows Total Transmitted Solar Radiation Rate" [W]
COL_Q_SOL_TRANS = "I_solar_total"

# Optional (debug / alignment):
COL_T_SUPPLY_IDF = "T_supply_avg"
COL_MDOT_IDF = "mdot_water_total"

# Optional other columns that may be present; not required here:
COL_Q_SOL_TOTAL_ALT = "Q_solar_total"     # if present, we can compare in meta
COL_Q_HEAT_TOTAL = "Q_heating_total"      # not required

USE_T_SUPPLY_BASE_FROM_CSV_IF_AVAILABLE = True

# ---- Comfort bounds (used for Tmin/Tmax labels; not NN inputs) ----
OCC_TARGET_C = 22.0
UNOCC_TARGET_C = 15.56
DEADBAND_C = 0.5

# Occupancy mode:
#   "schedule" -> fixed weekly schedule
#   "internal_gains" -> infer occupancy from Q_internal_no_solar threshold
OCCUPANCY_MODE = "internal_gains"
Q_INT_OCC_THRESHOLD_W: Optional[float] = None

# ---- Control bounds based on the dataset statistics ----
T_SUPPLY_MIN_C_DEFAULT = 31.620013
T_SUPPLY_MAX_C_DEFAULT = 60.0

MDOT_MIN_KG_S_DEFAULT = 0.0
MDOT_MAX_KG_S_DEFAULT = 4.05  # small buffer above 4.031382

# ---- Actuator rate limits (per 10-minute step) ----
DT_SUPPLY_MAX_PER_STEP_C_DEFAULT = 4.0
DMDOT_MAX_PER_STEP_KG_S_DEFAULT = 0.8

# ---- RC simulation numerics ----
# Adaptive substep configuration
USE_ADAPTIVE_SUBSTEPS = True
SUBSTEPS_FIXED = 6              # Used only if USE_ADAPTIVE_SUBSTEPS = False
SUBSTEPS_MIN = 6                # Minimum substeps (used when mdot is very low/zero)
SUBSTEPS_MAX = 200              # Cap to prevent excessive computation
EULER_SAFETY_FACTOR = 0.3       # α in Δt_sub ≤ α * C_rad_sec / (mdot * cp)
MDOT_FLOOR_FOR_SUBSTEPS = 0.01  # Avoid division by zero; below this, use SUBSTEPS_MIN

STATE_CLIP_MIN_C = -50.0
STATE_CLIP_MAX_C = 150.0

CP_WATER = 4186.0  # J/kg/K

# Episode-level disturbance scaling (keep modest)
Q_INT_SCALE_RANGE = (0.95, 1.05)
Q_SOL_TRANS_SCALE_RANGE = (0.90, 1.10)

# ---- Solar injection configuration ----
# "air_only"      -> old behavior: Q_solar_air = A_sol * Q_trans; Q_solar_int = 0
# "split_air_int" -> suggested behavior: Q_solar_air = A_sol * Q_trans; Q_solar_int = (1-A_sol) * Q_trans
SOLAR_INJECTION_MODE = "split_air_int"

# If True, clip A_sol into [0,1] when using split mode (keeps it a valid fraction).
# With the identified A_sol=0.55985, this does nothing either way.
ENFORCE_A_SOL_FRACTION_BOUNDS = False

# =============================================================================
# Helpers
# =============================================================================

def summarize_series(x: pd.Series) -> Dict[str, float]:
    x = pd.to_numeric(x, errors="coerce").dropna()
    if len(x) == 0:
        return {"min": float("nan"), "max": float("nan"), "q99": float("nan")}
    return {
        "min": float(x.min()),
        "max": float(x.max()),
        "q99": float(x.quantile(0.99)),
    }


def schedule_occupied(ts: pd.Timestamp) -> bool:
    # Reasonable retail schedule; replace with the actual IDF schedule if desired.
    wd = ts.weekday()  # Mon=0
    h = ts.hour
    if wd < 5:      # Mon-Fri
        return (h >= 7) and (h < 21)
    elif wd == 5:   # Sat
        return (h >= 7) and (h < 22)
    else:           # Sun
        return (h >= 9) and (h < 19)


def comfort_bounds(
    ts: pd.Timestamp,
    q_internal_W: Optional[float] = None,
    q_int_threshold_W: Optional[float] = None
) -> Tuple[float, float, bool]:
    if OCCUPANCY_MODE == "internal_gains":
        if (q_internal_W is None) or (q_int_threshold_W is None):
            occ = schedule_occupied(ts)
        else:
            occ = bool(q_internal_W >= q_int_threshold_W)
    else:
        occ = schedule_occupied(ts)

    target = OCC_TARGET_C if occ else UNOCC_TARGET_C
    return float(target - DEADBAND_C), float(target + DEADBAND_C), occ


def reset_curve_T_supply(T_out_C: float) -> float:
    """
    Outdoor reset curve:
      -20C -> 60C
       20C -> 19C
    In the winter range (-26..8C), this yields roughly ~31..60C and saturates near 60C in cold weather.
    """
    T_low_out, T_high_out = -20.0, 20.0
    T_at_low, T_at_high = 60.0, 19.0

    frac = (T_high_out - T_out_C) / (T_high_out - T_low_out)
    T = T_at_high + (T_at_low - T_at_high) * frac
    return float(np.clip(T, T_at_high, T_at_low))


def clip_rate_limited(x: float, x_prev: float, dx_max: float, lo: float, hi: float) -> float:
    x_rl = np.clip(x, x_prev - dx_max, x_prev + dx_max)
    return float(np.clip(x_rl, lo, hi))


# =============================================================================
# RC Plant Simulator (N=3) with Adaptive Substeps
# =============================================================================

class RCPlantN3:
    """
    States: [T_air, T_env, T_int, T_rad1, T_rad2, T_ret]
    Controls:
      T_supply [C], mdot [kg/s]
    Disturbances:
      T_out [C],
      Q_internal [W] (no solar),
      Q_solar_trans [W] (transmitted solar through windows; EnergyPlus output)

    Solar injection modes:
      - air_only:
          Q_solar_air = A_sol * Q_solar_trans
          Q_solar_int = 0
      - split_air_int (recommended):
          Q_solar_air = A_sol * Q_solar_trans
          Q_solar_int = (1 - A_sol) * Q_solar_trans

    Numerical stability:
      - Uses adaptive substeps based on mass flow rate
      - At high mdot, more substeps are used to keep Euler stable
      - Stability condition: dt_sub < 2 * C_rad_sec / (mdot * cp)
      - We use safety factor α (typically 0.3) to stay well within stable regime
    """

    def __init__(
        self,
        params: Dict[str, float],
        dt_seconds: float,
        solar_mode: str = "split_air_int",
        enforce_a_sol_fraction_bounds: bool = False,
        # Adaptive substep parameters
        use_adaptive_substeps: bool = True,
        substeps_fixed: int = 6,
        substeps_min: int = 6,
        substeps_max: int = 200,
        euler_safety_factor: float = 0.3,
        mdot_floor: float = 0.01,
    ):
        self.dt = float(dt_seconds)
        
        # Adaptive substep settings
        self.use_adaptive_substeps = bool(use_adaptive_substeps)
        self.substeps_fixed = int(substeps_fixed)
        self.substeps_min = int(substeps_min)
        self.substeps_max = int(substeps_max)
        self.euler_safety_factor = float(euler_safety_factor)
        self.mdot_floor = float(mdot_floor)

        # RC parameters
        self.C_air = float(params["C_air"])
        self.C_env = float(params["C_env"])
        self.C_int = float(params["C_int"])
        self.C_rad = float(params["C_rad"])

        self.R_ex = float(params["R_ex"])
        self.R_ae = float(params["R_ae"])
        self.R_ai = float(params["R_ai"])

        self.K_rad = float(params["K_rad"])
        self.a_rad = float(params["a_rad"])

        self.A_sol_raw = float(params.get("A_sol", 1.0))
        self.solar_mode = str(solar_mode).strip().lower()
        self.enforce_a_sol_fraction_bounds = bool(enforce_a_sol_fraction_bounds)

        self.Cp_water = CP_WATER
        self.N = 3
        
        # Precompute for adaptive substeps
        self.C_rad_section = self.C_rad / self.N

        # Default initial state
        self.state = np.array([21.0, 21.0, 21.0, 45.0, 43.0, 41.0], dtype=float)
        
        # Diagnostics
        self.substep_clip_events = 0
        self.total_substeps_used = 0
        self.max_substeps_used = 0
        self.min_substeps_used = self.substeps_max  # Will be updated

        # One-time warnings
        self._warned_a_sol = False
        self._warned_solar_mode = False

    def _compute_substeps(self, mdot: float) -> int:
        """
        Compute required substeps for numerical stability.
        
        For the advection term: dT_rad/dt ~ (mdot * cp / C_rad_sec) * ΔT
        Stability requires: dt_sub < 2 * C_rad_sec / (mdot * cp)
        We use a safety factor (typically 0.2-0.5) to stay well within stable regime.
        
        With these parameters:
          C_rad_section ≈ 199,126 J/K
          At mdot = 4 kg/s: k ≈ 0.084 s⁻¹, dt_stable < 24s
          With safety factor 0.3: dt_sub < 7.2s -> need ~84 substeps for 600s
        """
        if not self.use_adaptive_substeps:
            return self.substeps_fixed
        
        mdot_eff = max(abs(mdot), self.mdot_floor)
        
        # Maximum stable substep size
        # dt_stable = 2 * C_rad_sec / (mdot * cp) is the theoretical limit
        # We multiply by safety factor to stay well within stable regime
        dt_stable = self.euler_safety_factor * 2.0 * self.C_rad_section / (mdot_eff * self.Cp_water)
        
        # Required number of substeps
        substeps_needed = int(np.ceil(self.dt / dt_stable))
        
        # Clamp to reasonable range
        substeps = max(self.substeps_min, min(substeps_needed, self.substeps_max))
        
        return substeps

    def reset_state(self, x0: np.ndarray) -> None:
        x0 = np.array(x0, dtype=float).reshape(-1)
        if x0.shape[0] != 6:
            raise ValueError("x0 must have 6 elements for N=3 plant.")
        self.state = x0.copy()
    
    def reset_diagnostics(self) -> None:
        """Reset diagnostic counters (call at start of each episode if desired)."""
        self.substep_clip_events = 0
        self.total_substeps_used = 0
        self.max_substeps_used = 0
        self.min_substeps_used = self.substeps_max

    def _effective_a_sol(self) -> float:
        A = self.A_sol_raw
        if self.solar_mode == "split_air_int":
            if self.enforce_a_sol_fraction_bounds:
                A_eff = float(np.clip(A, 0.0, 1.0))
            else:
                A_eff = float(A)
            if (not self._warned_a_sol) and (A_eff < 0.0 or A_eff > 1.0):
                print(
                    f"WARNING: A_sol={A_eff:.4f} outside [0,1] while using split_air_int mode. "
                    "This makes (1-A_sol) negative and breaks the fraction interpretation."
                )
                self._warned_a_sol = True
            return A_eff

        if self.solar_mode == "air_only":
            # Here A_sol is just a gain; could be >1 in principle, but warn anyway.
            if (not self._warned_a_sol) and (A < 0.0 or A > 1.5):
                print(
                    f"WARNING: A_sol={A:.4f} is outside typical range (0..1.5) for air_only mode."
                )
                self._warned_a_sol = True
            return float(A)

        if not self._warned_solar_mode:
            print(f"WARNING: Unknown solar_mode='{self.solar_mode}'. Falling back to 'air_only'.")
            self._warned_solar_mode = True
        self.solar_mode = "air_only"
        return float(A)

    def _radiator_heat_sections(self, T_air: float, T_rad_sections: np.ndarray) -> np.ndarray:
        """
        Radiator heat exchange per section [W].

        Uses a signed form:
          Q = K_rad * (T_rad - T_air) * |T_rad - T_air|^a

        This allows radiators to cool toward air when flow is off (more physical/stable).
        """
        dT = (T_rad_sections - T_air)
        return self.K_rad * dT * np.power(np.abs(dT) + 1e-9, self.a_rad)

    def _solar_split(self, Q_solar_trans_W: float) -> Tuple[float, float]:
        """
        Returns (Q_solar_air_W, Q_solar_int_W).
        """
        A = self._effective_a_sol()
        if self.solar_mode == "air_only":
            return float(A * Q_solar_trans_W), 0.0

        # split_air_int
        Q_air = float(A * Q_solar_trans_W)
        Q_int = float((1.0 - A) * Q_solar_trans_W)
        return Q_air, Q_int

    def get_derivatives(
        self,
        state: np.ndarray,
        T_supply: float,
        mdot: float,
        T_out: float,
        Q_internal: float,
        Q_solar_trans_W: float,
    ) -> np.ndarray:
        T_air, T_env, T_int = state[0], state[1], state[2]
        T_rad = state[3:]  # 3 sections

        C_rad_section = self.C_rad_section

        # Radiator heat to zone air
        Q_rad_sections = self._radiator_heat_sections(T_air, T_rad)
        Q_rad_total = float(np.sum(Q_rad_sections))

        # Water propagation through sections
        T_rad_in = np.concatenate(([T_supply], T_rad[:-1]))
        dT_rad_dt = (mdot * self.Cp_water * (T_rad_in - T_rad) - Q_rad_sections) / (C_rad_section + 1e-9)

        # Building network heat flows [W]
        Q_ea = (T_env - T_air) / (self.R_ae + 1e-12)  # envelope -> air
        Q_ia = (T_int - T_air) / (self.R_ai + 1e-12)  # internal mass -> air
        Q_oe = (T_out - T_env) / (self.R_ex + 1e-12)  # outdoor -> envelope

        # Solar split (both are in W)
        Q_solar_air, Q_solar_int = self._solar_split(Q_solar_trans_W)

        # State derivatives
        dT_air_dt = (Q_ea + Q_ia + Q_rad_total + Q_internal + Q_solar_air) / (self.C_air + 1e-12)
        dT_env_dt = (Q_oe - Q_ea) / (self.C_env + 1e-12)

        # Internal mass receives part of solar in split mode
        dT_int_dt = (-Q_ia + Q_solar_int) / (self.C_int + 1e-12)

        return np.concatenate(([dT_air_dt], [dT_env_dt], [dT_int_dt], dT_rad_dt))

    def step(
        self,
        T_supply: float,
        mdot: float,
        T_out: float,
        Q_internal: float,
        Q_solar_trans_W: float
    ) -> np.ndarray:
        # Compute adaptive substeps based on current flow rate
        substeps = self._compute_substeps(mdot)
        
        # Update diagnostics
        self.total_substeps_used += substeps
        self.max_substeps_used = max(self.max_substeps_used, substeps)
        self.min_substeps_used = min(self.min_substeps_used, substeps)
        
        dt_sub = self.dt / substeps
        for _ in range(substeps):
            dx = self.get_derivatives(self.state, T_supply, mdot, T_out, Q_internal, Q_solar_trans_W)
            unclipped = self.state + dx * dt_sub
            self.state = np.clip(unclipped, STATE_CLIP_MIN_C, STATE_CLIP_MAX_C)
            if not np.array_equal(unclipped, self.state):
                self.substep_clip_events += 1
        return self.state.copy()


# =============================================================================
# Control policy with the dataset-derived bounds
# =============================================================================

@dataclass
class ControlBounds:
    T_supply_min_C: float
    T_supply_max_C: float
    mdot_min_kg_s: float
    mdot_max_kg_s: float
    dT_supply_max_per_step_C: float
    dmdot_max_per_step_kg_s: float


class MixedPolicyController:
    """
    Generates realistic controls:
    - T_supply: track IDF baseline (from CSV if available, else reset curve) + small exploration
    - mdot: PI-like on temperature error with shutoff and safety clamps
    """

    def __init__(self, rng: np.random.Generator, policy_name: str, bounds: ControlBounds):
        self.rng = rng
        self.policy_name = policy_name
        self.b = bounds

        # Episode-level randomization
        self.supply_offset_C = float(rng.uniform(-1.5, 3.0))
        self.supply_noise_rho = float(rng.uniform(0.85, 0.97))
        self.supply_noise_std_C = float(rng.uniform(0.1, 0.6))
        self._supply_noise_state = 0.0

        # PI gains (scaled for mdot up to ~4 kg/s)
        self.kp = float(rng.uniform(0.6, 1.6))     # kg/s per degC
        self.ki = float(rng.uniform(0.03, 0.12))   # kg/s per (degC * step)
        self._int_e = 0.0

        # Pulses for excitation
        self.pulse_prob_per_day = float(rng.uniform(0.2, 1.0))
        self.pulse_mag_C = float(rng.uniform(2.0, 6.0))
        self.pulse_steps_remaining = 0
        self.pulse_delta_C = 0.0

        # Random-walk exploration policy
        self.walk_T_std = float(rng.uniform(0.2, 1.2))
        self.walk_m_std = float(rng.uniform(0.05, 0.45))

        self.T_supply_prev = float(np.mean([self.b.T_supply_min_C, self.b.T_supply_max_C]))
        self.mdot_prev = 0.0

    def reset(self, T_supply_init: float, mdot_init: float) -> None:
        self.T_supply_prev = float(T_supply_init)
        self.mdot_prev = float(mdot_init)
        self._int_e = 0.0
        self._supply_noise_state = 0.0
        self.pulse_steps_remaining = 0
        self.pulse_delta_C = 0.0

    def _maybe_start_pulse(self) -> None:
        if self.pulse_steps_remaining > 0:
            return
        p_step = self.pulse_prob_per_day / STEPS_PER_DAY
        if self.rng.random() < p_step:
            duration_steps = int(self.rng.integers(low=6, high=24))  # 1h to 4h
            sign = 1.0 if self.rng.random() < 0.5 else -1.0
            self.pulse_delta_C = sign * self.pulse_mag_C
            self.pulse_steps_remaining = duration_steps

    def compute_u(
        self,
        state: np.ndarray,
        T_out: float,
        Tmin: float,
        Tmax: float,
        T_supply_idf_base: Optional[float] = None
    ) -> Tuple[float, float]:
        T_air = float(state[0])

        # ---- T_supply baseline ----
        if T_supply_idf_base is not None:
            T_base = float(T_supply_idf_base)
        else:
            T_base = reset_curve_T_supply(T_out)

        # Exploration noise (AR1)
        eps = self.rng.normal(0.0, self.supply_noise_std_C)
        self._supply_noise_state = self.supply_noise_rho * self._supply_noise_state + eps

        if self.policy_name == "idf_base":
            T_cmd = T_base
        elif self.policy_name == "idf_base_explore":
            T_cmd = T_base + self.supply_offset_C + self._supply_noise_state
        elif self.policy_name == "idf_base_pulses":
            self._maybe_start_pulse()
            pulse = 0.0
            if self.pulse_steps_remaining > 0:
                pulse = self.pulse_delta_C
                self.pulse_steps_remaining -= 1
            T_cmd = T_base + self.supply_offset_C + self._supply_noise_state + pulse
        elif self.policy_name == "random_walk":
            T_cmd = self.T_supply_prev + self.rng.normal(0.0, self.walk_T_std)
        else:
            raise ValueError(f"Unknown policy_name: {self.policy_name}")

        # Rate-limit + clip
        T_supply = clip_rate_limited(
            x=T_cmd,
            x_prev=self.T_supply_prev,
            dx_max=self.b.dT_supply_max_per_step_C,
            lo=self.b.T_supply_min_C,
            hi=self.b.T_supply_max_C
        )

        # ---- mdot PI on temperature error ----
        T_sp = 0.5 * (Tmin + Tmax)

        if T_air > Tmax:
            self._int_e = 0.0
            mdot_cmd = 0.0
        else:
            e = max(0.0, T_sp - T_air)
            self._int_e = 0.98 * self._int_e + e
            mdot_cmd = self.kp * e + self.ki * self._int_e

        if self.policy_name == "random_walk":
            mdot_cmd = self.mdot_prev + self.rng.normal(0.0, self.walk_m_std)

        # Safety overrides (heating-only)
        if T_air < (Tmin - 1.5):
            T_supply = self.b.T_supply_max_C
            mdot_cmd = self.b.mdot_max_kg_s
        if T_air > (Tmax + 1.5):
            mdot_cmd = 0.0

        # Heating-only coupling safeguard
        if T_supply <= (T_air + 0.5):
            mdot_cmd = 0.0

        mdot = clip_rate_limited(
            x=mdot_cmd,
            x_prev=self.mdot_prev,
            dx_max=self.b.dmdot_max_per_step_kg_s,
            lo=self.b.mdot_min_kg_s,
            hi=self.b.mdot_max_kg_s
        )

        self.T_supply_prev = float(T_supply)
        self.mdot_prev = float(mdot)

        return float(T_supply), float(mdot)


# =============================================================================
# Data loading
# =============================================================================

def load_rc_params(path: str) -> Dict[str, float]:
    with open(path, "r") as f:
        data = json.load(f)
    params = data.get("identified_parameters", data)

    required = ["C_air", "C_env", "C_int", "C_rad", "R_ex", "R_ae", "R_ai", "K_rad", "a_rad"]
    missing = [k for k in required if k not in params]
    if missing:
        raise ValueError(f"RC params missing keys: {missing}")

    if "A_sol" not in params:
        print("WARNING: A_sol not found in params; defaulting A_sol=1.0")
        params["A_sol"] = 1.0

    return {k: float(v) for k, v in params.items()}


def load_processed_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    if COL_TIME not in df.columns:
        raise ValueError(f"Processed CSV must contain '{COL_TIME}' column.")

    df[COL_TIME] = pd.to_datetime(df[COL_TIME], errors="coerce", dayfirst=DATETIME_DAYFIRST)
    df = df.dropna(subset=[COL_TIME]).sort_values(COL_TIME).reset_index(drop=True)

    required = [COL_T_OUT, COL_Q_INT, COL_Q_SOL_TRANS]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Processed CSV missing required columns: {missing}")

    if FILTER_MONTHS:
        df = df[df[COL_TIME].dt.month.isin(MONTHS_KEEP)].reset_index(drop=True)

    # Numeric coercion
    for c in required + [COL_T_SUPPLY_IDF, COL_MDOT_IDF, COL_Q_SOL_TOTAL_ALT, COL_Q_HEAT_TOTAL]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=required).reset_index(drop=True)

    keep = [COL_TIME, COL_T_OUT, COL_Q_INT, COL_Q_SOL_TRANS]
    for opt in [COL_T_SUPPLY_IDF, COL_MDOT_IDF, COL_Q_SOL_TOTAL_ALT, COL_Q_HEAT_TOTAL]:
        if opt in df.columns:
            keep.append(opt)

    return df[keep].copy()


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    rng = np.random.default_rng(RANDOM_SEED)

    print("Loading RC parameters...")
    rc_params = load_rc_params(RC_PARAMS_PATH)

    print("Loading processed EnergyPlus CSV...")
    df_lib = load_processed_csv(PROCESSED_CSV_PATH)

    # Basic dt sanity check
    if len(df_lib) >= 3:
        deltas = df_lib[COL_TIME].diff().dropna().dt.total_seconds()
        med = float(deltas.median())
        if abs(med - DT_SECONDS) > 1e-6:
            print(
                f"WARNING: Median timestep in CSV is {med:.1f}s, but DT_SECONDS={DT_SECONDS}. "
                "Data will still be generated assuming constant DT_SECONDS."
            )

    # Stats for meta
    stats = {
        COL_T_OUT: summarize_series(df_lib[COL_T_OUT]),
        COL_Q_INT: summarize_series(df_lib[COL_Q_INT]),
        COL_Q_SOL_TRANS: summarize_series(df_lib[COL_Q_SOL_TRANS]),
    }
    for opt in [COL_T_SUPPLY_IDF, COL_MDOT_IDF, COL_Q_SOL_TOTAL_ALT, COL_Q_HEAT_TOTAL]:
        if opt in df_lib.columns:
            stats[opt] = summarize_series(df_lib[opt])

    # Optional check: if Q_solar_total exists, compare it to I_solar_total
    solar_consistency = None
    if (COL_Q_SOL_TOTAL_ALT in df_lib.columns) and (COL_Q_SOL_TRANS in df_lib.columns):
        a = df_lib[COL_Q_SOL_TRANS].astype(float)
        b = df_lib[COL_Q_SOL_TOTAL_ALT].astype(float)
        diff = (a - b).abs()
        solar_consistency = {
            "mean_abs_diff_W": float(diff.mean()),
            "q99_abs_diff_W": float(diff.quantile(0.99)),
            "note": "If these diffs are ~0, the two columns are effectively duplicates."
        }

    # Occupancy threshold from internal gains if desired
    q_int_threshold = None
    if OCCUPANCY_MODE == "internal_gains":
        if Q_INT_OCC_THRESHOLD_W is not None:
            q_int_threshold = float(Q_INT_OCC_THRESHOLD_W)
        else:
            q10 = float(df_lib[COL_Q_INT].quantile(0.10))
            q90 = float(df_lib[COL_Q_INT].quantile(0.90))
            q_int_threshold = q10 + 0.35 * (q90 - q10)

    # Control bounds
    bounds = ControlBounds(
        T_supply_min_C=float(T_SUPPLY_MIN_C_DEFAULT),
        T_supply_max_C=float(T_SUPPLY_MAX_C_DEFAULT),
        mdot_min_kg_s=float(MDOT_MIN_KG_S_DEFAULT),
        mdot_max_kg_s=float(MDOT_MAX_KG_S_DEFAULT),
        dT_supply_max_per_step_C=float(DT_SUPPLY_MAX_PER_STEP_C_DEFAULT),
        dmdot_max_per_step_kg_s=float(DMDOT_MAX_PER_STEP_KG_S_DEFAULT),
    )

    episode_len = EPISODE_DAYS * STEPS_PER_DAY
    max_start = len(df_lib) - episode_len - 1
    if max_start <= 0:
        raise ValueError("Not enough rows in processed CSV for requested episode length.")

    # Policy mix (mostly IDF-like)
    policy_names = ["idf_base_explore", "idf_base_pulses", "random_walk", "idf_base"]
    policy_probs = np.array([0.60, 0.25, 0.10, 0.05], dtype=float)
    policy_probs = policy_probs / policy_probs.sum()

    records = []

    # Global diagnostics
    plant_clip_events_total = 0
    plant_substeps_total = 0
    plant_max_substeps_global = 0
    plant_min_substeps_global = SUBSTEPS_MAX

    print(f"Generating: {N_EPISODES} episodes x {EPISODE_DAYS} days (dt={DT_SECONDS}s)")
    print(f"Solar injection mode: {SOLAR_INJECTION_MODE} (A_sol={rc_params.get('A_sol', None)})")
    print(f"Adaptive substeps: {USE_ADAPTIVE_SUBSTEPS} (min={SUBSTEPS_MIN}, max={SUBSTEPS_MAX}, safety={EULER_SAFETY_FACTOR})")

    for ep in tqdm(range(N_EPISODES), desc="Episodes"):
        start_idx = int(rng.integers(low=0, high=max_start))
        ep_df = df_lib.iloc[start_idx:start_idx + episode_len].reset_index(drop=True)

        # Episode-level scaling
        q_int_scale = float(rng.uniform(*Q_INT_SCALE_RANGE))
        q_sol_scale = float(rng.uniform(*Q_SOL_TRANS_SCALE_RANGE))

        plant = RCPlantN3(
            rc_params,
            DT_SECONDS,
            solar_mode=SOLAR_INJECTION_MODE,
            enforce_a_sol_fraction_bounds=ENFORCE_A_SOL_FRACTION_BOUNDS,
            use_adaptive_substeps=USE_ADAPTIVE_SUBSTEPS,
            substeps_fixed=SUBSTEPS_FIXED,
            substeps_min=SUBSTEPS_MIN,
            substeps_max=SUBSTEPS_MAX,
            euler_safety_factor=EULER_SAFETY_FACTOR,
            mdot_floor=MDOT_FLOOR_FOR_SUBSTEPS,
        )

        # Initial state near start comfort midpoint
        ts0 = ep_df.loc[0, COL_TIME]
        q0 = float(ep_df.loc[0, COL_Q_INT]) * q_int_scale
        Tmin0, Tmax0, _ = comfort_bounds(ts0, q_internal_W=q0, q_int_threshold_W=q_int_threshold)
        T0 = 0.5 * (Tmin0 + Tmax0) + float(rng.normal(0.0, 0.5))

        Trad_mean = T0 + float(rng.uniform(5.0, 20.0))
        x0 = np.array([
            T0,
            T0 + float(rng.normal(0.0, 0.3)),
            T0 + float(rng.normal(0.0, 0.3)),
            Trad_mean + 2.0,
            Trad_mean + 0.0,
            Trad_mean - 2.0,
        ], dtype=float)
        plant.reset_state(x0)

        policy = str(rng.choice(policy_names, p=policy_probs))
        ctrl = MixedPolicyController(rng, policy, bounds)

        # Initialize controller memory
        T_out0 = float(ep_df.loc[0, COL_T_OUT])
        if USE_T_SUPPLY_BASE_FROM_CSV_IF_AVAILABLE and (COL_T_SUPPLY_IDF in ep_df.columns):
            T_supply_init = float(ep_df.loc[0, COL_T_SUPPLY_IDF])
        else:
            T_supply_init = reset_curve_T_supply(T_out0)

        T_supply_init = float(np.clip(T_supply_init, bounds.T_supply_min_C, bounds.T_supply_max_C))
        ctrl.reset(T_supply_init, 0.0)

        for k in range(episode_len):
            ts = ep_df.loc[k, COL_TIME]

            T_out = float(ep_df.loc[k, COL_T_OUT])
            Q_int = float(ep_df.loc[k, COL_Q_INT]) * q_int_scale

            # Solar transmitted through windows [W]
            Q_solar_trans = float(ep_df.loc[k, COL_Q_SOL_TRANS]) * q_sol_scale

            # Physical guards
            Q_int = max(0.0, Q_int)
            Q_solar_trans = max(0.0, Q_solar_trans)

            Tmin, Tmax, occ = comfort_bounds(ts, q_internal_W=Q_int, q_int_threshold_W=q_int_threshold)

            T_supply_idf_base = None
            if USE_T_SUPPLY_BASE_FROM_CSV_IF_AVAILABLE and (COL_T_SUPPLY_IDF in ep_df.columns):
                T_supply_idf_base = float(ep_df.loc[k, COL_T_SUPPLY_IDF])
                if np.isfinite(T_supply_idf_base):
                    T_supply_idf_base = float(np.clip(T_supply_idf_base, bounds.T_supply_min_C, bounds.T_supply_max_C))
                else:
                    T_supply_idf_base = None

            state_k = plant.state.copy()
            T_supply, mdot = ctrl.compute_u(
                state=state_k,
                T_out=T_out,
                Tmin=Tmin,
                Tmax=Tmax,
                T_supply_idf_base=T_supply_idf_base
            )

            T_ret_k = float(state_k[5])

            # Derived solar split terms (debug/aux)
            Q_solar_air_k, Q_solar_int_k = plant._solar_split(Q_solar_trans)

            # Heating proxy (useful as an RL cost surrogate / debug)
            Q_heat_proxy_k = float(max(0.0, mdot) * CP_WATER * max(0.0, T_supply - T_ret_k))

            state_k1 = plant.step(T_supply, mdot, T_out, Q_int, Q_solar_trans)

            rec = {
                "Timestamp": ts,
                "episode_id": ep,
                "step_in_episode": k,
                "policy": policy,

                "Occupied": int(occ),
                "Tmin": float(Tmin),
                "Tmax": float(Tmax),

                # State k
                "T_air_k": float(state_k[0]),
                "T_env_k": float(state_k[1]),
                "T_int_k": float(state_k[2]),
                "T_rad1_k": float(state_k[3]),
                "T_rad2_k": float(state_k[4]),
                "T_ret_k": float(state_k[5]),

                # Controls
                "T_supply_k": float(T_supply),
                "mdot_k": float(mdot),

                # Disturbances
                "T_out_k": float(T_out),
                "Q_internal_k": float(Q_int),

                # Legacy name kept for backward compatibility:
                "I_solar_k": float(Q_solar_trans),

                # Clear name (same value as I_solar_k):
                "Q_solar_trans_k": float(Q_solar_trans),

                # Optional IDF baseline supply (debug)
                "T_supply_idf_base_k": float(T_supply_idf_base) if T_supply_idf_base is not None else np.nan,

                # Helpful derived terms
                "Q_solar_air_k": float(Q_solar_air_k),
                "Q_solar_int_k": float(Q_solar_int_k),
                "Q_heat_proxy_k": float(Q_heat_proxy_k),

                # State k+1
                "T_air_k1": float(state_k1[0]),
                "T_env_k1": float(state_k1[1]),
                "T_int_k1": float(state_k1[2]),
                "T_rad1_k1": float(state_k1[3]),
                "T_rad2_k1": float(state_k1[4]),
                "T_ret_k1": float(state_k1[5]),
            }
            records.append(rec)

        # Accumulate episode diagnostics
        plant_clip_events_total += int(plant.substep_clip_events)
        plant_substeps_total += int(plant.total_substeps_used)
        plant_max_substeps_global = max(plant_max_substeps_global, plant.max_substeps_used)
        if plant.min_substeps_used < plant.substeps_max:  # Only update if actually used
            plant_min_substeps_global = min(plant_min_substeps_global, plant.min_substeps_used)

    df_out = pd.DataFrame(records)
    df_out.to_csv(OUTPUT_CSV_PATH, index=False)

    clip_pct = 100.0 * plant_clip_events_total / max(1, plant_substeps_total)
    avg_substeps = float(plant_substeps_total) / max(1, N_EPISODES * episode_len)

    meta = {
        "created_utc": pd.Timestamp.utcnow().isoformat(),
        "dt_seconds": DT_SECONDS,
        "episode_days": EPISODE_DAYS,
        "n_episodes": N_EPISODES,
        "n_samples": int(len(df_out)),
        "source_csv": PROCESSED_CSV_PATH,
        "solar_injection": {
            "mode": SOLAR_INJECTION_MODE,
            "enforce_A_sol_in_0_1": ENFORCE_A_SOL_FRACTION_BOUNDS,
            "note": (
                "In split_air_int mode: Q_solar_air = A_sol * Q_trans, "
                "Q_solar_int = (1 - A_sol) * Q_trans."
            )
        },
        "csv_signal_semantics": {
            COL_Q_SOL_TRANS: {
                "meaning": "Enclosure Windows Total Transmitted Solar Radiation Rate",
                "units": "W",
                "note": "This is transmitted solar heat rate through windows (not irradiance)."
            },
            "A_sol": {
                "meaning": (
                    "dimensionless solar split parameter. "
                    "In split_air_int mode it is interpreted as the fraction of transmitted solar "
                    "that goes directly to the air node."
                ),
                "units": "-",
            }
        },
        "processed_csv_stats": stats,
        "solar_column_consistency_check": solar_consistency,
        "occupancy": {
            "mode": OCCUPANCY_MODE,
            "q_internal_threshold_W": float(q_int_threshold) if q_int_threshold is not None else None,
            "occ_target_C": OCC_TARGET_C,
            "unocc_target_C": UNOCC_TARGET_C,
            "deadband_C": DEADBAND_C,
        },
        "disturbance_scaling": {
            "Q_int_scale_range": list(Q_INT_SCALE_RANGE),
            "Q_solar_trans_scale_range": list(Q_SOL_TRANS_SCALE_RANGE),
        },
        "control_bounds": {
            "T_supply_min_C": bounds.T_supply_min_C,
            "T_supply_max_C": bounds.T_supply_max_C,
            "mdot_min_kg_s": bounds.mdot_min_kg_s,
            "mdot_max_kg_s": bounds.mdot_max_kg_s,
            "dT_supply_max_per_step_C": bounds.dT_supply_max_per_step_C,
            "dmdot_max_per_step_kg_s": bounds.dmdot_max_per_step_kg_s,
        },
        "plant": {
            "N_sections": 3,
            "n_states": 6,
            "integration": {
                "method": "explicit_euler",
                "use_adaptive_substeps": USE_ADAPTIVE_SUBSTEPS,
                "substeps_fixed": SUBSTEPS_FIXED,
                "substeps_min": SUBSTEPS_MIN,
                "substeps_max": SUBSTEPS_MAX,
                "euler_safety_factor": EULER_SAFETY_FACTOR,
                "mdot_floor_for_substeps": MDOT_FLOOR_FOR_SUBSTEPS,
            },
            "state_clip_min_C": STATE_CLIP_MIN_C,
            "state_clip_max_C": STATE_CLIP_MAX_C,
            "diagnostics": {
                "substep_clip_events_total": int(plant_clip_events_total),
                "substep_clip_rate_pct": float(clip_pct),
                "total_substeps": int(plant_substeps_total),
                "avg_substeps_per_step": float(avg_substeps),
                "max_substeps_observed": int(plant_max_substeps_global),
                "min_substeps_observed": int(plant_min_substeps_global),
            },
        },
        "rc_params_path": RC_PARAMS_PATH,
        "rc_params_used": rc_params,
        "schema": {
            # Canonical feature set (recommended)
            "features_xk_canonical": [
                "T_air_k", "T_env_k", "T_int_k", "T_rad1_k", "T_rad2_k", "T_ret_k",
                "T_supply_k", "mdot_k",
                "T_out_k", "Q_internal_k", "Q_solar_trans_k"
            ],
            # Legacy feature set (if downstream code expects I_solar_k)
            "features_xk_legacy": [
                "T_air_k", "T_env_k", "T_int_k", "T_rad1_k", "T_rad2_k", "T_ret_k",
                "T_supply_k", "mdot_k",
                "T_out_k", "Q_internal_k", "I_solar_k"
            ],
            "labels_yk": [
                "T_air_k1", "T_env_k1", "T_int_k1", "T_rad1_k1", "T_rad2_k1", "T_ret_k1"
            ],
            "aux_columns": [
                "Timestamp", "episode_id", "step_in_episode", "policy",
                "Occupied", "Tmin", "Tmax",
                "T_supply_idf_base_k",
                "Q_solar_air_k", "Q_solar_int_k",
                "Q_heat_proxy_k"
            ],
            "aliases": {
                "I_solar_k": "Q_solar_trans_k"
            }
        }
    }

    with open(OUTPUT_META_PATH, "w") as f:
        json.dump(meta, f, indent=2)

    print("\nSaved:")
    print(f"  Dataset : {OUTPUT_CSV_PATH}")
    print(f"  Metadata: {OUTPUT_META_PATH}")
    print(f"Rows: {len(df_out)}")

    print("\nNumerical stability diagnostics:")
    print(f"  Total substeps executed: {plant_substeps_total:,}")
    print(f"  Avg substeps per step  : {avg_substeps:.1f}")
    print(f"  Min substeps observed  : {plant_min_substeps_global}")
    print(f"  Max substeps observed  : {plant_max_substeps_global}")
    print(f"  Clip events            : {plant_clip_events_total:,} ({clip_pct:.6f}%)")
    
    if clip_pct > 0.1:
        print(f"\n  WARNING: Clip rate {clip_pct:.4f}% exceeds 0.1% threshold.")
        print(f"           Consider increasing SUBSTEPS_MAX or reducing EULER_SAFETY_FACTOR.")
    else:
        print(f"\n  OK: Clip rate {clip_pct:.6f}% is within acceptable range (<0.1%).")

    print("\nPreview:")
    cols_preview = [
        "Timestamp", "episode_id", "policy",
        "T_air_k", "T_supply_k", "mdot_k",
        "T_out_k", "Q_internal_k", "Q_solar_trans_k",
        "Q_solar_air_k", "Q_solar_int_k",
        "T_air_k1"
    ]
    print(df_out[cols_preview].head())


if __name__ == "__main__":
    main()

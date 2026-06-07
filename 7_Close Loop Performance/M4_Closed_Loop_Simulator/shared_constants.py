# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 19:04:25 2026

@author: nmi03
"""

#!/usr/bin/env python3
"""
Shared constants and utility functions for RAMC Phase 3 closed-loop simulation.
===============================================================================

Extracted from closed_loop_simulator.py so that downstream modules (e.g.,
M6_NMPC_Direct_Shooting) can import occupancy logic, comfort bounds, and
stage cost functions WITHOUT pulling in the linearization/QP dependencies.

These functions and constants have ZERO dependency on:
  - linearize_nn (M2)
  - mpc_controller_hydronic (M3)

Author: Nima Monghasemi
Date: 2026-03-11 (extracted from closed_loop_simulator.py)
"""

import numpy as np
import pandas as pd

# =============================================================================
# Physical Constants
# =============================================================================

DT_SECONDS = 600
DT_MINUTES = 10
ND_DISTURBANCES = 3

# Fallback Q_internal values (used when CSV lacks Q_internal_W column)
Q_INTERNAL_OCCUPIED_W = 50000.0
Q_INTERNAL_UNOCCUPIED_W = 4000.0

# =============================================================================
# Comfort Setpoints — ALIGNED TO PHASE 2 TRAINING
# =============================================================================

OCC_TARGET_C = 21.0
UNOCC_TARGET_C = 15.56
DEADBAND_C = 1.0

# =============================================================================
# Actuator Limits
# =============================================================================

T_SUPPLY_MIN = 32.0
T_SUPPLY_MAX = 60.0
MDOT_MIN = 0.0
MDOT_MAX = 4.05

# =============================================================================
# Energy Cost Rate — MATCHED TO PHASE 2 TRAINING
# =============================================================================

ENERGY_COST_RATE = 0.9


# =============================================================================
# Occupancy Schedule
# =============================================================================

def get_occupancy_status(ts: pd.Timestamp) -> bool:
    """Check if timestamp is during occupied hours."""
    if not isinstance(ts, pd.Timestamp):
        ts = pd.to_datetime(ts)

    wd = ts.weekday()
    h = ts.hour

    if wd < 5:
        return (h >= 7) and (h < 21)
    elif wd == 5:
        return (h >= 7) and (h < 22)
    else:
        return (h >= 9) and (h < 19)


def get_comfort_bounds(occupied: bool) -> tuple:
    """Get comfort bounds based on occupancy."""
    if occupied:
        T_center = OCC_TARGET_C
    else:
        T_center = UNOCC_TARGET_C
    return (T_center - DEADBAND_C, T_center + DEADBAND_C)


# =============================================================================
# RAMC-Style Stage Cost
# =============================================================================

def _softplus_np(z: np.ndarray) -> np.ndarray:
    """Numerically stable softplus."""
    return np.log1p(np.exp(-np.abs(z))) + np.maximum(z, 0.0)


def stage_cost_ramc_np(
    T_air: np.ndarray,
    T_ret: np.ndarray,
    T_supply: np.ndarray,
    mdot: np.ndarray,
    Tmin: np.ndarray,
    Tmax: np.ndarray,
    *,
    comfort_beta: float = 0.5,
    dt_minutes: float = 10.0,
    energy_cost_rate: float = 0.9,
    Cp_water: float = 4186.0,
    w_comfort: float = 63.0,
    w_energy: float = 1.0,
    cost_scale: float = 1.0,
) -> tuple:
    """
    RAMC-style stage cost computation.

    Same functional form as Phase 2 training cost.

    Returns:
        (total_cost, comfort_cost, energy_cost, Q_W)
    """
    T_air = np.asarray(T_air, dtype=float)
    T_ret = np.asarray(T_ret, dtype=float)
    T_supply = np.asarray(T_supply, dtype=float)
    mdot = np.asarray(mdot, dtype=float)
    Tmin = np.asarray(Tmin, dtype=float)
    Tmax = np.asarray(Tmax, dtype=float)

    beta = float(comfort_beta)
    comfort = (beta * _softplus_np((Tmin - T_air) / beta) +
               beta * _softplus_np((T_air - Tmax) / beta))

    mdot_eff = np.maximum(mdot, 0.0)
    Q_W = mdot_eff * Cp_water * np.maximum(T_supply - T_ret, 0.0)

    dt_hours = float(dt_minutes) / 60.0
    energy_kWh = Q_W * dt_hours / 1000.0
    energy_cost = energy_kWh * float(energy_cost_rate)

    total = float(cost_scale) * (float(w_comfort) * comfort + float(w_energy) * energy_cost)

    return total, comfort, energy_cost, Q_W

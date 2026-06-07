#!/usr/bin/env python3
"""
Generate Phase 3 Weather Scenarios from Training Data
=====================================================

The training CSV contains 300 independent 7-day January episodes.
We select ONE episode as the nominal scenario and derive cold-snap
and forecast-error variants from it.

This guarantees all weather disturbance magnitudes (T_out, Q_solar) are
within the NN's trained range (except the intentional cold-snap perturbation).
The Q_internal forecast corruption may push Q_internal_W slightly beyond the
training envelope due to the 15% scaling and additive noise — this is
intentional realism, but is flagged in the verification summary.

An imperfect Q_internal forecast is applied to ALL scenarios. Internal-gains
uncertainty is always present in real buildings, regardless of weather, so
the Q_internal corruption is injected into every forecast CSV before saving.
The same corruption (same seed, shift, scale, and noise realization) is
applied to all three forecast CSVs, so every model receives the identical
corrupted forecast. The Q_internal mismatch is therefore constant across
scenarios — only the weather mismatch varies.

Because of this, the "nominal" scenario is not a perfect forecast: it is
nominal weather (no T_out/Q_solar mismatch) with a realistic internal-gains
forecast error. RAMC's advantage is expected to be smallest under nominal
weather and to grow as weather mismatch is added (cold snap / forecast error).

Experiment matrix: 4 models × 3 scenarios × 5 seeds = 60 experiments.

Output CSV columns: timestamp, T_out_C, Q_solar_W, Q_internal_W
  - T_out_C:      Outdoor temperature (°C)
  - Q_solar_W:    Transmitted solar I_solar (W) — A_sol applied by RC plant
  - Q_internal_W: Internal heat gains (W)

Author: Nima Monghasemi
Date: 2026-02-15
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple

# =============================================================================
# Configuration
# =============================================================================

TRAINING_CSV = Path(
    r"C:\Users\nmi03\OneDrive - Mälardalens universitet\Studying Folder"
    r"\New Project for Risk Aware Model then Control\6_Neural Network Architecture"
    r"\RAMC_training_data_N3.csv"
)

OUTPUT_DIR = Path(
    r"C:\Users\nmi03\OneDrive - Mälardalens universitet\Studying Folder"
    r"\New Project for Risk Aware Model then Control\7_Close Loop Performance"
    r"\M1_Weather_Data\data_ramc_epw"
)

# Episode selection: percentile of coldness (0=coldest, 100=warmest)
# We pick ~15th percentile: cold enough to stress the system,
# but not the absolute coldest (so cold snap still makes sense)
COLDNESS_PERCENTILE = 15

# Scenario parameters
STEPS_PER_WEEK = 1008
DT_SECONDS = 600

# Cold snap
COLD_SNAP_START_DAY = 2
COLD_SNAP_DURATION_HOURS = 48
COLD_SNAP_OFFSET_C = -10.0
COLD_SNAP_RAMP_HOURS = 3

# Forecast error
FORECAST_BIAS_C = 1.5
FORECAST_NOISE_STD_C = 1.0
FORECAST_AR_COEF = 0.9

# ─── Q_internal forecast error (applied to ALL scenarios) ────────────
# In reality, occupant-driven heat gains are largely unpredictable.
# This corruption model has three physically-motivated components:
#
# 1. Schedule timing shift: occupancy events happen earlier/later
#    than forecast (meetings run late, early closings, etc.)
# 2. Magnitude scaling: actual loads differ from predicted
#    (>1 = overestimate -> MPC expects free heating -> under-heats -> cold risk)
# 3. Additive AR(1) noise: fast variations from doors, equipment cycling

QINT_SHIFT_STEPS = 3                  # +30 min lag (forecast is late)
QINT_SCALE = 1.15                     # 15% overestimate
QINT_NOISE_FRAC = 0.10                # Noise std = 10% of mean occupied Q_internal
QINT_AR_COEF = 0.85                   # Temporal correlation of noise
QINT_SEED = 99                        # Fixed seed (independent of scenario seeds)


# =============================================================================
# Episode Selection
# =============================================================================

def select_episode(df: pd.DataFrame, percentile: float = COLDNESS_PERCENTILE) -> pd.DataFrame:
    """
    Select a single episode at the given coldness percentile.

    Args:
        df: Full training DataFrame
        percentile: 0=coldest, 100=warmest. 15 = fairly cold.

    Returns:
        DataFrame with 1008 rows for the selected episode
    """
    # Compute mean T_out per episode
    ep_stats = df.groupby('episode_id').agg(
        mean_T_out=('T_out_k', 'mean'),
        min_T_out=('T_out_k', 'min'),
        max_T_out=('T_out_k', 'max'),
        std_T_out=('T_out_k', 'std'),
        max_Isolar=('I_solar_k', 'max'),
        n_steps=('T_out_k', 'count'),
    ).reset_index()

    # Sort by mean temperature (coldest first)
    ep_stats = ep_stats.sort_values('mean_T_out').reset_index(drop=True)

    # Select episode at given percentile
    target_idx = int(len(ep_stats) * percentile / 100.0)
    target_idx = max(0, min(target_idx, len(ep_stats) - 1))

    selected = ep_stats.iloc[target_idx]
    ep_id = int(selected['episode_id'])

    print(f"\nEpisode selection (percentile={percentile}%):")
    print(f"  Selected episode_id: {ep_id}")
    print(f"  Mean T_out: {selected['mean_T_out']:.1f}°C")
    print(f"  T_out range: [{selected['min_T_out']:.1f}, {selected['max_T_out']:.1f}]°C")
    print(f"  T_out std: {selected['std_T_out']:.1f}°C")
    print(f"  Max I_solar: {selected['max_Isolar']:.0f} W")
    print(f"  Steps: {selected['n_steps']:.0f}")

    # Show context: surrounding episodes
    print(f"\n  Context (nearby episodes by coldness):")
    start = max(0, target_idx - 2)
    end = min(len(ep_stats), target_idx + 3)
    for i in range(start, end):
        row = ep_stats.iloc[i]
        marker = " <<<" if i == target_idx else ""
        print(f"    rank {i+1}/{len(ep_stats)}: ep={int(row['episode_id'])}, "
              f"mean_T={row['mean_T_out']:.1f}°C, "
              f"range=[{row['min_T_out']:.1f}, {row['max_T_out']:.1f}]{marker}")

    # Extract episode data
    ep_df = df[df['episode_id'] == ep_id].copy().reset_index(drop=True)

    assert len(ep_df) == STEPS_PER_WEEK, \
        f"Episode {ep_id} has {len(ep_df)} steps, expected {STEPS_PER_WEEK}"

    return ep_df


# =============================================================================
# Convert Episode to Scenario Format
# =============================================================================

def episode_to_weather(ep_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert a training episode to the weather CSV format.

    Maps:
      T_out_k      -> T_out_C
      I_solar_k    -> Q_solar_W  (same as Q_solar_trans_k, verified identical)
      Q_internal_k -> Q_internal_W
    """
    weather = pd.DataFrame({
        'timestamp': ep_df['Timestamp'],
        'T_out_C': ep_df['T_out_k'],
        'Q_solar_W': ep_df['I_solar_k'],        # = Q_solar_trans_k (verified)
        'Q_internal_W': ep_df['Q_internal_k'],
    })

    return weather


# =============================================================================
# Scenario Creation
# =============================================================================

def create_cold_snap(nominal: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Cold snap on truth only. Forecast = nominal (unseen by MPC)."""
    forecast = nominal.copy()
    truth = nominal.copy()

    steps_per_hour = 6
    snap_start = COLD_SNAP_START_DAY * (STEPS_PER_WEEK // 7)
    snap_duration = COLD_SNAP_DURATION_HOURS * steps_per_hour
    snap_end = min(snap_start + snap_duration, len(truth))
    ramp_steps = COLD_SNAP_RAMP_HOURS * steps_per_hour

    T_mod = truth['T_out_C'].values.copy()

    for i in range(len(truth)):
        if snap_start <= i < snap_end:
            T_mod[i] += COLD_SNAP_OFFSET_C
        elif snap_start - ramp_steps <= i < snap_start:
            progress = (i - (snap_start - ramp_steps)) / ramp_steps
            T_mod[i] += COLD_SNAP_OFFSET_C * progress
        elif snap_end <= i < snap_end + ramp_steps:
            progress = 1.0 - (i - snap_end) / ramp_steps
            T_mod[i] += COLD_SNAP_OFFSET_C * progress

    truth['T_out_C'] = T_mod

    print(f"\n  Cold snap applied:")
    print(f"    Steps {snap_start}-{snap_end} ({COLD_SNAP_DURATION_HOURS}h)")
    print(f"    Drop: {COLD_SNAP_OFFSET_C}°C")
    print(f"    Truth T_out: [{truth['T_out_C'].min():.1f}, {truth['T_out_C'].max():.1f}]°C")
    print(f"    Forecast T_out: [{forecast['T_out_C'].min():.1f}, "
          f"{forecast['T_out_C'].max():.1f}]°C (unchanged)")

    return truth, forecast


def create_forecast_error(
    nominal: pd.DataFrame, seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Truth = nominal. Forecast has warm bias + AR(1) noise on T_out."""
    truth = nominal.copy()
    forecast = nominal.copy()

    rng = np.random.RandomState(seed)
    n = len(forecast)

    noise = np.zeros(n)
    noise[0] = rng.normal(0, FORECAST_NOISE_STD_C)
    innovation_std = FORECAST_NOISE_STD_C * np.sqrt(1 - FORECAST_AR_COEF ** 2)
    for i in range(1, n):
        noise[i] = FORECAST_AR_COEF * noise[i - 1] + rng.normal(0, innovation_std)

    forecast['T_out_C'] = forecast['T_out_C'] + FORECAST_BIAS_C + noise

    print(f"\n  Forecast error applied:")
    print(f"    Bias: +{FORECAST_BIAS_C}°C, AR(1) ρ={FORECAST_AR_COEF}, "
          f"σ={FORECAST_NOISE_STD_C}°C")
    print(f"    Truth T_out: [{truth['T_out_C'].min():.1f}, "
          f"{truth['T_out_C'].max():.1f}]°C")
    print(f"    Forecast T_out: [{forecast['T_out_C'].min():.1f}, "
          f"{forecast['T_out_C'].max():.1f}]°C")

    return truth, forecast


# =============================================================================
# Q_internal Corruption (applied to ALL forecast CSVs)
# =============================================================================

def _shift_late_no_wrap(Q: np.ndarray, k: int) -> np.ndarray:
    """
    Shift Q forward by k steps (forecast lags truth) WITHOUT circular
    wrap-around.  The first k values are held at Q[0] (boundary pad).

    For k <= 0, returns an unmodified copy.

    This avoids the artefact where np.roll would inject end-of-week
    values into the beginning of the episode.
    """
    if k <= 0:
        return Q.copy()
    out = np.empty_like(Q)
    out[:k] = Q[0]       # pad: forecast hasn't "seen" the new week yet
    out[k:] = Q[:-k]     # lagged truth
    return out


def corrupt_q_internal(
    forecast: pd.DataFrame,
    truth_q_internal: np.ndarray,
    shift_steps: int = QINT_SHIFT_STEPS,
    scale: float = QINT_SCALE,
    noise_frac: float = QINT_NOISE_FRAC,
    ar_coef: float = QINT_AR_COEF,
    seed: int = QINT_SEED,
) -> pd.DataFrame:
    """
    Apply realistic Q_internal forecast error to a forecast DataFrame.

    This function is called on EVERY forecast CSV before saving, because
    internal gains uncertainty is always present regardless of weather.

    The corruption is computed relative to truth_q_internal (the nominal
    episode's Q_internal_W), NOT the forecast's current Q_internal_W.
    This ensures the same corruption pattern across all scenarios.

    Corruption model (with shift_steps = k > 0 meaning "forecast is late"):

        Q_shifted[t] = Q_truth[t - k]          (for t >= k)
        Q_shifted[t] = Q_truth[0]              (for t < k, boundary pad)
        Q_forecast[t] = clamp( scale * Q_shifted[t] + AR_noise[t],  0, inf )

    The forecast at time t reflects truth from k steps ago — i.e., the
    forecast lags behind reality. This is implemented via
    _shift_late_no_wrap() to avoid circular wrap-around artefacts at
    the week boundary.

    Why scale > 1 is the dangerous direction:
        Overestimate -> MPC expects more free heating from occupants ->
        MPC under-provisions supply heat -> cold violations (primary metric).
        This is where RAMC's risk-aware training should shine.

    Args:
        forecast:          Forecast DataFrame to modify (returned as copy)
        truth_q_internal:  Original (truth) Q_internal_W array from nominal episode
        shift_steps:       Lag in timesteps (+ve = forecast is late)
        scale:             Multiplicative factor (>1 = overestimate)
        noise_frac:        Noise std as fraction of mean occupied Q_internal
        ar_coef:           AR(1) coefficient for noise
        seed:              Random seed (SAME seed for all scenarios -> same corruption)

    Returns:
        Modified forecast DataFrame with corrupted Q_internal_W
    """
    forecast = forecast.copy()
    Q_truth = truth_q_internal.copy()
    n = len(Q_truth)

    # Validate length match
    assert len(forecast) == n, (
        f"Forecast length ({len(forecast)}) != truth Q_internal length ({n})"
    )

    # 1. Schedule timing shift (no wrap-around, boundary-padded)
    Q_shifted = _shift_late_no_wrap(Q_truth, shift_steps)

    # 2. Magnitude scaling
    Q_corrupted = scale * Q_shifted

    # 3. Additive AR(1) noise
    if noise_frac > 0:
        # Compute noise std from occupied-hours mean
        Q_occupied_mean = float(
            Q_truth[Q_truth > np.median(Q_truth)].mean()
        )
        noise_std_W = noise_frac * Q_occupied_mean

        rng = np.random.RandomState(seed)
        noise = np.zeros(n)
        innovation_std = noise_std_W * np.sqrt(1.0 - ar_coef ** 2)
        noise[0] = rng.normal(0, noise_std_W)
        for i in range(1, n):
            noise[i] = ar_coef * noise[i - 1] + rng.normal(0, innovation_std)
        Q_corrupted = Q_corrupted + noise

    # 4. Physical clamp: Q_internal >= 0
    forecast['Q_internal_W'] = np.maximum(Q_corrupted, 0.0)

    return forecast


def print_q_internal_corruption_summary(
    truth_q: np.ndarray,
    forecast_q: np.ndarray,
    scenario_name: str,
):
    """Print corruption statistics for verification."""
    err = forecast_q - truth_q
    occ_mask = truth_q > np.median(truth_q)
    err_occ = err[occ_mask]

    print(f"    Q_internal corruption ({scenario_name}):")
    print(f"      Truth range:    [{truth_q.min():.0f}, {truth_q.max():.0f}] W")
    print(f"      Forecast range: [{forecast_q.min():.0f}, {forecast_q.max():.0f}] W")
    print(f"      Error (all):      bias={err.mean():+.0f} W, "
          f"RMSE={np.sqrt((err**2).mean()):.0f} W")
    if occ_mask.any():
        print(f"      Error (occupied): bias={err_occ.mean():+.0f} W, "
              f"RMSE={np.sqrt((err_occ**2).mean()):.0f} W")


# =============================================================================
# Main
# =============================================================================

def generate_all_scenarios():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PHASE 3 WEATHER SCENARIOS FROM TRAINING DATA")
    print("(with Q_internal forecast corruption on ALL scenarios)")
    print("=" * 70)

    # ── Load training data ──────────────────────────────────────────────
    print(f"\nLoading: {TRAINING_CSV.name}")
    df = pd.read_csv(TRAINING_CSV, parse_dates=['Timestamp'])
    print(f"  Shape: {df.shape}")
    print(f"  Episodes: {df['episode_id'].nunique()}")

    # ── Select representative cold episode ──────────────────────────────
    ep_df = select_episode(df, percentile=COLDNESS_PERCENTILE)

    # ── Convert to weather format ───────────────────────────────────────
    nominal = episode_to_weather(ep_df)

    # ── Extract truth Q_internal for corruption (same for all scenarios)
    truth_q_internal = nominal['Q_internal_W'].values.copy()

    Q_occ_mean = float(
        truth_q_internal[truth_q_internal > np.median(truth_q_internal)].mean()
    )
    print(f"\n  Q_internal forecast corruption (applied to ALL scenarios):")
    print(f"    Shift: {QINT_SHIFT_STEPS} steps "
          f"({QINT_SHIFT_STEPS * (DT_SECONDS // 60)} min, no wrap-around)")
    print(f"    Scale: {QINT_SCALE:.2f} (>1 = overestimate -> cold risk)")
    print(f"    Noise: {QINT_NOISE_FRAC*100:.0f}% of occupied mean "
          f"({QINT_NOISE_FRAC * Q_occ_mean:.0f} W)")
    print(f"    AR(1) ρ: {QINT_AR_COEF}")
    print(f"    Seed: {QINT_SEED} (fixed across all scenarios)")

    # ── Save episode metadata ───────────────────────────────────────────
    metadata = {
        'source': str(TRAINING_CSV),
        'episode_id': int(ep_df['episode_id'].iloc[0]),
        'coldness_percentile': COLDNESS_PERCENTILE,
        'mean_T_out_C': float(nominal['T_out_C'].mean()),
        'min_T_out_C': float(nominal['T_out_C'].min()),
        'max_T_out_C': float(nominal['T_out_C'].max()),
        'max_Q_solar_W': float(nominal['Q_solar_W'].max()),
        'solar_column_used': 'I_solar_k (verified identical to Q_solar_trans_k)',
        'cold_snap_offset_C': COLD_SNAP_OFFSET_C,
        'forecast_bias_C': FORECAST_BIAS_C,
        'forecast_ar_coef': FORECAST_AR_COEF,
        'forecast_noise_std_C': FORECAST_NOISE_STD_C,
        'n_steps': len(nominal),
        'dt_seconds': DT_SECONDS,
        'q_internal_corruption': {
            'applied_to': 'ALL forecast CSVs',
            'shift_steps': QINT_SHIFT_STEPS,
            'shift_minutes': QINT_SHIFT_STEPS * (DT_SECONDS // 60),
            'shift_method': 'boundary-padded (no circular wrap-around)',
            'scale': QINT_SCALE,
            'noise_frac': QINT_NOISE_FRAC,
            'ar_coef': QINT_AR_COEF,
            'seed': QINT_SEED,
            'rationale': (
                'Internal gains uncertainty is always present in real buildings. '
                'Scale > 1 (overestimate) causes MPC to under-heat, creating '
                'cold violations where RAMC risk-aware training should excel. '
                'The "nominal" scenario has Q_internal mismatch only; cold snap '
                'and forecast error add weather mismatch on top.'
            ),
        },
    }

    with open(OUTPUT_DIR / "scenario_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"\n  Metadata saved: scenario_metadata.json")

    # =================================================================
    # Scenario 1: NOMINAL
    # =================================================================
    print("\n" + "-" * 50)
    print("Scenario 1: NOMINAL")
    print("  Truth: original episode (Q_internal = truth)")
    print("  Forecast: Q_internal corrupted (T_out and Q_solar = truth)")
    print("  NOTE: 'Nominal' means nominal WEATHER — Q_internal mismatch is present.")

    nominal_truth = nominal.copy()
    nominal_forecast = corrupt_q_internal(nominal.copy(), truth_q_internal)

    nominal_truth.to_csv(OUTPUT_DIR / "nominal_truth.csv", index=False)
    nominal_forecast.to_csv(OUTPUT_DIR / "nominal_forecast.csv", index=False)

    print(f"  Saved: nominal_truth.csv, nominal_forecast.csv")
    print(f"  T_out: [{nominal_truth['T_out_C'].min():.1f}, "
          f"{nominal_truth['T_out_C'].max():.1f}]°C")
    print(f"  Q_solar: [{nominal_truth['Q_solar_W'].min():.0f}, "
          f"{nominal_truth['Q_solar_W'].max():.0f}] W")
    print_q_internal_corruption_summary(
        truth_q_internal,
        nominal_forecast['Q_internal_W'].values,
        'nominal',
    )

    # =================================================================
    # Scenario 2: COLD SNAP
    # =================================================================
    print("\n" + "-" * 50)
    print("Scenario 2: COLD SNAP")
    print("  Truth: T_out -10°C drop (Q_internal = nominal truth)")
    print("  Forecast: T_out = nominal + Q_internal corrupted")

    cs_truth, cs_forecast = create_cold_snap(nominal)
    cs_forecast = corrupt_q_internal(cs_forecast, truth_q_internal)

    cs_truth.to_csv(OUTPUT_DIR / "cold_snap_truth.csv", index=False)
    cs_forecast.to_csv(OUTPUT_DIR / "cold_snap_forecast.csv", index=False)

    print(f"  Saved: cold_snap_truth.csv, cold_snap_forecast.csv")
    print_q_internal_corruption_summary(
        truth_q_internal,
        cs_forecast['Q_internal_W'].values,
        'cold_snap',
    )

    # =================================================================
    # Scenario 3: FORECAST ERROR
    # =================================================================
    print("\n" + "-" * 50)
    print("Scenario 3: FORECAST ERROR")
    print("  Truth: T_out = nominal (Q_internal = nominal truth)")
    print("  Forecast: T_out biased + Q_internal corrupted")

    fe_truth, fe_forecast = create_forecast_error(nominal)
    fe_forecast = corrupt_q_internal(fe_forecast, truth_q_internal)

    fe_truth.to_csv(OUTPUT_DIR / "forecast_error_truth.csv", index=False)
    fe_forecast.to_csv(OUTPUT_DIR / "forecast_error_forecast.csv", index=False)

    print(f"  Saved: forecast_error_truth.csv, forecast_error_forecast.csv")
    print_q_internal_corruption_summary(
        truth_q_internal,
        fe_forecast['Q_internal_W'].values,
        'forecast_error',
    )

    # =================================================================
    # Verification Summary
    # =================================================================
    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)

    training_ranges = {
        'T_out_k': (df['T_out_k'].min(), df['T_out_k'].max()),
        'I_solar_k': (df['I_solar_k'].min(), df['I_solar_k'].max()),
        'Q_internal_k': (df['Q_internal_k'].min(), df['Q_internal_k'].max()),
    }

    print(f"\nTraining ranges:")
    for col, (lo, hi) in training_ranges.items():
        print(f"  {col}: [{lo:.1f}, {hi:.1f}]")

    print(f"\nScenario ranges (with Q_internal corruption):")
    for name in ['nominal', 'cold_snap', 'forecast_error']:
        for suffix in ['truth', 'forecast']:
            fpath = OUTPUT_DIR / f"{name}_{suffix}.csv"
            dfw = pd.read_csv(fpath)
            T_range = (dfw['T_out_C'].min(), dfw['T_out_C'].max())
            Q_sol_range = (dfw['Q_solar_W'].min(), dfw['Q_solar_W'].max())
            Q_int_range = (dfw['Q_internal_W'].min(), dfw['Q_internal_W'].max())

            # Check ALL three channels against training ranges
            T_ok = (T_range[0] >= training_ranges['T_out_k'][0] - 0.1
                    and T_range[1] <= training_ranges['T_out_k'][1] + 0.1)
            Qsol_ok = (Q_sol_range[0] >= training_ranges['I_solar_k'][0] - 0.1
                       and Q_sol_range[1] <= training_ranges['I_solar_k'][1] + 0.1)
            Qint_ok = (Q_int_range[0] >= training_ranges['Q_internal_k'][0] - 0.1
                       and Q_int_range[1] <= training_ranges['Q_internal_k'][1] + 0.1)

            T_flag = "" if T_ok else "OOD"
            Qsol_flag = "" if Qsol_ok else "OOD"
            Qint_flag = "" if Qint_ok else "OOD"

            print(
                f"  {name}_{suffix}: "
                f"T=[{T_range[0]:.1f},{T_range[1]:.1f}]{T_flag}  "
                f"Qsol=[{Q_sol_range[0]:.0f},{Q_sol_range[1]:.0f}]{Qsol_flag}  "
                f"Qint=[{Q_int_range[0]:.0f},{Q_int_range[1]:.0f}]{Qint_flag}"
            )

        # Show Q_internal mismatch per scenario
        truth_df = pd.read_csv(OUTPUT_DIR / f"{name}_truth.csv")
        fcast_df = pd.read_csv(OUTPUT_DIR / f"{name}_forecast.csv")
        qerr = fcast_df['Q_internal_W'].values - truth_df['Q_internal_W'].values
        print(
            f"    -> Q_int forecast error: bias={qerr.mean():+.0f}W, "
            f"RMSE={np.sqrt((qerr**2).mean()):.0f}W"
        )

    print(f"\nOutput directory: {OUTPUT_DIR}")
    print("Matrix: 4 models × 3 scenarios × 5 seeds = 60 experiments")
    print("run_experimental_matrix.py: NO CHANGES NEEDED")
    print("DONE")


if __name__ == '__main__':
    generate_all_scenarios()

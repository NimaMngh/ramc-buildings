"""
Central path definitions for all RAMC assignment scripts.
==========================================================

Every cross-folder path is defined HERE and ONLY here.
Assignment scripts import from this module; they never hardcode paths.

Usage in any assignment script:
    from shared.paths import *

IMPORTANT — READ-ONLY CONVENTION:
    Paths under EXISTING_* prefixes point into folders 1–7.
    Assignment code must NEVER write to these locations.
    All outputs go into each assignment's own results/ folder.

Verified against: dir /s output dated 2026-03-11
"""

from pathlib import Path
import sys

# ============================================================
# 1. PROJECT ROOT
# ============================================================
# paths.py lives at: <PROJECT_ROOT>/8_Assignments/shared/paths.py
# So parents[2] is the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Quick sanity: the project root should contain folder "6_Neural Network Architecture"
assert (PROJECT_ROOT / "6_Neural Network Architecture").is_dir(), (
    f"PROJECT_ROOT resolved to {PROJECT_ROOT} but expected folder "
    f"'6_Neural Network Architecture' not found. Check file location."
)

# ============================================================
# 2. EXISTING DATA ASSETS  (READ ONLY — never write to these)
# ============================================================

# --- Phase 1: System Identification ---
RC_PARAMS_JSON = (
    PROJECT_ROOT
    / "3_Master Code for System Identification"
    / "results_N3_DE_optimized.json"
)

PROCESSED_CSV = (
    PROJECT_ROOT
    / "2_Dataset Creation for RC model"
    / "ASHRAE901_RetailStandalone_STD2022GBaseline_Vasteras_processed.csv"
)

EPW_FILE = (
    PROJECT_ROOT
    / "1_Building Simulation"
    / "Vasteras2007_2021.epw"
)

# --- Phase 1: Validation ---
APRIL_VALIDATION_JSON = (
    PROJECT_ROOT
    / "4_Universal Simulator Validation for RC"
    / "validation_results_N3_april.json"
)

RESIDUAL_ANALYSIS_JSON = (
    PROJECT_ROOT
    / "4_Universal Simulator Validation for RC"
    / "residual_analysis_results.json"
)

# --- Phase 2: Training Data ---
TRAINING_DATA_CSV = (
    PROJECT_ROOT
    / "5_Data Generation for Neural Network"
    / "RAMC_training_data_N3.csv"
)

TRAINING_DATA_META = (
    PROJECT_ROOT
    / "5_Data Generation for Neural Network"
    / "RAMC_training_data_N3_meta.json"
)

# --- Phase 2: Neural Network Results ---
RESULTS_DIR = (
    PROJECT_ROOT
    / "6_Neural Network Architecture"
    / "results"
    / "RAMC_FULL_cvar_20260307_115447"
)

DECOMPOSED_EVAL_CSV = RESULTS_DIR / "decomposed_evaluation_full.csv"
DECOMPOSED_EVAL_JSON = RESULTS_DIR / "decomposed_evaluation_full.json"
DECOMPOSED_EVAL_TXT = RESULTS_DIR / "decomposed_evaluation_summary.txt"
POOLED_TAIL_JSON = RESULTS_DIR / "pooled_tail_statistics.json"
ROLLOUT_COMPARISON_JSON = RESULTS_DIR / "rollout_comparison.json"
ROLLOUT_METRICS_JSON = RESULTS_DIR / "rollout_metrics.json"

# --- Phase 3: Weather Scenarios ---
SCENARIO_DIR = (
    PROJECT_ROOT
    / "7_Close Loop Performance"
    / "M1_Weather_Data"
    / "data_ramc_epw"
)

SCENARIO_METADATA = SCENARIO_DIR / "scenario_metadata.json"

# Scenario files: truth (what the plant experiences) and forecast (what NMPC sees)
SCENARIOS = {
    "nominal": {
        "truth": SCENARIO_DIR / "nominal_truth.csv",
        "forecast": SCENARIO_DIR / "nominal_forecast.csv",
    },
    "cold_snap": {
        "truth": SCENARIO_DIR / "cold_snap_truth.csv",
        "forecast": SCENARIO_DIR / "cold_snap_forecast.csv",
    },
    "forecast_error": {
        "truth": SCENARIO_DIR / "forecast_error_truth.csv",
        "forecast": SCENARIO_DIR / "forecast_error_forecast.csv",
    },
}

# --- Phase 3: NMPC Results ---
NMPC_RESULTS_DIR = (
    PROJECT_ROOT
    / "7_Close Loop Performance"
    / "M6_NMPC_Direct_Shooting"
    / "results_NMPC_20260307_185109"
)

NMPC_AGGREGATE_CSV = NMPC_RESULTS_DIR / "matrix" / "aggregate_metrics.csv"
NMPC_PAIRED_DIFF_CSV = NMPC_RESULTS_DIR / "matrix" / "paired_differences.csv"
NMPC_ALL_RESULTS_JSON = NMPC_RESULTS_DIR / "matrix" / "all_results.json"
NMPC_CONFIG_JSON = NMPC_RESULTS_DIR / "matrix" / "config.json"

# ============================================================
# 3. CHECKPOINT FILES  (READ ONLY)
# ============================================================
# All checkpoints live in RESULTS_DIR. Three variants per model:
#   *_best.pth       — best validation loss (primary)
#   *_best_cost.pth  — best cost metric
#   *_best_risk.pth  — best risk metric

# --- The 4 models used in Phase 3 closed-loop evaluation ---
CHECKPOINT_FIDELITY = RESULTS_DIR / "Fidelity_Baseline_rollout_a1.0_best.pth"
CHECKPOINT_RAMC_1E4 = RESULTS_DIR / "RAMC_lambda_0.0001_op_cvar_rollout_a1.0_best.pth"
CHECKPOINT_RAMC_5E4 = RESULTS_DIR / "RAMC_lambda_0.0005_op_cvar_rollout_a1.0_best.pth"
CHECKPOINT_RAMC_15E4 = RESULTS_DIR / "RAMC_lambda_0.0015_op_cvar_rollout_a1.0_best.pth"
CHECKPOINT_RAW_MSE = RESULTS_DIR / "Raw_MSE_Baseline_best.pth"

# --- Full λ grid checkpoints (needed for A3 broader sweep) ---
# Keys are λ values as strings matching the filename convention
ALL_CHECKPOINTS = {
    "Raw_MSE": RESULTS_DIR / "Raw_MSE_Baseline_best.pth",
    "Fidelity": RESULTS_DIR / "Fidelity_Baseline_rollout_a1.0_best.pth",
    "5e-05": RESULTS_DIR / "RAMC_lambda_5e-05_op_cvar_rollout_a1.0_best.pth",
    "0.0001": RESULTS_DIR / "RAMC_lambda_0.0001_op_cvar_rollout_a1.0_best.pth",
    "0.0002": RESULTS_DIR / "RAMC_lambda_0.0002_op_cvar_rollout_a1.0_best.pth",
    "0.0003": RESULTS_DIR / "RAMC_lambda_0.0003_op_cvar_rollout_a1.0_best.pth",
    "0.0005": RESULTS_DIR / "RAMC_lambda_0.0005_op_cvar_rollout_a1.0_best.pth",
    "0.0007": RESULTS_DIR / "RAMC_lambda_0.0007_op_cvar_rollout_a1.0_best.pth",
    "0.001": RESULTS_DIR / "RAMC_lambda_0.001_op_cvar_rollout_a1.0_best.pth",
    "0.0015": RESULTS_DIR / "RAMC_lambda_0.0015_op_cvar_rollout_a1.0_best.pth",
    "0.002": RESULTS_DIR / "RAMC_lambda_0.002_op_cvar_rollout_a1.0_best.pth",
    "0.005": RESULTS_DIR / "RAMC_lambda_0.005_op_cvar_rollout_a1.0_best.pth",
}

# Convenience: the 4 Phase 3 models as a dict (used by A1, A2, A4)
PHASE3_MODELS = {
    "Fidelity": CHECKPOINT_FIDELITY,
    "RAMC_1e-4": CHECKPOINT_RAMC_1E4,
    "RAMC_5e-4": CHECKPOINT_RAMC_5E4,
    "RAMC_1.5e-3": CHECKPOINT_RAMC_15E4,
}

# Convenience: the focused pair for A1 stress tests
STRESS_TEST_MODELS = {
    "Fidelity": CHECKPOINT_FIDELITY,
    "RAMC_1.5e-3": CHECKPOINT_RAMC_15E4,
}

# ============================================================
# 4. EXISTING CODE MODULES  (IMPORT from these, never modify)
# ============================================================
NN_ARCH_DIR = PROJECT_ROOT / "6_Neural Network Architecture"
CLOSED_LOOP_DIR = PROJECT_ROOT / "7_Close Loop Performance" / "M4_Closed_Loop_Simulator"
NMPC_DIR = PROJECT_ROOT / "7_Close Loop Performance" / "M6_NMPC_Direct_Shooting"
WEATHER_DIR = PROJECT_ROOT / "7_Close Loop Performance" / "M1_Weather_Data"
SYSID_DIR = PROJECT_ROOT / "3_Master Code for System Identification"
DATAGEN_DIR = PROJECT_ROOT / "5_Data Generation for Neural Network"


def setup_imports():
    """
    Add existing code directories to sys.path so assignment scripts
    can import modules like:
        from thermal_dynamics_net import ThermalDynamicsNet
        from rc_ground_truth import RCGroundTruthModel
        from nmpc_direct_shooting import NMPCDirectShooting
        from closed_loop_nmpc import ClosedLoopNMPCSimulator
        from load_ramc_model import load_ramc_model
        from ramc_losses import ...
    
    Call this once at the top of each assignment script:
        from shared.paths import *
        setup_imports()
    """
    dirs_to_add = [
        str(NN_ARCH_DIR),
        str(CLOSED_LOOP_DIR),
        str(NMPC_DIR),
        str(WEATHER_DIR),
        str(SYSID_DIR),
        str(DATAGEN_DIR),
    ]
    for d in dirs_to_add:
        if d not in sys.path:
            sys.path.insert(0, d)


# ============================================================
# 5. ASSIGNMENT OUTPUT DIRECTORIES
# ============================================================
ASSIGNMENTS_DIR = PROJECT_ROOT / "8_Assignments"
SHARED_DIR = ASSIGNMENTS_DIR / "shared"

# Each assignment's results directory (created on demand)
A1_DIR = ASSIGNMENTS_DIR / "A1_external_validity"
A2_DIR = ASSIGNMENTS_DIR / "A2_benchmark_sufficiency"
A3_DIR = ASSIGNMENTS_DIR / "A3_lambda_selection"
A4_DIR = ASSIGNMENTS_DIR / "A4_controller_robustness"
A5_DIR = ASSIGNMENTS_DIR / "A5_mechanistic_bridge"
A6_DIR = ASSIGNMENTS_DIR / "A6_pareto_analysis"
A7_DIR = ASSIGNMENTS_DIR / "A7_sysid_transparency"


def get_results_dir(assignment_dir: Path, *subdirs: str) -> Path:
    """
    Create and return a results subdirectory for an assignment.
    
    Usage:
        out = get_results_dir(A1_DIR, "raw")          # A1_external_validity/results/raw/
        out = get_results_dir(A3_DIR, "figures")       # A3_lambda_selection/results/figures/
        out = get_results_dir(A2_DIR)                  # A2_benchmark_sufficiency/results/
    """
    path = assignment_dir / "results"
    for sub in subdirs:
        path = path / sub
    path.mkdir(parents=True, exist_ok=True)
    return path


# ============================================================
# 6. CONSTANTS  (shared across assignments)
# ============================================================

# NMPC default settings (from run_nmpc_matrix.py / Table 9 of results doc)
NMPC_DEFAULTS = {
    "horizon": 24,           # steps (4 hours)
    "block_size": 4,         # steps (40-minute blocks)
    "adam_iters": 25,
    "learning_rate": 0.05,
    "grad_clip": 5.0,
}

# Comfort and cost parameters (must match training and evaluation exactly)
COMFORT_BAND = (20.0, 22.0)  # °C, occupied hours
W_COLD = 63.0
W_HOT = 30.0
W_ENERGY = 0.9               # SEK/kWh
W_TERMINAL = 20.0

# Actuator limits
T_SUPPLY_RANGE = (32.0, 60.0)  # °C
MDOT_RANGE = (0.0, 4.05)       # kg/s

# Simulation
STEPS_PER_EPISODE = 1008      # 7 days × 144 steps/day (10-min intervals)
DT_MINUTES = 10

# Phase 3 seeds (from Table 9)
PHASE3_SEEDS = [42, 123, 456, 789, 1000]

# Reduced seed set for focused experiments (A1, A4)
REDUCED_SEEDS = [42, 123, 456]


# ============================================================
# 7. VERIFICATION UTILITY
# ============================================================

def verify_paths(verbose: bool = True) -> dict:
    """
    Check that all critical paths exist. Returns a dict of
    {path_name: (path, exists_bool)}. Prints a summary if verbose.
    
    Run this after setting up the project to catch path issues early:
        python -m shared.paths
    """
    critical_paths = {
        "RC_PARAMS_JSON": RC_PARAMS_JSON,
        "PROCESSED_CSV": PROCESSED_CSV,
        "TRAINING_DATA_CSV": TRAINING_DATA_CSV,
        "DECOMPOSED_EVAL_CSV": DECOMPOSED_EVAL_CSV,
        "DECOMPOSED_EVAL_JSON": DECOMPOSED_EVAL_JSON,
        "POOLED_TAIL_JSON": POOLED_TAIL_JSON,
        "SCENARIO_METADATA": SCENARIO_METADATA,
        "NMPC_AGGREGATE_CSV": NMPC_AGGREGATE_CSV,
        "NMPC_PAIRED_DIFF_CSV": NMPC_PAIRED_DIFF_CSV,
        "NMPC_ALL_RESULTS_JSON": NMPC_ALL_RESULTS_JSON,
        "CHECKPOINT_FIDELITY": CHECKPOINT_FIDELITY,
        "CHECKPOINT_RAMC_1E4": CHECKPOINT_RAMC_1E4,
        "CHECKPOINT_RAMC_5E4": CHECKPOINT_RAMC_5E4,
        "CHECKPOINT_RAMC_15E4": CHECKPOINT_RAMC_15E4,
        "CHECKPOINT_RAW_MSE": CHECKPOINT_RAW_MSE,
    }
    
    # Add all λ-grid checkpoints
    for key, path in ALL_CHECKPOINTS.items():
        critical_paths[f"CKPT_{key}"] = path
    
    # Add scenario files
    for scenario_name, files in SCENARIOS.items():
        for kind, path in files.items():
            critical_paths[f"SCENARIO_{scenario_name}_{kind}"] = path
    
    # Add code directories
    code_dirs = {
        "NN_ARCH_DIR": NN_ARCH_DIR,
        "CLOSED_LOOP_DIR": CLOSED_LOOP_DIR,
        "NMPC_DIR": NMPC_DIR,
    }
    
    results = {}
    missing = []
    found = []
    
    for name, path in {**critical_paths, **code_dirs}.items():
        exists = path.exists()
        results[name] = (path, exists)
        if exists:
            found.append(name)
        else:
            missing.append(name)
    
    if verbose:
        print(f"Path verification: {len(found)}/{len(results)} found")
        if missing:
            print(f"\n  MISSING ({len(missing)}):")
            for name in missing:
                print(f"    {name}")
                print(f"      {results[name][0]}")
        else:
            print("  All paths verified OK.")
    
    return results


# ============================================================
# 8. SELF-TEST
# ============================================================
if __name__ == "__main__":
    print(f"PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"RESULTS_DIR:  {RESULTS_DIR}")
    print(f"SCENARIO_DIR: {SCENARIO_DIR}")
    print()
    verify_paths(verbose=True)

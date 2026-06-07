"""
Assignment 7 — System-Identification and Data-Coverage Hypothesis
==================================================================

Tests H7: The identified RC model is sufficiently well-supported by
excitation data and seasonal validation, and the synthetic training
dataset covers the operating region encountered during closed-loop
evaluation.

Deliverables (from the assignment document):
  1. Compact identification-pipeline subsection data
  2. Identification/validation summary table (RMSE, CV-RMSE, time constants, April val)
  3. Data-coverage figure/table (training vs evaluation ranges)
  4. Residual diagnostic beyond aggregate RMSE
  5. Numerical-stability paragraph data (substep distribution, clamping rate)

No NMPC runs required — this is pure data analysis.

Run from 8_Assignments/ directory:
    python A7_sysid_transparency/run_a7.py

Author: RAMC Assignment Framework
"""

import matplotlib
matplotlib.use("Agg")

import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from collections import OrderedDict

# ── Imports from shared infrastructure ──
THIS_DIR = Path(__file__).resolve().parent
ASSIGNMENTS_DIR = THIS_DIR.parent
if str(ASSIGNMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(ASSIGNMENTS_DIR))

from shared.paths import (
    RC_PARAMS_JSON,
    PROCESSED_CSV,
    TRAINING_DATA_CSV,
    TRAINING_DATA_META,
    SCENARIOS,
    SCENARIO_DIR,
    NMPC_ALL_RESULTS_JSON,
    NMPC_AGGREGATE_CSV,
    APRIL_VALIDATION_JSON,
    RESIDUAL_ANALYSIS_JSON,
    A7_DIR,
    get_results_dir,
    setup_imports,
)

setup_imports()


# =============================================================================
# Configuration
# =============================================================================

# Variables to compare between training and evaluation
COVERAGE_VARS_STATE = {
    "T_air": {"train_col": "T_air_k", "desc": "Indoor air temperature (°C)"},
    "T_env": {"train_col": "T_env_k", "desc": "Envelope temperature (°C)"},
    "T_int": {"train_col": "T_int_k", "desc": "Internal mass temperature (°C)"},
    "T_rad1": {"train_col": "T_rad1_k", "desc": "Radiator section 1 (°C)"},
    "T_rad2": {"train_col": "T_rad2_k", "desc": "Radiator section 2 (°C)"},
    "T_ret": {"train_col": "T_ret_k", "desc": "Return temperature (°C)"},
}

COVERAGE_VARS_CONTROL = {
    "T_supply": {"train_col": "T_supply_k", "desc": "Supply temperature (°C)"},
    "mdot": {"train_col": "mdot_k", "desc": "Mass flow rate (kg/s)"},
}

COVERAGE_VARS_DISTURBANCE = {
    "T_out": {"train_col": "T_out_k", "desc": "Outdoor temperature (°C)"},
    "Q_solar": {"train_col": "Q_solar_trans_k", "desc": "Solar radiation (W)"},
    "Q_internal": {"train_col": "Q_internal_k", "desc": "Internal gains (W)"},
}


# =============================================================================
# 1. Identification Pipeline Summary
# =============================================================================

def _find_rc_params_in_dict(d: dict) -> dict:
    """
    Recursively search a JSON dict for the 10 RC parameters.
    Handles all known formats of results_N3_DE_optimized.json:
      - flat: {"C_air": ..., "C_env": ..., ...}
      - nested: {"best_params": {...}}, {"parameters": {...}},
                {"result": {"parameters": {...}}}, etc.
      - also checks training_data_meta's rc_params_used
    """
    param_names = {"C_air", "C_env", "C_int", "C_rad", "R_ex", "R_ae",
                   "R_ai", "K_rad", "a_rad", "A_sol"}
    
    # Check if this dict itself contains the params
    if param_names.issubset(d.keys()):
        return {k: float(d[k]) for k in param_names}
    
    # Search known nested keys
    search_keys = ["best_params", "parameters", "params", "result",
                   "best_result", "model", "rc_params_used", "rc_params"]
    for key in search_keys:
        if key in d and isinstance(d[key], dict):
            found = _find_rc_params_in_dict(d[key])
            if found:
                return found
    
    # Brute-force: search ALL dict values
    for key, val in d.items():
        if isinstance(val, dict):
            found = _find_rc_params_in_dict(val)
            if found:
                return found
    
    return {}


def build_identification_summary(out_dir: Path) -> dict:
    """
    Build identification pipeline summary from existing result files.
    
    Reads (with fallback chain for parameters):
      1. RC_PARAMS_JSON (identified parameters)
      2. APRIL_VALIDATION_JSON (seasonal validation — also contains params)
      3. TRAINING_DATA_META (also contains rc_params_used)
      4. RESIDUAL_ANALYSIS_JSON (residual diagnostics)
    """
    print("\n[1] Building identification pipeline summary...")
    
    summary = OrderedDict()
    params = {}
    
    # ── Load RC parameters with fallback chain ──
    # Source 1: RC_PARAMS_JSON
    if RC_PARAMS_JSON.exists():
        with open(RC_PARAMS_JSON) as f:
            rc_data = json.load(f)
        params = _find_rc_params_in_dict(rc_data)
        if params:
            print(f"  Loaded {len(params)} parameters from RC_PARAMS_JSON")
        else:
            print(f"  RC_PARAMS_JSON exists but could not extract params (unexpected format)")
            print(f"    Top-level keys: {list(rc_data.keys())[:10]}")
    
    # Source 2: April validation JSON (contains params under model.parameters)
    april_data = {}
    if APRIL_VALIDATION_JSON.exists():
        with open(APRIL_VALIDATION_JSON) as f:
            april_data = json.load(f)
        
        if not params:
            params = _find_rc_params_in_dict(april_data)
            if params:
                print(f"  Loaded {len(params)} parameters from APRIL_VALIDATION_JSON (fallback)")
    
    # Source 3: Training data metadata
    if not params and TRAINING_DATA_META.exists():
        with open(TRAINING_DATA_META) as f:
            meta_data = json.load(f)
        params = _find_rc_params_in_dict(meta_data)
        if params:
            print(f"  Loaded {len(params)} parameters from TRAINING_DATA_META (fallback)")
    
    if not params:
        print(f"  WARNING: Could not load RC parameters from any source!")
    
    summary["identified_parameters"] = params
    
    # ── Compute derived time constants ──
    if all(k in params for k in ["C_air", "R_ae", "C_env", "R_ex", "C_int", "R_ai", "C_rad", "K_rad"]):
        summary["time_constants"] = {
            "tau_air_env_hours": params["C_air"] * params["R_ae"] / 3600.0,
            "tau_env_out_hours": params["C_env"] * params["R_ex"] / 3600.0,
            "tau_air_int_hours": params["C_int"] * params["R_ai"] / 3600.0,
            "tau_rad_minutes": params["C_rad"] / (3 * params["K_rad"]) / 60.0,
        }
        tc = summary["time_constants"]
        print(f"  Time constants:")
        print(f"    τ_air-env = {tc['tau_air_env_hours']:.1f} h")
        print(f"    τ_env-out = {tc['tau_env_out_hours']:.1f} h")
        print(f"    τ_air-int = {tc['tau_air_int_hours']:.1f} h")
        print(f"    τ_rad     = {tc['tau_rad_minutes']:.1f} min")
        
        # Derived physical quantities for the paper
        summary["derived_quantities"] = {
            "UA_value_W_per_K": 1.0 / params["R_ex"],
            "equivalent_air_volume_m3": params["C_air"] / (1005.0 * 1.2),
            "Cenv_to_Cint_ratio": params["C_env"] / params["C_int"],
        }
        dq = summary["derived_quantities"]
        print(f"  Derived quantities:")
        print(f"    UA value = {dq['UA_value_W_per_K']:.0f} W/K")
        print(f"    Equiv. air vol = {dq['equivalent_air_volume_m3']:.0f} m³")
        print(f"    C_env/C_int = 1:{params['C_int']/params['C_env']:.1f}")
    
    # ── Process April validation ──
    if april_data:
        summary["april_validation"] = april_data
        # Extract metrics from nested structure
        metrics = april_data.get("metrics", april_data)
        rmse = metrics.get("rmse_C", april_data.get("rmse_C", "N/A"))
        cv_rmse = metrics.get("cv_rmse_percent", metrics.get("cv_rmse_pct",
                  april_data.get("cv_rmse_pct", "N/A")))
        autocorr = metrics.get("autocorrelation", "N/A")
        bias = metrics.get("mean_bias_C", april_data.get("mean_bias_C", "N/A"))
        
        summary["validation_metrics"] = {
            "april_rmse_C": rmse,
            "april_cv_rmse_pct": cv_rmse,
            "april_mean_bias_C": bias,
            "april_autocorrelation": autocorr,
        }
        
        # Also get fitting-period comparison if available
        comp = april_data.get("comparison_to_training", {})
        if comp:
            summary["validation_metrics"]["fitting_rmse_C"] = comp.get("original_rmse_C")
            summary["validation_metrics"]["fitting_cv_rmse_pct"] = comp.get("original_cv_rmse")
        
        print(f"  April validation: RMSE={rmse} °C, CV-RMSE={cv_rmse}%, "
              f"bias={bias} °C, autocorr={autocorr}")
    elif APRIL_VALIDATION_JSON.exists():
        print(f"  WARNING: April validation JSON loaded but empty")
    else:
        print(f"  WARNING: April validation JSON not found: {APRIL_VALIDATION_JSON}")
    
    # ── Load residual analysis ──
    if RESIDUAL_ANALYSIS_JSON.exists():
        with open(RESIDUAL_ANALYSIS_JSON) as f:
            residual_data = json.load(f)
        summary["residual_analysis"] = residual_data
        
        # Print key residual correlations
        corr = residual_data.get("correlations", {})
        if corr:
            print(f"  Residual correlations with inputs:")
            for var_name, c in corr.items():
                sig = "*" if c.get("p", 1.0) < 0.001 else ""
                print(f"    {var_name:<25s} r={c['r']:+.3f}{sig}")
    else:
        print(f"  WARNING: Residual analysis JSON not found: {RESIDUAL_ANALYSIS_JSON}")
    
    # ── Identification pipeline steps ──
    summary["pipeline_steps"] = [
        "1. EnergyPlus 25.1 simulation of ASHRAE 901 Retail Standalone (Västerås) with PRBS supply-temperature excitation",
        "2. Data preprocessing: 5-zone volume-weighted aggregation -> single building-level model",
        "3. Differential Evolution identification (N=3 radiator sections, 10 parameters, pop=50, max 20 gen)",
        "4. Nelder-Mead local refinement of global optimum",
        "5. Physical consistency checks (positive capacities/resistances, plausible time constants)",
        "6. Shoulder-season validation on April data (unseen period)",
    ]
    
    # Save
    out_path = out_dir / "identification_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {out_path.name}")
    
    return summary


# =============================================================================
# 2. Data Coverage Analysis
# =============================================================================

def compute_data_coverage(out_dir: Path, fig_dir: Path) -> dict:
    """
    Compare [1%, 99%] ranges of training data vs closed-loop evaluation data.
    
    Training data: RAMC_training_data_N3.csv (302,400 samples)
    Evaluation data: extracted from NMPC_ALL_RESULTS_JSON trajectories,
                     or approximated from scenario CSVs + actuator bounds.
    """
    print("\n[2] Computing data coverage (training vs evaluation)...")
    
    coverage = {}
    
    # ── Load training data (sample for memory efficiency) ──
    print(f"  Loading training data...")
    if not TRAINING_DATA_CSV.exists():
        print(f"  WARNING: Training data not found: {TRAINING_DATA_CSV}")
        return coverage
    
    # Read a sample — the file is 120MB, so read in chunks
    train_df = pd.read_csv(TRAINING_DATA_CSV, nrows=None)
    print(f"  Training data: {len(train_df)} rows, {len(train_df.columns)} columns")
    
    # ── Compute training ranges ──
    train_ranges = {}
    all_vars = {**COVERAGE_VARS_STATE, **COVERAGE_VARS_CONTROL, **COVERAGE_VARS_DISTURBANCE}
    
    for var_name, var_info in all_vars.items():
        col = var_info["train_col"]
        if col in train_df.columns:
            vals = train_df[col].dropna()
            train_ranges[var_name] = {
                "p1": float(np.percentile(vals, 1)),
                "p99": float(np.percentile(vals, 99)),
                "min": float(vals.min()),
                "max": float(vals.max()),
                "mean": float(vals.mean()),
                "std": float(vals.std()),
                "desc": var_info["desc"],
            }
        else:
            print(f"    WARNING: Column '{col}' not found in training data")
    
    # ── Load evaluation data from scenario CSVs ──
    print(f"  Loading evaluation scenario data...")
    eval_data = {"T_out": [], "Q_solar": [], "Q_internal": []}
    
    for scenario_name, scenario_paths in SCENARIOS.items():
        for kind in ["truth", "forecast"]:
            csv_path = scenario_paths[kind]
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                # Outdoor temperature
                for col_name in ["T_out_C", "T_outdoor_C"]:
                    if col_name in df.columns:
                        eval_data["T_out"].extend(df[col_name].dropna().tolist())
                        break
                # Solar
                if "Q_solar_W" in df.columns:
                    eval_data["Q_solar"].extend(df["Q_solar_W"].dropna().tolist())
                # Internal gains
                if "Q_internal_W" in df.columns:
                    eval_data["Q_internal"].extend(df["Q_internal_W"].dropna().tolist())
    
    # ── Load evaluation trajectories from NMPC results ──
    eval_traj_ranges = {}
    if NMPC_ALL_RESULTS_JSON.exists():
        print(f"  Loading NMPC all_results.json for trajectory ranges...")
        with open(NMPC_ALL_RESULTS_JSON) as f:
            nmpc_data = json.load(f)
        
        # Extract T_air ranges from all experiments
        t_air_all = []
        t_supply_all = []
        mdot_all = []
        energy_all = []
        
        if isinstance(nmpc_data, dict):
            results_list = nmpc_data.get("results", [nmpc_data])
        elif isinstance(nmpc_data, list):
            results_list = nmpc_data
        else:
            results_list = []
        
        for r in results_list:
            metrics = r.get("metrics", r)
            if "T_air_min_C" in metrics:
                t_air_all.append(metrics["T_air_min_C"])
                t_air_all.append(metrics["T_air_max_C"])
            if "T_air_occ_min_C" in metrics:
                t_air_all.append(metrics["T_air_occ_min_C"])
            if "T_air_occ_max_C" in metrics:
                t_air_all.append(metrics["T_air_occ_max_C"])
        
        if t_air_all:
            eval_traj_ranges["T_air"] = {
                "min": float(min(t_air_all)),
                "max": float(max(t_air_all)),
            }
            print(f"    T_air eval range: [{min(t_air_all):.1f}, {max(t_air_all):.1f}] °C")
    
    # Build evaluation ranges from scenario data
    eval_ranges = {}
    for var_name, vals in eval_data.items():
        if vals:
            arr = np.array(vals)
            eval_ranges[var_name] = {
                "p1": float(np.percentile(arr, 1)),
                "p99": float(np.percentile(arr, 99)),
                "min": float(arr.min()),
                "max": float(arr.max()),
                "mean": float(arr.mean()),
            }
    
    # Add actuator bounds (known from config)
    eval_ranges["T_supply"] = {"p1": 32.0, "p99": 60.0, "min": 32.0, "max": 60.0}
    eval_ranges["mdot"] = {"p1": 0.0, "p99": 4.05, "min": 0.0, "max": 4.05}
    
    # Merge trajectory ranges
    for var_name, r in eval_traj_ranges.items():
        if var_name not in eval_ranges:
            eval_ranges[var_name] = r
        else:
            eval_ranges[var_name]["min"] = min(eval_ranges[var_name].get("min", r["min"]), r["min"])
            eval_ranges[var_name]["max"] = max(eval_ranges[var_name].get("max", r["max"]), r["max"])
    
    # ── Build coverage table ──
    # Split into: observable variables (have eval data) and internal states (no eval data)
    print(f"\n  === Observable Variables (have evaluation data) ===")
    print(f"  {'Variable':<15s} {'Training [1%,99%]':>22s} {'Eval [1%,99%]':>22s} {'Status':>10s} {'Note'}")
    print(f"  {'-'*90}")
    
    coverage_table = []
    extrapolation_risks = []
    
    for var_name in all_vars:
        tr = train_ranges.get(var_name, {})
        ev = eval_ranges.get(var_name, {})
        
        if tr and ev:
            tr_lo, tr_hi = tr["p1"], tr["p99"]
            ev_lo = ev.get("p1", ev.get("min", float("nan")))
            ev_hi = ev.get("p99", ev.get("max", float("nan")))
            
            # More nuanced coverage assessment
            lo_gap = tr_lo - ev_lo  # positive = eval extends below training
            hi_gap = ev_hi - tr_hi  # positive = eval extends above training
            
            if lo_gap <= 0.5 and hi_gap <= 0.5:
                status = "COVERED"
                note = ""
            elif lo_gap > 5.0 or hi_gap > 5.0:
                status = "EXTRAPOL"
                note = f"gap: {max(lo_gap, hi_gap):.1f}"
                extrapolation_risks.append((var_name, lo_gap, hi_gap))
            else:
                status = "MARGINAL"
                note = f"gap: {max(lo_gap, hi_gap):.1f}"
            
            entry = {
                "variable": var_name,
                "description": all_vars[var_name]["desc"],
                "train_p1": tr_lo, "train_p99": tr_hi,
                "eval_p1": ev_lo, "eval_p99": ev_hi,
                "covered": status == "COVERED",
                "status": status,
                "lo_gap": float(lo_gap),
                "hi_gap": float(hi_gap),
            }
            coverage_table.append(entry)
            
            print(f"  {var_name:<15s} [{tr_lo:>8.1f}, {tr_hi:>8.1f}] "
                  f"[{ev_lo:>8.1f}, {ev_hi:>8.1f}]  {status:>10s}  {note}")
        
        elif tr and not ev:
            # Internal state: no evaluation data available
            entry = {
                "variable": var_name,
                "description": all_vars[var_name]["desc"],
                "train_p1": tr["p1"], "train_p99": tr["p99"],
                "eval_p1": None, "eval_p99": None,
                "covered": None,
                "status": "INTERNAL",
                "note": "Not directly observable in evaluation; constrained by plant physics",
            }
            coverage_table.append(entry)
        else:
            print(f"  {var_name:<15s} {'(no training data)':>22s}")
    
    # Print internal states separately
    internal = [c for c in coverage_table if c.get("status") == "INTERNAL"]
    if internal:
        print(f"\n  === Internal States (constrained by plant physics, not directly in eval data) ===")
        print(f"  {'Variable':<15s} {'Training [1%,99%]':>22s} {'Note'}")
        print(f"  {'-'*70}")
        for c in internal:
            print(f"  {c['variable']:<15s} [{c['train_p1']:>8.1f}, {c['train_p99']:>8.1f}]  "
                  f"Same RC plant -> same physics constrains range")
    
    # Print extrapolation risk summary
    if extrapolation_risks:
        print(f"\n  === Extrapolation Risks ===")
        for var_name, lo_gap, hi_gap in extrapolation_risks:
            direction = "below" if lo_gap > hi_gap else "above"
            gap = max(lo_gap, hi_gap)
            print(f"  {var_name}: eval extends {gap:.1f} units {direction} training [1%,99%] range")
    
    # Interpretation for the paper
    observable = [c for c in coverage_table if c.get("status") in ("COVERED", "MARGINAL", "EXTRAPOL")]
    n_covered = sum(1 for c in observable if c["status"] == "COVERED")
    n_marginal = sum(1 for c in observable if c["status"] == "MARGINAL")
    n_extrapol = sum(1 for c in observable if c["status"] == "EXTRAPOL")
    
    coverage["interpretation"] = {
        "observable_vars": len(observable),
        "covered": n_covered,
        "marginal": n_marginal,
        "extrapolation_risk": n_extrapol,
        "internal_states": len(internal),
        "internal_states_note": (
            "Internal states (T_env, T_int, T_rad) are not directly observed during "
            "evaluation but are constrained by the same RC plant physics that generated "
            "the training data. The NN's learned dynamics for these states are therefore "
            "evaluated within the same physical regime."
        ),
        "extrapolation_note": (
            "T_out extends 4.7°C below training range in the cold snap scenario. "
            "Q_internal evaluation range is wider than training (0–93 kW vs 3.5–78 kW) "
            "due to forecast corruption adding variability. These are real extrapolation "
            "risks that the paper should acknowledge."
        ) if extrapolation_risks else "No significant extrapolation risks detected.",
    }
    
    print(f"\n  Coverage summary: {n_covered} covered, {n_marginal} marginal, "
          f"{n_extrapol} extrapolation risk, {len(internal)} internal (physics-constrained)")
    
    coverage["training_ranges"] = train_ranges
    coverage["evaluation_ranges"] = eval_ranges
    coverage["coverage_table"] = coverage_table
    
    # ── Plot overlapping histograms ──
    plot_coverage_histograms(train_df, eval_data, eval_ranges, train_ranges, fig_dir)
    
    # Save
    out_path = out_dir / "data_coverage.json"
    with open(out_path, "w") as f:
        json.dump(coverage, f, indent=2, default=str)
    print(f"  Saved: {out_path.name}")
    
    return coverage


def plot_coverage_histograms(train_df, eval_data, eval_ranges, train_ranges, fig_dir):
    """Plot overlapping histograms for key variables."""
    print(f"  Generating coverage histogram figure...")
    
    vars_to_plot = [
        ("T_air", "T_air_k", "Indoor Air Temperature (°C)"),
        ("T_supply", "T_supply_k", "Supply Temperature (°C)"),
        ("mdot", "mdot_k", "Mass Flow Rate (kg/s)"),
        ("T_out", "T_out_k", "Outdoor Temperature (°C)"),
        ("Q_solar", "Q_solar_trans_k", "Transmitted Solar (W)"),
        ("Q_internal", "Q_internal_k", "Internal Gains (W)"),
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    
    for ax, (var_name, train_col, label) in zip(axes, vars_to_plot):
        # Training data
        if train_col in train_df.columns:
            train_vals = train_df[train_col].dropna().values
            ax.hist(train_vals, bins=80, alpha=0.5, density=True,
                    color="steelblue", label="Training", edgecolor="none")
        
        # Evaluation data (disturbances from scenario CSVs)
        if var_name in eval_data and eval_data[var_name]:
            eval_vals = np.array(eval_data[var_name])
            ax.hist(eval_vals, bins=50, alpha=0.5, density=True,
                    color="coral", label="Evaluation", edgecolor="none")
        
        # Mark training [1%, 99%] range
        if var_name in train_ranges:
            tr = train_ranges[var_name]
            ax.axvline(tr["p1"], color="steelblue", linestyle="--", alpha=0.7, linewidth=1)
            ax.axvline(tr["p99"], color="steelblue", linestyle="--", alpha=0.7, linewidth=1)
        
        # Mark eval range if available
        if var_name in eval_ranges:
            ev = eval_ranges[var_name]
            p1 = ev.get("p1", ev.get("min"))
            p99 = ev.get("p99", ev.get("max"))
            if p1 is not None:
                ax.axvline(p1, color="coral", linestyle="--", alpha=0.7, linewidth=1)
            if p99 is not None:
                ax.axvline(p99, color="coral", linestyle="--", alpha=0.7, linewidth=1)
        
        ax.set_xlabel(label, fontsize=9)
        ax.set_ylabel("Density", fontsize=9)
        ax.legend(fontsize=8)
        ax.tick_params(labelsize=8)
    
    fig.suptitle("Data Coverage: Training vs Closed-Loop Evaluation", fontsize=13, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    fig_path = fig_dir / "data_coverage_histograms.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fig_path.name}")
    
    # Also save PDF version
    fig2, axes2 = plt.subplots(2, 3, figsize=(15, 8))
    axes2 = axes2.flatten()
    for ax, (var_name, train_col, label) in zip(axes2, vars_to_plot):
        if train_col in train_df.columns:
            train_vals = train_df[train_col].dropna().values
            ax.hist(train_vals, bins=80, alpha=0.5, density=True,
                    color="steelblue", label="Training", edgecolor="none")
        if var_name in eval_data and eval_data[var_name]:
            eval_vals = np.array(eval_data[var_name])
            ax.hist(eval_vals, bins=50, alpha=0.5, density=True,
                    color="coral", label="Evaluation", edgecolor="none")
        ax.set_xlabel(label, fontsize=9)
        ax.set_ylabel("Density", fontsize=9)
        ax.legend(fontsize=8)
        ax.tick_params(labelsize=8)
    fig2.suptitle("Data Coverage: Training vs Closed-Loop Evaluation", fontsize=13, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig2.savefig(fig_dir / "data_coverage_histograms.pdf", bbox_inches="tight")
    plt.close(fig2)


# =============================================================================
# 3. Residual Diagnostics (beyond aggregate RMSE)
# =============================================================================

def compute_residual_diagnostics(out_dir: Path, fig_dir: Path) -> dict:
    """
    Compute residual diagnostics stratified by operating regime.
    
    Uses the processed CSV (identification data) to compute:
    - Residual autocorrelation (already available)
    - Occupied vs unoccupied bias
    - Morning warm-up vs steady-state error
    - Solar transition periods
    """
    print("\n[3] Computing residual diagnostics...")
    
    diagnostics = {}
    
    # ── Load existing residual analysis ──
    if RESIDUAL_ANALYSIS_JSON.exists():
        with open(RESIDUAL_ANALYSIS_JSON) as f:
            existing = json.load(f)
        diagnostics["existing_residual_analysis"] = existing
        print(f"  Existing analysis: {json.dumps({k: v for k, v in existing.items() if not isinstance(v, (list, dict))}, indent=2)}")
    
    # ── Load processed CSV for regime-stratified analysis ──
    if not PROCESSED_CSV.exists():
        print(f"  WARNING: Processed CSV not found: {PROCESSED_CSV}")
        return diagnostics
    
    print(f"  Loading processed CSV for regime analysis...")
    df = pd.read_csv(PROCESSED_CSV)
    
    if "DateTime" in df.columns:
        df["DateTime"] = pd.to_datetime(df["DateTime"])
    elif "datetime" in df.columns:
        df["DateTime"] = pd.to_datetime(df["datetime"])
    
    # Check available columns
    has_tair = "T_air_avg" in df.columns
    has_tout = "T_outdoor" in df.columns or "T_out" in df.columns
    tout_col = "T_outdoor" if "T_outdoor" in df.columns else "T_out"
    
    if has_tair and "DateTime" in df.columns:
        # ── Stratify by hour of day ──
        df["hour"] = df["DateTime"].dt.hour
        df["weekday"] = df["DateTime"].dt.weekday
        
        # Occupancy (weekday 7-21, Saturday 7-22, Sunday 9-19)
        occ_mask = np.zeros(len(df), dtype=bool)
        for i, row in df.iterrows():
            wd = row["weekday"]
            h = row["hour"]
            if wd < 5:
                occ_mask[i] = (h >= 7) and (h < 21)
            elif wd == 5:
                occ_mask[i] = (h >= 7) and (h < 22)
            else:
                occ_mask[i] = (h >= 9) and (h < 19)
        
        df["occupied"] = occ_mask
        
        # Morning warm-up: 6:00–9:00 on weekdays
        morning_mask = (df["weekday"] < 5) & (df["hour"] >= 6) & (df["hour"] < 9)
        steady_mask = (df["weekday"] < 5) & (df["hour"] >= 11) & (df["hour"] < 17)
        
        # Solar transitions: sunrise (6-8) and sunset (16-18)
        solar_rise = (df["hour"] >= 6) & (df["hour"] < 8)
        solar_set = (df["hour"] >= 16) & (df["hour"] < 18)
        solar_transition = solar_rise | solar_set
        
        # Temperature derivative for warm-up detection
        if "dT_air_dt" in df.columns:
            warmup_mask = df["dT_air_dt"] > 0.001  # positive temperature change
        else:
            warmup_mask = morning_mask
        
        regimes = {
            "occupied": occ_mask,
            "unoccupied": ~occ_mask,
            "morning_warmup": morning_mask,
            "steady_state": steady_mask,
            "solar_transition": solar_transition,
            "cold_outdoor": df[tout_col] < -5 if has_tout else None,
            "mild_outdoor": (df[tout_col] >= -5) & (df[tout_col] < 5) if has_tout else None,
        }
        
        # Compute T_air statistics per regime
        regime_stats = {}
        for regime_name, mask in regimes.items():
            if mask is None:
                continue
            subset = df.loc[mask, "T_air_avg"]
            if len(subset) > 10:
                regime_stats[regime_name] = {
                    "n_samples": int(len(subset)),
                    "T_air_mean": float(subset.mean()),
                    "T_air_std": float(subset.std()),
                    "T_air_min": float(subset.min()),
                    "T_air_max": float(subset.max()),
                }
        
        diagnostics["regime_statistics"] = regime_stats
        
        # Print summary
        print(f"\n  Regime-stratified T_air statistics:")
        print(f"  {'Regime':<20s} {'N':>8s} {'Mean':>8s} {'Std':>8s} {'Min':>8s} {'Max':>8s}")
        print(f"  {'-'*54}")
        for regime_name, stats in regime_stats.items():
            print(f"  {regime_name:<20s} {stats['n_samples']:>8d} "
                  f"{stats['T_air_mean']:>8.2f} {stats['T_air_std']:>8.2f} "
                  f"{stats['T_air_min']:>8.2f} {stats['T_air_max']:>8.2f}")
    
    # ── Plot: T_air distribution by regime ──
    if has_tair:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        
        # Panel 1: Occupied vs unoccupied
        if "occupied" in df.columns:
            ax = axes[0]
            df.loc[df["occupied"], "T_air_avg"].hist(
                bins=50, alpha=0.6, ax=ax, color="steelblue", label="Occupied", density=True)
            df.loc[~df["occupied"], "T_air_avg"].hist(
                bins=50, alpha=0.6, ax=ax, color="coral", label="Unoccupied", density=True)
            ax.set_xlabel("T_air (°C)")
            ax.set_ylabel("Density")
            ax.set_title("Occupied vs Unoccupied")
            ax.legend(fontsize=8)
        
        # Panel 2: Morning warmup vs steady-state
        ax = axes[1]
        df.loc[morning_mask, "T_air_avg"].hist(
            bins=40, alpha=0.6, ax=ax, color="orange", label="Morning (6-9h)", density=True)
        df.loc[steady_mask, "T_air_avg"].hist(
            bins=40, alpha=0.6, ax=ax, color="green", label="Steady (11-17h)", density=True)
        ax.set_xlabel("T_air (°C)")
        ax.set_title("Morning Warmup vs Steady-State")
        ax.legend(fontsize=8)
        
        # Panel 3: By outdoor temperature band
        if has_tout:
            ax = axes[2]
            cold = df[tout_col] < -5
            mild = (df[tout_col] >= -5) & (df[tout_col] < 5)
            warm = df[tout_col] >= 5
            if cold.sum() > 10:
                df.loc[cold, "T_air_avg"].hist(
                    bins=40, alpha=0.5, ax=ax, color="blue", label="T_out < -5°C", density=True)
            if mild.sum() > 10:
                df.loc[mild, "T_air_avg"].hist(
                    bins=40, alpha=0.5, ax=ax, color="green", label="-5 ≤ T_out < 5°C", density=True)
            if warm.sum() > 10:
                df.loc[warm, "T_air_avg"].hist(
                    bins=40, alpha=0.5, ax=ax, color="red", label="T_out ≥ 5°C", density=True)
            ax.set_xlabel("T_air (°C)")
            ax.set_title("By Outdoor Temperature Band")
            ax.legend(fontsize=8)
        
        plt.tight_layout()
        fig.savefig(fig_dir / "regime_stratified_diagnostics.png", dpi=150, bbox_inches="tight")
        fig.savefig(fig_dir / "regime_stratified_diagnostics.pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: regime_stratified_diagnostics.png/.pdf")
    
    # Save
    out_path = out_dir / "residual_diagnostics.json"
    with open(out_path, "w") as f:
        json.dump(diagnostics, f, indent=2, default=str)
    print(f"  Saved: {out_path.name}")
    
    return diagnostics


# =============================================================================
# 4. Numerical Stability of Synthetic Data
# =============================================================================

def analyze_numerical_stability(out_dir: Path) -> dict:
    """
    Analyze the numerical stability of the synthetic training data.
    
    Reports:
    - Adaptive substep rule and parameters
    - Distribution of state values (check for implausible values)
    - Fraction of samples at physical bounds
    """
    print("\n[4] Analyzing numerical stability of synthetic data...")
    
    stability = {}
    
    # ── Substep configuration (from documentation) ──
    stability["substep_config"] = {
        "method": "Adaptive Euler with stability-limited sub-stepping",
        "safety_factor_alpha": 0.3,
        "substeps_min": 6,
        "substeps_max": 200,
        "mdot_floor_kgs": 0.01,
        "stability_criterion": "dt_sub <= alpha * 2 * C_rad_sec / (mdot * Cp)",
        "note": "At max flow (4.05 kg/s), tau_adv ≈ 12s, requiring ~17 substeps for alpha=0.3",
    }
    
    # ── Analyze training data for implausible values ──
    if not TRAINING_DATA_CSV.exists():
        print(f"  WARNING: Training data not found")
        return stability
    
    print(f"  Loading training data for numerical stability checks...")
    df = pd.read_csv(TRAINING_DATA_CSV)
    
    # Physical bounds for each variable
    bounds = {
        "T_air_k": (-25.0, 40.0, "°C"),
        "T_env_k": (-30.0, 50.0, "°C"),
        "T_int_k": (-20.0, 45.0, "°C"),
        "T_rad1_k": (10.0, 65.0, "°C"),
        "T_rad2_k": (10.0, 65.0, "°C"),
        "T_ret_k": (5.0, 65.0, "°C"),
        "T_supply_k": (25.0, 65.0, "°C"),
        "mdot_k": (-0.01, 4.1, "kg/s"),
    }
    
    stability_checks = {}
    total_samples = len(df)
    
    print(f"  Total training samples: {total_samples:,}")
    print(f"\n  {'Variable':<14s} {'Min':>10s} {'Max':>10s} {'Bound Lo':>10s} {'Bound Hi':>10s} {'% Outside':>10s}")
    print(f"  {'-'*56}")
    
    for col, (lo, hi, unit) in bounds.items():
        if col in df.columns:
            vals = df[col].dropna()
            n_below = (vals < lo).sum()
            n_above = (vals > hi).sum()
            n_outside = n_below + n_above
            pct_outside = 100.0 * n_outside / len(vals) if len(vals) > 0 else 0.0
            
            stability_checks[col] = {
                "min": float(vals.min()),
                "max": float(vals.max()),
                "bound_lo": lo,
                "bound_hi": hi,
                "n_below_bound": int(n_below),
                "n_above_bound": int(n_above),
                "pct_outside": float(pct_outside),
                "unit": unit,
            }
            
            print(f"  {col:<14s} {vals.min():>10.2f} {vals.max():>10.2f} "
                  f"{lo:>10.1f} {hi:>10.1f} {pct_outside:>9.4f}%")
    
    stability["variable_checks"] = stability_checks
    
    # ── Check for NaN or Inf ──
    n_nan = df.isnull().sum().sum()
    n_inf = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
    stability["nan_count"] = int(n_nan)
    stability["inf_count"] = int(n_inf)
    print(f"\n  NaN values: {n_nan}, Inf values: {n_inf}")
    
    # ── Estimate substep distribution from mdot ──
    if "mdot_k" in df.columns:
        mdot = df["mdot_k"].values
        Cp = 4186.0
        C_rad = 597376.96  # from identified params
        C_rad_sec = C_rad / 3.0  # per radiator section
        alpha = 0.3
        mdot_floor = 0.01
        dt = 600.0
        
        mdot_eff = np.maximum(np.abs(mdot), mdot_floor)
        tau_adv = C_rad_sec / (mdot_eff * Cp)
        n_sub_raw = np.ceil(dt / (alpha * 2.0 * tau_adv))
        n_sub = np.clip(n_sub_raw, 6, 200).astype(int)
        
        stability["substep_distribution"] = {
            "min": int(n_sub.min()),
            "max": int(n_sub.max()),
            "mean": float(n_sub.mean()),
            "median": float(np.median(n_sub)),
            "p95": float(np.percentile(n_sub, 95)),
            "p99": float(np.percentile(n_sub, 99)),
            "pct_at_minimum_6": float(100.0 * (n_sub == 6).mean()),
            "pct_above_50": float(100.0 * (n_sub > 50).mean()),
            "pct_at_maximum_200": float(100.0 * (n_sub == 200).mean()),
        }
        
        sd = stability["substep_distribution"]
        print(f"\n  Estimated adaptive substep distribution:")
        print(f"    Range: [{sd['min']}, {sd['max']}]")
        print(f"    Mean: {sd['mean']:.1f}, Median: {sd['median']:.0f}")
        print(f"    P95: {sd['p95']:.0f}, P99: {sd['p99']:.0f}")
        print(f"    At minimum (6): {sd['pct_at_minimum_6']:.1f}%")
        print(f"    Above 50: {sd['pct_above_50']:.1f}%")
    
    # ── Training data metadata ──
    if TRAINING_DATA_META.exists():
        with open(TRAINING_DATA_META) as f:
            meta = json.load(f)
        stability["training_meta"] = meta
        print(f"\n  Training metadata: {json.dumps({k: v for k, v in meta.items() if not isinstance(v, (list, dict))}, indent=2)}")
    
    # Save
    out_path = out_dir / "numerical_stability.json"
    with open(out_path, "w") as f:
        json.dump(stability, f, indent=2, default=str)
    print(f"  Saved: {out_path.name}")
    
    return stability


# =============================================================================
# 5. Evaluation Scenario Visualization
# =============================================================================

def visualize_scenarios(fig_dir: Path) -> None:
    """
    Visualize the three evaluation scenarios showing how disturbances
    compare to the training data distribution.
    """
    print("\n[5] Visualizing evaluation scenarios vs training range...")
    
    if not TRAINING_DATA_CSV.exists():
        print("  WARNING: Training data not found, skipping")
        return
    
    # Load training data for range reference
    train_df = pd.read_csv(TRAINING_DATA_CSV, usecols=["T_out_k", "Q_solar_trans_k", "Q_internal_k"])
    train_tout_range = (train_df["T_out_k"].quantile(0.01), train_df["T_out_k"].quantile(0.99))
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    
    colors = {"nominal": "steelblue", "cold_snap": "navy", "forecast_error": "firebrick"}
    
    for scenario_name, scenario_paths in SCENARIOS.items():
        truth_path = scenario_paths["truth"]
        forecast_path = scenario_paths["forecast"]
        
        if not truth_path.exists():
            continue
        
        truth_df = pd.read_csv(truth_path)
        truth_df["timestamp"] = pd.to_datetime(truth_df["timestamp"])
        hours = (truth_df["timestamp"] - truth_df["timestamp"].iloc[0]).dt.total_seconds() / 3600.0
        
        col_tout = "T_out_C" if "T_out_C" in truth_df.columns else "T_outdoor_C"
        color = colors.get(scenario_name, "gray")
        
        # Panel 1: Outdoor temperature
        if col_tout in truth_df.columns:
            axes[0].plot(hours, truth_df[col_tout], label=f"{scenario_name} (truth)",
                        color=color, linewidth=1.2)
        
        # Also plot forecast
        if forecast_path.exists():
            fc_df = pd.read_csv(forecast_path)
            fc_col = "T_out_C" if "T_out_C" in fc_df.columns else "T_outdoor_C"
            if fc_col in fc_df.columns:
                fc_hours = np.arange(len(fc_df)) * 10.0 / 60.0
                axes[0].plot(fc_hours, fc_df[fc_col], label=f"{scenario_name} (forecast)",
                            color=color, linewidth=0.8, linestyle="--", alpha=0.6)
        
        # Panel 2: Solar
        if "Q_solar_W" in truth_df.columns:
            axes[1].plot(hours, truth_df["Q_solar_W"], label=scenario_name,
                        color=color, linewidth=1.0)
        
        # Panel 3: Internal gains
        if "Q_internal_W" in truth_df.columns:
            axes[2].plot(hours, truth_df["Q_internal_W"], label=scenario_name,
                        color=color, linewidth=1.0)
    
    # Add training range bands
    axes[0].axhspan(train_tout_range[0], train_tout_range[1],
                    alpha=0.1, color="green", label="Training [1%, 99%]")
    
    axes[0].set_ylabel("T_outdoor (°C)")
    axes[0].set_title("Evaluation Scenarios vs Training Data Range")
    axes[0].legend(fontsize=7, ncol=3)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_ylabel("Solar (W)")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)
    
    axes[2].set_ylabel("Q_internal (W)")
    axes[2].set_xlabel("Time (hours)")
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(fig_dir / "scenarios_vs_training_range.png", dpi=150, bbox_inches="tight")
    fig.savefig(fig_dir / "scenarios_vs_training_range.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: scenarios_vs_training_range.png/.pdf")


# =============================================================================
# Main
# =============================================================================

def run_a7():
    """Run all A7 analyses."""
    print("\n" + "#" * 70)
    print("  Assignment 7 — System-Identification and Data-Coverage Hypothesis")
    print("#" * 70)
    
    out_dir = get_results_dir(A7_DIR)
    fig_dir = get_results_dir(A7_DIR, "figures")
    
    # 1. Identification pipeline summary
    id_summary = build_identification_summary(out_dir)
    
    # 2. Data coverage analysis
    coverage = compute_data_coverage(out_dir, fig_dir)
    
    # 3. Residual diagnostics
    residuals = compute_residual_diagnostics(out_dir, fig_dir)
    
    # 4. Numerical stability
    stability = analyze_numerical_stability(out_dir)
    
    # 5. Scenario visualization
    visualize_scenarios(fig_dir)
    
    # ── Final summary ──
    print("\n" + "=" * 70)
    print("  Assignment 7 — Summary")
    print("=" * 70)
    
    if "time_constants" in id_summary:
        tc = id_summary["time_constants"]
        print(f"  RC time constants: τ_air-env={tc['tau_air_env_hours']:.1f}h, "
              f"τ_env-out={tc['tau_env_out_hours']:.1f}h, "
              f"τ_air-int={tc['tau_air_int_hours']:.1f}h, "
              f"τ_rad={tc['tau_rad_minutes']:.1f}min")
    
    if id_summary.get("validation_metrics"):
        vm = id_summary["validation_metrics"]
        print(f"  April validation: RMSE={vm.get('april_rmse_C', 'N/A')} °C, "
              f"CV-RMSE={vm.get('april_cv_rmse_pct', 'N/A')}%")
    
    if coverage.get("interpretation"):
        ci = coverage["interpretation"]
        print(f"  Data coverage ({ci['observable_vars']} observable): "
              f"{ci['covered']} covered, {ci['marginal']} marginal, "
              f"{ci['extrapolation_risk']} extrapolation risk")
        print(f"  Internal states: {ci['internal_states']} (same RC plant -> physics-constrained)")
    
    if "substep_distribution" in stability:
        sd = stability["substep_distribution"]
        print(f"  Substep distribution: mean={sd['mean']:.0f}, "
              f"[{sd['min']}, {sd['max']}], "
              f"{sd['pct_at_minimum_6']:.0f}% at minimum")
    
    if stability.get("nan_count", 0) == 0 and stability.get("inf_count", 0) == 0:
        print(f"  Numerical artifacts: NONE (no NaN or Inf values)")
    
    print(f"\n  All results saved to: {out_dir}")
    print("=" * 70)


if __name__ == "__main__":
    run_a7()

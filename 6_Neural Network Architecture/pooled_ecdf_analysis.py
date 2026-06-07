# -*- coding: utf-8 -*-
"""
Pooled Tail ECDF Analysis — Standalone Post-Training Script
============================================================
Generates Figure 6 from the paper: empirical CDF of pooled induced one-step
stage costs under K perturbations on the test set.

Compares Raw MSE, Fidelity, and selected RAMC models.

Run AFTER training is complete:
    python pooled_ecdf_analysis.py

Configuration:
    - Update RESULTS_DIR to point to your trained model checkpoints
    - Update CSV_DATA_PATH to point to your training data CSV
    - Update MODELS_TO_COMPARE to select which models to include

Author: Nima Monghasemi
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt

from thermal_dynamics_net import (
    create_dataloaders,
    ThermalDynamicsNet,
    build_clamp_bounds_from_dataset,
)
from ramc_losses import (
    sample_gaussian_perturbations,
    stage_cost_ramc,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

# Update this path to the trained models directory
RESULTS_DIR = r"results\RAMC_FULL_cvar_20260212_211938"
CSV_DATA_PATH = "RAMC_training_data_N3.csv"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Output directory (defaults to RESULTS_DIR)
OUTPUT_DIR = None  # Set to a path, or None to use RESULTS_DIR

# Evaluation settings
EVAL_K = 256          # Number of perturbation samples (match paper)
CVAR_ALPHA = 0.9      # CVaR confidence level
MAX_BATCHES = None    # None = use all test batches; set integer to limit
SEED = 42

# Model architecture (must match training)
MODEL_CONFIG = {
    "state_dim": 6,
    "control_dim": 2,
    "disturbance_dim": 3,
    "hidden_dims": [256, 256, 128],
    "dropout_rate": 0.0,
}

# Data split (must match training)
DATA_CONFIG = {
    "batch_size": 2048,
    "train_split": 0.7,
    "val_split": 0.2,
    "test_split": 0.1,
}

# Perturbation configuration (must match training — RECALIBRATED)
PERTURBATION_CONFIG = {
    "sigma_state": 1.0,
    "sigma_rad_scale": 0.5,
    "sigma_T_supply": 0.5,
    "sigma_mdot": 0.01,
    "sigma_T_out": 1.0,
    "sigma_Q_solar": 500.0,
    "sigma_Q_internal": 5000.0,      # RECALIBRATED for building-level gains
    "clamp_physical": True,
    "p_occupancy_flip": 0.02,
    "q_internal_nominal": 40000.0,   # RECALIBRATED for building-level gains
    "use_antithetic": True,
}

# Stage cost configuration (must match training)
COST_CONFIG = {
    "comfort_bounds": (20.0, 22.0),
    "comfort_hinge": "softplus",
    "comfort_beta": 0.5,
    "t_ret_index": 5,
    "dt_minutes": 10.0,
    "energy_cost_rate": 0.9,
    "w_comfort": 63.0,
    "w_energy": 1.0,
    "cost_scale": 1.0,
}


# UPDATE: Include the extreme model to show the visual shift
MODELS_TO_COMPARE = [
    ("Raw_MSE_Baseline",              "Raw MSE"),
    ("Fidelity_Baseline",             "Fidelity (λ=0)"),
    ("RAMC_lambda_0.0002_op_cvar",    "RAMC (λ=2×10⁻⁴)"),
    ("RAMC_lambda_0.001_op_cvar",     "RAMC (λ=1×10⁻³)"),
    ("RAMC_lambda_0.005_op_cvar",     "RAMC (λ=5×10⁻³)"), # Keep this for visual evidence!
]

PLOT_STYLES = {
    "Raw MSE":           {"color": "#d62728", "linestyle": "-",  "linewidth": 2.5, "alpha": 0.7}, # Red
    "Fidelity (λ=0)":    {"color": "#1f77b4", "linestyle": "--", "linewidth": 2.0, "alpha": 0.9}, # Blue Dashed
    "RAMC (λ=2×10⁻⁴)":   {"color": "#ff7f0e", "linestyle": "-",  "linewidth": 2.0, "alpha": 1.0}, # Orange
    "RAMC (λ=1×10⁻³)":   {"color": "#2ca02c", "linestyle": "-",  "linewidth": 2.0, "alpha": 1.0}, # Green
    "RAMC (λ=5×10⁻³)":   {"color": "#9467bd", "linestyle": "-",  "linewidth": 2.0, "alpha": 1.0}, # Purple
}

# Default style for models not in PLOT_STYLES
DEFAULT_STYLE = {"color": "gray", "linestyle": "-", "linewidth": 1.5, "alpha": 0.8}


# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def load_model_from_checkpoint(
    checkpoint_path: str,
    model_config: dict,
    device: str,
) -> ThermalDynamicsNet:
    """Load a trained model from checkpoint with normalization restoration."""
    model = ThermalDynamicsNet(**model_config)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)

    if hasattr(model, "enable_normalization_if_stats_present"):
        model.enable_normalization_if_stats_present()

    model.eval()
    return model


@torch.no_grad()
def collect_pooled_costs(
    model: ThermalDynamicsNet,
    loader,
    K: int,
    perturbation_config: dict,
    cost_config: dict,
    clamp_bounds: dict,
    device: str,
    max_batches: Optional[int] = None,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Collect pooled induced costs under perturbations on a data loader.

    Returns:
        total_costs: [N_total] pooled total stage costs
        comfort_costs: [N_total] pooled comfort component
        energy_costs: [N_total] pooled energy component
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model.eval()

    all_total = []
    all_comfort = []
    all_energy = []

    for bi, batch in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break

        if isinstance(batch, (list, tuple)) and len(batch) == 6:
            states, controls, disturbances, targets, Tmin, Tmax = batch
            Tmin = Tmin.to(device) if Tmin is not None else None
            Tmax = Tmax.to(device) if Tmax is not None else None
        else:
            states, controls, disturbances, targets = batch[:4]
            Tmin, Tmax = None, None

        states = states.to(device)
        controls = controls.to(device)
        disturbances = disturbances.to(device)

        B = states.size(0)

        # Sample perturbations
        s_p, c_p, d_p = sample_gaussian_perturbations(
            states, controls, disturbances,
            num_perturbations=K,
            clamp_bounds=clamp_bounds,
            **perturbation_config,
        )

        # Flatten [B, K, dim] -> [B*K, dim]
        s_flat = s_p.reshape(B * K, -1)
        c_flat = c_p.reshape(B * K, -1)
        d_flat = d_p.reshape(B * K, -1)

        # Forward pass
        preds = model(s_flat, c_flat, d_flat)

        # Expand bounds
        Tmin_flat = Tmin.view(B, 1).expand(B, K).reshape(B * K) if Tmin is not None else None
        Tmax_flat = Tmax.view(B, 1).expand(B, K).reshape(B * K) if Tmax is not None else None

        # Compute stage costs with components
        costs, components = stage_cost_ramc(
            preds, c_flat,
            Tmin=Tmin_flat,
            Tmax=Tmax_flat,
            return_components=True,
            **cost_config,
        )

        all_total.append(costs.cpu().numpy())
        all_comfort.append(components["comfort"].cpu().numpy())
        all_energy.append(components["energy_cost"].cpu().numpy())

    total_costs = np.concatenate(all_total)
    comfort_costs = np.concatenate(all_comfort)
    energy_costs = np.concatenate(all_energy)

    return total_costs, comfort_costs, energy_costs


def compute_tail_statistics(
    costs: np.ndarray,
    alpha: float = 0.9,
) -> Dict[str, float]:
    """Compute tail statistics for a cost array."""
    costs = costs[np.isfinite(costs)]
    if len(costs) == 0:
        return {"mean": np.nan, "std": np.nan, "var": np.nan, "cvar": np.nan,
                "p50": np.nan, "p90": np.nan, "p95": np.nan, "p99": np.nan}

    var_alpha = float(np.quantile(costs, alpha))
    tail_mask = costs >= var_alpha
    cvar_alpha = float(np.mean(costs[tail_mask])) if tail_mask.any() else var_alpha

    return {
        "mean": float(np.mean(costs)),
        "std": float(np.std(costs)),
        "var": var_alpha,
        "cvar": cvar_alpha,
        "p50": float(np.median(costs)),
        "p90": float(np.quantile(costs, 0.90)),
        "p95": float(np.quantile(costs, 0.95)),
        "p99": float(np.quantile(costs, 0.99)),
    }


def plot_pooled_ecdf(
    costs_dict: Dict[str, np.ndarray],
    alpha: float = 0.9,
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (5.0, 3.5), # Slightly wider for inset
    title: Optional[str] = None,
):
    """
    Plot empirical CDF with a ZOOM INSET to show the tail difference.
    """
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    with plt.rc_context({"font.size": 9, "legend.fontsize": 7.5}):
        fig, ax = plt.subplots(figsize=figsize)

        # 1. Main Plot
        for label, costs in costs_dict.items():
            costs = np.asarray(costs).ravel()
            costs = costs[np.isfinite(costs)]
            if costs.size == 0: continue

            xs = np.sort(costs)
            ys = np.arange(1, xs.size + 1, dtype=float) / float(xs.size)
            style = PLOT_STYLES.get(label, DEFAULT_STYLE)
            ax.plot(xs, ys, label=label, **style)

        ax.set_xlabel("Induced one-step stage cost")
        ax.set_ylabel("Empirical CDF")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right")
        
        # Set X-limit to 99.5th percentile to avoid extreme outliers compressing the view
        all_c = np.concatenate([c.ravel() for c in costs_dict.values()])
        p995 = np.percentile(all_c, 99.5)
        ax.set_xlim(0, p995)

        # 2. Zoomed Inset (The "Fix")
        # Zoom in on x=[VaR_0.8, Max] and y=[0.85, 1.0]
        axins = inset_axes(ax, width="40%", height="35%", loc="center right", borderpad=2)
        
        for label, costs in costs_dict.items():
            costs = np.sort(costs[np.isfinite(costs)])
            ys = np.arange(1, costs.size + 1, dtype=float) / float(costs.size)
            style = PLOT_STYLES.get(label, DEFAULT_STYLE)
            
            # Only plot the tail in the inset to save rendering time
            tail_idx = int(0.8 * len(costs)) 
            axins.plot(costs[tail_idx:], ys[tail_idx:], **style)

        # Set inset limits to focus on the divergence
        p85 = np.percentile(all_c, 85)
        axins.set_xlim(p85, p995)
        axins.set_ylim(0.85, 1.01)
        axins.grid(True, alpha=0.2)
        axins.set_title("Tail Zoom (Top 15%)", fontsize=7)
        
        # Draw box to show where the zoom comes from
        from mpl_toolkits.axes_grid1.inset_locator import mark_inset
        mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5", alpha=0.5)

        if save_path:
            fig.savefig(save_path, dpi=600, bbox_inches="tight")
            print(f"  Saved ECDF with Zoom: {save_path}")

        plt.close(fig)


def plot_pooled_ecdf_decomposed(
    total_dict: Dict[str, np.ndarray],
    comfort_dict: Dict[str, np.ndarray],
    energy_dict: Dict[str, np.ndarray],
    alpha: float = 0.9,
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 3.2),
):
    """
    Three-panel ECDF: total, comfort, energy — for supplementary material.
    """
    with plt.rc_context({
        "font.size": 9,
        "axes.labelsize": 9,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
    }):
        fig, axes = plt.subplots(1, 3, figsize=figsize)

        panels = [
            (axes[0], total_dict, "Total stage cost", "(a)"),
            (axes[1], comfort_dict, "Comfort penalty", "(b)"),
            (axes[2], energy_dict, "Energy cost", "(c)"),
        ]

        for ax, costs_dict, xlabel, tag in panels:
            for label, costs in costs_dict.items():
                costs = np.asarray(costs).ravel()
                costs = costs[np.isfinite(costs)]
                if costs.size == 0:
                    continue
                xs = np.sort(costs)
                ys = np.arange(1, xs.size + 1, dtype=float) / float(xs.size)
                style = PLOT_STYLES.get(label, DEFAULT_STYLE)
                ax.plot(xs, ys, label=label, **style)

                var_alpha = float(np.quantile(costs, alpha))
                ax.axvline(var_alpha, color=style["color"], linestyle=":", linewidth=1.0, alpha=0.5)

            ax.axhline(alpha, color="gray", linestyle=":", linewidth=0.8, alpha=0.4)
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Empirical CDF")
            ax.grid(True, alpha=0.3)
            ax.text(0.03, 0.97, tag, transform=ax.transAxes, fontsize=11,
                    fontweight="bold", va="top", ha="left")

            all_c = np.concatenate([np.asarray(c).ravel() for c in costs_dict.values()])
            valid = all_c[np.isfinite(all_c)]
            if len(valid) > 0:
                ax.set_xlim(0, float(np.percentile(valid, 99.5)))

        # Single legend at the bottom
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=len(labels),
                   frameon=True, fontsize=8, bbox_to_anchor=(0.5, -0.02))

        fig.tight_layout(rect=[0, 0.06, 1, 1])

        if save_path:
            fig.savefig(save_path, dpi=600, bbox_inches="tight")
            pdf_path = save_path.rsplit(".", 1)[0] + ".pdf"
            fig.savefig(pdf_path, bbox_inches="tight")
            print(f"  Saved decomposed ECDF: {save_path}")

        plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================

def main():
    output_dir = OUTPUT_DIR or RESULTS_DIR
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print("POOLED TAIL ECDF ANALYSIS")
    print("=" * 80)
    print(f"Results dir : {RESULTS_DIR}")
    print(f"Data file   : {CSV_DATA_PATH}")
    print(f"Device      : {DEVICE}")
    print(f"K           : {EVAL_K}")
    print(f"CVaR α      : {CVAR_ALPHA}")
    print(f"Max batches : {MAX_BATCHES or 'all'}")
    print()

    # --- Load data ---
    print("Loading data...")
    train_loader, val_loader, test_loader, full_dataset, split_indices = create_dataloaders(
        csv_file_path=CSV_DATA_PATH,
        batch_size=DATA_CONFIG["batch_size"],
        train_split=DATA_CONFIG["train_split"],
        val_split=DATA_CONFIG["val_split"],
        test_split=DATA_CONFIG["test_split"],
        device=DEVICE,
        split_mode="time",
    )
    print(f"  Test samples: {len(split_indices['test'])}")

    # --- Build clamp bounds from TRAIN only ---
    clamp_bounds = build_clamp_bounds_from_dataset(
        full_dataset, qlo=0.001, qhi=0.999, indices=split_indices["train"],
    )
    clamp_bounds["dist"] = {
        "T_out": (-30.0, 40.0),
        "Q_solar": (0.0, 40000.0),
        "Q_internal": (0.0, 100000.0),
    }

    # --- Collect costs for each model ---
    print("\nCollecting pooled costs for each model...")

    total_costs_dict = {}
    comfort_costs_dict = {}
    energy_costs_dict = {}
    tail_stats_all = {}

    for ckpt_stem, display_name in MODELS_TO_COMPARE:
        ckpt_path = os.path.join(RESULTS_DIR, f"{ckpt_stem}_best.pth")
        if not os.path.exists(ckpt_path):
            print(f"  [SKIP] Checkpoint not found: {ckpt_path}")
            continue

        print(f"  Processing: {display_name} ...")
        model = load_model_from_checkpoint(ckpt_path, MODEL_CONFIG, DEVICE)

        total, comfort, energy = collect_pooled_costs(
            model=model,
            loader=test_loader,
            K=EVAL_K,
            perturbation_config=PERTURBATION_CONFIG,
            cost_config=COST_CONFIG,
            clamp_bounds=clamp_bounds,
            device=DEVICE,
            max_batches=MAX_BATCHES,
            seed=SEED,
        )

        total_costs_dict[display_name] = total
        comfort_costs_dict[display_name] = comfort
        energy_costs_dict[display_name] = energy

        # Compute tail stats
        stats = compute_tail_statistics(total, alpha=CVAR_ALPHA)
        stats["comfort_cvar"] = compute_tail_statistics(comfort, alpha=CVAR_ALPHA)["cvar"]
        stats["energy_cvar"] = compute_tail_statistics(energy, alpha=CVAR_ALPHA)["cvar"]
        tail_stats_all[display_name] = stats

        print(f"    n={len(total):,}  Mean={stats['mean']:.4f}  "
              f"VaR₀.₉={stats['var']:.4f}  CVaR₀.₉={stats['cvar']:.4f}")

    if len(total_costs_dict) < 2:
        print("\nERROR: Need at least 2 models for comparison. Check checkpoint paths.")
        return

    # --- Print summary table ---
    print("\n" + "-" * 80)
    print(f"POOLED TAIL STATISTICS (α={CVAR_ALPHA}, K={EVAL_K})")
    print("-" * 80)
    print(f"{'Model':<25} {'n samples':>12} {'Mean':>10} {'VaR₀.₉':>10} {'CVaR₀.₉':>10}")
    print("-" * 80)
    for name, stats in tail_stats_all.items():
        n = len(total_costs_dict[name])
        print(f"{name:<25} {n:>12,} {stats['mean']:>10.4f} {stats['var']:>10.4f} {stats['cvar']:>10.4f}")
    print("-" * 80)

    # Improvements vs fidelity baseline
    if "Fidelity (λ=0)" in tail_stats_all:
        base = tail_stats_all["Fidelity (λ=0)"]
        print(f"\nRelative to Fidelity (λ=0):")
        print(f"{'Model':<25} {'ΔMean':>12} {'ΔVaR₀.₉':>12} {'ΔCVaR₀.₉':>12}")
        print("-" * 65)
        for name, stats in tail_stats_all.items():
            if name == "Fidelity (λ=0)":
                continue
            d_mean = 100 * (stats["mean"] - base["mean"]) / abs(base["mean"])
            d_var = 100 * (stats["var"] - base["var"]) / abs(base["var"])
            d_cvar = 100 * (stats["cvar"] - base["cvar"]) / abs(base["cvar"])
            print(f"{name:<25} {d_mean:>+11.2f}% {d_var:>+11.2f}% {d_cvar:>+11.2f}%")
        print("-" * 65)

    # --- Generate plots ---
    print("\nGenerating plots...")

    # Main figure (Figure 6 in paper)
    plot_pooled_ecdf(
        total_costs_dict,
        alpha=CVAR_ALPHA,
        save_path=os.path.join(output_dir, "tail_ecdf_comparison_TEST.png"),
    )

    # Supplementary: decomposed ECDF
    plot_pooled_ecdf_decomposed(
        total_costs_dict, comfort_costs_dict, energy_costs_dict,
        alpha=CVAR_ALPHA,
        save_path=os.path.join(output_dir, "tail_ecdf_decomposed_TEST.png"),
    )

    # --- Save statistics to JSON ---
    json_output = {
        "generated": datetime.now().isoformat(),
        "config": {
            "eval_K": EVAL_K,
            "cvar_alpha": CVAR_ALPHA,
            "max_batches": MAX_BATCHES,
            "seed": SEED,
            "perturbation_config": PERTURBATION_CONFIG,
            "cost_config": {k: list(v) if isinstance(v, tuple) else v
                           for k, v in COST_CONFIG.items()},
        },
        "models": {},
    }
    for name, stats in tail_stats_all.items():
        json_output["models"][name] = {
            "n_samples": int(len(total_costs_dict[name])),
            **stats,
        }

    json_path = os.path.join(output_dir, "pooled_tail_statistics.json")
    with open(json_path, "w") as f:
        json.dump(json_output, f, indent=2)
    print(f"  Saved statistics: {json_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  1 12:26:02 2026

@author: nmi03
"""

# tail_diagnostics.py
import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from ramc_losses import sample_gaussian_perturbations, stage_cost_ramc

def plot_cost_ecdf_comparison(
    costs_dict,
    alpha=0.9,
    save_path=None,
    show=False,
    figsize=(4.0, 2.8),
):
    """
    ECDF plot that handles overlapping curves.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    with plt.rc_context({
        "font.size": 9,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
    }):
        fig, ax = plt.subplots(figsize=figsize)
        
        # Define colors and styles to distinguish overlapping curves
        styles = {
            "Raw MSE":           {"color": "#d62728", "linestyle": "-",  "linewidth": 2.5, "alpha": 0.7},
            "Fidelity (Î»=0)":   {"color": "#1f77b4", "linestyle": "--", "linewidth": 2.0, "alpha": 0.9},
            "RAMC (Î»=3Ã—10â»â´)": {"color": "#2ca02c", "linestyle": "-",  "linewidth": 2.0, "alpha": 1.0},
            "RAMC (Î»=5Ã—10â»â´)": {"color": "#ff7f0e", "linestyle": "-",  "linewidth": 2.0, "alpha": 1.0},
            "RAMC (Î»=10â»Â³)":   {"color": "#ff7f0e", "linestyle": "-",  "linewidth": 2.0, "alpha": 1.0},
            "RAMC (Î»=2Ã—10â»Â³)": {"color": "#9467bd", "linestyle": "-",  "linewidth": 2.0, "alpha": 1.0},
            "RAMC (Î»=5Ã—10â»Â³)": {"color": "#9467bd", "linestyle": "-",  "linewidth": 2.0, "alpha": 1.0},
        }
        
        var_values = {}

        for label, costs in costs_dict.items():
            costs = np.asarray(costs).reshape(-1)
            costs = costs[np.isfinite(costs)]
            if costs.size == 0:
                continue

            xs = np.sort(costs)
            ys = np.arange(1, xs.size + 1, dtype=float) / float(xs.size)

            # Get style
            style = styles.get(label, {"color": "gray", "linestyle": "-", "linewidth": 1.5, "alpha": 0.8})
            
            ax.plot(xs, ys, label=label, **style)

            # VaR marker
            var_alpha = np.quantile(costs, alpha)
            var_values[label] = var_alpha
            ax.axvline(
                var_alpha,
                color=style["color"],
                linestyle=":",
                linewidth=1.2,
                alpha=0.6,
            )

        # Alpha reference line
        ax.axhline(alpha, color="gray", linestyle=":", linewidth=0.8, alpha=0.4)

        ax.set_xlabel("Induced one-step stage cost")
        ax.set_ylabel("Empirical CDF")
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=True, loc="lower right")
        
        # Focus on interesting region
        all_costs = np.concatenate([np.asarray(c).reshape(-1) for c in costs_dict.values()])
        ax.set_xlim(0, np.percentile(all_costs, 99.5))

        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=600, bbox_inches="tight")
            print(f"[tail_diagnostics] Saved ECDF plot to: {save_path}")

        if show:
            plt.show()
        
        plt.close(fig)


@torch.no_grad()
def collect_cost_samples(
    model,
    loader,
    loss_config: dict,
    device: str,
    max_batches: int = 10,
    seed: int = 42,
):
    """
    Collect a large pool of induced one-step costs under perturbations on a loader.
    Returns a 1D numpy array of costs across (samples x K).
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model.eval()
    costs_all = []

    K = int(loss_config.get("num_perturbations", 32))

    # ── FIX 2a: Extract flag once before the loop so the override is
    # applied consistently to every batch without repeated dict lookups.
    ignore_bounds = loss_config.get("ignore_dataset_bounds", False)

    for bi, batch in enumerate(loader):
        if bi >= max_batches:
            break

        if isinstance(batch, (list, tuple)) and len(batch) == 6:
            states, controls, disturbances, targets, Tmin, Tmax = batch
            if Tmin is not None: Tmin = Tmin.to(device)
            if Tmax is not None: Tmax = Tmax.to(device)
        else:
            states, controls, disturbances, targets = batch[:4]
            Tmin, Tmax = None, None

        # ── FIX 2b: Honour ignore_dataset_bounds ─────────────────────────
        # stage_cost_ramc already handles Tmin/Tmax is None correctly —
        # it falls back to comfort_bounds from loss_config (20, 22) °C.
        if ignore_bounds:
            Tmin = None
            Tmax = None

        states = states.to(device)
        controls = controls.to(device)
        disturbances = disturbances.to(device)

        # Sample perturbations
        s_p, c_p, d_p = sample_gaussian_perturbations(
            states, controls, disturbances,
            num_perturbations=K,
            sigma_state=loss_config.get("sigma_state", 1.0),
            sigma_rad_scale=loss_config.get("sigma_rad_scale", 0.5),
            sigma_T_supply=loss_config.get("sigma_T_supply", 0.5),
            sigma_mdot=loss_config.get("sigma_mdot", 0.01),
            sigma_T_out=loss_config.get("sigma_T_out", 1.0),
            sigma_Q_solar=loss_config.get("sigma_Q_solar", 500.0),
            sigma_Q_internal=loss_config.get("sigma_Q_internal", 5000.0),
            clamp_physical=loss_config.get("clamp_physical", True),
            clamp_bounds=loss_config.get("clamp_bounds", None),
            p_occupancy_flip=loss_config.get("p_occupancy_flip", 0.0),
            q_internal_nominal=loss_config.get("q_internal_nominal", 40000.0),
            use_antithetic=loss_config.get("use_antithetic", True),
        )

        B = states.size(0)
        # Flatten: [B, K, dim] -> [B*K, dim]
        s_flat = s_p.reshape(B*K, -1)
        c_flat = c_p.reshape(B*K, -1)
        d_flat = d_p.reshape(B*K, -1)

        # Predict next state
        preds = model(s_flat, c_flat, d_flat)

        # Expand bounds if necessary
        Tmin_flat = Tmin.view(B, 1).expand(B, K).reshape(B * K) if Tmin is not None else None
        Tmax_flat = Tmax.view(B, 1).expand(B, K).reshape(B * K) if Tmax is not None else None

        # Compute cost
        costs, _ = stage_cost_ramc(
            preds, c_flat,
            Tmin=Tmin_flat,
            Tmax=Tmax_flat,
            comfort_bounds=loss_config.get("comfort_bounds", (20.0, 22.0)),
            comfort_hinge=loss_config.get("comfort_hinge", "softplus"),
            comfort_beta=loss_config.get("comfort_beta", 0.5),
            t_ret_index=loss_config.get("t_ret_index", 5),
            dt_minutes=loss_config.get("dt_minutes", 10.0),
            energy_cost_rate=loss_config.get("energy_cost_rate", 0.9),
            w_comfort=loss_config.get("w_comfort", 63.0),
            w_energy=loss_config.get("w_energy", 1.0),
            cost_scale=loss_config.get("cost_scale", 1.0),
            return_components=True,
        )

        costs_all.append(costs.detach().cpu().numpy())

    costs_all = np.concatenate(costs_all, axis=0)  # shape [N_total * K]
    return costs_all


def summarize_tail(costs: np.ndarray, alpha: float = 0.9):
    """Return mean, VaR_alpha, CVaR_alpha."""
    mean = float(np.mean(costs))
    var = float(np.quantile(costs, alpha))
    cvar = float(np.mean(costs[costs >= var]))  # empirical expected shortfall
    return mean, var, cvar


def plot_cost_tail_comparison(costs_dict, alpha=0.9, save_path=None):
    """
    Two-panel diagnostic: histogram + ECDF with VaR markers.
    
    costs_dict: {name: 1D array of costs}
    """
    # Define colors for consistent styling
    colors = {
        "Raw MSE": "#d62728",             # red
        "Fidelity (Î»=0)": "#1f77b4",      # blue
        "RAMC (Î»=10â»Â³)": "#ff7f0e",       # orange
        "RAMC (Î»=2Ã—10â»Â³)": "#2ca02c",     # green
    }
    
    plt.figure(figsize=(12, 5))

    # Histogram (log y helps visualize tails)
    plt.subplot(1, 2, 1)
    for name, arr in costs_dict.items():
        color = colors.get(name, None)
        plt.hist(arr, bins=80, alpha=0.5, density=True, label=name, log=True, color=color)
    plt.title("Induced Cost Distribution (Log Scale)")
    plt.xlabel("Cost")
    plt.ylabel("Log Density")
    plt.legend()
    plt.grid(alpha=0.3)

    # ECDF (Empirical Cumulative Distribution Function)
    plt.subplot(1, 2, 2)
    for name, arr in costs_dict.items():
        xs = np.sort(arr)
        ys = np.arange(1, len(xs) + 1) / len(xs)
        color = colors.get(name, None)
        plt.plot(xs, ys, label=name, linewidth=2, color=color)
        
        # Mark VaR
        var = np.quantile(arr, alpha)
        plt.axvline(var, linestyle="--", alpha=0.5, color=color if color else None)
        
    plt.axhline(alpha, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
    plt.title(f"ECDF (VaR at Î±={alpha})")
    plt.xlabel("Cost")
    plt.ylabel("CDF")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Tail diagnostic plot saved to: {save_path}")
    plt.show()
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  8 20:39:52 2026

@author: nmi03
"""

import os, json
import numpy as np
import matplotlib.pyplot as plt

RESULTS_DIR = r"RAMC_FULL_cvar\RAMC_FULL_cvar_20260211"
JSON_PATH = os.path.join(RESULTS_DIR, "rollout_comparison.json")

# Which RAMC lambda values to highlight (numeric values)
HIGHLIGHT_LAMBDAS = [3e-4, 5e-3]
HIGHLIGHT_COLORS = ["#2ca02c", "#9467bd"]  # match ECDF colors for consistency

USE_LOGY = True  # set False if you prefer linear

def parse_lambda(name: str):
    """Extract lambda value from model name."""
    if "RAMC" not in name:
        return None
    for token in ["=", "λ="]:
        if token in name:
            try:
                return float(name.split(token)[-1].strip())
            except:
                return None
    return None

def format_lambda(lam: float):
    if np.isclose(lam, 3e-4):
        return r"$\lambda=3\times10^{-4}$"
    if np.isclose(lam, 5e-3):
        return r"$\lambda=5\times10^{-3}$"
    # Generic fallback
    if lam >= 1e-3:
        exp = int(np.floor(np.log10(lam)))
        coeff = lam / (10**exp)
        return rf"$\lambda={coeff:.0f}\times10^{{{exp}}}$"
    return rf"$\lambda={lam:g}$"

with open(JSON_PATH, "r") as f:
    data = json.load(f)

# Helper: consistent ordering
def sort_key(name):
    if "Raw MSE" in name: return (-2, 0.0)
    if "Fidelity" in name: return (-1, 0.0)
    if "RAMC" in name:
        try:
            lam = float(name.split("=")[1].strip())
            return (0, lam)
        except:
            return (0, 999)
    return (1, 999)

names = sorted(list(data.keys()), key=sort_key)

H = data[names[0]]["horizon"]
steps = np.arange(1, H+1)
hours = steps * (10/60)  # 10-min timestep

plt.rcParams.update({
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.25,
})

fig, axes = plt.subplots(2, 1, figsize=(7.5, 6.0), sharex=True)

# --- Panel 1: T_air ---
ax = axes[0]
for name in names:
    y = np.array(data[name]["T_air_rmse_per_step"], dtype=float)

    if "Raw MSE" in name:
        ax.plot(hours, y, color="#d62728", lw=2.8, label="Raw MSE")
    elif "Fidelity" in name:
        ax.plot(hours, y, color="black", lw=2.6, ls="--", label="Fidelity (λ=0)")
    else:
        # Check if this RAMC model should be highlighted
        lam = parse_lambda(name)
        is_highlight = False
        color_idx = None
        
        if lam is not None:
            for i, target_lam in enumerate(HIGHLIGHT_LAMBDAS):
                if np.isclose(lam, target_lam):
                    is_highlight = True
                    color_idx = i
                    break
        
        if is_highlight:
            ax.plot(hours, y, color=HIGHLIGHT_COLORS[color_idx], lw=2.6, ls="-", 
                   label=f"RAMC {format_lambda(lam)}")
        else:
            # RAMC family (thin, faint)
            ax.plot(hours, y, color="#1f77b4", lw=1.0, alpha=0.25)

ax.set_ylabel(r"$\mathrm{RMSE}(\hat{T}_{\mathrm{air}}-T_{\mathrm{air}})\ [^\circ\mathrm{C}]$")
if USE_LOGY:
    ax.set_yscale("log")
ax.legend(ncols=2, frameon=True)

# --- Panel 2: T_ret ---
ax = axes[1]
for name in names:
    y = np.array(data[name]["T_ret_rmse_per_step"], dtype=float)

    if "Raw MSE" in name:
        ax.plot(hours, y, color="#d62728", lw=2.8, label="Raw MSE")
    elif "Fidelity" in name:
        ax.plot(hours, y, color="black", lw=2.6, ls="--", label="Fidelity (λ=0)")
    else:
        # Check if this RAMC model should be highlighted
        lam = parse_lambda(name)
        is_highlight = False
        color_idx = None
        
        if lam is not None:
            for i, target_lam in enumerate(HIGHLIGHT_LAMBDAS):
                if np.isclose(lam, target_lam):
                    is_highlight = True
                    color_idx = i
                    break
        
        if is_highlight:
            ax.plot(hours, y, color=HIGHLIGHT_COLORS[color_idx], lw=2.6, ls="-", 
                   label=f"RAMC {format_lambda(lam)}")
        else:
            ax.plot(hours, y, color="#1f77b4", lw=1.0, alpha=0.25)

ax.set_xlabel(r"Rollout time $h\Delta t$ [h], $\Delta t=10$ min")
ax.set_ylabel(r"$\mathrm{RMSE}(\hat{T}_{\mathrm{ret}}-T_{\mathrm{ret}})\ [^\circ\mathrm{C}]$")
if USE_LOGY:
    ax.set_yscale("log")

# Avoid duplicate legends
# ax.legend()  # omit; top legend is enough

plt.tight_layout()

out_png = os.path.join(RESULTS_DIR, "Fig6_rollout_consistency_2panel.png")
out_pdf = os.path.join(RESULTS_DIR, "Fig6_rollout_consistency_2panel.pdf")
plt.savefig(out_png, dpi=900, bbox_inches="tight")
plt.savefig(out_pdf, bbox_inches="tight")
print("Saved:", out_png)
print("Saved:", out_pdf)
plt.show()
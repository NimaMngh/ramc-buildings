# -*- coding: utf-8 -*-
"""
Created on Thu Jan  8 15:37:32 2026

@author: nmi03
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# === UPDATE THIS ===
RESULTS_DIR = r"RAMC_FULL_cvar\RAMC_FULL_cvar_20260211"
CSV_PATH = os.path.join(RESULTS_DIR, "decomposed_evaluation_full.csv")

# Set to the num_seeds used to create the decomposed CSV
N_SEEDS = 3

def ci95(std):
    return 1.96 * std / np.sqrt(N_SEEDS)

df = pd.read_csv(CSV_PATH)

# Identify models
df_mse = df[df["lambda"] < 0].copy()
df_fid = df[np.isclose(df["lambda"], 0.0)].copy()
df_ramc = df[df["lambda"] > 0].copy().sort_values("lambda")

assert len(df_fid) == 1, "Expected exactly one Fidelity baseline row (lambda=0)."
fid = df_fid.iloc[0]

# Convenience: means + CI
def col_mean_ci(prefix):
    return df_ramc[f"{prefix}_mean"].values, ci95(df_ramc[f"{prefix}_std"].values)

lam = df_ramc["lambda"].values

# Metrics
comfort_mean, comfort_ci = col_mean_ci("risk_comfort")
energy_mean, energy_ci   = col_mean_ci("risk_energy")
tair_mean, tair_ci       = col_mean_ci("t_air_rmse")

# Baselines
fid_comfort = float(fid["risk_comfort_mean"])
fid_energy  = float(fid["risk_energy_mean"])
fid_tair    = float(fid["t_air_rmse_mean"])

mse_comfort = float(df_mse["risk_comfort_mean"].iloc[0]) if len(df_mse) else None
mse_tair    = float(df_mse["t_air_rmse_mean"].iloc[0]) if len(df_mse) else None

# --- Plot ---
plt.rcParams.update({
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "legend.fontsize": 9
})

fig, axes = plt.subplots(2, 2, figsize=(10, 8)) 

# Helper to add panel tags (a), (b)... in the corner (standard IEEE)
def add_panel_tag(ax, text):
    ax.text(0.02, 0.96, text, transform=ax.transAxes, 
            fontsize=11, fontweight='bold', va='top', ha='left')

# (a) Comfort CVaR vs lambda
ax = axes[0, 0]
ax.errorbar(lam, comfort_mean, yerr=comfort_ci, fmt="o-", capsize=3, label="RAMC")
ax.set_xscale("log")
ax.set_xlabel(r"$\lambda$ (risk weight)")
ax.set_ylabel("Conditional CVaR₀.₉ (comfort cost)")
ax.axhline(fid_comfort, ls="--", lw=1.2, color="tab:orange", label=r"Fidelity ($\lambda=0$)")
if mse_comfort is not None:
    ax.axhline(mse_comfort, ls=":", lw=1.2, color="gray", label="Raw MSE")

# === CHANGE HERE: Moved to 'lower left' to avoid the curve ===
ax.legend(loc='lower left') 
add_panel_tag(ax, "(a)")

# (b) Decomposition (Relative change vs Fidelity)
ax = axes[0, 1]
comfort_rel = 100 * (comfort_mean - fid_comfort) / abs(fid_comfort)
energy_rel  = 100 * (energy_mean  - fid_energy)  / abs(fid_energy)

ax.plot(lam, comfort_rel, "o-", label=r"$\Delta\,\mathrm{Cond.\;CVaR}_{0.9}(c_{\mathrm{comfort}})$")
ax.plot(lam, energy_rel,  "s-", label=r"$\Delta\,\mathrm{Cond.\;CVaR}_{0.9}(c_{\mathrm{energy}})$")
ax.set_xscale("log")
ax.set_xlabel(r"$\lambda$ (risk weight)")
ax.set_ylabel("Change vs fidelity baseline (%)")
ax.axhline(0, color="k", lw=1, alpha=0.6)
ax.legend(loc='best')
add_panel_tag(ax, "(b)")

# (c) Pareto: comfort risk vs T_air RMSE
ax = axes[1, 0]
ax.errorbar(tair_mean, comfort_mean, xerr=tair_ci, yerr=comfort_ci,
            fmt="o", capsize=3, alpha=0.9, label="RAMC")

# Baseline markers
ax.scatter([fid_tair], [fid_comfort], marker="^", s=90, color="tab:orange", label=r"Fidelity ($\lambda=0$)")
if mse_tair is not None and mse_comfort is not None:
    ax.scatter([mse_tair], [mse_comfort], marker="s", s=90, color="gray", label="Raw MSE")

ax.set_xlabel("T_air one-step RMSE (°C)")
ax.set_ylabel("Conditional CVaR₀.₉ (comfort cost)")
ax.legend(loc='best')
add_panel_tag(ax, "(c)")

# (d) T_air RMSE vs lambda
ax = axes[1, 1]
ax.errorbar(lam, tair_mean, yerr=tair_ci, fmt="o-", capsize=3, label="RAMC")
ax.set_xscale("log")
ax.set_xlabel(r"$\lambda$ (risk weight)")
ax.set_ylabel("T_air one-step RMSE (°C)")
ax.axhline(fid_tair, ls="--", lw=1.2, color="tab:orange", label=r"Fidelity ($\lambda=0$)")
if mse_tair is not None:
    ax.axhline(mse_tair, ls=":", lw=1.2, color="gray", label="Raw MSE")
ax.legend(loc='best')
add_panel_tag(ax, "(d)")

plt.tight_layout()

out_png = os.path.join(RESULTS_DIR, "Fig4_openloop_2x2_comfort_IEEE.png")
out_pdf = os.path.join(RESULTS_DIR, "Fig4_openloop_2x2_comfort_IEEE.pdf")
plt.savefig(out_png, dpi=900, bbox_inches="tight")
plt.savefig(out_pdf, bbox_inches="tight")
print("Saved:", out_png)
print("Saved:", out_pdf)
plt.show()
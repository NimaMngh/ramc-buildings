# -*- coding: utf-8 -*-
"""
Fig4 — One-step relative decomposition (single panel)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_DIR = r"results\RAMC_FULL_cvar_20260212_211938"
CSV_PATH = os.path.join(RESULTS_DIR, "decomposed_evaluation_full.csv")
N_SEEDS = 3

def ci95(std):
    return 1.96 * std / np.sqrt(N_SEEDS)

df = pd.read_csv(CSV_PATH)
df_fid  = df[np.isclose(df["lambda"], 0.0)].copy()
df_ramc = df[df["lambda"] > 0].copy().sort_values("lambda")
fid = df_fid.iloc[0]

lam = df_ramc["lambda"].values
comfort_mean = df_ramc["risk_comfort_mean"].values
energy_mean  = df_ramc["risk_energy_mean"].values
fid_comfort  = float(fid["risk_comfort_mean"])
fid_energy   = float(fid["risk_energy_mean"])

comfort_rel = 100 * (comfort_mean - fid_comfort) / abs(fid_comfort)
energy_rel  = 100 * (energy_mean  - fid_energy)  / abs(fid_energy)

plt.rcParams.update({
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "font.family": "serif",
})

fig, ax = plt.subplots(figsize=(4.5, 3.2))

ax.plot(lam, comfort_rel, "o-", markersize=5, color="tab:blue",
        label=r"$\Delta\,$Cond. CVaR$_{0.9}(c_{\mathrm{comfort}})$")
ax.plot(lam, energy_rel,  "s-", markersize=5, color="tab:orange",
        label=r"$\Delta\,$Cond. CVaR$_{0.9}(c_{\mathrm{energy}})$")

ax.set_xscale("log")
ax.set_xlabel(r"$\lambda$ (risk weight)")
ax.set_ylabel("Change vs fidelity baseline (%)")
ax.axhline(0, color="k", lw=1, alpha=0.6)

ax.legend(loc="lower left", framealpha=0.9, fontsize=9)

fig.tight_layout()

out_png = os.path.join(RESULTS_DIR, "Fig4_decomposition_single_IEEE.png")
out_pdf = os.path.join(RESULTS_DIR, "Fig4_decomposition_single_IEEE.pdf")
fig.savefig(out_png, dpi=900, bbox_inches="tight")
fig.savefig(out_pdf, bbox_inches="tight")
print("Saved:", out_png)
print("Saved:", out_pdf)
plt.show()

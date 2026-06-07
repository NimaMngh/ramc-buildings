# -*- coding: utf-8 -*-
"""
Fig5 — Rollout time-series (2-panel) — Revised highlights
==========================================================
Changes:
  - Highlighted models: λ=2e-4 (best T_ret) and λ=1e-3 (best no-regret)
  - RESTORED: Original legend positioning (centered between plots)
  - RESTORED: Original frame/style for legend
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt

# UPDATE THIS PATH TO MATCH YOUR RESULTS FOLDER IF NEEDED
RESULTS_DIR = r"results\RAMC_FULL_cvar_20260212_211938"
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
JSON_PATH = os.path.join(RESULTS_DIR, "rollout_comparison.json")

# --- EXPERT FEEDBACK UPDATES ---
# Highlight λ=2e-4 (Best T_ret) and λ=1e-3 (Best No-Regret)
HIGHLIGHT_LAMBDAS = [2e-4, 1e-3]
HIGHLIGHT_COLORS  = ["#ff7f0e", "#2ca02c"] # Orange, Green
HIGHLIGHT_LABELS  = [
    r"RAMC $\lambda=2\times10^{-4}$",
    r"RAMC $\lambda=10^{-3}$",
]

USE_LOGY = True 

def parse_lambda(name: str):
    if "RAMC" not in name:
        return None
    for token in ["λ=", "="]:
        if token in name:
            try:
                return float(name.split(token)[-1].strip())
            except:
                continue
    return None

# --- Mock Data Loading if file doesn't exist (for reproducibility) ---
if not os.path.exists(JSON_PATH):
    print("Warning: JSON not found, using mock data for visualization.")
    steps = np.arange(1, 25)
    data = {}
    names_mock = ["Raw MSE", "Fidelity", "RAMC λ=2e-4", "RAMC λ=1e-3", "RAMC λ=1e-5"]
    for n in names_mock:
        data[n] = {
            "horizon": 24,
            "T_air_rmse_per_step": np.exp(-0.1 * steps) + np.random.rand(24)*0.1,
            "T_ret_rmse_per_step": np.exp(-0.15 * steps) + np.random.rand(24)*0.1
        }
else:
    with open(JSON_PATH, "r") as f:
        data = json.load(f)

def sort_key(name):
    if "Raw MSE" in name: return (-2, 0.0)
    if "Fidelity" in name: return (-1, 0.0)
    if "RAMC" in name:
        try:
            lam = float(name.split("=")[-1].strip())
            return (0, lam)
        except:
            return (0, 999)
    return (1, 999)

names = sorted(list(data.keys()), key=sort_key)

if names:
    H = data[names[0]]["horizon"]
    steps = np.arange(1, H + 1)
    hours = steps * (10 / 60)
else:
    H = 24
    steps = np.arange(1, 25)
    hours = steps * (10/60)

plt.rcParams.update({
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "font.family": "serif",
})

fig, axes = plt.subplots(2, 1, figsize=(7.5, 6.0), sharex=True)

# Store final T_ret values for annotation
fid_tret_final = None
highlight_tret_finals = {}

for panel_idx, (ax, state_key, ylabel) in enumerate([
    (axes[0], "T_air_rmse_per_step",
     r"RMSE$(\hat{T}_{\mathrm{air}} - T_{\mathrm{air}})$ [°C]"),
    (axes[1], "T_ret_rmse_per_step",
     r"RMSE$(\hat{T}_{\mathrm{ret}} - T_{\mathrm{ret}})$ [°C]"),
]):
    # First pass: draw faint RAMC curves (background)
    for name in names:
        if name not in data: continue
        lam = parse_lambda(name)
        if lam is None or lam <= 0:
            continue
        # Skip highlighted models in this pass
        is_highlight = any(np.isclose(lam, hl, atol=1e-6) for hl in HIGHLIGHT_LAMBDAS)
        if is_highlight:
            continue
        y = np.array(data[name][state_key], dtype=float)
        ax.plot(hours, y, color="#1f77b4", lw=0.8, alpha=0.2)

    # Second pass: draw baselines and highlights (foreground)
    for name in names:
        if name not in data: continue
        y = np.array(data[name][state_key], dtype=float)

        if "Raw MSE" in name:
            ax.plot(hours, y, color="#d62728", lw=2.8, label="Raw MSE",
                    zorder=5)

        elif "Fidelity" in name:
            ax.plot(hours, y, color="black", lw=2.6, ls="--",
                    label=r"Fidelity ($\lambda\!=\!0$)", zorder=5)
            if panel_idx == 1:
                fid_tret_final = y[-1]

        else:
            lam = parse_lambda(name)
            if lam is None:
                continue
            for hi, target_lam in enumerate(HIGHLIGHT_LAMBDAS):
                if np.isclose(lam, target_lam, atol=1e-6):
                    ax.plot(hours, y, color=HIGHLIGHT_COLORS[hi], lw=2.6,
                            label=HIGHLIGHT_LABELS[hi], zorder=6)
                    if panel_idx == 1:
                        highlight_tret_finals[target_lam] = y[-1]
                    break

    ax.set_ylabel(ylabel)
    if USE_LOGY:
        ax.set_yscale("log")

# Add faint curve label dummy (just once, as context explanation)
axes[0].plot([], [], color="#1f77b4", lw=1.0, alpha=0.35,
             label="Other RAMC models")

# --- LEGEND MODIFICATION START (RESTORED ORIGINAL) ---

# 1. Apply tight_layout first to organize the plot elements
plt.tight_layout()

# 1. Increase vertical space to make room for the legend
plt.subplots_adjust(hspace=0.5) 

# 2. Create the shared legend
# Nudge the y-anchor up to 0.53 to pull it away from the lower plot
handles, labels = axes[0].get_legend_handles_labels()
# Filter duplicates if needed (though original code didn't need to)
by_label = dict(zip(labels, handles))
fig.legend(by_label.values(), by_label.keys(), loc='center', bbox_to_anchor=(0.5, 0.53),
           ncols=2, frameon=True, fontsize=10, framealpha=0.95,
           edgecolor='gray')

# --- LEGEND MODIFICATION END ---


# --- Annotation Logic for Bottom Panel ---
ax_bot = axes[1]

# Sort highlights by their final values for better positioning
sorted_highlights = sorted(
    [(lam, highlight_tret_finals.get(lam), color) 
     for lam, color in zip(HIGHLIGHT_LAMBDAS, HIGHLIGHT_COLORS) 
     if lam in highlight_tret_finals],
    key=lambda x: x[1] if x[1] else 0
)

# Vertical positions for annotations
anno_y_positions = [0.88, 1.05] 

for idx, (target_lam, ramc_val, color) in enumerate(sorted_highlights):
    if ramc_val is not None and fid_tret_final is not None:
        pct = (fid_tret_final - ramc_val) / fid_tret_final * 100

        if pct > 0:
            # Safe check index for position
            pos_idx = idx if idx < len(anno_y_positions) else 0
            ax_bot.annotate(
                f"−{pct:.0f}%",
                xy=(hours[-1], ramc_val),
                xytext=(hours[-1] + 0.25, ramc_val * anno_y_positions[pos_idx]),
                fontsize=9.5, fontweight="bold", color=color,
                arrowprops=dict(arrowstyle="->", color=color, lw=1.4,
                              connectionstyle="arc3,rad=0.2"),
                bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                          edgecolor=color, linewidth=1.5, alpha=0.95),
                ha='left', va='center'
            )

if fid_tret_final is not None:
    ax_bot.text(
        hours[-1] + 0.05, fid_tret_final, "ref",
        fontsize=8.5, color="black", va="center", ha="left",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                  edgecolor="black", linewidth=0.8, alpha=0.7)
    )

axes[1].set_xlabel(r"Rollout time $h\Delta t$ [h], $\Delta t = 10$ min")

# Extend x-axis slightly to accommodate annotations
axes[1].set_xlim(0, hours[-1] * 1.15)

out_png = os.path.join(RESULTS_DIR, "Fig5_rollout_2panel_IEEE_Revised.png")
out_pdf = os.path.join(RESULTS_DIR, "Fig5_rollout_2panel_IEEE_Revised.pdf")

# Use bbox_inches='tight' to ensure the external legend isn't cropped
plt.savefig(out_png, dpi=900, bbox_inches="tight")
plt.savefig(out_pdf, bbox_inches="tight")
print("Saved:", out_png)
print("Saved:", out_pdf)
plt.show()
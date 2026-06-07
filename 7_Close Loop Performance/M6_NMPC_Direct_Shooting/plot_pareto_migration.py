#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_pareto_migration.py
========================

Unified single-panel Pareto energy–comfort figure with migration arrows.

For each model, an arrow path connects:
  nominal (●) -> forecast error (■) -> cold snap (▲)

The arrow direction and length show how each model's operating point
shifts under increasing weather stress.

Place this file in:
  7_Close Loop Performance/M6_NMPC_Direct_Shooting/

It reads:
  - results_NMPC_20260307_185109/matrix/aggregate_metrics.csv   (Phase3 models)
  - ../../8_Assignments/A2_benchmark_sufficiency/results/benchmark_summary.json

It saves to:
  - results_NMPC_20260307_185109/figures_final/pareto_energy_comfort_unified.{pdf,png}

Author: Nima (MDU Future Energy Center)
Date: 2026-03-18
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from pathlib import Path


# =============================================================================
# Paths — adjust if your directory layout differs
# =============================================================================
SCRIPT_DIR   = Path(__file__).resolve().parent
RESULTS_DIR  = SCRIPT_DIR / "results_NMPC_20260307_185109"
MATRIX_DIR   = RESULTS_DIR / "matrix"
OUTPUT_DIR   = RESULTS_DIR / "figures_final"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Phase3 aggregate metrics (4 neural models × 3 scenarios × 5 seeds)
PHASE3_CSV   = MATRIX_DIR / "aggregate_metrics.csv"

# A2 benchmark summary (margins + RC exact, 3 seeds each)
BENCH_JSON   = (SCRIPT_DIR.parent.parent
                / "8_Assignments" / "A2_benchmark_sufficiency"
                / "results" / "benchmark_summary.json")

# Fallback: if the relative path doesn't work, try same directory
if not BENCH_JSON.exists():
    BENCH_JSON = SCRIPT_DIR / "benchmark_summary.json"
if not BENCH_JSON.exists():
    raise FileNotFoundError(
        f"Cannot find benchmark_summary.json.\n"
        f"  Tried: {SCRIPT_DIR.parent.parent / '8_Assignments' / 'A2_benchmark_sufficiency' / 'results' / 'benchmark_summary.json'}\n"
        f"  Tried: {SCRIPT_DIR / 'benchmark_summary.json'}\n"
        f"  Copy it next to this script or fix the BENCH_JSON path above."
    )


# =============================================================================
# Load data
# =============================================================================
print(f"Loading Phase3 data:  {PHASE3_CSV}")
phase3_agg = pd.read_csv(PHASE3_CSV)

print(f"Loading benchmarks:   {BENCH_JSON}")
with open(BENCH_JSON) as f:
    bench_summary = json.load(f)

# Build unified dataframe
rows = []
for _, r in phase3_agg.iterrows():
    rows.append({
        'model':       r['model'],
        'scenario':    r['scenario'],
        'energy_mean': r['total_energy_kWh_mean'],
        'cvar90_mean': r['cvar90_cold_occ_C_mean'],
    })
for entry in bench_summary:
    rows.append({
        'model':       entry['model_label'],
        'scenario':    entry['scenario'],
        'energy_mean': entry['total_energy_kWh_mean'],
        'cvar90_mean': entry['cvar90_cold_occ_C_mean'],
    })
df = pd.DataFrame(rows)
print(f"Unified data: {len(df)} rows, {df['model'].nunique()} models")


# =============================================================================
# Visual configuration
# =============================================================================
MODEL_COLORS = {
    'Fidelity_Baseline_rollout':  '#2166ac',
    'RAMC_lambda_0.0001_rollout': '#e67e22',
    'RAMC_lambda_0.0015_rollout': '#c0392b',
    'Fidelity_margin_0.3':        '#2ecc71',
    'Fidelity_margin_0.5':        '#00796b',
    'Fidelity_margin_1.0':        '#004d40',
    'RC_Exact':                   '#7f8c8d',
}

MODEL_LABELS = {
    'Fidelity_Baseline_rollout':  'Fidelity baseline ($\\lambda$=0)',
    'RAMC_lambda_0.0001_rollout': 'RAMC $\\lambda$=10$^{-4}$',
    'RAMC_lambda_0.0015_rollout': 'RAMC $\\lambda$=1.5\u00d710$^{-3}$',
    'Fidelity_margin_0.3':        'Margin +0.3\u2009\u00b0C',
    'Fidelity_margin_0.5':        'Margin +0.5\u2009\u00b0C',
    'Fidelity_margin_1.0':        'Margin +1.0\u2009\u00b0C',
    'RC_Exact':                   'RC exact model',
}

# Scenario path order: increasing stress
SCENARIO_PATH = ['nominal', 'forecast_error', 'cold_snap']
SCENARIO_MARKERS = {
    'nominal':        'o',
    'cold_snap':      '^',
    'forecast_error': 's',
}
SCENARIO_LABELS = {
    'nominal':        'Nominal',
    'cold_snap':      'Cold snap',
    'forecast_error': 'Forecast error',
}

STABLE_MODELS = [
    'Fidelity_Baseline_rollout',
    'RAMC_lambda_0.0001_rollout',
    'RAMC_lambda_0.0015_rollout',
    'RC_Exact',
]


# =============================================================================
# Helper
# =============================================================================
def get_pt(model, scenario):
    sub = df[(df['model'] == model) & (df['scenario'] == scenario)]
    if sub.empty:
        return None
    return (sub['energy_mean'].values[0], sub['cvar90_mean'].values[0])


# =============================================================================
# Build figure  –  LaTeX-friendly sizing & font control
# =============================================================================
FONTSIZE_AXIS_LABEL  = 13
FONTSIZE_TICK        = 11
FONTSIZE_LEGEND      = 9.5
FONTSIZE_LEGEND_TITLE = 10.5

fig, ax = plt.subplots(figsize=(10, 6.5))

# Global tick size
ax.tick_params(axis='both', which='major', labelsize=FONTSIZE_TICK)

# --- Stable models: arrows + markers ---
for model in STABLE_MODELS:
    pts = [get_pt(model, sc) for sc in SCENARIO_PATH]
    if any(p is None for p in pts):
        continue

    col = MODEL_COLORS[model]

    # Arrows between consecutive scenarios
    for i in range(len(pts) - 1):
        x0, y0 = pts[i]
        x1, y1 = pts[i + 1]
        arrow = FancyArrowPatch(
            (x0, y0), (x1, y1),
            arrowstyle='->', mutation_scale=14,
            color=col, linewidth=1.8, alpha=0.65,
            connectionstyle='arc3,rad=0.0',
            zorder=3,
        )
        ax.add_patch(arrow)

    # Markers at each scenario
    for i, sc in enumerate(SCENARIO_PATH):
        x, y = pts[i]
        mk = SCENARIO_MARKERS[sc]
        ms = 110 if sc == 'cold_snap' else (90 if sc == 'forecast_error' else 80)
        ax.scatter(x, y, marker=mk, s=ms, color=col,
                   edgecolors='black', linewidths=0.9, zorder=6, alpha=0.95)


# =============================================================================
# Axes
# =============================================================================
ax.set_xlabel('Total heating energy (kWh)', fontsize=FONTSIZE_AXIS_LABEL)
ax.set_ylabel('CVaR$_{0.9}$ cold violation (\u00b0C)', fontsize=FONTSIZE_AXIS_LABEL)
ax.grid(True, alpha=0.15, zorder=0)
ax.set_xlim(3300, 6700)
ax.set_ylim(-0.03, 0.82)


# =============================================================================
# Legend
# =============================================================================
# Model legend (color + arrow line)
model_handles = []
for model in STABLE_MODELS:
    model_handles.append(Line2D([0], [0], marker='o', color=MODEL_COLORS[model],
                                markerfacecolor=MODEL_COLORS[model],
                                markeredgecolor='black', markersize=8,
                                linewidth=1.8, alpha=0.7,
                                label=MODEL_LABELS[model]))

# Scenario legend (shape)
scenario_handles = [
    Line2D([0], [0], marker=SCENARIO_MARKERS[sc], color='w',
           markerfacecolor='#555', markeredgecolor='black', markersize=10,
           label=SCENARIO_LABELS[sc])
    for sc in SCENARIO_PATH
]

leg1 = ax.legend(handles=model_handles, loc='upper right',
                 fontsize=FONTSIZE_LEGEND, framealpha=0.92, edgecolor='#ccc',
                 title='Model', title_fontsize=FONTSIZE_LEGEND_TITLE,
                 handletextpad=0.5, borderpad=0.6)
ax.add_artist(leg1)
ax.legend(handles=scenario_handles, loc='lower right',
          fontsize=FONTSIZE_LEGEND, framealpha=0.92, edgecolor='#ccc',
          title='Scenario (\u25cf\u2192\u25a0\u2192\u25b2)',
          title_fontsize=FONTSIZE_LEGEND_TITLE,
          handletextpad=0.5, borderpad=0.6)


# =============================================================================
# Save
# =============================================================================
for ext in ['pdf', 'png']:
    out = OUTPUT_DIR / f"pareto_energy_comfort_unified.{ext}"
    fig.savefig(str(out), dpi=300, bbox_inches='tight')
    print(f"  Saved: {out}")

plt.close(fig)
print("\nDone.")

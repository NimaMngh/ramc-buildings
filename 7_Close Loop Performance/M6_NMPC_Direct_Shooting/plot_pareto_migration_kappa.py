#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_pareto_migration_kappa.py
==============================
[KAPPA VERSION — v2]

Pareto energy–comfort migration figure, sized for the MDU thesis template.

Text area:  113 mm × 199 mm  (≈ 4.45 × 7.83 in)
  -> figure width  = 4.45 in  (full \textwidth)

Font: Helvetica (matches \helvet / \sffamily in the thesis template).
Legends placed BELOW the axes to avoid overlap with data.

Outputs -> figures_kappa/
  pareto_energy_comfort_unified_kappa.pdf
  pareto_energy_comfort_unified_kappa.png

Author : Nima (MDU Future Energy Center)
Tag    : KAPPA
Date   : 2026-04-01
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

# ═══════════════════════════════════════════════════════════════════════
# Kappa thesis page geometry
# ═══════════════════════════════════════════════════════════════════════
# stock: 239 mm × 169 mm
# margins: left 25 mm + right 25 mm + binding 6 mm = 56 mm horizontal
# text width  = 169 - 56 = 113 mm ≈ 4.449 in
# text height = 239 - 40 = 199 mm ≈ 7.835 in
TEXTWIDTH_IN  = 113.0 / 25.4
TEXTHEIGHT_IN = 199.0 / 25.4

FIG_WIDTH  = TEXTWIDTH_IN
# Slightly taller to accommodate the legend strip below the axes
FIG_HEIGHT = FIG_WIDTH / 1.18          # ≈ 3.77 in — axes + bottom legends

# ═══════════════════════════════════════════════════════════════════════
# Font sizes — Helvetica, tuned for 113 mm text width at 11 pt body
# ═══════════════════════════════════════════════════════════════════════
FONTSIZE_AXIS_LABEL   = 9
FONTSIZE_TICK         = 7.5
FONTSIZE_LEGEND       = 6.5
FONTSIZE_LEGEND_TITLE = 7.5

matplotlib.rcParams.update({
    'font.family':      'sans-serif',
    'font.sans-serif':  ['Helvetica', 'Arial', 'DejaVu Sans'],
    'mathtext.fontset': 'custom',
    'mathtext.rm':      'Helvetica',
    'mathtext.it':      'Helvetica:italic',
    'mathtext.bf':      'Helvetica:bold',
    'axes.unicode_minus': False,
    'pdf.fonttype':     42,
    'ps.fonttype':      42,
    'savefig.dpi':      300,
    'savefig.bbox':     'tight',
    'savefig.pad_inches': 0.02,
})


# ═══════════════════════════════════════════════════════════════════════
# Paths
# ═══════════════════════════════════════════════════════════════════════
SCRIPT_DIR   = Path(__file__).resolve().parent
RESULTS_DIR  = SCRIPT_DIR / "results_NMPC_20260307_185109"
MATRIX_DIR   = RESULTS_DIR / "matrix"
OUTPUT_DIR   = RESULTS_DIR / "figures_kappa"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PHASE3_CSV = MATRIX_DIR / "aggregate_metrics.csv"

BENCH_JSON = (SCRIPT_DIR.parent.parent
              / "8_Assignments" / "A2_benchmark_sufficiency"
              / "results" / "benchmark_summary.json")
if not BENCH_JSON.exists():
    BENCH_JSON = SCRIPT_DIR / "benchmark_summary.json"
if not BENCH_JSON.exists():
    raise FileNotFoundError(
        f"Cannot find benchmark_summary.json.\n"
        f"  Copy it next to this script or fix the BENCH_JSON path."
    )


# ═══════════════════════════════════════════════════════════════════════
# Load data
# ═══════════════════════════════════════════════════════════════════════
print(f"[KAPPA] Loading Phase3 data : {PHASE3_CSV}")
phase3_agg = pd.read_csv(PHASE3_CSV)

print(f"[KAPPA] Loading benchmarks  : {BENCH_JSON}")
with open(BENCH_JSON) as f:
    bench_summary = json.load(f)

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
print(f"[KAPPA] Unified data: {len(df)} rows, {df['model'].nunique()} models")


# ═══════════════════════════════════════════════════════════════════════
# Visual identity
# ═══════════════════════════════════════════════════════════════════════
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
    'Fidelity_Baseline_rollout':  'Fidelity baseline ($\\lambda\\!=\\!0$)',
    'RAMC_lambda_0.0001_rollout': 'RAMC $\\lambda\\!=\\!10^{-4}$',
    'RAMC_lambda_0.0015_rollout': 'RAMC $\\lambda\\!=\\!1.5{\\times}10^{-3}$',
    'Fidelity_margin_0.3':        'Margin +0.3\u2009\u00b0C',
    'Fidelity_margin_0.5':        'Margin +0.5\u2009\u00b0C',
    'Fidelity_margin_1.0':        'Margin +1.0\u2009\u00b0C',
    'RC_Exact':                   'RC exact model',
}

SCENARIO_PATH    = ['nominal', 'forecast_error', 'cold_snap']
SCENARIO_MARKERS = {'nominal': 'o', 'forecast_error': 's', 'cold_snap': '^'}
SCENARIO_LABELS  = {'nominal': 'Nominal', 'forecast_error': 'Forecast error',
                    'cold_snap': 'Cold snap'}

STABLE_MODELS = [
    'Fidelity_Baseline_rollout',
    'RAMC_lambda_0.0001_rollout',
    'RAMC_lambda_0.0015_rollout',
    'Fidelity_margin_0.3',
    'Fidelity_margin_0.5',
    'Fidelity_margin_1.0',
    'RC_Exact',
]


# ═══════════════════════════════════════════════════════════════════════
# Helper
# ═══════════════════════════════════════════════════════════════════════
def get_pt(model, scenario):
    sub = df[(df['model'] == model) & (df['scenario'] == scenario)]
    if sub.empty:
        return None
    return (sub['energy_mean'].values[0], sub['cvar90_mean'].values[0])


# ═══════════════════════════════════════════════════════════════════════
# Build figure — KAPPA sizing, legends below axes
# ═══════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

ax.tick_params(axis='both', which='major', labelsize=FONTSIZE_TICK)

# ── Markers & arrows ────────────────────────────────────────────────
MARKER_SIZE_BASE = 45
for model in STABLE_MODELS:
    pts = [get_pt(model, sc) for sc in SCENARIO_PATH]
    if any(p is None for p in pts):
        continue

    col = MODEL_COLORS[model]

    for i in range(len(pts) - 1):
        x0, y0 = pts[i]
        x1, y1 = pts[i + 1]
        arrow = FancyArrowPatch(
            (x0, y0), (x1, y1),
            arrowstyle='->', mutation_scale=10,
            color=col, linewidth=1.2, alpha=0.65,
            connectionstyle='arc3,rad=0.0', zorder=3,
        )
        ax.add_patch(arrow)

    for i, sc in enumerate(SCENARIO_PATH):
        x, y = pts[i]
        mk = SCENARIO_MARKERS[sc]
        ms = {'cold_snap': MARKER_SIZE_BASE * 1.25,
              'forecast_error': MARKER_SIZE_BASE * 1.05,
              'nominal': MARKER_SIZE_BASE}[sc]
        ax.scatter(x, y, marker=mk, s=ms, color=col,
                   edgecolors='black', linewidths=0.6, zorder=6, alpha=0.95)

# ── Axes ─────────────────────────────────────────────────────────────
ax.set_xlabel('Total heating energy (kWh)', fontsize=FONTSIZE_AXIS_LABEL)
ax.set_ylabel('CVaR$_{0.9}$ cold violation (\u00b0C)',
              fontsize=FONTSIZE_AXIS_LABEL)
ax.grid(True, alpha=0.15, linewidth=0.4, zorder=0)
ax.set_xlim(3300, 6700)
ax.set_ylim(-0.03, 0.82)

# ── Legends — placed BELOW the axes ─────────────────────────────────
# Model legend: 2 rows × 4 columns, anchored below the plot
model_handles = []
for model in STABLE_MODELS:
    model_handles.append(
        Line2D([0], [0], marker='o', color=MODEL_COLORS[model],
               markerfacecolor=MODEL_COLORS[model],
               markeredgecolor='black', markersize=5,
               linewidth=1.2, alpha=0.7,
               label=MODEL_LABELS[model])
    )

# Scenario legend handles — appended to the same legend for compactness
scenario_handles = [
    Line2D([0], [0], marker=SCENARIO_MARKERS[sc], color='w',
           markerfacecolor='#555', markeredgecolor='black', markersize=6,
           linestyle='None',
           label=SCENARIO_LABELS[sc])
    for sc in SCENARIO_PATH
]

# Separator: invisible handle to visually group scenarios
separator = Line2D([0], [0], color='w', marker='None', linestyle='None',
                   label='')

all_handles = model_handles + [separator] + scenario_handles
all_labels  = [h.get_label() for h in all_handles]

leg = ax.legend(
    all_handles, all_labels,
    loc='upper center',
    bbox_to_anchor=(0.5, -0.13),
    ncol=4,
    fontsize=FONTSIZE_LEGEND,
    framealpha=0.92,
    edgecolor='#ccc',
    handletextpad=0.4,
    borderpad=0.5,
    labelspacing=0.4,
    handlelength=1.5,
    columnspacing=1.0,
)

# Bold the scenario entries in the legend to set them apart
# (Scenarios are the last 3 visible entries)
for i, text in enumerate(leg.get_texts()):
    # The separator is at index len(model_handles), scenarios follow
    if i > len(model_handles):
        text.set_fontstyle('italic')


# ═══════════════════════════════════════════════════════════════════════
# Save
# ═══════════════════════════════════════════════════════════════════════
for ext in ['pdf', 'png']:
    out = OUTPUT_DIR / f"pareto_energy_comfort_unified_kappa.{ext}"
    fig.savefig(str(out), dpi=300, bbox_inches='tight', pad_inches=0.02)
    print(f"[KAPPA] Saved: {out}")

plt.close(fig)

print(f"\n[KAPPA] Figure size: {FIG_WIDTH:.3f} x {FIG_HEIGHT:.3f} in "
      f"({FIG_WIDTH*25.4:.1f} x {FIG_HEIGHT*25.4:.1f} mm)")
print(f"[KAPPA] Text area:  {TEXTWIDTH_IN:.3f} x {TEXTHEIGHT_IN:.3f} in "
      f"(113.0 x 199.0 mm)")
print("[KAPPA] Done.")

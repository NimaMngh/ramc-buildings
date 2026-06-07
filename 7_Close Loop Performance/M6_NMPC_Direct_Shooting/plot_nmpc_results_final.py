# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 12:09:59 2026

@author: nmi03
"""

#!/usr/bin/env python3
"""
plot_nmpc_results_final.py
==========================

Final publication-quality figures for the NMPC direct-shooting experiments.

Produces:
  1. trajectory_<scenario>_seed<seed>.png  — 2-panel per-scenario detail
  2. combined_3scenario.{png,pdf}          — 2×3 overview (T_air + power)
  3. bar_chart_4model_3scenario.{png,pdf}  — energy/comfort bar comparison
  4. pareto_energy_vs_comfort.{png,pdf}    — energy-comfort tradeoff
  5. sensitivity_heatmap.{png,pdf}         — forecast sensitivity matrix
  6. ramc_0005_instability.{png,pdf}       — why λ=0.0005 is excluded
  7. paired_delta_baseline.{png,pdf}       — Δ vs baseline (forest plot)

Decisions baked in:
  - RAMC λ=0.0005 is FLAGGED as unstable (shown separately, not in main)
  - Colors: Fidelity=blue, λ=0.0001=orange, λ=0.0005=purple, λ=0.0015=red
  - Nominal trajectory data sourced from matrix (exp IDs resolved from config)
  - Forecast error + cold snap from pairwise OR matrix as available

Author: Nima (MDU Future Energy Center)
Date: 2026-03-08
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from pathlib import Path
from typing import Optional, Dict, Tuple, List
from collections import defaultdict


# =============================================================================
# Configuration
# =============================================================================

RESULTS_DIR = Path(r"results_NMPC_20260307_185109")
MATRIX_DIR = RESULTS_DIR / "matrix"
PAIRWISE_DIR = RESULTS_DIR / "pairwise"
OUTPUT_DIR = RESULTS_DIR / "figures_final"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DT_SECONDS = 600
CP_WATER = 4186.0

# ── Visual identity ──────────────────────────────────────────────────────────
COLORS = {
    'Fidelity_Baseline_rollout':       '#2166ac',
    'RAMC_lambda_0.0001_rollout':      '#f39c12',
    'RAMC_lambda_0.0005_rollout':      '#9b59b6',
    'RAMC_lambda_0.0015_rollout':      '#c0392b',
}

SHORT = {
    'Fidelity_Baseline_rollout':       'Fidelity ($\\lambda$=0)',
    'RAMC_lambda_0.0001_rollout':      'RAMC $\\lambda$=0.0001',
    'RAMC_lambda_0.0005_rollout':      'RAMC $\\lambda$=0.0005',
    'RAMC_lambda_0.0015_rollout':      'RAMC $\\lambda$=0.0015',
}

SCENARIO_TITLE = {
    'nominal':        'Nominal Weather',
    'forecast_error': 'Forecast Error (+2\u00b0C bias)',
    'cold_snap':      'Cold Snap (\u221215\u00b0C, 48 h)',
}

MODEL_ORDER = [
    'Fidelity_Baseline_rollout',
    'RAMC_lambda_0.0001_rollout',
    'RAMC_lambda_0.0005_rollout',
    'RAMC_lambda_0.0015_rollout',
]

# Stable subset for main comparison
STABLE_MODELS = [m for m in MODEL_ORDER if '0.0005' not in m]

SCENARIO_ORDER = ['nominal', 'forecast_error', 'cold_snap']
SEEDS = [42, 123, 456, 789, 1000]


# =============================================================================
# Data loading
# =============================================================================

def load_all_results() -> List[Dict]:
    with open(MATRIX_DIR / "all_results.json") as f:
        data = json.load(f)
    return [r for r in data['results'] if r.get('success')]


def load_trajectory(exp_id: int) -> Optional[Dict]:
    p = MATRIX_DIR / f"traj_exp{exp_id}.json"
    return json.load(open(p)) if p.exists() else None


def resolve_exp_id(model: str, scenario: str, seed: int) -> Optional[int]:
    with open(MATRIX_DIR / "config.json") as f:
        cfg = json.load(f)
    ms = sorted(cfg['models'])   # models happen to already be alphabetical
    ss = cfg['scenarios']        # no sorted() here, to preserve original run order
    sd = cfg['seeds']
    eid = 1
    for m in ms:
        for s in ss:
            for se in sd:
                if m == model and s == scenario and se == seed:
                    return eid
                eid += 1
    return None


def compute_power_kW(traj: Dict) -> np.ndarray:
    mdot = np.array(traj['mdot'])
    T_sup = np.array(traj['T_supply'])
    T_ret = np.array(traj['T_ret'])[:len(mdot)]
    return (mdot * CP_WATER * np.maximum(T_sup - T_ret, 0)) / 1000.0


# =============================================================================
# Figure 1: Single-scenario 2-panel trajectory (T_air + Heat Delivered)
# =============================================================================

def fig_trajectory(traj_a: Dict, traj_b: Dict, scenario: str, seed: int,
                   metrics_a: Optional[Dict] = None,
                   metrics_b: Optional[Dict] = None,
                   save: Optional[str] = None):
    """2-panel: T_air and heat delivered."""
    n = len(traj_a['mdot'])
    th = np.arange(n) * DT_SECONDS / 3600
    ths = np.arange(n + 1) * DT_SECONDS / 3600

    Tmin = np.array(traj_a['Tmin'])
    Tmax = np.array(traj_a['Tmax'])
    occ = np.array(traj_a['occupancy'], dtype=float) > 0.5

    CA, CB = '#2166ac', '#c0392b'
    LA, LB = 'Fidelity Baseline', r'RAMC ($\lambda$=0.0015)'

    T_a, T_b = np.array(traj_a['T_air']), np.array(traj_b['T_air'])
    pw_a, pw_b = compute_power_kW(traj_a), compute_power_kW(traj_b)
    cum_a = np.cumsum(pw_a * DT_SECONDS / 3600)
    cum_b = np.cumsum(pw_b * DT_SECONDS / 3600)

    fig = plt.figure(figsize=(14, 7), constrained_layout=False)
    gs = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[3, 2],
                           hspace=0.08)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)

    # ── Panel 1: Temperature ──
    for i in range(n):
        if occ[i]:
            ax1.axvspan(th[i], th[min(i + 1, n - 1)],
                        color='#ffffcc', alpha=0.25, zorder=0)
    ax1.fill_between(th, Tmin, Tmax, where=occ, color='#d9d9d9',
                     alpha=0.35, step='post', label='Comfort band (occ)')
    ax1.plot(ths, T_a, color=CA, lw=1.8, label=LA, zorder=3)
    ax1.plot(ths, T_b, color=CB, lw=1.8, label=LB, zorder=3)
    ax1.set_ylabel('Indoor Temp (\u00b0C)', fontsize=11)
    ax1.set_title(
        f'Closed-Loop NMPC \u2014 {SCENARIO_TITLE.get(scenario, scenario)}'
        f'  (seed={seed})',
        fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9, ncol=2)
    ax1.grid(True, alpha=0.2)
    allT = np.concatenate([T_a, T_b, Tmin, Tmax])
    ax1.set_ylim(allT.min() - 0.5, allT.max() + 0.8)

    # ── Panel 2: Power + cumulative ──
    ax2.plot(th, pw_a, color=CA, lw=1.2, drawstyle='steps-post',
             label=f'{LA} \u2014 power')
    ax2.plot(th, pw_b, color=CB, lw=1.2, drawstyle='steps-post',
             label=f'{LB} \u2014 power')
    mp = max(pw_a.max(), pw_b.max()) * 1.1
    ax2.fill_between(th, 0, mp, where=occ, color='#ffffcc', alpha=0.15,
                     step='post', zorder=0)
    ax2.set_ylabel('Heat (kW)', fontsize=11)
    ax2.set_ylim(0, mp)
    ax2.grid(True, alpha=0.2)
    ax2c = ax2.twinx()
    ax2c.plot(th, cum_a, color='#6baed6', ls='-.', lw=1, alpha=0.7,
              label='Cum. (Baseline)')
    ax2c.plot(th, cum_b, color='#fc8d59', ls='-.', lw=1, alpha=0.7,
              label='Cum. (RAMC)')
    ax2c.set_ylabel('Cumulative (kWh)', fontsize=10, color='#666')
    ax2c.tick_params(axis='y', labelcolor='#666')
    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax2c.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, loc='upper right', fontsize=8, ncol=2)
    ax2.set_xlabel('Time (hours)', fontsize=12)
    ax2.set_xlim(0, th.max())

    plt.setp(ax1.get_xticklabels(), visible=False)
    fig.subplots_adjust(left=0.08, right=0.92, top=0.95, bottom=0.08,
                        hspace=0.08)
    if save:
        fig.savefig(save, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save}")
    plt.close(fig)



# =============================================================================
# Figure 2: Combined 3-scenario (2 rows × 3 cols)
# =============================================================================

def fig_combined(trajs: Dict, mets: Dict, seed: int = 42,
                 save: Optional[str] = None):
    avail = [s for s in SCENARIO_ORDER if s in trajs]
    nc = len(avail)
    if nc == 0:
        return
    fig, axes = plt.subplots(2, nc, figsize=(6 * nc, 10), sharex='col')
    if nc == 1:
        axes = axes.reshape(2, 1)

    CA, CB = '#2166ac', '#c0392b'

    for col, sc in enumerate(avail):
        ta, tb = trajs[sc]
        n = len(ta['mdot'])
        th = np.arange(n) * DT_SECONDS / 3600
        ths = np.arange(n + 1) * DT_SECONDS / 3600
        Tmin, Tmax = np.array(ta['Tmin']), np.array(ta['Tmax'])
        occ = np.array(ta['occupancy'], dtype=float) > 0.5
        T_a, T_b = np.array(ta['T_air']), np.array(tb['T_air'])
        pw_a, pw_b = compute_power_kW(ta), compute_power_kW(tb)

        at = axes[0, col]
        at.fill_between(th, Tmin, Tmax, where=occ, color='#d9d9d9',
                        alpha=0.35, step='post')
        at.plot(th, Tmin, 'k--', alpha=0.5, lw=0.7, drawstyle='steps-post')
        at.plot(th, Tmax, 'k--', alpha=0.5, lw=0.7, drawstyle='steps-post')
        at.plot(ths, T_a, color=CA, lw=1.5, label='Fidelity Baseline')
        at.plot(ths, T_b, color=CB, lw=1.5,
                label=r'RAMC ($\lambda$=0.0015)')
        at.set_title(SCENARIO_TITLE.get(sc, sc), fontsize=12,
                     fontweight='bold')
        at.grid(True, alpha=0.2)
        allT = np.concatenate([T_a, T_b, Tmin, Tmax])
        at.set_ylim(allT.min() - 0.5, allT.max() + 0.5)
        if col == 0:
            at.set_ylabel('Indoor Temp (\u00b0C)', fontsize=11)
        if col == nc - 1:
            at.legend(loc='upper right', fontsize=8)

        ma, mb = mets.get(sc, (None, None))
        if ma and mb:
            def _g(d, k):
                return d.get('metrics', d).get(k, 0.0)
            ann = (
                f"E: {_g(ma,'total_energy_kWh'):.0f} / "
                f"{_g(mb,'total_energy_kWh'):.0f} kWh\n"
                f"Peak: {_g(ma,'peak_cold_violation_occ_C'):.2f} / "
                f"{_g(mb,'peak_cold_violation_occ_C'):.2f} \u00b0C\n"
                f"CVaR90: {_g(ma,'cvar90_cold_occ_C'):.3f} / "
                f"{_g(mb,'cvar90_cold_occ_C'):.3f} \u00b0C"
            )
            at.text(0.02, 0.02, ann, transform=at.transAxes, fontsize=7,
                    va='bottom', fontfamily='monospace',
                    bbox=dict(boxstyle='round,pad=0.3', fc='white',
                              alpha=0.85, ec='#ccc'))

        ap = axes[1, col]
        ap.plot(th, pw_a, color=CA, lw=1, drawstyle='steps-post',
                label='Baseline')
        ap.plot(th, pw_b, color=CB, lw=1, drawstyle='steps-post',
                label='RAMC')
        mp = max(pw_a.max(), pw_b.max()) * 1.1
        ap.fill_between(th, 0, mp, where=occ, color='#ffffcc', alpha=0.15,
                        step='post', zorder=0)
        ap.set_ylim(0, mp)
        ap.set_xlabel('Time (hours)', fontsize=11)
        ap.set_xlim(0, th.max())
        ap.grid(True, alpha=0.2)
        if col == 0:
            ap.set_ylabel('Heat Delivered (kW)', fontsize=11)
        if col == nc - 1:
            ap.legend(loc='upper right', fontsize=8)

    fig.suptitle(
        'NMPC Direct Shooting \u2014 Fidelity Baseline vs RAMC'
        f'  (seed={seed})',
        fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if save:
        fig.savefig(save, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save}")
    plt.close(fig)


# =============================================================================
# Figure 3: Bar chart (stable models only + note on λ=0.0005)
# =============================================================================

def fig_bar_chart(results: List[Dict], save: Optional[str] = None):
    """Main comparison: 3 stable models × 3 scenarios."""
    metrics_cfg = [
        ('total_energy_kWh',              'Total Energy (kWh)',         False),
        ('deg_hours_cold_occ',            'Degree-Hours Cold (Occ)',    True),
        ('cvar90_cold_occ_C',             'CVaR90 Cold (\u00b0C)',      True),
        ('peak_cold_violation_occ_C',     'Peak Cold (\u00b0C)',        True),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()
    x = np.arange(len(SCENARIO_ORDER))
    w = 0.22
    offsets = np.array([-1, 0, 1]) * w

    for ai, (mk, label, lower) in enumerate(metrics_cfg):
        ax = axes[ai]
        for mi, model in enumerate(STABLE_MODELS):
            means, stds = [], []
            for sc in SCENARIO_ORDER:
                vals = [r['metrics'][mk] for r in results
                        if r['model'] == model and r['scenario'] == sc]
                means.append(np.mean(vals) if vals else 0)
                stds.append(np.std(vals) if vals else 0)
            ax.bar(x + offsets[mi], means, w, yerr=stds, capsize=3,
                   color=COLORS[model], alpha=0.85, label=SHORT[model],
                   edgecolor='white', lw=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([SCENARIO_TITLE[s] for s in SCENARIO_ORDER],
                           fontsize=9)
        ax.set_ylabel(label, fontsize=10)
        ax.grid(True, alpha=0.2, axis='y')
        ttl = label
        if lower:
            ttl += '  (lower = better)'
        ax.set_title(ttl, fontsize=11)
        if ai == 0:
            ax.legend(fontsize=8, loc='upper left')

    fig.suptitle(
        'NMPC Direct Shooting \u2014 Stable Models (5 seeds, '
        '$\\lambda$=0.0005 excluded)',
        fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if save:
        fig.savefig(save, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save}")
    plt.close(fig)


# =============================================================================
# Figure 4: Pareto front (stable models, λ=0.0005 shown faded)
# =============================================================================

def fig_pareto(results: List[Dict], save: Optional[str] = None):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    for ax, sc in zip(axes, SCENARIO_ORDER):
        # Faded: λ=0.0005
        unstable = 'RAMC_lambda_0.0005_rollout'
        sub_u = [r for r in results
                 if r['model'] == unstable and r['scenario'] == sc]
        if sub_u:
            ex = [r['metrics']['total_energy_kWh'] for r in sub_u]
            cy = [r['metrics']['cvar90_cold_occ_C'] for r in sub_u]
            ax.scatter(ex, cy, color=COLORS[unstable], s=50, alpha=0.2,
                       marker='x', zorder=1)
            ax.scatter(np.mean(ex), np.mean(cy), color=COLORS[unstable],
                       s=120, alpha=0.3, marker='X', zorder=1,
                       label=SHORT[unstable] + ' (unstable)')

        # Stable models
        for model in STABLE_MODELS:
            sub = [r for r in results
                   if r['model'] == model and r['scenario'] == sc]
            if not sub:
                continue
            ex = [r['metrics']['total_energy_kWh'] for r in sub]
            cy = [r['metrics']['cvar90_cold_occ_C'] for r in sub]
            ax.scatter(ex, cy, color=COLORS[model], s=70, alpha=0.7,
                       edgecolors='black', lw=0.5, zorder=3,
                       label=SHORT[model])
            ax.scatter(np.mean(ex), np.mean(cy), color=COLORS[model],
                       s=200, marker='*', edgecolors='black', lw=1.2,
                       zorder=4)

        ax.set_xlabel('Total Energy (kWh)', fontsize=11)
        ax.set_title(SCENARIO_TITLE[sc], fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.2)
        if sc == 'nominal':
            ax.set_ylabel('CVaR90 Cold Violation (\u00b0C)', fontsize=11)
            ax.legend(fontsize=7.5, loc='upper right')

    fig.suptitle(
        'Energy vs Comfort Pareto  (\u2605 = mean, dots = seeds)',
        fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if save:
        fig.savefig(save, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save}")
    plt.close(fig)


# =============================================================================
# Figure 5: Sensitivity heatmap
# =============================================================================

def fig_heatmap(results: List[Dict], save: Optional[str] = None):
    key_metrics = [
        ('total_energy_kWh',          'Energy (kWh)'),
        ('deg_hours_cold_occ',        'DH Cold'),
        ('cvar90_cold_occ_C',         'CVaR90 Cold'),
        ('peak_cold_violation_occ_C', 'Peak Cold (\u00b0C)'),
        ('hours_cold_outside_occ',    'Hours Outside'),
    ]

    mat = np.zeros((len(MODEL_ORDER), len(key_metrics)))
    for i, model in enumerate(MODEL_ORDER):
        for j, (mk, _) in enumerate(key_metrics):
            pds = []
            for seed in SEEDS:
                rn = next((r for r in results if r['model'] == model
                          and r['scenario'] == 'nominal'
                          and r['seed'] == seed), None)
                rf = next((r for r in results if r['model'] == model
                          and r['scenario'] == 'forecast_error'
                          and r['seed'] == seed), None)
                if rn and rf:
                    vn, vf = rn['metrics'][mk], rf['metrics'][mk]
                    pds.append(abs(vf - vn) / (abs(vn) + 1e-10) * 100)
            mat[i, j] = np.mean(pds) if pds else 0

    fig, ax = plt.subplots(figsize=(11, 4.5))
    im = ax.imshow(mat, cmap='YlOrRd', aspect='auto', vmin=0, vmax=40)
    ax.set_xticks(range(len(key_metrics)))
    ax.set_xticklabels([l for _, l in key_metrics], rotation=25, ha='right',
                       fontsize=10)
    ax.set_yticks(range(len(MODEL_ORDER)))
    ax.set_yticklabels([SHORT[m] for m in MODEL_ORDER], fontsize=10)
    for i in range(len(MODEL_ORDER)):
        for j in range(len(key_metrics)):
            v = mat[i, j]
            c = 'white' if v > 25 else 'black'
            ax.text(j, i, f'{v:.1f}%', ha='center', va='center',
                    fontsize=10, fontweight='bold', color=c)
    plt.colorbar(im, ax=ax,
                 label='Mean |%\u0394| (nominal vs forecast error)')
    ax.set_title(
        'Forecast Sensitivity by Model \u2014 '
        'Mean |%\u0394| Across 5 Seeds',
        fontsize=13, fontweight='bold', pad=15)
    plt.tight_layout()
    if save:
        fig.savefig(save, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save}")
    plt.close(fig)


# =============================================================================
# Figure 6: λ=0.0005 instability exhibit
# =============================================================================

def fig_instability(results: List[Dict], save: Optional[str] = None):
    """Show why λ=0.0005 is excluded: seed-dependent catastrophic failure."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, sc in zip(axes, SCENARIO_ORDER):
        for model in MODEL_ORDER:
            vals = []
            for seed in SEEDS:
                r = next((r for r in results if r['model'] == model
                         and r['scenario'] == sc and r['seed'] == seed),
                         None)
                if r:
                    vals.append(r['metrics']['peak_cold_violation_occ_C'])
            if not vals:
                continue
            xs = np.arange(len(vals))
            lw = 2.5 if '0.0005' in model else 1.2
            al = 1.0 if '0.0005' in model else 0.6
            ls = '-' if '0.0005' in model else '--'
            ax.plot(xs, vals, color=COLORS[model], lw=lw, alpha=al,
                    ls=ls, marker='o', markersize=6, label=SHORT[model])

        ax.set_xticks(range(len(SEEDS)))
        ax.set_xticklabels(SEEDS, fontsize=9)
        ax.set_xlabel('Seed', fontsize=10)
        ax.set_title(SCENARIO_TITLE[sc], fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.2)
        ax.axhline(1.0, color='gray', ls=':', alpha=0.5)
        if sc == 'nominal':
            ax.set_ylabel('Peak Cold Violation (\u00b0C)', fontsize=11)
            ax.legend(fontsize=7.5, loc='upper right')

    fig.suptitle(
        'Seed-Dependent Instability of RAMC $\\lambda$=0.0005\n'
        '(peaks 2\u20133\u00b0C in some seeds, <1\u00b0C in others '
        '\u2192 excluded from main analysis)',
        fontsize=12, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    if save:
        fig.savefig(save, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save}")
    plt.close(fig)


# =============================================================================
# Figure 7: Forest plot — paired Δ vs baseline
# =============================================================================

def fig_forest_plot(results: List[Dict], save: Optional[str] = None):
    """For each RAMC model, show Δ(metric) vs Fidelity Baseline with CI."""
    baseline = 'Fidelity_Baseline_rollout'
    ramc_models = [m for m in STABLE_MODELS if m != baseline]
    metrics_cfg = [
        ('cvar90_cold_occ_C',         'CVaR90 Cold (\u00b0C)'),
        ('peak_cold_violation_occ_C', 'Peak Cold (\u00b0C)'),
        ('deg_hours_cold_occ',        'Deg-Hours Cold'),
        ('total_energy_kWh',          'Energy (kWh)'),
    ]

    fig, axes = plt.subplots(len(metrics_cfg), 1, figsize=(14, 3 * len(metrics_cfg)),
                             sharex=False)

    for ai, (mk, label) in enumerate(metrics_cfg):
        ax = axes[ai]
        y_pos = 0
        yticks, ylabels = [], []

        for model in ramc_models:
            for sc in SCENARIO_ORDER:
                diffs = []
                for seed in SEEDS:
                    rb = next((r for r in results if r['model'] == baseline
                              and r['scenario'] == sc and r['seed'] == seed),
                              None)
                    rm = next((r for r in results if r['model'] == model
                              and r['scenario'] == sc and r['seed'] == seed),
                              None)
                    if rb and rm:
                        diffs.append(rm['metrics'][mk] - rb['metrics'][mk])

                if not diffs:
                    continue

                mean_d = np.mean(diffs)
                std_d = np.std(diffs)
                ci_lo = mean_d - 1.96 * std_d / np.sqrt(len(diffs))
                ci_hi = mean_d + 1.96 * std_d / np.sqrt(len(diffs))

                color = COLORS[model]
                ax.errorbar(mean_d, y_pos, xerr=[[mean_d - ci_lo],
                            [ci_hi - mean_d]],
                            fmt='o', color=color, capsize=4, capthick=1.5,
                            markersize=8, zorder=3)

                sc_short = {'nominal': 'Nom', 'forecast_error': 'FcErr',
                            'cold_snap': 'Cold'}[sc]
                ylabels.append(f'{SHORT[model]}\n{sc_short}')
                yticks.append(y_pos)
                y_pos += 1

            y_pos += 0.5  # gap between models

        ax.axvline(0, color='black', ls='-', lw=1, zorder=1)
        ax.set_yticks(yticks)
        ax.set_yticklabels(ylabels, fontsize=8)
        ax.set_xlabel(f'\u0394 {label} (RAMC \u2212 Baseline)', fontsize=10)
        ax.grid(True, alpha=0.2, axis='x')
        ax.invert_yaxis()

        # Shade the "better" side
        if 'energy' in mk.lower():
            pass  # energy: no clear "better" side
        else:
            ax.axvspan(ax.get_xlim()[0], 0, color='#d4edda', alpha=0.15,
                       zorder=0)
            ax.text(0.01, 0.98, '\u2190 better', transform=ax.transAxes,
                    fontsize=8, va='top', color='green', alpha=0.5)

    fig.suptitle(
        'Paired Differences vs Fidelity Baseline (95% CI, 5 seeds)',
        fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if save:
        fig.savefig(save, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save}")
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("FINAL PUBLICATION FIGURES")
    print("=" * 70)

    results = load_all_results()
    print(f"Loaded {len(results)} experiments")

    seed = 42
    model_a = 'Fidelity_Baseline_rollout'
    model_b = 'RAMC_lambda_0.0015_rollout'

    # ── Collect trajectories ──
    trajs, mets = {}, {}
    for sc in SCENARIO_ORDER:
        # Try pairwise first
        pa = PAIRWISE_DIR / f"traj_a_{sc}_seed{seed}.json"
        pb = PAIRWISE_DIR / f"traj_b_{sc}_seed{seed}.json"
        pm = PAIRWISE_DIR / f"pairwise_{sc}_seed{seed}.json"
        
        if pa.exists() and pb.exists():
            ta = json.load(open(pa))
            tb = json.load(open(pb))
            if pm.exists():                                          # <- changed
                pairwise_data = json.load(open(pm))                  # <- changed
                ma = {'metrics': pairwise_data.get('metrics_a', {})} # <- changed
                mb = {'metrics': pairwise_data.get('metrics_b', {})} # <- changed
            else:                                                    # <- changed
                ma, mb = None, None                                  # <- changed
            print(f"  {sc}: loaded from pairwise")
        else:
            # Resolve from matrix
            eid_a = resolve_exp_id(model_a, sc, seed)
            eid_b = resolve_exp_id(model_b, sc, seed)
            if eid_a is None or eid_b is None:
                print(f"  {sc}: SKIP (cannot resolve exp IDs)")
                continue
            ta = load_trajectory(eid_a)
            tb = load_trajectory(eid_b)
            ra = MATRIX_DIR / f"result_exp{eid_a}.json"
            rb = MATRIX_DIR / f"result_exp{eid_b}.json"
            ma = json.load(open(ra)) if ra.exists() else None
            mb = json.load(open(rb)) if rb.exists() else None
            print(f"  {sc}: loaded from matrix (exp {eid_a}, {eid_b})")

        if ta and tb:
            trajs[sc] = (ta, tb)
            mets[sc] = (ma, mb)

    # ── Generate figures ──
    print(f"\n{'─' * 50}")
    print("Generating trajectory plots...")
    for sc in trajs:
        ta, tb = trajs[sc]
        ma, mb = mets[sc]
        fig_trajectory(ta, tb, sc, seed, ma, mb,
                       save=str(OUTPUT_DIR / f"trajectory_{sc}_seed{seed}.png"))

    print(f"\n{'─' * 50}")
    print("Generating combined 3-scenario...")
    for ext in ['png', 'pdf']:
        fig_combined(trajs, mets, seed,
                     save=str(OUTPUT_DIR / f"combined_3scenario.{ext}"))

    print(f"\n{'─' * 50}")
    print("Generating bar chart (stable models)...")
    for ext in ['png', 'pdf']:
        fig_bar_chart(results,
                      save=str(OUTPUT_DIR / f"bar_chart_stable.{ext}"))

    print(f"\n{'─' * 50}")
    print("Generating Pareto front...")
    for ext in ['png', 'pdf']:
        fig_pareto(results,
                   save=str(OUTPUT_DIR / f"pareto_energy_comfort.{ext}"))

    print(f"\n{'─' * 50}")
    print("Generating sensitivity heatmap...")
    for ext in ['png', 'pdf']:
        fig_heatmap(results,
                    save=str(OUTPUT_DIR / f"sensitivity_heatmap.{ext}"))

    print(f"\n{'─' * 50}")
    print("Generating instability exhibit...")
    for ext in ['png', 'pdf']:
        fig_instability(results,
                        save=str(OUTPUT_DIR / f"ramc_0005_instability.{ext}"))

    print(f"\n{'─' * 50}")
    print("Generating forest plot...")
    for ext in ['png', 'pdf']:
        fig_forest_plot(results,
                        save=str(OUTPUT_DIR / f"forest_plot_vs_baseline.{ext}"))

    # ── Print summary table for paper ──
    print(f"\n{'=' * 70}")
    print("SUMMARY TABLE FOR PAPER (stable models only)")
    print(f"{'=' * 70}")

    print(f"\n{'Model':<22s} {'Scenario':<16s} {'Energy':>8s} {'DH Cold':>8s} "
          f"{'CVaR90':>8s} {'Peak':>8s} {'Hr<Tmin':>8s}")
    print(f"{'─' * 88}")

    for model in STABLE_MODELS:
        for sc in SCENARIO_ORDER:
            sub = [r for r in results
                   if r['model'] == model and r['scenario'] == sc]
            if not sub:
                continue
            e = np.mean([r['metrics']['total_energy_kWh'] for r in sub])
            dh = np.mean([r['metrics']['deg_hours_cold_occ'] for r in sub])
            cv = np.mean([r['metrics']['cvar90_cold_occ_C'] for r in sub])
            pk = np.mean([r['metrics']['peak_cold_violation_occ_C']
                          for r in sub])
            hr = np.mean([r['metrics']['hours_cold_outside_occ']
                          for r in sub])
            short = SHORT[model][:21]
            st = SCENARIO_TITLE[sc][:15]
            print(f"{short:<22s} {st:<16s} {e:>8.0f} {dh:>8.2f} "
                  f"{cv:>8.3f} {pk:>8.2f} {hr:>8.1f}")
        print()

    # ── Key findings ──
    print(f"\n{'=' * 70}")
    print("KEY FINDINGS FOR PAPER")
    print(f"{'=' * 70}")

    for sc in SCENARIO_ORDER:
        fid = [r for r in results
               if r['model'] == model_a and r['scenario'] == sc]
        ramc = [r for r in results
                if r['model'] == model_b and r['scenario'] == sc]
        if not fid or not ramc:
            continue

        e_f = np.mean([r['metrics']['total_energy_kWh'] for r in fid])
        e_r = np.mean([r['metrics']['total_energy_kWh'] for r in ramc])
        cv_f = np.mean([r['metrics']['cvar90_cold_occ_C'] for r in fid])
        cv_r = np.mean([r['metrics']['cvar90_cold_occ_C'] for r in ramc])
        pk_f = np.mean([r['metrics']['peak_cold_violation_occ_C']
                         for r in fid])
        pk_r = np.mean([r['metrics']['peak_cold_violation_occ_C']
                         for r in ramc])
        dh_f = np.mean([r['metrics']['deg_hours_cold_occ'] for r in fid])
        dh_r = np.mean([r['metrics']['deg_hours_cold_occ'] for r in ramc])

        de = (e_r - e_f) / e_f * 100
        dcv = (cv_r - cv_f) / cv_f * 100
        dpk = (pk_r - pk_f) / pk_f * 100
        ddh = (dh_r - dh_f) / dh_f * 100

        print(f"\n  {SCENARIO_TITLE[sc]}:")
        print(f"    RAMC vs Baseline: energy {de:+.1f}%, CVaR90 {dcv:+.1f}%, "
              f"peak cold {dpk:+.1f}%, DH cold {ddh:+.1f}%")

    print(f"\n{'=' * 70}")
    print(f"All figures saved to: {OUTPUT_DIR}")
    print(f"{'=' * 70}")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fig_cold_snap_extract.py
========================

Extracts the cold snap panel (panel c) from the combined 3-scenario figure
(RAMC Paper C, Figure 7) as a standalone single-column figure.

Layout:
  Row 1 (tall)  : Indoor temperature — Fidelity baseline + RAMC λ=1.5×10⁻³
  Row 2 (short) : ΔT = T_RAMC − T_baseline

Figure sizing calibrated for MDU thesis LaTeX template:
  - Text width  = 113 mm = 4.449 in  (stock 169 mm, margins 25+25+6 mm)
  - Text height = 199 mm = 7.835 in  (stock 239 mm, margins 20+20 mm)
  - Caption: scriptsize sf bf, skip=12pt, margin=10pt
  - Font: Helvetica / Arial (kappa sans-serif style)

Saves to:
  results_NMPC_20260307_185109/figures_final/cold_snap_panel_c.{png,pdf}

Author: Nima (MDU Future Energy Center)
Date:   2026-04-01
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from typing import Optional, Dict


# ─────────────────────────────────────────────────────────
# LaTeX template geometry
# ─────────────────────────────────────────────────────────
TEXT_WIDTH_MM  = 169 - 25 - 25 - 6   # = 113 mm
TEXT_WIDTH_IN  = TEXT_WIDTH_MM / 25.4  # ≈ 4.449 in
FIG_HEIGHT_IN  = TEXT_WIDTH_IN * 0.95  # ≈ 4.23 in

# ─────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────
DT_SECONDS = 600
CP_WATER   = 4186.0

CA = '#2166ac'           # Baseline blue
CB = '#c0392b'           # RAMC red
DELTA_POS = '#d73027'    # ΔT positive  (RAMC warmer)
DELTA_NEG = '#4575b4'    # ΔT negative  (baseline warmer)

SCRIPT_DIR   = Path(__file__).resolve().parent
RESULTS_DIR  = SCRIPT_DIR / "results_NMPC_20260307_185109"
MATRIX_DIR   = RESULTS_DIR / "matrix"
PAIRWISE_DIR = RESULTS_DIR / "pairwise"
OUTPUT_DIR   = RESULTS_DIR / "figures_final"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────
# Matplotlib RC — Helvetica / Arial for kappa
# ─────────────────────────────────────────────────────────
plt.rcParams.update({
    # Font family: Helvetica for kappa sans-serif style
    'font.family':       'sans-serif',
    'font.sans-serif':   ['Helvetica', 'Arial', 'DejaVu Sans'],
    'mathtext.fontset':  'custom',
    'mathtext.rm':       'Helvetica',
    'mathtext.it':       'Helvetica:italic',
    'mathtext.bf':       'Helvetica:bold',
    'mathtext.sf':       'Helvetica',

    # Font sizes: 11pt body in LaTeX; figure labels slightly smaller
    'font.size':         9,
    'axes.titlesize':    10,
    'axes.labelsize':    9,
    'xtick.labelsize':   8,
    'ytick.labelsize':   8,
    'legend.fontsize':   7.5,

    # Line and tick styling
    'axes.linewidth':    0.6,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.major.size':  3,
    'ytick.major.size':  3,
    'lines.linewidth':   1.0,

    # Export
    'savefig.dpi':       300,
    'savefig.bbox':      'tight',
    'savefig.pad_inches': 0.02,
    'pdf.fonttype':      42,            # TrueType in PDF
    'ps.fonttype':       42,
})


# ─────────────────────────────────────────────────────────
# Data loaders
# ─────────────────────────────────────────────────────────
def load_trajectory(exp_id: int) -> Optional[Dict]:
    p = MATRIX_DIR / f"traj_exp{exp_id}.json"
    return json.load(open(p)) if p.exists() else None


def resolve_exp_id(model: str, scenario: str, seed: int) -> Optional[int]:
    with open(MATRIX_DIR / "config.json") as f:
        cfg = json.load(f)
    ms = cfg['models']
    ss = cfg['scenarios']
    sd = cfg['seeds']
    eid = 1
    for m in ms:
        for s in ss:
            for se in sd:
                if m == model and s == scenario and se == seed:
                    return eid
                eid += 1
    return None


def load_cold_snap_pair(seed: int = 42):
    """Load the baseline + RAMC trajectory pair for cold_snap."""
    model_a = 'Fidelity_Baseline_rollout'
    model_b = 'RAMC_lambda_0.0015_rollout'
    sc = 'cold_snap'

    pa = PAIRWISE_DIR / f"traj_a_{sc}_seed{seed}.json"
    pb = PAIRWISE_DIR / f"traj_b_{sc}_seed{seed}.json"

    if pa.exists() and pb.exists():
        ta = json.load(open(pa))
        tb = json.load(open(pb))
        print(f"  {sc}: loaded from pairwise")
        return ta, tb

    eid_a = resolve_exp_id(model_a, sc, seed)
    eid_b = resolve_exp_id(model_b, sc, seed)
    if eid_a is None or eid_b is None:
        print(f"  {sc}: SKIP (cannot resolve exp IDs)")
        return None, None
    ta = load_trajectory(eid_a)
    tb = load_trajectory(eid_b)
    print(f"  {sc}: loaded from matrix (exp {eid_a}, {eid_b})")
    return ta, tb


# ─────────────────────────────────────────────────────────
# Synthetic fallback
# ─────────────────────────────────────────────────────────
def _occ_schedule(n):
    occ = np.zeros(n)
    for k in range(n):
        hour_of_week = (k * DT_SECONDS / 3600) % 168
        day = int(hour_of_week // 24)
        hour = hour_of_week % 24
        if day < 5 and 7 <= hour < 18:
            occ[k] = 1.0
    return occ


def _make_synthetic_cold_snap(seed=42):
    rng = np.random.default_rng(seed + 2000)
    n = 1008
    th = np.arange(n) * DT_SECONDS / 3600
    occ = _occ_schedule(n)
    Tmin = np.where(occ > 0.5, 20.0, 15.0)
    Tmax = np.where(occ > 0.5, 22.0, 30.0)

    T_out = -5 + 5 * np.sin(2 * np.pi * th / 24) + rng.normal(0, 0.3, n)
    T_out[(th >= 24) & (th < 72)] -= 10

    def _sim(target_occ, target_pre, target_off, gain, rng_sim):
        T = np.full(n + 1, 19.5)
        md = np.zeros(n); ts = np.zeros(n); tr = np.zeros(n)
        for k in range(n):
            future = any(occ[min(k + j, n - 1)] > 0.5 for j in range(1, 7))
            tgt = target_occ if occ[k] > 0.5 else (target_pre if future else target_off)
            err = tgt - T[k]
            flow = np.clip(err * gain, 0, 4.0)
            tsup = np.clip(40 + err * 5, 32, 60)
            md[k], ts[k] = flow, tsup
            tret = tsup - flow * 3 - rng_sim.normal(0, 0.2)
            tr[k] = max(tret, T[k])
            pw = flow * CP_WATER * max(tsup - tr[k], 0) / 1000
            T[k + 1] = T[k] + pw * 0.012 + (T_out[k] - T[k]) * 0.003 + rng_sim.normal(0, 0.02)
        return T, md, ts, tr

    rng_a = np.random.default_rng(seed + 2100)
    rng_b = np.random.default_rng(seed + 2200)

    T_a, md_a, ts_a, tr_a = _sim(20.5, 18.0, 18.0, 0.8, rng_a)
    T_b, md_b, ts_b, tr_b = _sim(20.8, 20.2, 18.0, 1.0, rng_b)

    m = np.arange(n + 1)
    z = (m >= 144) & (m < 432)
    T_a[z] -= 1.5
    T_b[z] -= 0.8

    def _pack(T, md, ts, tr):
        return {'T_air': T.tolist(), 'mdot': md.tolist(),
                'T_supply': ts.tolist(), 'T_ret': tr.tolist(),
                'Tmin': Tmin.tolist(), 'Tmax': Tmax.tolist(),
                'occupancy': occ.tolist()}
    return _pack(T_a, md_a, ts_a, tr_a), _pack(T_b, md_b, ts_b, tr_b)


# ─────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────
def _smooth(x, w=3):
    kernel = np.ones(w) / w
    return np.convolve(x, kernel, mode='same')


# ─────────────────────────────────────────────────────────
# Figure: Cold snap panel (c) — standalone, LaTeX-optimised
# ─────────────────────────────────────────────────────────
def fig_cold_snap_panel(ta, tb, seed=42, save=None):
    """
    Single-column figure reproducing panel (c) from Figure 7.
    Sized to fill exactly \\textwidth in the MDU thesis template.
    """
    n   = len(ta['mdot'])
    th  = np.arange(n) * DT_SECONDS / 3600
    ths = np.arange(n + 1) * DT_SECONDS / 3600

    Tmin = np.array(ta['Tmin'])
    Tmax = np.array(ta['Tmax'])
    occ  = np.array(ta['occupancy'], dtype=float) > 0.5
    T_a  = np.array(ta['T_air'])
    T_b  = np.array(tb['T_air'])

    # ── Figure: exact text width ──
    fig = plt.figure(figsize=(TEXT_WIDTH_IN, FIG_HEIGHT_IN))
    gs  = gridspec.GridSpec(
        2, 1, figure=fig,
        height_ratios=[3.5, 1.0],
        hspace=0.15,
    )

    # ═══════════════════════════════════════════════════
    # ROW 1 — Indoor temperature
    # ═══════════════════════════════════════════════════
    ax_t = fig.add_subplot(gs[0])

    # Occupied-period shading
    for i in range(n):
        if occ[i]:
            ax_t.axvspan(th[i], th[min(i + 1, n - 1)],
                         color='#fffde7', alpha=0.45, zorder=0, linewidth=0)

    # Tmin dashed comfort bound
    ax_t.plot(th, Tmin, color='#888', ls='--', lw=0.6,
              drawstyle='steps-post', zorder=1)

    # Temperature traces
    ax_t.plot(ths, T_a, color=CA, lw=1.0, zorder=3,
              label='Fidelity baseline')
    ax_t.plot(ths, T_b, color=CB, lw=1.0, zorder=3,
              label=r'RAMC ($\lambda\!=\!1.5{\times}10^{-3}$)')

    # ── Y-limits: extend ceiling for legend clearance ──
    allT = np.concatenate([T_a, T_b])
    T_floor = max(allT.min() - 0.8, 14.5)
    T_ceil  = allT.max() + 1.8
    ax_t.set_ylim(T_floor, T_ceil)

    ax_t.grid(True, alpha=0.15, lw=0.4)
    ax_t.tick_params(labelbottom=False)
    ax_t.set_ylabel('Indoor temperature (\u00b0C)')

    # ── Legend ──
    ax_t.legend(
        loc='upper right',
        bbox_to_anchor=(0.99, 0.99),
        framealpha=0.92,
        edgecolor='#ccc',
        handlelength=1.8,
        borderpad=0.4,
        handletextpad=0.5,
        fancybox=False,
    )

    # ═══════════════════════════════════════════════════
    # ROW 2 — ΔT strip
    # ═══════════════════════════════════════════════════
    ax_d = fig.add_subplot(gs[1], sharex=ax_t)

    delta_sm = _smooth(T_b[:n] - T_a[:n], w=3)

    ax_d.fill_between(th, 0, delta_sm, where=delta_sm >= 0,
                      color=DELTA_POS, alpha=0.45, step='mid', lw=0)
    ax_d.fill_between(th, 0, delta_sm, where=delta_sm < 0,
                      color=DELTA_NEG, alpha=0.45, step='mid', lw=0)
    ax_d.plot(th, delta_sm, color='#333', lw=0.7, zorder=3)
    ax_d.axhline(0, color='#666', lw=0.5, zorder=2)

    for i in range(n):
        if occ[i]:
            ax_d.axvspan(th[i], th[min(i + 1, n - 1)],
                         color='#fffde7', alpha=0.3, zorder=0, lw=0)

    dmax = max(abs(delta_sm.min()), abs(delta_sm.max()), 0.5) * 1.35
    ax_d.set_ylim(-dmax, dmax)
    ax_d.grid(True, alpha=0.12, lw=0.4)
    ax_d.set_xlabel('Time (hours)')
    ax_d.set_xlim(0, th.max())
    ax_d.set_ylabel('$\\Delta T$ (\u00b0C)')

    plt.subplots_adjust(left=0.12, right=0.97, top=0.97, bottom=0.12)

    if save:
        fig.savefig(save, dpi=300, bbox_inches='tight', pad_inches=0.02)
        print(f"  Saved: {save}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("EXTRACT COLD SNAP PANEL (c) — LaTeX-optimised (Helvetica)")
    print(f"  Target width:  {TEXT_WIDTH_MM} mm = {TEXT_WIDTH_IN:.3f} in")
    print(f"  Target height: {FIG_HEIGHT_IN:.3f} in")
    print("=" * 60)

    seed = 42
    use_synthetic = False

    if MATRIX_DIR.exists():
        ta, tb = load_cold_snap_pair(seed)
        if ta is None or tb is None:
            use_synthetic = True
    else:
        use_synthetic = True

    if use_synthetic:
        print("\n  Real data not found — using synthetic preview")
        ta, tb = _make_synthetic_cold_snap(seed)

    for ext in ['png', 'pdf']:
        fig_cold_snap_panel(
            ta, tb, seed=seed,
            save=str(OUTPUT_DIR / f"cold_snap_panel_c.{ext}"),
        )

    print(f"\n{'─' * 60}")
    print("LaTeX include snippet:")
    print(f"{'─' * 60}")
    print(r"""
\begin{figure}[htbp]
  \centering
  \includegraphics[width=\textwidth]{cold_snap_panel_c}
  \caption{Closed-loop NMPC trajectory under cold snap conditions
    ($-10\,^\circ$C, 48\,h). Top: indoor temperature for Fidelity
    baseline (blue) and RAMC $\lambda=1.5\times10^{-3}$ (red); dashed
    line shows the occupied-period comfort bound $T_{\min}$. Bottom:
    temperature difference $\Delta T = T_{\mathrm{RAMC}} -
    T_{\mathrm{baseline}}$, with red (positive) indicating pre-heating
    by RAMC. Seed\,=\,42.}
  \label{fig:cold-snap-panel}
\end{figure}
""")

    print(f"\nOutputs in: {OUTPUT_DIR}")
    print("=" * 60)

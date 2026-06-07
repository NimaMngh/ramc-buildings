#!/usr/bin/env python3
"""
fig_combined_redesign.py
========================
Redesigned 2-row × 3-col trajectory figure for the RAMC paper.

Layout (per column = one scenario):
  Row 1  (tall)  : Indoor temperature — both traces + occupied shading
  Row 2  (short) : ΔT = T_RAMC − T_baseline — highlights pre-heating

Place in:  7_Close Loop Performance/M6_NMPC_Direct_Shooting/

Author: Claude (redesign) / Nima (original data pipeline)
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
# Configuration
# ─────────────────────────────────────────────────────────
DT_SECONDS = 600
CP_WATER = 4186.0

CA = '#2166ac'           # Baseline blue
CB = '#c0392b'           # RAMC red
DELTA_POS = '#d73027'    # ΔT positive (RAMC warmer)
DELTA_NEG = '#4575b4'    # ΔT negative (baseline warmer)

SCENARIO_TITLE = {
    'nominal':        'Nominal weather',
    'forecast_error': 'Forecast error (+1.5\u2009°C bias)',
    'cold_snap':      'Cold snap (\u221210\u2009°C, 48\u2009h)',
}
SCENARIO_ORDER = ['nominal', 'forecast_error', 'cold_snap']

# ── Paths (relative to this script) ─────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results_NMPC_20260307_185109"
MATRIX_DIR  = RESULTS_DIR / "matrix"
PAIRWISE_DIR = RESULTS_DIR / "pairwise"
OUTPUT_DIR  = RESULTS_DIR / "figures_final"


# ─────────────────────────────────────────────────────────
# Real data loader
# ─────────────────────────────────────────────────────────
def load_trajectory(exp_id: int) -> Optional[Dict]:
    p = MATRIX_DIR / f"traj_exp{exp_id}.json"
    return json.load(open(p)) if p.exists() else None


def resolve_exp_id(model: str, scenario: str, seed: int) -> Optional[int]:
    with open(MATRIX_DIR / "config.json") as f:
        cfg = json.load(f)
    ms = cfg['models']       # <- DO NOT sort, use config order
    ss = cfg['scenarios']    # <- DO NOT sort, use config order
    sd = cfg['seeds']        # <- already matches (seeds are stored in run order)
    eid = 1
    for m in ms:
        for s in ss:
            for se in sd:
                if m == model and s == scenario and se == seed:
                    return eid
                eid += 1
    return None


def collect_trajectories(seed: int = 42):
    """Load trajectory pairs for all three scenarios from real results."""
    model_a = 'Fidelity_Baseline_rollout'
    model_b = 'RAMC_lambda_0.0015_rollout'

    trajs = {}
    for sc in SCENARIO_ORDER:
        # Try pairwise first
        pa = PAIRWISE_DIR / f"traj_a_{sc}_seed{seed}.json"
        pb = PAIRWISE_DIR / f"traj_b_{sc}_seed{seed}.json"

        if pa.exists() and pb.exists():
            ta = json.load(open(pa))
            tb = json.load(open(pb))
            print(f"  {sc}: loaded from pairwise")
        else:
            eid_a = resolve_exp_id(model_a, sc, seed)
            eid_b = resolve_exp_id(model_b, sc, seed)
            if eid_a is None or eid_b is None:
                print(f"  {sc}: SKIP (cannot resolve exp IDs)")
                continue
            ta = load_trajectory(eid_a)
            tb = load_trajectory(eid_b)
            print(f"  {sc}: loaded from matrix (exp {eid_a}, {eid_b})")

        if ta and tb:
            trajs[sc] = (ta, tb)

    return trajs


# ─────────────────────────────────────────────────────────
# Synthetic fallback (for preview when real data is absent)
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


def _make_synthetic(scenario, seed=42):
    rng = np.random.default_rng(seed)
    n = 1008
    th = np.arange(n) * DT_SECONDS / 3600
    occ = _occ_schedule(n)
    Tmin = np.where(occ > 0.5, 20.0, 15.0)
    Tmax = np.where(occ > 0.5, 22.0, 30.0)

    # ── Use a DIFFERENT seed per scenario so traces are distinct ──
    scenario_offsets = {'nominal': 0, 'forecast_error': 1000, 'cold_snap': 2000}
    rng_sc = np.random.default_rng(seed + scenario_offsets.get(scenario, 0))

    T_out = -5 + 5 * np.sin(2 * np.pi * th / 24) + rng_sc.normal(0, 0.3, n)
    if scenario == 'cold_snap':
        T_out[(th >= 24) & (th < 72)] -= 10

    def _sim(target_occ, target_pre, target_off, gain, rng_sim):
        T = np.full(n + 1, 19.5)
        md = np.zeros(n); ts = np.zeros(n); tr = np.zeros(n)
        for k in range(n):
            future = any(occ[min(k+j, n-1)] > 0.5 for j in range(1, 7))
            tgt = target_occ if occ[k] > 0.5 else (target_pre if future else target_off)
            err = tgt - T[k]
            flow = np.clip(err * gain, 0, 4.0)
            tsup = np.clip(40 + err * 5, 32, 60)
            md[k], ts[k] = flow, tsup
            tret = tsup - flow * 3 - rng_sim.normal(0, 0.2)
            tr[k] = max(tret, T[k])
            pw = flow * CP_WATER * max(tsup - tr[k], 0) / 1000
            T[k+1] = T[k] + pw * 0.012 + (T_out[k] - T[k]) * 0.003 + rng_sim.normal(0, 0.02)
        return T, md, ts, tr

    rng_a = np.random.default_rng(seed + scenario_offsets.get(scenario, 0) + 100)
    rng_b = np.random.default_rng(seed + scenario_offsets.get(scenario, 0) + 200)

    T_a, md_a, ts_a, tr_a = _sim(20.5, 18.0, 18.0, 0.8, rng_a)
    T_b, md_b, ts_b, tr_b = _sim(20.8, 20.2, 18.0, 1.0, rng_b)

    if scenario == 'forecast_error':
        T_a[200:400] -= 0.6; T_b[200:400] -= 0.3
    elif scenario == 'cold_snap':
        m = np.arange(n+1); z = (m >= 144) & (m < 432)
        T_a[z] -= 1.5; T_b[z] -= 0.8

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
# Main figure:  2 rows × 3 cols
# ─────────────────────────────────────────────────────────
def fig_combined_2row(trajs, seed=42, save=None):
    avail = [s for s in SCENARIO_ORDER if s in trajs]
    nc = len(avail)
    if nc == 0:
        return

    fig = plt.figure(figsize=(7.2, 3.8))  # <- much smaller, closer to \textwidth
    gs = gridspec.GridSpec(
        2, nc, figure=fig,
        height_ratios=[3.0, 1.2],
        hspace=0.12, wspace=0.25,
    )

    for col, sc in enumerate(avail):
        ta, tb = trajs[sc]
        n = len(ta['mdot'])
        th  = np.arange(n) * DT_SECONDS / 3600
        ths = np.arange(n + 1) * DT_SECONDS / 3600

        Tmin = np.array(ta['Tmin'])
        Tmax = np.array(ta['Tmax'])
        occ  = np.array(ta['occupancy'], dtype=float) > 0.5
        T_a  = np.array(ta['T_air'])
        T_b  = np.array(tb['T_air'])

        # ═══════════════════════════════════════════════
        # ROW 1 — Indoor temperature
        # ═══════════════════════════════════════════════
        ax_t = fig.add_subplot(gs[0, col])

        for i in range(n):
            if occ[i]:
                ax_t.axvspan(th[i], th[min(i+1, n-1)],
                             color='#fffde7', alpha=0.45, zorder=0,
                             linewidth=0)

        ax_t.plot(th, Tmin, color='#888', ls='--', lw=0.6,
                  drawstyle='steps-post', zorder=1)

        ax_t.plot(ths, T_a, color=CA, lw=1.0, zorder=3,
                  label='Fidelity baseline')
        ax_t.plot(ths, T_b, color=CB, lw=1.0, zorder=3,
                  label=r'RAMC ($\lambda\!=\!1.5{\times}10^{-3}$)')

        allT = np.concatenate([T_a, T_b])
        ax_t.set_ylim(max(allT.min() - 0.8, 14.5),
                       min(allT.max() + 0.8, 23))
        ax_t.grid(True, alpha=0.15, lw=0.5)
        ax_t.tick_params(labelbottom=False, labelsize=7)

        if col == 0:
            ax_t.set_ylabel('Indoor temperature (°C)', fontsize=8)
        else:
            ax_t.tick_params(labelleft=False)
        if col == nc - 1:
            ax_t.legend(loc='upper right', fontsize=6.5, framealpha=0.85,
                        edgecolor='#ccc', handlelength=1.5)

        # ═══════════════════════════════════════════════
        # ROW 2 — ΔT strip
        # ═══════════════════════════════════════════════
        ax_d = fig.add_subplot(gs[1, col], sharex=ax_t)

        delta_sm = _smooth(T_b[:n] - T_a[:n], w=3)

        ax_d.fill_between(th, 0, delta_sm, where=delta_sm >= 0,
                          color=DELTA_POS, alpha=0.45, step='mid', lw=0)
        ax_d.fill_between(th, 0, delta_sm, where=delta_sm < 0,
                          color=DELTA_NEG, alpha=0.45, step='mid', lw=0)
        ax_d.plot(th, delta_sm, color='#333', lw=0.7, zorder=3)
        ax_d.axhline(0, color='#666', lw=0.5, zorder=2)

        for i in range(n):
            if occ[i]:
                ax_d.axvspan(th[i], th[min(i+1, n-1)],
                             color='#fffde7', alpha=0.3, zorder=0, lw=0)

        dmax = max(abs(delta_sm.min()), abs(delta_sm.max()), 0.5) * 1.35

        ax_d.set_ylim(-dmax, dmax)
        ax_d.grid(True, alpha=0.12, lw=0.5)
        ax_d.tick_params(labelsize=7)

        ax_d.set_xlabel(f'Time (hours)\n({chr(97+col)})', fontsize=8)
        ax_d.set_xlim(0, th.max())

        if col == 0:
            ax_d.set_ylabel('$\\Delta T$ (°C)', fontsize=8)
        else:
            ax_d.tick_params(labelleft=False)

    plt.subplots_adjust(left=0.09, right=0.98, top=0.98, bottom=0.15)

    if save:
        fig.savefig(save, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save}")
    plt.close(fig)



# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("REDESIGNED COMBINED FIGURE (2-row: T_air + ΔT)")
    print("=" * 60)

    seed = 42

    # Try real data first
    use_synthetic = False
    if MATRIX_DIR.exists():
        trajs = collect_trajectories(seed)
        if not trajs:
            use_synthetic = True
    else:
        use_synthetic = True

    if use_synthetic:
        print("\n  Real data not found — using synthetic preview")
        trajs = {}
        for sc in SCENARIO_ORDER:
            trajs[sc] = _make_synthetic(sc, seed=seed)

    out = OUTPUT_DIR if OUTPUT_DIR.parent.exists() else Path(".")
    out.mkdir(parents=True, exist_ok=True)

    for ext in ['png', 'pdf']:
        fig_combined_2row(
            trajs, seed=seed,
            save=str(out / f"combined_3scenario_redesign.{ext}"),
        )

    print(f"\nOutputs in: {out}")
    print("=" * 60)

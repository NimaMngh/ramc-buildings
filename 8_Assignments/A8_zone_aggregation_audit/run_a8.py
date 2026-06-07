"""
Assignment 8 — Zone-Aggregation Risk Audit (P0/P1, R1.1)
=========================================================

Quantifies whether single-zone (volume-weighted) aggregation of the 5-zone
EnergyPlus retail building hides localized cold violations, and builds the
quantitative argument for why a centralized hydronic plant with one
supply-temperature actuator is structurally limited to scalar feedback
aggregation — making the choice of aggregation rule a comfort-policy
decision, not a methodological shortcut.

Pipeline:
  1. Parse the annual EnergyPlus ESO (per-zone air temperature + per-zone
     people heating rate).
  2. Match the nominal-scenario week against the ESO outdoor-temperature
     trace using scenario_metadata.json.
  3. Compute T_vol, T_occ, T_minzone, occupancy flags, and CDH metrics
     for the full January window and the matched nominal week.
  4. Compute the inter-zone spread distribution and the uniform-shift
     counterfactual (cost of edge-biased aggregation).
  5. Save summary JSONs/CSVs and publication figures.

Outputs (relative to this file):
  results/audit_summary.json
  results/audit_results.csv
  results/spread_analysis.json
  results/figures/audit_figure.{png,pdf}
  results/figures/spread_figure.{png,pdf}
  results/raw/eso_parsed.csv
  results/raw/january_timeseries.csv
  results/raw/nominal_week_timeseries.csv

Run from 8_Assignments/ directory:
    python A8_zone_aggregation_audit/run_a8.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# CONFIG — edit these paths to point to the files in your project tree
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

ESO_PATH = (
    PROJECT_ROOT / "1_Building Simulation"
    / "ASHRAE901_RetailStandalone_STD2022GBaseline_Vasteras.eso"
)
NOMINAL_CSV = (
    PROJECT_ROOT / "7_Close Loop Performance" / "M1_Weather_Data"
    / "data_ramc_epw" / "nominal_truth.csv"
)
SCENARIO_META = (
    PROJECT_ROOT / "7_Close Loop Performance" / "M1_Weather_Data"
    / "data_ramc_epw" / "scenario_metadata.json"
)

# Results land next to this file
RESULTS_DIR = Path(__file__).resolve().parent / "results"
FIG_DIR     = RESULTS_DIR / "figures"
RAW_DIR     = RESULTS_DIR / "raw"


# =============================================================================
# Building constants (verified against DataPreprocessing.py & .eio)
# =============================================================================

ZONE_VOLUMES = {
    "BACK_SPACE":    2317.33,
    "CORE_RETAIL":   9762.95,
    "POINT_OF_SALE":  919.94,
    "FRONT_RETAIL":   919.94,
    "FRONT_ENTRY":     73.20,
}
ZONES = list(ZONE_VOLUMES.keys())

# Variable IDs from ESO header
VAR_IDS = {
    "T_out":           7,
    "T_BACK_SPACE":    720, "T_CORE_RETAIL":  721, "T_POINT_OF_SALE": 722,
    "T_FRONT_RETAIL":  723, "T_FRONT_ENTRY":  724,
    "P_BACK_SPACE":    49,  "P_CORE_RETAIL":  52,  "P_POINT_OF_SALE": 55,
    "P_FRONT_RETAIL":  58,  "P_FRONT_ENTRY":  61,
}

T_MIN_COMFORT = 20.0  # from main_experiment.py comfort_bounds[0]
T_MAX_COMFORT = 22.0  # comfort_bounds[1]
DT_MINUTES    = 10
DT_HOURS      = DT_MINUTES / 60.0
OCC_THRESHOLD_W = 1.0   # any People > 1 W counts as occupied


# =============================================================================
# Stage 1 — ESO parser
# =============================================================================

def parse_eso(eso_path: Path, wanted_ids: dict) -> pd.DataFrame:
    """Stream-parse an EnergyPlus ESO. Returns one row per TimeStep record."""
    wanted_set = {str(v) for v in wanted_ids.values()}
    id_to_name = {str(v): k for k, v in wanted_ids.items()}
    rows, current, in_data = [], None, False

    print(f"  Parsing: {eso_path}")
    with open(eso_path, "r", encoding="latin-1", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not in_data:
                if line == "End of Data Dictionary":
                    in_data = True
                continue
            if line.startswith("End of Data"):
                break
            if not line:
                continue
            comma = line.find(",")
            if comma < 0:
                continue
            head, rest = line[:comma], line[comma + 1:]
            if head == "2":
                if current is not None:
                    rows.append(current)
                parts = [p.strip() for p in rest.split(",")]
                if len(parts) < 8:
                    current = None
                    continue
                try:
                    current = {
                        "day_sim":   int(parts[0]),
                        "month":     int(parts[1]),
                        "day":       int(parts[2]),
                        "hour":      int(parts[4]),
                        "start_min": float(parts[5]),
                        "end_min":   float(parts[6]),
                        "daytype":   parts[7] if len(parts) > 7 else "",
                    }
                except ValueError:
                    current = None
            elif head in ("1", "3", "4", "5", "6"):
                if current is not None:
                    rows.append(current)
                current = None
            else:
                if current is None or head not in wanted_set:
                    continue
                try:
                    current[id_to_name[head]] = float(rest)
                except ValueError:
                    pass
    if current is not None:
        rows.append(current)

    df = pd.DataFrame(rows)
    zone_T_cols = [f"T_{z}" for z in ZONES]
    df = df.dropna(subset=zone_T_cols).reset_index(drop=True)
    # EnergyPlus uses 1-indexed hours with end-of-interval semantics
    total_min = (df["hour"].astype(int) - 1) * 60 + df["end_min"].astype(int)
    df["datetime"] = (
        pd.to_datetime(dict(year=2024,
                            month=df["month"].astype(int),
                            day=df["day"].astype(int)), errors="coerce")
        + pd.to_timedelta(total_min, unit="m")
    )
    df = df.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
    return df


# =============================================================================
# Stage 2 — match nominal week
# =============================================================================

def match_nominal_week(eso_df: pd.DataFrame, nominal_df: pd.DataFrame):
    """Locate the nominal scenario's calendar week in the ESO via RMSE on T_out."""
    eso_T = eso_df["T_out"].values
    nom_T = nominal_df["T_out_C"].values
    N = len(nom_T)
    if len(eso_T) < N:
        raise ValueError(f"ESO has {len(eso_T)} steps, need {N}.")
    best_i, best_rmse = -1, np.inf
    for i in range(len(eso_T) - N + 1):
        diff = eso_T[i : i + N] - nom_T
        rmse = float(np.sqrt(np.mean(diff * diff)))
        if rmse < best_rmse:
            best_rmse, best_i = rmse, i
    return best_i, best_rmse


# =============================================================================
# Stage 3 — aggregations and summaries
# =============================================================================

def compute_aggregations(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    zone_T_cols = [f"T_{z}" for z in ZONES]
    zone_P_cols = [f"P_{z}" for z in ZONES]
    w = np.array([ZONE_VOLUMES[z] for z in ZONES])
    T = out[zone_T_cols].values
    P = out[zone_P_cols].values

    out["T_vol"]       = (T * w).sum(axis=1) / w.sum()
    out["T_minzone"]   = T.min(axis=1)
    out["argmin_zone"] = [ZONES[i] for i in np.argmin(T, axis=1)]
    P_sum = P.sum(axis=1)
    out["total_people_W"] = P_sum
    with np.errstate(divide="ignore", invalid="ignore"):
        out["T_occ"] = np.where(
            P_sum > OCC_THRESHOLD_W,
            (T * P).sum(axis=1) / np.maximum(P_sum, 1e-9),
            np.nan,
        )
    out["occupied"] = P_sum > OCC_THRESHOLD_W
    for j, z in enumerate(ZONES):
        out[f"occ_{z}"] = P[:, j] > OCC_THRESHOLD_W
    return out


def cdh(temperature, occupied, T_min, dt_hours):
    valid = occupied & np.isfinite(temperature)
    return float(np.maximum(T_min - temperature[valid], 0.0).sum() * dt_hours)


def cdh_anyzone_local(df, T_min, dt_hours):
    total = 0.0
    for z in ZONES:
        T_z = df[f"T_{z}"].values
        occ_z = df[f"occ_{z}"].values
        valid = occ_z & np.isfinite(T_z)
        total += float(np.maximum(T_min - T_z[valid], 0.0).sum() * dt_hours)
    return total


def summarize_window(df: pd.DataFrame, label: str) -> dict:
    occ = df["occupied"].values
    n_occ, n_total = int(occ.sum()), len(df)
    cdh_vol     = cdh(df["T_vol"].values,     occ, T_MIN_COMFORT, DT_HOURS)
    cdh_occ     = cdh(df["T_occ"].values,     occ, T_MIN_COMFORT, DT_HOURS)
    cdh_minzone = cdh(df["T_minzone"].values, occ, T_MIN_COMFORT, DT_HOURS)
    cdh_any     = cdh_anyzone_local(df, T_MIN_COMFORT, DT_HOURS)
    hidden      = cdh_minzone - cdh_vol
    occ_df      = df[df["occupied"]]
    worst       = occ_df["argmin_zone"].value_counts().to_dict()
    return {
        "window": label, "n_timesteps": n_total, "n_occupied_timesteps": n_occ,
        "frac_occupied": n_occ / max(n_total, 1),
        "T_out_mean_C": float(df["T_out"].mean()),
        "T_out_min_C":  float(df["T_out"].min()),
        "T_out_max_C":  float(df["T_out"].max()),
        "CDH_vol": cdh_vol, "CDH_occ": cdh_occ, "CDH_minzone": cdh_minzone,
        "CDH_anyzone_local": cdh_any, "HiddenCDH": hidden,
        "HiddenCDH_ratio_vs_vol":
            (hidden / cdh_vol) if cdh_vol > 1e-9
            else (float("inf") if hidden > 0 else 0.0),
        "worst_zone_during_occ": worst,
    }


# =============================================================================
# Stage 4 — spread distribution and counterfactual
# =============================================================================

def spread_stats(df_occ: pd.DataFrame) -> dict:
    delta = (df_occ["T_vol"] - df_occ["T_minzone"]).values
    return {
        "mean": float(delta.mean()), "median": float(np.median(delta)),
        "p75": float(np.percentile(delta, 75)),
        "p90": float(np.percentile(delta, 90)),
        "p95": float(np.percentile(delta, 95)),
        "p99": float(np.percentile(delta, 99)),
        "max": float(delta.max()), "values": delta,
    }


def counterfactual(df_occ: pd.DataFrame, T_target: float) -> dict:
    """Uniform-shift counterfactual under min-zone-protect control policy."""
    Tm = df_occ["T_minzone"].values
    Delta = np.maximum(T_target - Tm, 0.0)
    Tm_shift = Tm + Delta
    cdh_min_after = float(np.maximum(T_MIN_COMFORT - Tm_shift, 0).sum() * DT_HOURS)
    hdh_per_zone = {
        z: float(np.maximum(df_occ[f"T_{z}"].values + Delta - T_MAX_COMFORT, 0).sum() * DT_HOURS)
        for z in ZONES
    }
    vol_total = sum(ZONE_VOLUMES.values())
    hdh_vw = sum(hdh_per_zone[z] * ZONE_VOLUMES[z] for z in ZONES) / vol_total
    active = Delta > 1e-9
    return {
        "T_target": T_target,
        "frac_steps_active": float(active.mean()),
        "mean_Delta_when_active": float(Delta[active].mean()) if active.any() else 0.0,
        "max_Delta": float(Delta.max()),
        "CDH_minzone_after": cdh_min_after,
        "HDH_per_zone": hdh_per_zone,
        "HDH_volume_weighted": hdh_vw,
    }


# =============================================================================
# Figures
# =============================================================================

def fig_audit(week_df, week_summary, save_stem):
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                              gridspec_kw={"height_ratios": [3, 1]})
    t = week_df["datetime"]
    ax = axes[0]
    ax.plot(t, week_df["T_vol"],     label="T_vol (volume-weighted)", lw=1.5, color="#2C3E50")
    ax.plot(t, week_df["T_occ"],     label="T_occ (occupancy-weighted)", lw=1.0, color="#3498DB", alpha=0.9)
    ax.plot(t, week_df["T_minzone"], label="T_minzone (coldest zone)", lw=1.5, color="#E74C3C")
    ax.axhline(T_MIN_COMFORT, color="k", ls="--", lw=0.8, alpha=0.6,
               label=f"T_min = {T_MIN_COMFORT}°C")
    occ = week_df["occupied"].astype(bool).values
    diff = np.diff(occ.astype(int), prepend=0, append=0)
    starts, ends = np.where(diff == 1)[0], np.where(diff == -1)[0]
    for s, e in zip(starts, ends):
        if e - s > 1:
            ax.axvspan(t.iloc[s], t.iloc[min(e, len(t)-1)],
                       color="#F1C40F", alpha=0.10, zorder=0)
    ax.set_ylabel("Zone-aggregate temperature (°C)")
    ax.set_title(f"Nominal week zone audit — HiddenCDH = {week_summary['HiddenCDH']:.2f} °C·h "
                 f"(CDH_minzone = {week_summary['CDH_minzone']:.2f}, "
                 f"CDH_vol = {week_summary['CDH_vol']:.2f})")
    ax.legend(loc="lower right", fontsize=9); ax.grid(alpha=0.3)

    ax2 = axes[1]
    z2y = {z: i for i, z in enumerate(ZONES)}
    occ_only = week_df[occ]
    ys = [z2y[z] for z in occ_only["argmin_zone"]]
    ax2.scatter(occ_only["datetime"], ys, s=6, color="#E74C3C", alpha=0.7)
    ax2.set_yticks(list(z2y.values()))
    ax2.set_yticklabels(list(z2y.keys()), fontsize=8)
    ax2.set_ylabel("Coldest zone"); ax2.set_xlabel("Time"); ax2.grid(alpha=0.3)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        plt.savefig(f"{save_stem}.{ext}", dpi=140 if ext == "png" else None)
    plt.close()


def fig_spread(week_occ, full_occ, week_spread, jan_spread, sweep, save_stem):
    fig = plt.figure(figsize=(13, 9))
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.28)

    # Panel A: ECDF
    ax = fig.add_subplot(gs[0, 0])
    for name, s, color in [("Nominal week", week_spread, "#E74C3C"),
                            ("Full January", jan_spread, "#2C3E50")]:
        x = np.sort(s["values"])
        y = np.arange(1, len(x) + 1) / len(x)
        ax.plot(x, y, lw=1.8, color=color, label=f"{name} (n={len(x)})")
    ax.axvline(jan_spread["p90"], color="#2C3E50", ls=":", lw=1, alpha=0.6)
    ax.axvline(jan_spread["p95"], color="#2C3E50", ls="--", lw=1, alpha=0.6)
    ax.text(jan_spread["p90"], 0.02, f"  p90={jan_spread['p90']:.2f}",
            color="#2C3E50", fontsize=9, va="bottom")
    ax.text(jan_spread["p95"], 0.10, f"  p95={jan_spread['p95']:.2f}",
            color="#2C3E50", fontsize=9, va="bottom")
    ax.set_xlabel(r"$\delta(k) = T_{\rm vol}(k) - T_{\rm minzone}(k)$  (°C)")
    ax.set_ylabel("ECDF over occupied steps")
    ax.set_title("Panel A — inter-zone spread (physical, no controller)")
    ax.legend(loc="lower right", fontsize=9); ax.grid(alpha=0.3); ax.set_xlim(left=0)

    # Panel B: nominal-week histogram
    ax = fig.add_subplot(gs[0, 1])
    ax.hist(week_spread["values"], bins=40, color="#E74C3C", alpha=0.85, edgecolor="white")
    ax.axvline(week_spread["mean"], color="k", ls="--", lw=1.2,
               label=f"mean = {week_spread['mean']:.2f} °C")
    ax.axvline(week_spread["p90"], color="k", ls=":", lw=1,
               label=f"p90 = {week_spread['p90']:.2f} °C")
    ax.set_xlabel(r"$\delta$ on nominal week (°C)"); ax.set_ylabel("Count")
    ax.set_title("Panel B — nominal-week spread distribution")
    ax.legend(fontsize=9); ax.grid(alpha=0.3, axis="y")

    # Panel C: trade-off curve
    ax = fig.add_subplot(gs[1, 0])
    targets = [r["T_target"] for r in sweep]
    cdh_after = [r["CDH_minzone_after"] for r in sweep]
    hdh_core  = [r["HDH_per_zone"]["CORE_RETAIL"] for r in sweep]
    hdh_vw    = [r["HDH_volume_weighted"] for r in sweep]
    ax2 = ax.twinx()
    p1, = ax.plot(targets, cdh_after, "o-", color="#3498DB", lw=2,
                   label=r"CDH$_{\rm minzone}$ after shift")
    p2, = ax2.plot(targets, hdh_core, "s-", color="#E74C3C", lw=2,
                    label="HDH on CORE_RETAIL (70% of vol.)")
    p3, = ax2.plot(targets, hdh_vw, "^--", color="#F39C12", lw=2, alpha=0.8,
                    label="HDH vol.-weighted (whole building)")
    ax.set_xlabel(r"Min-zone protection target $T_{\rm target}$  (°C)")
    ax.set_ylabel("CDH$_{\\rm minzone}$ after shift (°C·h)", color="#3498DB")
    ax2.set_ylabel("Resulting hot degree-hours (°C·h)", color="#E74C3C")
    ax.set_title("Panel C — controller trade-off (nominal week, uniform-shift approx.)")
    ax.tick_params(axis="y", labelcolor="#3498DB")
    ax2.tick_params(axis="y", labelcolor="#E74C3C")
    ax.grid(alpha=0.3)
    ax.legend([p1, p2, p3], [p.get_label() for p in (p1, p2, p3)],
              loc="upper left", fontsize=9)

    # Panel D: per-zone overheat at the protective operating point
    ax = fig.add_subplot(gs[1, 1])
    protective = sweep[-1]
    zone_hdh = [protective["HDH_per_zone"][z] for z in ZONES]
    colors = ["#7F8C8D" if z != "CORE_RETAIL" else "#E74C3C" for z in ZONES]
    bars = ax.barh(ZONES, zone_hdh, color=colors, edgecolor="white")
    for b, v, z in zip(bars, zone_hdh, ZONES):
        vol_pct = 100 * ZONE_VOLUMES[z] / sum(ZONE_VOLUMES.values())
        ax.text(v + 0.5, b.get_y() + b.get_height() / 2,
                f"{v:.1f} °C·h  ({vol_pct:.1f}% of vol.)", va="center", fontsize=8)
    ax.set_xlabel(f"Hot degree-hours under T_target = {protective['T_target']:.1f} °C")
    ax.set_title("Panel D — per-zone overheat induced by min-zone protection")
    ax.invert_yaxis(); ax.grid(alpha=0.3, axis="x")
    xmax = max(zone_hdh) * 1.45 if max(zone_hdh) > 0 else 1
    ax.set_xlim(0, xmax)

    plt.suptitle(
        "Inter-zone spread and the cost of edge-biased aggregation\n"
        "(Centralized hydronic plant: one supply-temperature actuator; "
        "feedback aggregation is a comfort-policy choice)",
        fontsize=11, y=0.995,
    )
    for ext in ("png", "pdf"):
        plt.savefig(f"{save_stem}.{ext}", dpi=140 if ext == "png" else None,
                     bbox_inches="tight")
    plt.close()


# =============================================================================
# Main
# =============================================================================

def main():
    for d in (RESULTS_DIR, FIG_DIR, RAW_DIR):
        d.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("A8 — Zone-Aggregation Risk Audit (P0/P1, addresses R1.1)")
    print("=" * 70)

    # ── Stage 1: parse ESO ─────────────────────────────────────────────────
    print("\n[Stage 1] Parsing ESO...")
    df = parse_eso(ESO_PATH, VAR_IDS)
    print(f"  {len(df)} TimeStep records, date range "
          f"{df['datetime'].min()} -> {df['datetime'].max()}")
    df.to_csv(RAW_DIR / "eso_parsed.csv", index=False)

    # ── Stage 2: load scenario, match nominal week ────────────────────────
    print("\n[Stage 2] Matching nominal week against ESO...")
    nom = pd.read_csv(NOMINAL_CSV)
    with open(SCENARIO_META) as f:
        meta = json.load(f)
    start_idx, rmse = match_nominal_week(df, nom)
    week_df = df.iloc[start_idx : start_idx + len(nom)].reset_index(drop=True)
    print(f"  Match: episode_id={meta['episode_id']}, start_idx={start_idx}, "
          f"RMSE={rmse:.4f}°C")
    print(f"  Window: {week_df['datetime'].iloc[0]} -> {week_df['datetime'].iloc[-1]}")

    # ── Stage 3: aggregations + summaries ──────────────────────────────────
    print("\n[Stage 3] Computing zone aggregations...")
    jan_full = compute_aggregations(df)
    week     = compute_aggregations(week_df)
    summaries = [
        summarize_window(jan_full, "full_january"),
        summarize_window(week,     "nominal_week"),
    ]
    pd.DataFrame(summaries).to_csv(RESULTS_DIR / "audit_results.csv", index=False)
    jan_full.to_csv(RAW_DIR / "january_timeseries.csv", index=False)
    week.to_csv(RAW_DIR / "nominal_week_timeseries.csv", index=False)

    audit_summary = {
        "scenario_metadata": meta,
        "match": {
            "start_idx_in_eso":      int(start_idx),
            "match_rmse_T_out_C":    float(rmse),
            "matched_datetime_start": str(week_df["datetime"].iloc[0]),
            "matched_datetime_end":   str(week_df["datetime"].iloc[-1]),
        },
        "config": {
            "T_min_comfort_C": T_MIN_COMFORT,
            "T_max_comfort_C": T_MAX_COMFORT,
            "dt_minutes":      DT_MINUTES,
            "occ_threshold_W": OCC_THRESHOLD_W,
            "zone_volumes_m3": ZONE_VOLUMES,
        },
        "summaries": summaries,
    }
    with open(RESULTS_DIR / "audit_summary.json", "w") as f:
        json.dump(audit_summary, f, indent=2, default=float)

    # ── Stage 4: spread + counterfactual ───────────────────────────────────
    print("\n[Stage 4] Computing inter-zone spread + counterfactual...")
    week_occ = week[week["occupied"].astype(bool)].reset_index(drop=True)
    full_occ = jan_full[jan_full["occupied"].astype(bool)].reset_index(drop=True)
    ws, js = spread_stats(week_occ), spread_stats(full_occ)
    targets = np.arange(18.0, 21.01, 0.5)
    sweep_week = [counterfactual(week_occ, T) for T in targets]
    sweep_full = [counterfactual(full_occ, T) for T in targets]

    spread_out = {
        "scope": "occupied steps only",
        "T_min_comfort": T_MIN_COMFORT, "T_max_comfort": T_MAX_COMFORT,
        "dt_minutes": DT_MINUTES,
        "spread_distribution": {
            "nominal_week": {k: v for k, v in ws.items() if k != "values"},
            "full_january": {k: v for k, v in js.items() if k != "values"},
        },
        "counterfactual_sweep_nominal_week": sweep_week,
        "counterfactual_sweep_full_january": sweep_full,
        "approximation_note": (
            "Counterfactual uses a uniform-shift approximation: under a "
            "centralized hydronic actuator, a supply-temperature elevation "
            "Δ is assumed to raise all zone temperatures by Δ. This is "
            "pessimistic for CORE_RETAIL (which in practice responds more "
            "than peripheral zones to a uniform supply elevation due to "
            "lower envelope loss) and optimistic for FRONT_ENTRY. The "
            "approximation is appropriate for first-order trade-off "
            "characterization in the small-perturbation regime."
        ),
    }
    with open(RESULTS_DIR / "spread_analysis.json", "w") as f:
        json.dump(spread_out, f, indent=2, default=float)

    # ── Stage 5: figures ──────────────────────────────────────────────────
    print("\n[Stage 5] Generating figures...")
    fig_audit(week, summaries[1], FIG_DIR / "audit_figure")
    fig_spread(week_occ, full_occ, ws, js, sweep_week, FIG_DIR / "spread_figure")

    # ── Console summary ────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("HEADLINE RESULTS")
    print("=" * 70)
    for s in summaries:
        print(f"\n[{s['window']}]")
        print(f"  CDH_vol     = {s['CDH_vol']:.3f} °C·h")
        print(f"  CDH_minzone = {s['CDH_minzone']:.3f} °C·h")
        print(f"  HiddenCDH   = {s['HiddenCDH']:.3f} °C·h  "
              f"({s['HiddenCDH_ratio_vs_vol']:.2f}× CDH_vol)")
        print(f"  Worst zone  : {s['worst_zone_during_occ']}")
    print(f"\n  Spread p90 (nominal week) : {ws['p90']:.3f} °C")
    print(f"  Spread p90 (full January) : {js['p90']:.3f} °C")
    print(f"\nAll outputs written under: {RESULTS_DIR}")


if __name__ == "__main__":
    main()

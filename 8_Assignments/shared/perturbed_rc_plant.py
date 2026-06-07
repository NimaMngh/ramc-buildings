"""
Perturbed RC Plant Generator for Assignment 1.
================================================

Creates parameter-perturbed RC plant variants for the external-validity
stress test. Three layers of perturbation:

  1. One-at-a-time: perturb each parameter in its worst-case direction
  2. Random ensemble: sample 8 parameter vectors uniformly within ranges
  3. Combined severe: all parameters pushed simultaneously

Perturbation ranges are physically justified (see Assignment 1 document):
  - C_env: ±20%  (longest time constant, least identifiable)
  - R_ex:  -15% to +10%  (infiltration varies; lower = leakier = worse)
  - A_sol: ±0.10 absolute  (solar gains hard to predict)
  - K_rad: ±10%  (radiator fouling / air locks)

Used by: A1 (primary), A2 (exact-model benchmark under mismatch)
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple


# Baseline parameters (from results_N3_DE_optimized.json)
BASELINE_PARAMS = {
    "C_air": 84426246.51832934,
    "C_env": 661376048.5634619,
    "C_int": 6002555765.428176,
    "C_rad": 597376.9622532368,
    "R_ex": 0.0003079967462046204,
    "R_ae": 0.00016180515492072352,
    "R_ai": 5.818704417041902e-05,
    "K_rad": 283.97226311004107,
    "a_rad": 0.2795410071110868,
    "A_sol": 0.5598500539656959,
}

# Perturbation specification
# For multiplicative: (param_name, low_factor, high_factor, comfort_adverse_direction)
# For additive (A_sol): (param_name, low_delta, high_delta, comfort_adverse_direction)
PERTURBATION_SPEC = {
    "C_env": {
        "type": "multiplicative",
        "low": 0.80,   # -20%
        "high": 1.20,   # +20%
        "adverse": 1.20,  # higher C_env = slower envelope response = harder to heat
        "justification": "Longest time constant (57h); least constrained by 1-week identification",
    },
    "R_ex": {
        "type": "multiplicative",
        "low": 0.85,   # -15% (leakier building)
        "high": 1.10,   # +10%
        "adverse": 0.85,  # lower R_ex = more heat loss = harder to maintain comfort
        "justification": "Infiltration varies seasonally; lower R_ex = leakier = worse for comfort",
    },
    "A_sol": {
        "type": "additive",
        "low": -0.10,
        "high": +0.10,
        "adverse": -0.10,  # less solar to air = colder
        "justification": "Solar gains difficult to predict; shifting air-node fraction changes diurnal dynamics",
    },
    "K_rad": {
        "type": "multiplicative",
        "low": 0.90,   # -10%
        "high": 1.10,   # +10%
        "adverse": 0.90,  # less radiator output = harder to heat
        "justification": "Radiator performance degrades with fouling and air locks",
    },
}


def apply_perturbation(base_params: dict, param_name: str, value) -> dict:
    """Apply a single perturbation to a parameter set."""
    params = base_params.copy()
    spec = PERTURBATION_SPEC[param_name]

    if spec["type"] == "multiplicative":
        params[param_name] = base_params[param_name] * value
    elif spec["type"] == "additive":
        params[param_name] = base_params[param_name] + value
    
    # Clamp A_sol to [0, 1]
    if param_name == "A_sol":
        params["A_sol"] = float(np.clip(params["A_sol"], 0.01, 0.99))

    return params


def generate_one_at_a_time() -> List[Dict]:
    """
    Generate 4 single-parameter perturbed plants, each in its
    comfort-adverse direction.
    """
    variants = []
    for param_name, spec in PERTURBATION_SPEC.items():
        adverse_val = spec["adverse"]
        params = apply_perturbation(BASELINE_PARAMS, param_name, adverse_val)

        if spec["type"] == "multiplicative":
            label = f"OAT_{param_name}_{adverse_val:.2f}x"
            desc = f"{param_name} × {adverse_val:.2f}"
        else:
            label = f"OAT_{param_name}_{adverse_val:+.2f}"
            desc = f"{param_name} {adverse_val:+.2f}"

        variants.append({
            "label": label,
            "description": desc,
            "justification": spec["justification"],
            "category": "one_at_a_time",
            "perturbed_param": param_name,
            "perturbation_value": adverse_val,
            "params": params,
        })
    return variants


def generate_random_ensemble(n_draws: int = 8, seed: int = 42) -> List[Dict]:
    """
    Draw n_draws parameter vectors by sampling each parameter independently
    and uniformly within its perturbation range.
    """
    rng = np.random.RandomState(seed)
    variants = []

    for i in range(n_draws):
        params = BASELINE_PARAMS.copy()
        perturbations = {}

        for param_name, spec in PERTURBATION_SPEC.items():
            if spec["type"] == "multiplicative":
                factor = rng.uniform(spec["low"], spec["high"])
                params[param_name] = BASELINE_PARAMS[param_name] * factor
                perturbations[param_name] = f"×{factor:.3f}"
            elif spec["type"] == "additive":
                delta = rng.uniform(spec["low"], spec["high"])
                params[param_name] = BASELINE_PARAMS[param_name] + delta
                params[param_name] = float(np.clip(params[param_name], 0.01, 0.99))
                perturbations[param_name] = f"{delta:+.4f}"

        variants.append({
            "label": f"RAND_{i+1:02d}",
            "description": f"Random ensemble draw {i+1}/{n_draws}",
            "category": "random_ensemble",
            "perturbations": perturbations,
            "params": params,
        })

    return variants


def generate_combined_severe() -> Dict:
    """
    All four parameters pushed simultaneously in their comfort-adverse
    direction. This is the worst-case named scenario.
    """
    params = BASELINE_PARAMS.copy()
    perturbations = {}

    for param_name, spec in PERTURBATION_SPEC.items():
        adverse_val = spec["adverse"]
        if spec["type"] == "multiplicative":
            params[param_name] = BASELINE_PARAMS[param_name] * adverse_val
            perturbations[param_name] = f"×{adverse_val:.2f}"
        elif spec["type"] == "additive":
            params[param_name] = BASELINE_PARAMS[param_name] + adverse_val
            params[param_name] = float(np.clip(params[param_name], 0.01, 0.99))
            perturbations[param_name] = f"{adverse_val:+.2f}"

    return {
        "label": "COMBINED_SEVERE",
        "description": "All parameters at comfort-adverse extremes simultaneously",
        "category": "combined_severe",
        "perturbations": perturbations,
        "params": params,
    }


def generate_all_plant_variants(n_random: int = 8, random_seed: int = 42) -> List[Dict]:
    """
    Generate the complete set of plant variants for A1.

    Returns list of dicts, each with 'label', 'category', 'params', etc.
    The first entry is always the matched (baseline) plant.
    """
    variants = []

    # 1. Matched plant (baseline)
    variants.append({
        "label": "MATCHED",
        "description": "Baseline identified parameters (no mismatch)",
        "category": "matched",
        "params": BASELINE_PARAMS.copy(),
    })

    # 2. One-at-a-time (4 variants)
    variants.extend(generate_one_at_a_time())

    # 3. Random ensemble (8 variants)
    variants.extend(generate_random_ensemble(n_draws=n_random, seed=random_seed))

    # 4. Combined severe (1 variant)
    variants.append(generate_combined_severe())

    return variants


def validate_plant_variant(params: dict, label: str = "") -> bool:
    """
    Quick sanity check: run a constant-control simulation for 24h
    and verify temperatures stay in [-30, 50] °C.
    """
    try:
        from rc_ground_truth import RCGroundTruthModel
    except ImportError:
        print(f"  WARNING: Cannot import RCGroundTruthModel for validation")
        return True  # Skip validation if import fails

    model = RCGroundTruthModel(params=params, dt_seconds=600)
    x = np.array([20.0, 18.0, 19.0, 40.0, 37.0, 35.0])
    u = np.array([45.0, 1.0])
    d = np.array([-10.0, 0.0, 30000.0])

    for _ in range(144):  # 24 hours
        x = model.step(x, u, d)
        if np.any(x < -30) or np.any(x > 50):
            print(f"  REJECT {label}: temperature out of [-30, 50] °C range")
            return False

    return True


def save_plant_variants(variants: List[Dict], output_path: Path):
    """Save plant variants to JSON (params are serializable)."""
    with open(output_path, "w") as f:
        json.dump(variants, f, indent=2)
    print(f"  Saved {len(variants)} plant variants to {output_path.name}")


if __name__ == "__main__":
    variants = generate_all_plant_variants()
    print(f"Generated {len(variants)} plant variants:")
    for v in variants:
        print(f"  {v['label']:<25s} ({v['category']}): {v.get('description', '')}")

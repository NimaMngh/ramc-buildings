"""
Verification script for the RAMC assignments infrastructure.
Run this ONCE after setting up the 8_Assignments folder to confirm
all paths resolve and all imports will work.

Usage (from the 8_Assignments directory):
    python verify_setup.py

Or from the project root:
    python 8_Assignments/verify_setup.py
"""

import sys
from pathlib import Path

# Ensure we can import from shared/
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

def main():
    print("=" * 65)
    print("  RAMC Assignments — Infrastructure Verification")
    print("=" * 65)
    
    # ---- Step 1: Import paths ----
    print("\n[1/5] Importing shared.paths ...")
    try:
        from shared.paths import (
            PROJECT_ROOT, RESULTS_DIR, SCENARIO_DIR,
            RC_PARAMS_JSON, TRAINING_DATA_CSV, DECOMPOSED_EVAL_CSV,
            ALL_CHECKPOINTS, SCENARIOS, PHASE3_MODELS,
            NMPC_AGGREGATE_CSV,
            NN_ARCH_DIR, CLOSED_LOOP_DIR, NMPC_DIR,
            setup_imports, verify_paths, get_results_dir,
            COMFORT_BAND, PHASE3_SEEDS,
            A1_DIR, A2_DIR, A3_DIR,
        )
        print(f"  OK — PROJECT_ROOT = {PROJECT_ROOT}")
    except Exception as e:
        print(f"  FAIL — {e}")
        print("  Fix shared/paths.py before proceeding.")
        return False
    
    # ---- Step 2: Verify all paths ----
    print("\n[2/5] Verifying file paths ...")
    results = verify_paths(verbose=True)
    missing = [k for k, (p, exists) in results.items() if not exists]
    
    # ---- Step 3: Check key data files are readable ----
    print("\n[3/5] Checking key data files ...")
    data_checks = {
        "RC params JSON": RC_PARAMS_JSON,
        "Decomposed eval CSV": DECOMPOSED_EVAL_CSV,
        "NMPC aggregate CSV": NMPC_AGGREGATE_CSV,
    }
    for label, path in data_checks.items():
        if path.exists():
            size_kb = path.stat().st_size / 1024
            print(f"  OK — {label}: {size_kb:.1f} KB")
        else:
            print(f"  SKIP — {label} not found")
    
    # ---- Step 4: Test setup_imports ----
    print("\n[4/5] Testing setup_imports() ...")
    setup_imports()
    
    importable = []
    not_importable = []
    
    modules_to_check = [
        ("thermal_dynamics_net", "ThermalDynamicsNet"),
        ("ramc_losses", None),
        ("rc_ground_truth", "RCGroundTruthModel"),
        ("load_ramc_model", "load_ramc_model"),
        ("nmpc_direct_shooting", None),
        ("closed_loop_nmpc", "ClosedLoopNMPCSimulator"),
    ]
    
    for module_name, class_name in modules_to_check:
        try:
            mod = __import__(module_name)
            if class_name:
                assert hasattr(mod, class_name), f"{class_name} not found in {module_name}"
            importable.append(module_name)
            print(f"  OK — import {module_name}" + (f" ({class_name})" if class_name else ""))
        except Exception as e:
            not_importable.append((module_name, str(e)))
            print(f"  FAIL — import {module_name}: {e}")
    
    # ---- Step 5: Test results directory creation ----
    print("\n[5/5] Testing results directory creation ...")
    try:
        test_dir = get_results_dir(A3_DIR, "test_verification")
        assert test_dir.is_dir()
        # Clean up
        test_dir.rmdir()
        print(f"  OK — get_results_dir works")
    except Exception as e:
        print(f"  FAIL — {e}")
    
    # ---- Summary ----
    print("\n" + "=" * 65)
    ok = len(missing) == 0 and len(not_importable) == 0
    if ok:
        print("  ALL CHECKS PASSED — ready to start Assignment 3")
    else:
        if missing:
            print(f"  {len(missing)} missing path(s) — check directory structure")
        if not_importable:
            print(f"  {len(not_importable)} failed import(s) — check sys.path or dependencies")
        print("  Fix issues above before running assignment scripts.")
    print("=" * 65)
    
    return ok


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TARGETS = ["PHIF", "SW", "VSH"]
SEEDS = [42, 7, 123]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run balanced per-well sampling sensitivity for WL-PCQR.")
    parser.add_argument("--max-per-well", type=int, default=320)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    methods = ["global_rf_cqr", "wl_pcqr"]
    for target in TARGETS:
        for seed in SEEDS:
            out = ROOT / "analysis" / f"wlpcqr_maxperwell{args.max_per_well}_{target}_seed{seed}_20260516"
            marker = out / f"{target.lower()}_wlpcqr_target_well_aggregate_metrics.csv"
            if marker.exists() and not args.force:
                print(f"[skip] {marker}")
                continue
            cmd = [
                sys.executable,
                "scripts/run_wlpcqr_methods.py",
                "--out",
                str(out),
                "--target",
                target,
                "--seed",
                str(seed),
                "--max-per-well",
                str(args.max_per_well),
                "--methods",
                *methods,
                "--local-calibration-frac",
                "0.20",
                "--local-calibration-strategy",
                "stratified",
                "--exclude-features",
                "DEPTH",
            ]
            print("[run]", " ".join(cmd))
            subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()

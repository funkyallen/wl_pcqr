from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def run_command(cmd: list[str], log_path: Path) -> float:
    start = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(
            cmd,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    elapsed = time.perf_counter() - start
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with code {proc.returncode}: {' '.join(cmd)}. See {log_path}")
    return elapsed


def read_first_metric(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(path)
    rows = pd.read_csv(path)
    if rows.empty:
        raise RuntimeError(f"Empty metric file: {path}")
    return rows.iloc[0].to_dict()


def classical_spec(method: str, label: str, target: str, seed: int, out_dir: Path, args: argparse.Namespace) -> dict[str, object]:
    run_dir = out_dir / f"{target.lower()}_{method}_seed{seed}"
    cmd = [
        sys.executable,
        "scripts/run_wlpcqr_methods.py",
        "--out",
        str(run_dir),
        "--target",
        target,
        "--seed",
        str(seed),
        "--max-per-well",
        str(args.max_per_well),
        "--methods",
        method,
        "--local-calibration-frac",
        str(args.local_calibration_frac),
        "--local-calibration-strategy",
        args.local_calibration_strategy,
        "--exclude-features",
        *args.exclude_features,
    ]
    return {
        "method": method,
        "method_label": label,
        "family": "classical",
        "cmd": cmd,
        "run_dir": run_dir,
        "metrics": run_dir / f"{target.lower()}_wlpcqr_target_well_aggregate_metrics.csv",
    }


def make_specs(args: argparse.Namespace) -> list[dict[str, object]]:
    classical = [
        ("wl_pcqr", "WL-PCQR"),
        ("global_rf_cqr", "Global RF-CQR"),
        ("extratrees_pcqr", "ExtraTrees-PCQR"),
        ("ngboost_cqr_signed", "NGBoost PI"),
        ("hgb_cqr_signed", "HGB-CQR"),
        ("lightgbm_cqr_signed", "LightGBM Quantile+CQR"),
        ("catboost_cqr_signed", "CatBoost Quantile+CQR"),
    ]
    specs = [
        classical_spec(method, label, args.target, args.seed, args.out, args)
        for method, label in classical
    ]
    if args.methods:
        wanted = set(args.methods)
        specs = [spec for spec in specs if spec["method"] in wanted]
    return specs


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark wall-clock time for WL-PCQR and compact CQR baselines.")
    parser.add_argument("--out", type=Path, default=ROOT / "analysis" / "runtime_benchmark")
    parser.add_argument("--target", default="PHIF", choices=["PHIF", "SW", "VSH"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-per-well", type=int, default=160)
    parser.add_argument("--local-calibration-frac", type=float, default=0.20)
    parser.add_argument("--local-calibration-strategy", choices=["random", "stratified", "contiguous"], default="stratified")
    parser.add_argument("--exclude-features", nargs="*", default=["DEPTH"])
    parser.add_argument("--methods", nargs="*", default=[])
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    specs = make_specs(args)
    if not specs:
        raise RuntimeError("No methods selected.")

    for index, spec in enumerate(specs, start=1):
        method = str(spec["method"])
        print(f"[{index}/{len(specs)}] {method}")
        run_dir = Path(spec["run_dir"])
        run_dir.mkdir(parents=True, exist_ok=True)
        seconds = run_command(list(spec["cmd"]), args.out / f"{method}.log")
        metrics = read_first_metric(Path(spec["metrics"]))
        row = {
            "target": args.target,
            "seed": args.seed,
            "method": method,
            "method_label": spec["method_label"],
            "family": spec["family"],
            "wall_time_sec": seconds,
            "wall_time_min": seconds / 60.0,
            "run_dir": str(run_dir),
        }
        for col in [
            "splits",
            "n",
            "rmse",
            "coverage",
            "width",
            "winkler",
            "macro_well_coverage",
            "p10_well_coverage",
            "worst_well_coverage",
            "macro_well_winkler",
        ]:
            if col in metrics:
                row[col] = metrics[col]
        rows.append(row)
        pd.DataFrame(rows).to_csv(args.out / "runtime_results_partial.csv", index=False)

    result = pd.DataFrame(rows)
    result["rank_winkler"] = result["winkler"].rank(method="min", ascending=True).astype(int)
    result["rank_time"] = result["wall_time_sec"].rank(method="min", ascending=True).astype(int)
    result["time_per_split_sec"] = result["wall_time_sec"] / result["splits"].astype(float)
    result["estimated_3target_3seed_min"] = result["wall_time_min"] * 9.0
    result.to_csv(args.out / "runtime_results.csv", index=False)

    with (args.out / "runtime_results.md").open("w", encoding="utf-8", newline="") as f:
        f.write("# Runtime Benchmark\n\n")
        f.write(f"- target: `{args.target}`\n")
        f.write(f"- seed: `{args.seed}`\n")
        f.write(f"- max_per_well: `{args.max_per_well}`\n")
        f.write(f"- local calibration: `{args.local_calibration_strategy}` at `{args.local_calibration_frac}`\n")
        f.write(f"- excluded features: `{', '.join(args.exclude_features) if args.exclude_features else 'none'}`\n\n")
        cols = [
            "method",
            "method_label",
            "family",
            "wall_time_sec",
            "time_per_split_sec",
            "estimated_3target_3seed_min",
            "winkler",
            "coverage",
            "width",
            "worst_well_coverage",
            "rank_winkler",
            "rank_time",
        ]
        f.write(result[cols].sort_values("rank_winkler").to_markdown(index=False, floatfmt=".4f"))
        f.write("\n")

    print(result.sort_values("rank_winkler")[["method", "wall_time_sec", "winkler", "coverage", "width", "rank_winkler", "rank_time"]].to_string(index=False))


if __name__ == "__main__":
    main()

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    rows = []
    for path in ROOT.glob("analysis/wlpcqr_maxperwell320_*_seed*_20260516/*_aggregate_metrics.csv"):
        rows.append(pd.read_csv(path))
    if not rows:
        raise FileNotFoundError("No max_per_well=320 aggregate files found.")

    raw = pd.concat(rows, ignore_index=True)
    out_dir = ROOT / "analysis" / "final_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "maxperwell320_sensitivity_raw.csv"
    mean_path = out_dir / "maxperwell320_sensitivity_mean.csv"
    raw.to_csv(raw_path, index=False)

    cols = [
        "coverage",
        "p10_well_coverage",
        "worst_well_coverage",
        "width",
        "winkler",
        "rmse",
        "macro_well_coverage",
        "macro_well_winkler",
    ]
    summary = raw.groupby(["method", "method_label"], as_index=False)[cols].mean().sort_values("winkler")
    summary.to_csv(mean_path, index=False)
    print(summary.to_string(index=False))
    print(f"Wrote {raw_path}")
    print(f"Wrote {mean_path}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import wlpcqr_core as base


ROOT = Path(__file__).resolve().parents[1]
TARGETS = ["PHIF", "SW", "VSH"]
SEEDS = [42, 7, 123]


def cqr_scores(y: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    return np.maximum.reduce([lower - y, y - upper, np.zeros_like(y, dtype=float)])


def evaluate_split(y: np.ndarray, pred: np.ndarray, lower: np.ndarray, upper: np.ndarray, alpha: float) -> dict[str, float]:
    metrics = {}
    metrics.update(base.point_metrics(y, pred))
    metrics.update(base.interval_metrics(y, lower, upper, alpha))
    return metrics


def run_one(target: str, seed: int, max_per_well: int, alpha: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    base.ACTIVE_FEATURES = [f for f in base.FEATURES if f != "DEPTH"]
    df = base.load_dataset(ROOT / "data" / "raw" / "train.csv", target, max_per_well=max_per_well, seed=seed)
    wells = sorted(df["WELLNUM"].astype(int).unique())
    split_rows = []
    pred_rows = []

    for test_well in wells:
        source_wells = [w for w in wells if w != test_well]
        score_parts = []
        for cal_well in source_wells:
            train_wells = [w for w in source_wells if w != cal_well]
            train = df[df["WELLNUM"].isin(train_wells)]
            cal = df[df["WELLNUM"].astype(int) == cal_well]
            _, cal_lower, cal_upper = base.fit_predict_tree_quantiles(train, cal, target, "rf", alpha, seed)
            score_parts.append(cqr_scores(cal[target].to_numpy(dtype=float), cal_lower, cal_upper))

        qhat = base.conformal_quantile(np.concatenate(score_parts), alpha)
        train_final = df[df["WELLNUM"].isin(source_wells)]
        test = df[df["WELLNUM"].astype(int) == test_well].copy()
        pred, lower, upper = base.fit_predict_tree_quantiles(train_final, test, target, "rf", alpha, seed)
        lower, upper = base.clip_interval(lower - qhat, upper + qhat)
        y = test[target].to_numpy(dtype=float)
        metrics = evaluate_split(y, pred, lower, upper, alpha)
        metrics.update(
            {
                "target": target,
                "seed": seed,
                "method": "source_only_rf_cqr",
                "method_label": "Source-only RF-CQR",
                "test_well": int(test_well),
                "n": int(len(test)),
                "qhat": float(qhat),
            }
        )
        split_rows.append(metrics)

        pred_frame = test[["WELLNUM", "DEPTH", target]].copy()
        pred_frame["seed"] = seed
        pred_frame["method"] = "source_only_rf_cqr"
        pred_frame["pred"] = pred
        pred_frame["lower"] = lower
        pred_frame["upper"] = upper
        pred_frame["width"] = upper - lower
        pred_frame["covered"] = (y >= lower) & (y <= upper)
        pred_rows.append(pred_frame)

    split = pd.DataFrame(split_rows)
    weights = split["n"].to_numpy(dtype=float)
    aggregate = {
        "target": target,
        "seed": seed,
        "method": "source_only_rf_cqr",
        "method_label": "Source-only RF-CQR",
        "splits": int(len(split)),
        "n": int(split["n"].sum()),
    }
    for col in ["rmse", "mae", "r2", "corr", "coverage", "width", "width_std", "width_cv", "winkler"]:
        aggregate[col] = float(np.average(split[col], weights=weights))
    well_cov = split["coverage"].to_numpy(dtype=float)
    aggregate["macro_well_coverage"] = float(np.mean(well_cov))
    aggregate["p10_well_coverage"] = float(np.quantile(well_cov, 0.10))
    aggregate["worst_well_coverage"] = float(np.min(well_cov))
    aggregate["macro_well_width"] = float(np.mean(split["width"]))
    aggregate["macro_well_winkler"] = float(np.mean(split["winkler"]))
    return pd.DataFrame([aggregate]), pd.concat(pred_rows, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run source-only residual-pooling RF-CQR baseline.")
    parser.add_argument("--out", default=str(ROOT / "analysis" / "source_only_baseline"))
    parser.add_argument("--max-per-well", type=int, default=160)
    parser.add_argument("--alpha", type=float, default=0.10)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_aggs = []
    for target in TARGETS:
        for seed in SEEDS:
            agg, pred = run_one(target, seed, args.max_per_well, args.alpha)
            all_aggs.append(agg)
            tag = f"{target.lower()}_seed{seed}"
            agg.to_csv(out_dir / f"{tag}_aggregate_metrics.csv", index=False)
            pred.to_csv(out_dir / f"{tag}_predictions.csv", index=False)
            print(agg.to_string(index=False))

    raw = pd.concat(all_aggs, ignore_index=True)
    raw.to_csv(out_dir / "source_only_raw.csv", index=False)
    cols = ["coverage", "p10_well_coverage", "worst_well_coverage", "width", "winkler", "rmse"]
    summary = raw.groupby(["method", "method_label"], as_index=False)[cols].mean()
    summary.to_csv(out_dir / "source_only_mean.csv", index=False)
    final_dir = ROOT / "analysis" / "final_results"
    final_dir.mkdir(parents=True, exist_ok=True)
    raw.to_csv(final_dir / "source_only_raw.csv", index=False)
    summary.to_csv(final_dir / "source_only_mean.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

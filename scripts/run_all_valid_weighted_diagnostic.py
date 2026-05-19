from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import pandas as pd

import run_wlpcqr_methods as local
import wlpcqr_core as base


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "analysis" / "all_valid_weighted_diagnostic"
TARGETS = ["PHIF", "SW", "VSH"]
METHOD_LABELS = {
    "global_rf_cqr": "Global RF-CQR",
    "wl_pcqr": "WL-PCQR",
}


def stratified_capped_indices(n: int, frac: float, cap: int) -> tuple[np.ndarray, np.ndarray]:
    if n < 2:
        raise ValueError("At least two target-well samples are required.")
    n_cal = int(round(n * frac))
    n_cal = min(max(n_cal, 1), n - 1, cap)
    all_idx = np.arange(n)
    raw = np.linspace(0, n - 1, n_cal)
    cal_idx = np.unique(np.round(raw).astype(int))
    if len(cal_idx) < n_cal:
        remaining = np.setdiff1d(all_idx, cal_idx)
        need = n_cal - len(cal_idx)
        cal_idx = np.sort(np.concatenate([cal_idx, remaining[:need]]))
    test_idx = np.setdiff1d(all_idx, cal_idx)
    return cal_idx, test_idx


def weighted_mean(group: pd.DataFrame, col: str) -> float:
    weights = group["n"].to_numpy(dtype=float)
    values = group[col].to_numpy(dtype=float)
    if float(np.sum(weights)) <= 0.0:
        return float(np.mean(values))
    return float(np.average(values, weights=weights))


def record_metrics(
    target: str,
    seed: int,
    well: object,
    method: str,
    y: np.ndarray,
    pred: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    alpha: float,
    train_n: int,
    calibration_n: int,
    selected_level: str,
    qhat: float,
) -> dict[str, object]:
    rec: dict[str, object] = {
        "protocol": "ALL_VALID_EVAL_CAPPED_CAL",
        "target": target,
        "seed": seed,
        "well": str(well),
        "method": method,
        "method_label": METHOD_LABELS[method],
        "alpha": alpha,
        "target_coverage": 1.0 - alpha,
        "train_n": int(train_n),
        "calibration_n": int(calibration_n),
        "n": int(len(y)),
        "selected_level": selected_level,
        "qhat": float(qhat),
    }
    rec.update(base.point_metrics(y, pred))
    rec.update(base.interval_metrics(y, lower, upper, alpha))
    return rec


def evaluate_target(
    target: str,
    seed: int,
    alpha: float,
    calibration_frac: float,
    calibration_cap: int,
    clip_intervals: bool,
) -> list[dict[str, object]]:
    data = base.load_dataset(ROOT / "data" / "raw" / "train.csv", target, max_per_well=0, seed=seed)
    rows: list[dict[str, object]] = []
    for test_well in sorted(data["WELLNUM"].unique()):
        split_start = time.perf_counter()
        train = data[data["WELLNUM"] != test_well].copy()
        target_well = data[data["WELLNUM"] == test_well].copy().sort_values("DEPTH").reset_index(drop=True)
        cal_idx, test_idx = stratified_capped_indices(len(target_well), calibration_frac, calibration_cap)
        cal = target_well.iloc[cal_idx].copy().reset_index(drop=True)
        test = target_well.iloc[test_idx].copy().sort_values("DEPTH").reset_index(drop=True)
        eval_frame = pd.concat([cal, test], ignore_index=True)
        pred_all, lower_all, upper_all, leaf_all = local.fit_predict_tree_quantiles_with_leaves(
            train,
            eval_frame,
            target,
            "rf",
            alpha,
            seed + 37 * int(test_well),
        )
        n_cal = len(cal)
        y_cal = cal[target].to_numpy(dtype=float)
        y_test = test[target].to_numpy(dtype=float)
        pred_cal = pred_all[:n_cal]
        pred_test = pred_all[n_cal:]
        lower_cal = lower_all[:n_cal]
        upper_cal = upper_all[:n_cal]
        lower_test = lower_all[n_cal:]
        upper_test = upper_all[n_cal:]

        global_scores = np.maximum(lower_cal - y_cal, y_cal - upper_cal)
        global_qhat = base.conformal_quantile(global_scores, alpha)
        global_lower = lower_test - global_qhat
        global_upper = upper_test + global_qhat
        if clip_intervals:
            global_lower, global_upper = base.clip_interval(global_lower, global_upper)
        rows.append(
            record_metrics(
                target,
                seed,
                test_well,
                "global_rf_cqr",
                y_test,
                pred_test,
                global_lower,
                global_upper,
                alpha,
                len(train),
                n_cal,
                "uniform",
                global_qhat,
            )
        )

        wl_pred, wl_lower, wl_upper, wl_qhat, selected_level = local.wl_pcqr_path_interval(
            y_cal=y_cal,
            pred_cal=pred_cal,
            pred_test=pred_test,
            lower_cal=lower_cal,
            upper_cal=upper_cal,
            lower_test=lower_test,
            upper_test=upper_test,
            leaf_cal=leaf_all[:n_cal],
            leaf_test=leaf_all[n_cal:],
            alpha=alpha,
        )
        wl_lo = np.minimum(wl_lower, wl_upper)
        wl_hi = np.maximum(wl_lower, wl_upper)
        wl_lower, wl_upper = wl_lo, wl_hi
        if clip_intervals:
            wl_lower, wl_upper = base.clip_interval(wl_lower, wl_upper)
        rows.append(
            record_metrics(
                target,
                seed,
                test_well,
                "wl_pcqr",
                y_test,
                wl_pred,
                wl_lower,
                wl_upper,
                alpha,
                len(train),
                n_cal,
                selected_level,
                wl_qhat,
            )
        )
        elapsed = time.perf_counter() - split_start
        print(
            f"[all-valid] target={target} well={test_well} train={len(train)} "
            f"cal={n_cal} test={len(test)} elapsed={elapsed:.1f}s"
        )
    return rows


def aggregate_by_target(split_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    metric_cols = ["rmse", "mae", "coverage", "width", "width_std", "width_cv", "winkler"]
    for (target, method), group in split_df.groupby(["target", "method"], sort=True):
        rec: dict[str, object] = {
            "target": target,
            "method": method,
            "method_label": str(group["method_label"].iloc[0]),
            "splits": int(len(group)),
            "n": int(group["n"].sum()),
            "mean_calibration_n": float(group["calibration_n"].mean()),
        }
        for col in metric_cols:
            rec[col] = weighted_mean(group, col)
        rec["macro_well_coverage"] = float(group["coverage"].mean())
        rec["p10_well_coverage"] = float(np.percentile(group["coverage"].to_numpy(dtype=float), 10))
        rec["worst_well_coverage"] = float(group["coverage"].min())
        rec["worst_well"] = str(group.sort_values("coverage").iloc[0]["well"])
        rec["macro_well_width"] = float(group["width"].mean())
        rec["macro_well_winkler"] = float(group["winkler"].mean())
        rows.append(rec)
    return pd.DataFrame(rows).sort_values(["target", "macro_well_winkler", "macro_well_coverage"], ascending=[True, True, False])


def aggregate_summary(by_target_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    summary_cols = [
        "coverage",
        "width",
        "winkler",
        "macro_well_coverage",
        "p10_well_coverage",
        "worst_well_coverage",
        "macro_well_width",
        "macro_well_winkler",
        "rmse",
    ]
    for method, group in by_target_df.groupby("method", sort=True):
        rec: dict[str, object] = {
            "method": method,
            "method_label": str(group["method_label"].iloc[0]),
            "targets": int(group["target"].nunique()),
            "n": int(group["n"].sum()),
            "mean_calibration_n": float(group["mean_calibration_n"].mean()),
        }
        for col in summary_cols:
            rec[col] = float(group[col].mean())
        rows.append(rec)
    return pd.DataFrame(rows).sort_values(["macro_well_winkler", "macro_well_coverage"], ascending=[True, False])


def write_report(
    split_df: pd.DataFrame,
    by_target_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    alpha: float,
    seed: int,
    calibration_frac: float,
    calibration_cap: int,
    clip_intervals: bool,
) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "# All-Valid-Depth Well-Weighted Diagnostic",
        "",
        "This diagnostic addresses the reviewer concern that the 160-depth balanced protocol might hide full-depth complexity. "
        "It uses all valid source-well training depths and all non-calibration target-well evaluation depths after the target-bound label filter. "
        "Target-well calibration remains stratified and is capped to keep the WL-PCQR leave-one-out proximity path computationally bounded.",
        "",
        f"- Seed: `{seed}`",
        f"- Alpha: `{alpha}`",
        f"- Calibration fraction before cap: `{calibration_frac}`",
        f"- Calibration cap per target well: `{calibration_cap}`",
        f"- Intervals clipped to [0, 1]: `{clip_intervals}`",
        f"- Active features: `{', '.join(base.ACTIVE_FEATURES)}`",
        "",
        "## Method Summary",
        "",
        summary_df.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Target-Level Aggregates",
        "",
        by_target_df.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Per-Well Metrics",
        "",
        split_df.sort_values(["target", "method", "well"]).to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Interpretation",
        "",
        "The paper-facing reading should use the macro-well columns, because those give each target well equal weight despite very different depth counts. "
        "This is a diagnostic rather than the main benchmark: it tests whether the balanced-depth conclusion survives full-depth evaluation, not whether thousands of target labels are available for conformal path selection.",
    ]
    (OUT_DIR / "ALL_VALID_WEIGHTED_DIAGNOSTIC.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    alpha = 0.10
    seed = 42
    calibration_frac = 0.20
    calibration_cap = 160
    clip_intervals = True
    base.ACTIVE_FEATURES = [feature for feature in base.FEATURES if feature != "DEPTH"]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for target in TARGETS:
        rows.extend(evaluate_target(target, seed, alpha, calibration_frac, calibration_cap, clip_intervals))
    split_df = pd.DataFrame(rows)
    by_target_df = aggregate_by_target(split_df)
    summary_df = aggregate_summary(by_target_df)
    split_df.to_csv(OUT_DIR / "all_valid_weighted_split_metrics.csv", index=False)
    by_target_df.to_csv(OUT_DIR / "all_valid_weighted_by_target.csv", index=False)
    summary_df.to_csv(OUT_DIR / "all_valid_weighted_summary.csv", index=False)
    write_report(split_df, by_target_df, summary_df, alpha, seed, calibration_frac, calibration_cap, clip_intervals)
    print(f"Wrote {OUT_DIR}")
    print(summary_df.round(6).to_string(index=False))


if __name__ == "__main__":
    main()

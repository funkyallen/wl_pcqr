from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import run_wlpcqr_methods as local
import wlpcqr_core as base


ROOT = Path(__file__).resolve().parents[1]
TARGETS = ["PHIF", "SW", "VSH"]
SEEDS = [42, 7, 123]
ALPHA = 0.10
MAX_PER_WELL = 160
CAL_FRAC = 0.20
CAL_STRATEGY = "stratified"


ABLATION_LABELS = {
    "uniform_cqr": "Uniform target-well CQR",
    "raw_proximity": "Raw proximity score",
    "proximity_scale": "Proximity scale recalibration",
    "location_scale": "Proximity location-scale recalibration",
    "wl_pcqr": "WL-PCQR selected path",
}


def _interval_metrics_record(
    *,
    dataset: str,
    experiment: str,
    setting: str,
    target: str,
    seed: int,
    well: object,
    method: str,
    method_label: str,
    y: np.ndarray,
    pred: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    alpha: float,
    clip: bool,
    selected_level: str = "",
) -> dict[str, object]:
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    lo = np.minimum(lower, upper)
    hi = np.maximum(lower, upper)
    if clip:
        lo, hi = base.clip_interval(lo, hi)
    rec: dict[str, object] = {
        "dataset": dataset,
        "experiment": experiment,
        "setting": setting,
        "target": target,
        "seed": seed,
        "well": str(well),
        "method": method,
        "method_label": method_label,
        "alpha": alpha,
        "target_coverage": 1.0 - alpha,
        "clip_intervals": bool(clip),
        "selected_level": selected_level,
        "n": int(len(y)),
    }
    rec.update(base.point_metrics(y, pred))
    rec.update(base.interval_metrics(y, lo, hi, alpha))
    return rec


def _aggregate(split_df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for group_keys, group in split_df.groupby(keys, sort=True):
        if not isinstance(group_keys, tuple):
            group_keys = (group_keys,)
        weights = group["n"].to_numpy(dtype=float)
        rec: dict[str, object] = dict(zip(keys, group_keys))
        rec["splits"] = int(len(group))
        rec["n"] = int(group["n"].sum())
        for col in ["rmse", "mae", "coverage", "width", "width_cv", "winkler"]:
            rec[col] = float(np.average(group[col].to_numpy(dtype=float), weights=weights))
        rec["macro_well_coverage"] = float(group["coverage"].mean())
        rec["p10_well_coverage"] = float(np.percentile(group["coverage"].to_numpy(dtype=float), 10))
        rec["worst_well_coverage"] = float(group["coverage"].min())
        rec["macro_well_width"] = float(group["width"].mean())
        rec["macro_well_winkler"] = float(group["winkler"].mean())
        rows.append(rec)
    return pd.DataFrame(rows).sort_values(keys + ["winkler"]).reset_index(drop=True)


def _proximity_candidate(
    *,
    y_cal: np.ndarray,
    pred_cal: np.ndarray,
    pred_test: np.ndarray,
    lower_cal: np.ndarray,
    upper_cal: np.ndarray,
    lower_test: np.ndarray,
    upper_test: np.ndarray,
    leaf_cal: np.ndarray,
    leaf_test: np.ndarray,
    alpha: float,
    use_location: bool,
    use_recalibration: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_cal = len(y_cal)
    bias_cal = np.zeros(n_cal, dtype=float)
    bias_test = np.zeros(len(leaf_test), dtype=float)
    if use_location:
        residuals = y_cal - pred_cal
        for cal_pos, leaves in enumerate(leaf_cal):
            keep = np.arange(n_cal) != cal_pos
            weights = np.mean(leaf_cal[keep] == leaves[None, :], axis=1)
            bias_cal[cal_pos] = local.weighted_quantile(residuals[keep], weights, 0.5)
        for idx, leaves in enumerate(leaf_test):
            weights = np.mean(leaf_cal == leaves[None, :], axis=1)
            bias_test[idx] = local.weighted_quantile(residuals, weights, 0.5)

    shifted_lower_cal = lower_cal + bias_cal
    shifted_upper_cal = upper_cal + bias_cal
    shifted_lower_test = lower_test + bias_test
    shifted_upper_test = upper_test + bias_test
    shifted_pred_test = pred_test + bias_test
    scores = np.maximum(shifted_lower_cal - y_cal, y_cal - shifted_upper_cal)

    cal_qhats = np.empty(n_cal, dtype=float)
    for cal_pos, leaves in enumerate(leaf_cal):
        keep = np.arange(n_cal) != cal_pos
        weights = np.mean(leaf_cal[keep] == leaves[None, :], axis=1)
        cal_qhats[cal_pos] = local.weighted_quantile(scores[keep], weights, 1.0 - alpha)

    correction = 0.0
    if use_recalibration:
        recal_scores = np.maximum.reduce(
            [
                shifted_lower_cal - cal_qhats - y_cal,
                y_cal - (shifted_upper_cal + cal_qhats),
                np.zeros_like(y_cal),
            ]
        )
        correction = local.empirical_quantile(recal_scores, alpha) if use_location else base.conformal_quantile(recal_scores, alpha)

    qhats = np.empty(len(leaf_test), dtype=float)
    for idx, leaves in enumerate(leaf_test):
        weights = np.mean(leaf_cal == leaves[None, :], axis=1)
        qhats[idx] = local.weighted_quantile(scores, weights, 1.0 - alpha)
    lower = shifted_lower_test - qhats - correction
    upper = shifted_upper_test + qhats + correction
    return shifted_pred_test, lower, upper


def _candidate_intervals(
    *,
    y_cal: np.ndarray,
    pred_cal: np.ndarray,
    pred_test: np.ndarray,
    lower_cal: np.ndarray,
    upper_cal: np.ndarray,
    lower_test: np.ndarray,
    upper_test: np.ndarray,
    leaf_cal: np.ndarray,
    leaf_test: np.ndarray,
    alpha: float,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, str]]:
    scores = np.maximum(lower_cal - y_cal, y_cal - upper_cal)
    q_global = base.conformal_quantile(scores, alpha)
    out: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, str]] = {
        "uniform_cqr": (pred_test, lower_test - q_global, upper_test + q_global, "uniform"),
    }
    pred_raw, lower_raw, upper_raw = _proximity_candidate(
        y_cal=y_cal,
        pred_cal=pred_cal,
        pred_test=pred_test,
        lower_cal=lower_cal,
        upper_cal=upper_cal,
        lower_test=lower_test,
        upper_test=upper_test,
        leaf_cal=leaf_cal,
        leaf_test=leaf_test,
        alpha=alpha,
        use_location=False,
        use_recalibration=False,
    )
    out["raw_proximity"] = (pred_raw, lower_raw, upper_raw, "raw_proximity")

    pred_scale, lower_scale, upper_scale = _proximity_candidate(
        y_cal=y_cal,
        pred_cal=pred_cal,
        pred_test=pred_test,
        lower_cal=lower_cal,
        upper_cal=upper_cal,
        lower_test=lower_test,
        upper_test=upper_test,
        leaf_cal=leaf_cal,
        leaf_test=leaf_test,
        alpha=alpha,
        use_location=False,
        use_recalibration=True,
    )
    out["proximity_scale"] = (pred_scale, lower_scale, upper_scale, "scale")

    pred_locs, lower_locs, upper_locs = _proximity_candidate(
        y_cal=y_cal,
        pred_cal=pred_cal,
        pred_test=pred_test,
        lower_cal=lower_cal,
        upper_cal=upper_cal,
        lower_test=lower_test,
        upper_test=upper_test,
        leaf_cal=leaf_cal,
        leaf_test=leaf_test,
        alpha=alpha,
        use_location=True,
        use_recalibration=True,
    )
    out["location_scale"] = (pred_locs, lower_locs, upper_locs, "location-scale")

    pred_full, lower_full, upper_full, _, selected_level = local.wl_pcqr_path_interval(
        y_cal=y_cal,
        pred_cal=pred_cal,
        pred_test=pred_test,
        lower_cal=lower_cal,
        upper_cal=upper_cal,
        lower_test=lower_test,
        upper_test=upper_test,
        leaf_cal=leaf_cal,
        leaf_test=leaf_test,
        alpha=alpha,
    )
    out["wl_pcqr"] = (pred_full, lower_full, upper_full, selected_level)
    return out


def evaluate_ablation_for_target(target: str, seed: int) -> list[dict[str, object]]:
    base.ACTIVE_FEATURES = [feature for feature in base.FEATURES if feature != "DEPTH"]
    data = base.load_dataset(ROOT / "data" / "raw" / "train.csv", target, MAX_PER_WELL, seed)
    rows: list[dict[str, object]] = []
    for test_well in sorted(data["WELLNUM"].unique()):
        train = data[data["WELLNUM"] != test_well].copy()
        target_well = data[data["WELLNUM"] == test_well].copy().sort_values("DEPTH").reset_index(drop=True)
        cal_idx, test_idx = local.local_calibration_indices(
            len(target_well), CAL_FRAC, CAL_STRATEGY, seed + 7919 * int(test_well)
        )
        cal = target_well.iloc[cal_idx].copy().reset_index(drop=True)
        test = target_well.iloc[test_idx].copy().sort_values("DEPTH").reset_index(drop=True)
        eval_frame = pd.concat([cal, test], ignore_index=True)
        pred_all, lower_all, upper_all, leaf_all = local.fit_predict_tree_quantiles_with_leaves(
            train,
            eval_frame,
            target,
            "rf",
            ALPHA,
            seed + 37 * int(test_well),
        )
        n_cal = len(cal)
        candidates = _candidate_intervals(
            y_cal=cal[target].to_numpy(dtype=float),
            pred_cal=pred_all[:n_cal],
            pred_test=pred_all[n_cal:],
            lower_cal=lower_all[:n_cal],
            upper_cal=upper_all[:n_cal],
            lower_test=lower_all[n_cal:],
            upper_test=upper_all[n_cal:],
            leaf_cal=leaf_all[:n_cal],
            leaf_test=leaf_all[n_cal:],
            alpha=ALPHA,
        )
        y = test[target].to_numpy(dtype=float)
        for method, (pred, lower, upper, selected_level) in candidates.items():
            rows.append(
                _interval_metrics_record(
                    dataset="SPWLA_PDDA_2021",
                    experiment="mechanism_ablation",
                    setting="stratified20_no_depth",
                    target=target,
                    seed=seed,
                    well=test_well,
                    method=method,
                    method_label=ABLATION_LABELS[method],
                    y=y,
                    pred=pred,
                    lower=lower,
                    upper=upper,
                    alpha=ALPHA,
                    clip=True,
                    selected_level=selected_level,
                )
            )
    return rows


def run_mechanism_ablation(out_dir: Path) -> None:
    rows: list[dict[str, object]] = []
    for target in TARGETS:
        for seed in SEEDS:
            print(f"[ablation] target={target} seed={seed}")
            rows.extend(evaluate_ablation_for_target(target, seed))
    split_df = pd.DataFrame(rows)
    summary = _aggregate(split_df, ["experiment", "setting", "method", "method_label"])
    by_target = _aggregate(split_df, ["experiment", "setting", "target", "method", "method_label"])
    level_counts = (
        split_df[split_df["method"] == "wl_pcqr"]
        .groupby(["target", "selected_level"], sort=True)
        .size()
        .reset_index(name="count")
    )
    split_df.to_csv(out_dir / "mechanism_ablation_raw.csv", index=False)
    summary.to_csv(out_dir / "mechanism_ablation_summary.csv", index=False)
    by_target.to_csv(out_dir / "mechanism_ablation_by_target.csv", index=False)
    level_counts.to_csv(out_dir / "wlpcqr_selected_level_counts.csv", index=False)


def evaluate_wlpcqr_setting(
    *,
    target: str,
    seed: int,
    experiment: str,
    setting: str,
    alpha: float,
    include_depth: bool,
    clip: bool,
) -> list[dict[str, object]]:
    base.ACTIVE_FEATURES = list(base.FEATURES if include_depth else [feature for feature in base.FEATURES if feature != "DEPTH"])
    data = base.load_dataset(ROOT / "data" / "raw" / "train.csv", target, MAX_PER_WELL, seed)
    metric_rows, _ = local.evaluate_local(
        data=data,
        target=target,
        method="wl_pcqr",
        alpha=alpha,
        local_calibration_frac=CAL_FRAC,
        strategy=CAL_STRATEGY,
        seed=seed,
        stress_curves=[],
        clip_intervals=clip,
    )
    rows: list[dict[str, object]] = []
    for rec in metric_rows:
        rec = dict(rec)
        rec["dataset"] = "SPWLA_PDDA_2021"
        rec["experiment"] = experiment
        rec["setting"] = setting
        rec["method"] = "wl_pcqr"
        rec["method_label"] = "WL-PCQR"
        rows.append(rec)
    return rows


def run_sensitivity(out_dir: Path) -> None:
    rows: list[dict[str, object]] = []
    settings = [
        ("depth_feature", "no_depth", ALPHA, False, True),
        ("depth_feature", "with_depth", ALPHA, True, True),
        ("physical_bounds", "clip_to_0_1", ALPHA, False, True),
        ("physical_bounds", "no_clip", ALPHA, False, False),
        ("nominal_coverage", "alpha_0.05", 0.05, False, True),
        ("nominal_coverage", "alpha_0.10", 0.10, False, True),
        ("nominal_coverage", "alpha_0.20", 0.20, False, True),
    ]
    for experiment, setting, alpha, include_depth, clip in settings:
        for target in TARGETS:
            for seed in SEEDS:
                print(f"[sensitivity] {experiment}/{setting} target={target} seed={seed}")
                rows.extend(
                    evaluate_wlpcqr_setting(
                        target=target,
                        seed=seed,
                        experiment=experiment,
                        setting=setting,
                        alpha=alpha,
                        include_depth=include_depth,
                        clip=clip,
                    )
                )
    split_df = pd.DataFrame(rows)
    summary = _aggregate(split_df, ["experiment", "setting", "method", "method_label"])
    by_target = _aggregate(split_df, ["experiment", "setting", "target", "method", "method_label"])
    split_df.to_csv(out_dir / "sensitivity_raw.csv", index=False)
    summary.to_csv(out_dir / "sensitivity_summary.csv", index=False)
    by_target.to_csv(out_dir / "sensitivity_by_target.csv", index=False)


def write_markdown_report(out_dir: Path) -> None:
    ablation = pd.read_csv(out_dir / "mechanism_ablation_summary.csv")
    sensitivity = pd.read_csv(out_dir / "sensitivity_summary.csv")
    level_counts = pd.read_csv(out_dir / "wlpcqr_selected_level_counts.csv")
    lines = [
        "# WL-PCQR Ablation and Sensitivity Report",
        "",
        "## Mechanism Ablation",
        "",
        ablation[
            [
                "method_label",
                "coverage",
                "p10_well_coverage",
                "worst_well_coverage",
                "width",
                "winkler",
                "rmse",
            ]
        ].to_markdown(index=False, floatfmt=".4f"),
        "",
        "## WL-PCQR Selected Path Counts",
        "",
        level_counts.to_markdown(index=False),
        "",
        "## Sensitivity Summary",
        "",
        sensitivity[
            [
                "experiment",
                "setting",
                "coverage",
                "p10_well_coverage",
                "worst_well_coverage",
                "width",
                "winkler",
                "rmse",
            ]
        ].to_markdown(index=False, floatfmt=".4f"),
        "",
    ]
    (out_dir / "WLPCQR_ABLATION_SENSITIVITY_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    out_dir = ROOT / "analysis" / "ablation_sensitivity"
    out_dir.mkdir(parents=True, exist_ok=True)
    run_mechanism_ablation(out_dir)
    run_sensitivity(out_dir)
    write_markdown_report(out_dir)
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import run_wlpcqr_methods as local
import wlpcqr_core as base


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "submission_revision_diagnostics"
TARGETS = ["PHIF", "SW", "VSH"]
SEEDS = [42, 7, 123]
ALPHA = 0.10
MAX_PER_WELL = 160
MAIN_CAL_FRAC = 0.20


def _reset_features() -> None:
    base.ACTIVE_FEATURES = [feature for feature in base.FEATURES if feature != "DEPTH"]


def _weighted_mean(group: pd.DataFrame, column: str) -> float:
    weights = group["n"].to_numpy(dtype=float)
    values = group[column].to_numpy(dtype=float)
    return float(np.average(values, weights=weights))


def _aggregate_split(split_df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for group_keys, group in split_df.groupby(keys, sort=True):
        if not isinstance(group_keys, tuple):
            group_keys = (group_keys,)
        rec: dict[str, object] = dict(zip(keys, group_keys))
        rec["splits"] = int(len(group))
        rec["n"] = int(group["n"].sum())
        for metric in ["rmse", "mae", "coverage", "width", "winkler"]:
            rec[metric] = _weighted_mean(group, metric)
        rec["macro_well_coverage"] = float(group["coverage"].mean())
        rec["p10_well_coverage"] = float(np.percentile(group["coverage"].to_numpy(dtype=float), 10))
        rec["worst_well_coverage"] = float(group["coverage"].min())
        rec["macro_well_width"] = float(group["width"].mean())
        rec["macro_well_winkler"] = float(group["winkler"].mean())
        rows.append(rec)
    return pd.DataFrame(rows).sort_values(keys).reset_index(drop=True)


def vsh_outlier_by_well() -> pd.DataFrame:
    raw = pd.read_csv(ROOT / "data" / "raw" / "train.csv").replace(-9999.0, np.nan)
    rows: list[dict[str, object]] = []
    for well, group in raw[raw["VSH"].notna()].groupby("WELLNUM", sort=True):
        values = group["VSH"].astype(float)
        outside = (values < 0.0) | (values > 1.0)
        rows.append(
            {
                "well": int(well),
                "raw_vsh_count": int(len(values)),
                "removed_out_of_range_vsh": int(outside.sum()),
                "removal_ratio": float(outside.mean()),
                "min_raw_vsh": float(values.min()),
                "max_raw_vsh": float(values.max()),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "vsh_outlier_by_well.csv", index=False)
    return out


def split_sample_counts() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    for target in TARGETS:
        for seed in SEEDS:
            data = base.load_dataset(ROOT / "data" / "raw" / "train.csv", target, MAX_PER_WELL, seed)
            for well in sorted(data["WELLNUM"].unique()):
                target_well = data[data["WELLNUM"] == well].copy().sort_values("DEPTH").reset_index(drop=True)
                cal_idx, test_idx = local.local_calibration_indices(
                    len(target_well), MAIN_CAL_FRAC, "stratified", seed + 7919 * int(well)
                )
                rows.append(
                    {
                        "target": target,
                        "seed": seed,
                        "target_well": int(well),
                        "retained_target_well_n": int(len(target_well)),
                        "source_train_n": int(len(data) - len(target_well)),
                        "calibration_n": int(len(cal_idx)),
                        "evaluation_n": int(len(test_idx)),
                    }
                )
    full = pd.DataFrame(rows)
    summary_rows: list[dict[str, object]] = []
    for target, group in full.groupby("target", sort=True):
        summary_rows.append(
            {
                "target": target,
                "target_wells": int(group["target_well"].nunique()),
                "retained_per_well_min": int(group["retained_target_well_n"].min()),
                "retained_per_well_max": int(group["retained_target_well_n"].max()),
                "source_train_n_min": int(group["source_train_n"].min()),
                "source_train_n_max": int(group["source_train_n"].max()),
                "calibration_n_min": int(group["calibration_n"].min()),
                "calibration_n_max": int(group["calibration_n"].max()),
                "evaluation_n_min": int(group["evaluation_n"].min()),
                "evaluation_n_max": int(group["evaluation_n"].max()),
            }
        )
    summary = pd.DataFrame(summary_rows)
    full.to_csv(OUT / "main_protocol_split_sample_counts.csv", index=False)
    summary.to_csv(OUT / "main_protocol_split_sample_summary.csv", index=False)
    return full, summary


def calibration_5pct_sensitivity() -> tuple[pd.DataFrame, pd.DataFrame]:
    _reset_features()
    rows: list[dict[str, object]] = []
    for target in TARGETS:
        for seed in SEEDS:
            data = base.load_dataset(ROOT / "data" / "raw" / "train.csv", target, MAX_PER_WELL, seed)
            for strategy in ["contiguous", "random", "stratified"]:
                print(f"[5pct-calibration] target={target} seed={seed} strategy={strategy}")
                metric_rows, _ = local.evaluate_local(
                    data=data,
                    target=target,
                    method="wl_pcqr",
                    alpha=ALPHA,
                    local_calibration_frac=0.05,
                    strategy=strategy,
                    seed=seed,
                    stress_curves=[],
                    clip_intervals=True,
                )
                for rec in metric_rows:
                    rec = dict(rec)
                    rec["frac"] = 0.05
                    rec["strategy"] = strategy
                    rec["method"] = "wl_pcqr"
                    rec["method_label"] = "WL-PCQR"
                    rows.append(rec)
    split = pd.DataFrame(rows)
    per_target_seed = _aggregate_split(split, ["frac", "strategy", "target", "seed", "method", "method_label"])
    summary = (
        per_target_seed.groupby(["frac", "strategy", "method", "method_label"], sort=True)[
            ["coverage", "p10_well_coverage", "worst_well_coverage", "width", "winkler", "rmse"]
        ]
        .mean()
        .reset_index()
        .sort_values(["frac", "strategy"])
    )
    split.to_csv(OUT / "calibration_5pct_split_metrics.csv", index=False)
    per_target_seed.to_csv(OUT / "calibration_5pct_by_target_seed.csv", index=False)
    summary.to_csv(OUT / "calibration_5pct_summary.csv", index=False)
    return split, summary


def paired_bootstrap_winkler() -> pd.DataFrame:
    raw = pd.read_csv(OUT.parent / "reviewer_required_diagnostics" / "reviewer_nearest_baseline_split_metrics.csv")
    keep = ["WL-PCQR", "Global RF-CQR", "kNN Local RF-CQR", "RF-Leaf QRF+CQR", "GR-Mondrian RF-CQR"]
    raw = raw[raw["method_label"].isin(keep)].copy()
    rows: list[dict[str, object]] = []
    for keys, group in raw.groupby(["target", "seed", "method_label"], sort=True):
        target, seed, method = keys
        rows.append(
            {
                "target": target,
                "seed": int(seed),
                "method_label": method,
                "n": int(group["n"].sum()),
                "winkler": _weighted_mean(group, "winkler"),
                "coverage": _weighted_mean(group, "coverage"),
                "width": _weighted_mean(group, "width"),
            }
        )
    per_run = pd.DataFrame(rows)
    pivot = per_run.pivot(index=["target", "seed"], columns="method_label", values="winkler").dropna()
    rng = np.random.default_rng(20260519)
    out_rows: list[dict[str, object]] = []
    for comparator in [m for m in keep if m != "WL-PCQR"]:
        diff = (pivot[comparator] - pivot["WL-PCQR"]).to_numpy(dtype=float)
        boot = np.empty(10000, dtype=float)
        n = len(diff)
        for b in range(len(boot)):
            idx = rng.integers(0, n, size=n)
            boot[b] = float(np.mean(diff[idx]))
        out_rows.append(
            {
                "comparison": f"{comparator} minus WL-PCQR",
                "paired_units": int(n),
                "mean_winkler_reduction": float(np.mean(diff)),
                "ci95_low": float(np.percentile(boot, 2.5)),
                "ci95_high": float(np.percentile(boot, 97.5)),
                "wlpcqr_better_units": int((diff > 0).sum()),
            }
        )
    out = pd.DataFrame(out_rows)
    per_run.to_csv(OUT / "paired_winkler_per_target_seed.csv", index=False)
    out.to_csv(OUT / "paired_winkler_bootstrap_ci.csv", index=False)
    return out


def proximity_weight_diagnostics() -> tuple[pd.DataFrame, pd.DataFrame]:
    _reset_features()
    rows: list[dict[str, object]] = []
    for target in TARGETS:
        for seed in SEEDS:
            data = base.load_dataset(ROOT / "data" / "raw" / "train.csv", target, MAX_PER_WELL, seed)
            for well in sorted(data["WELLNUM"].unique()):
                print(f"[proximity-diagnostics] target={target} seed={seed} well={well}")
                train = data[data["WELLNUM"] != well].copy()
                target_well = data[data["WELLNUM"] == well].copy().sort_values("DEPTH").reset_index(drop=True)
                cal_idx, test_idx = local.local_calibration_indices(
                    len(target_well), MAIN_CAL_FRAC, "stratified", seed + 7919 * int(well)
                )
                cal = target_well.iloc[cal_idx].copy().reset_index(drop=True)
                test = target_well.iloc[test_idx].copy().sort_values("DEPTH").reset_index(drop=True)
                eval_frame = pd.concat([cal, test], ignore_index=True)
                _, _, _, leaves = local.fit_predict_tree_quantiles_with_leaves(
                    train, eval_frame, target, "rf", ALPHA, seed + 37 * int(well)
                )
                n_cal = len(cal)
                leaf_cal = leaves[:n_cal]
                leaf_test = leaves[n_cal:]
                eff_values: list[float] = []
                fallback_count = 0
                max_weight_values: list[float] = []
                for test_leaves in leaf_test:
                    weights = np.mean(leaf_cal == test_leaves[None, :], axis=1)
                    sum_w = float(np.sum(weights))
                    sum_w2 = float(np.sum(weights**2))
                    if sum_w <= base.EPS:
                        fallback_count += 1
                        eff = float(n_cal)
                        max_w = 1.0 / n_cal
                    else:
                        eff = (sum_w * sum_w) / sum_w2 if sum_w2 > 0 else float(n_cal)
                        max_w = float(np.max(weights / sum_w))
                    eff_values.append(float(eff))
                    max_weight_values.append(max_w)
                rows.append(
                    {
                        "target": target,
                        "seed": seed,
                        "well": int(well),
                        "calibration_n": int(n_cal),
                        "evaluation_n": int(len(test)),
                        "fallback_count": int(fallback_count),
                        "fallback_rate": float(fallback_count / max(len(test), 1)),
                        "effective_n_min": float(np.min(eff_values)),
                        "effective_n_median": float(np.median(eff_values)),
                        "effective_n_p10": float(np.percentile(eff_values, 10)),
                        "max_normalized_weight_median": float(np.median(max_weight_values)),
                        "max_normalized_weight_p90": float(np.percentile(max_weight_values, 90)),
                    }
                )
    split = pd.DataFrame(rows)
    summary = (
        split.groupby("target", sort=True)
        .agg(
            splits=("well", "size"),
            fallback_rate=("fallback_rate", "mean"),
            fallback_count=("fallback_count", "sum"),
            evaluation_n=("evaluation_n", "sum"),
            effective_n_min=("effective_n_min", "min"),
            effective_n_p10=("effective_n_p10", "mean"),
            effective_n_median=("effective_n_median", "mean"),
            max_normalized_weight_p90=("max_normalized_weight_p90", "mean"),
        )
        .reset_index()
    )
    total = {
        "target": "All",
        "splits": int(len(split)),
        "fallback_rate": float(split["fallback_count"].sum() / split["evaluation_n"].sum()),
        "fallback_count": int(split["fallback_count"].sum()),
        "evaluation_n": int(split["evaluation_n"].sum()),
        "effective_n_min": float(split["effective_n_min"].min()),
        "effective_n_p10": float(split["effective_n_p10"].mean()),
        "effective_n_median": float(split["effective_n_median"].mean()),
        "max_normalized_weight_p90": float(split["max_normalized_weight_p90"].mean()),
    }
    summary = pd.concat([summary, pd.DataFrame([total])], ignore_index=True)
    split.to_csv(OUT / "proximity_weight_split_diagnostics.csv", index=False)
    summary.to_csv(OUT / "proximity_weight_summary.csv", index=False)
    return split, summary


def hard_well_analysis() -> pd.DataFrame:
    raw = pd.read_csv(ROOT / "data" / "raw" / "train.csv").replace(-9999.0, np.nan)
    raw["RDEP_LOG10"] = np.where(raw["RDEP"] > 0, np.log10(raw["RDEP"]), np.nan)
    raw["RMED_LOG10"] = np.where(raw["RMED"] > 0, np.log10(raw["RMED"]), np.nan)
    active = [feature for feature in base.FEATURES if feature != "DEPTH"]
    cases = [("PHIF", 0), ("VSH", 1)]
    rows: list[dict[str, object]] = []
    for target, well in cases:
        target_valid = raw[raw[target].notna()].copy()
        retained = target_valid[(target_valid[target] >= 0.0) & (target_valid[target] <= 1.0)].copy()
        well_group = retained[retained["WELLNUM"] == well].copy()
        other_group = retained[retained["WELLNUM"] != well].copy()
        rec: dict[str, object] = {
            "target": target,
            "well": int(well),
            "retained_n": int(len(well_group)),
            "target_mean": float(well_group[target].mean()),
            "other_well_mean": float(other_group[target].mean()),
            "target_median": float(well_group[target].median()),
            "other_well_median": float(other_group[target].median()),
            "target_q10": float(well_group[target].quantile(0.10)),
            "target_q90": float(well_group[target].quantile(0.90)),
            "input_missing_rate": float(well_group[active].isna().mean().mean()),
            "other_input_missing_rate": float(other_group[active].isna().mean().mean()),
        }
        for feature in ["DEN", "NEU", "GR", "RDEP_LOG10", "RMED_LOG10"]:
            rec[f"{feature}_missing_rate"] = float(well_group[feature].isna().mean())
            rec[f"{feature}_other_missing_rate"] = float(other_group[feature].isna().mean())
        rows.append(rec)
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "hard_well_petrophysical_summary.csv", index=False)
    return out


def hyperparameter_table() -> pd.DataFrame:
    rows = [
        {"component": "RandomForestRegressor", "setting": "n_estimators", "value": "160"},
        {"component": "RandomForestRegressor", "setting": "min_samples_leaf", "value": "8"},
        {"component": "RandomForestRegressor", "setting": "max_features", "value": "0.75"},
        {"component": "ExtraTreesRegressor", "setting": "n_estimators", "value": "180"},
        {"component": "ExtraTreesRegressor", "setting": "min_samples_leaf", "value": "8"},
        {"component": "HistGradientBoostingRegressor", "setting": "learning_rate", "value": "0.05"},
        {"component": "HistGradientBoostingRegressor", "setting": "max_iter", "value": "100-110"},
        {"component": "LightGBM", "setting": "n_estimators / learning_rate", "value": "180 / 0.035"},
        {"component": "CatBoost", "setting": "iterations / depth / learning_rate", "value": "220 / 6 / 0.035"},
        {"component": "NGBoost", "setting": "distribution / estimators / learning_rate", "value": "Normal / 160 / 0.025"},
        {"component": "All experiments", "setting": "seeds", "value": "42, 7, 123"},
        {"component": "Preprocessing", "setting": "imputation", "value": "split-wise training median"},
    ]
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "hyperparameter_reproducibility_table.csv", index=False)
    return out


def write_report(tables: dict[str, pd.DataFrame]) -> None:
    lines = ["# Submission Revision Diagnostics", ""]
    for name, table in tables.items():
        lines.extend([f"## {name}", "", table.to_markdown(index=False, floatfmt=".4f"), ""])
    (OUT / "SUBMISSION_REVISION_DIAGNOSTICS.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    vsh = vsh_outlier_by_well()
    _, split_summary = split_sample_counts()
    _, cal5 = calibration_5pct_sensitivity()
    paired = paired_bootstrap_winkler()
    _, prox = proximity_weight_diagnostics()
    hard = hard_well_analysis()
    hyper = hyperparameter_table()
    write_report(
        {
            "VSH Outlier By Well": vsh,
            "Main Protocol Split Sample Summary": split_summary,
            "Five Percent Calibration Sensitivity": cal5,
            "Paired Winkler Bootstrap CI": paired,
            "Proximity Weight Summary": prox,
            "Hard-Well Petrophysical Summary": hard,
            "Hyperparameter Reproducibility Table": hyper,
        }
    )
    print((OUT / "SUBMISSION_REVISION_DIAGNOSTICS.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()

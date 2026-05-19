from __future__ import annotations

import argparse
import contextlib
import io
from pathlib import Path

import numpy as np
import pandas as pd

import run_mendeley_external_validation as mendeley
import run_wlpcqr_methods as local
import run_wlpcqr_within_depth_block as within
import wlpcqr_core as base


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "baseline_comparison"
FAILURE_ROWS: list[dict[str, object]] = []

SPWLA_FEATURES = [
    "DEPTH",
    "DTC",
    "DTS",
    "BS",
    "CALI",
    "DEN",
    "DENC",
    "GR",
    "NEU",
    "PEF",
    "RDEP_LOG10",
    "RMED_LOG10",
    "ROP",
]

TARGET_CAL_METHODS = [
    "wl_pcqr",
    "global_rf_cqr",
    "extratrees_pcqr",
    "ngboost_cqr_signed",
    "hgb_cqr_signed",
    "lightgbm_cqr_signed",
    "catboost_cqr_signed",
]

DEFAULT_SEEDS = [42]
SPWLA_TARGETS = ["PHIF", "SW", "VSH"]


def reset_spwla_features() -> None:
    base.FEATURES = list(SPWLA_FEATURES)
    base.ACTIVE_FEATURES = [feature for feature in SPWLA_FEATURES if feature != "DEPTH"]


def metric_columns(frame: pd.DataFrame) -> list[str]:
    candidates = [
        "coverage",
        "p10_well_coverage",
        "worst_well_coverage",
        "width",
        "winkler",
        "rmse",
        "macro_well_coverage",
        "macro_well_winkler",
    ]
    return [col for col in candidates if col in frame.columns]


def summarize(raw: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    metrics = metric_columns(raw)
    grouped = raw.groupby(group_cols, dropna=False)[metrics]
    mean = grouped.mean().add_suffix("_mean")
    sd = grouped.std(ddof=1).fillna(0.0).add_suffix("_sd")
    out = pd.concat([mean, sd], axis=1).reset_index()
    if "winkler_mean" in out.columns:
        out = out.sort_values(["dataset", "protocol", "winkler_mean", "coverage_mean"]).reset_index(drop=True)
        out["rank_winkler"] = out.groupby(["dataset", "protocol"])["winkler_mean"].rank(method="min", ascending=True).astype(int)
    return out


def cqr_scores(y: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    return np.maximum.reduce([lower - y, y - upper, np.zeros_like(y, dtype=float)])


def stratified_source_split(n: int, frac: float) -> tuple[np.ndarray, np.ndarray]:
    n_cal = min(max(int(round(n * frac)), 1), n - 1)
    all_idx = np.arange(n)
    cal_idx = np.unique(np.round(np.linspace(0, n - 1, n_cal)).astype(int))
    if len(cal_idx) < n_cal:
        rest = np.setdiff1d(all_idx, cal_idx)
        cal_idx = np.sort(np.concatenate([cal_idx, rest[: n_cal - len(cal_idx)]]))
    train_idx = np.setdiff1d(all_idx, cal_idx)
    return train_idx, cal_idx


def source_only_rf_cqr(data: pd.DataFrame, target: str, seed: int, alpha: float, dataset_name: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    wells = sorted(data["WELLNUM"].astype(int).unique())
    for test_well in wells:
        source_wells = [well for well in wells if well != test_well]
        score_parts: list[np.ndarray] = []
        if len(source_wells) >= 2:
            for cal_well in source_wells:
                train_wells = [well for well in source_wells if well != cal_well]
                train = data[data["WELLNUM"].astype(int).isin(train_wells)].copy()
                cal = data[data["WELLNUM"].astype(int) == cal_well].copy()
                _, cal_lower, cal_upper = base.fit_predict_tree_quantiles(train, cal, target, "rf", alpha, seed)
                score_parts.append(cqr_scores(cal[target].to_numpy(dtype=float), cal_lower, cal_upper))
        else:
            source = data[data["WELLNUM"].astype(int).isin(source_wells)].copy().sort_values("DEPTH").reset_index(drop=True)
            train_idx, cal_idx = stratified_source_split(len(source), 0.20)
            train = source.iloc[train_idx].copy()
            cal = source.iloc[cal_idx].copy()
            _, cal_lower, cal_upper = base.fit_predict_tree_quantiles(train, cal, target, "rf", alpha, seed)
            score_parts.append(cqr_scores(cal[target].to_numpy(dtype=float), cal_lower, cal_upper))
        qhat = base.conformal_quantile(np.concatenate(score_parts), alpha)
        train_final = data[data["WELLNUM"].astype(int).isin(source_wells)].copy()
        test = data[data["WELLNUM"].astype(int) == test_well].copy().sort_values("DEPTH").reset_index(drop=True)
        pred, lower, upper = base.fit_predict_tree_quantiles(train_final, test, target, "rf", alpha, seed)
        lower, upper = base.clip_interval(lower - qhat, upper + qhat)
        y = test[target].to_numpy(dtype=float)
        rec: dict[str, object] = {
            "dataset": dataset_name,
            "protocol": "cross_well_source_only",
            "well": str(test_well),
            "method": "source_only_rf_cqr",
            "method_label": "Source-only RF-CQR",
            "target": target,
            "seed": seed,
            "n": int(len(test)),
            "qhat": float(qhat),
        }
        rec.update(base.point_metrics(y, pred))
        rec.update(base.interval_metrics(y, lower, upper, alpha))
        rows.append(rec)
    split = pd.DataFrame(rows)
    agg = local.aggregate_metrics(split.assign(alpha=alpha, target_coverage=1.0 - alpha))
    agg["dataset"] = dataset_name
    agg["protocol"] = "cross_well"
    agg["seed"] = seed
    return agg


def run_target_cal_cross(data: pd.DataFrame, dataset_name: str, target: str, seed: int, alpha: float, frac: float) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for method in TARGET_CAL_METHODS:
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                metric_rows, _ = local.evaluate_local(
                    data=data,
                    target=target,
                    method=method,
                    alpha=alpha,
                    local_calibration_frac=frac,
                    strategy="stratified",
                    seed=seed,
                    stress_curves=[],
                    clip_intervals=True,
                )
            split = pd.DataFrame(metric_rows)
            agg = local.aggregate_metrics(split)
            agg["dataset"] = dataset_name
            agg["protocol"] = "cross_well"
            agg["seed"] = seed
            rows.append(agg)
        except Exception as exc:
            FAILURE_ROWS.append(
                {
                    "dataset": dataset_name,
                    "protocol": "cross_well",
                    "method": method,
                    "method_label": local.METHOD_LABELS.get(method, method),
                    "target": target,
                    "seed": seed,
                    "split_id": "all_target_wells",
                    "error": repr(exc),
                }
            )
    try:
        rows.append(source_only_rf_cqr(data, target, seed, alpha, dataset_name))
    except Exception as exc:
        FAILURE_ROWS.append(
            {
                "dataset": dataset_name,
                "protocol": "cross_well",
                "method": "source_only_rf_cqr",
                "method_label": "Source-only RF-CQR",
                "target": target,
                "seed": seed,
                "split_id": "all_target_wells",
                "error": repr(exc),
            }
        )
    return pd.concat(rows, ignore_index=True)


def run_within(data: pd.DataFrame, dataset_name: str, target: str, seed: int, alpha: float, blocks: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for method in TARGET_CAL_METHODS:
        for well, group in data.groupby("WELLNUM", sort=True):
            group = group.sort_values("DEPTH").reset_index(drop=True)
            block_ids = base.assign_depth_blocks(len(group), blocks)
            for block in sorted(np.unique(block_ids)):
                cal_block = within.adjacent_calibration_block(int(block), blocks)
                train = group[(block_ids != block) & (block_ids != cal_block)].copy().reset_index(drop=True)
                cal = group[block_ids == cal_block].copy().reset_index(drop=True)
                test = group[block_ids == block].copy().reset_index(drop=True)
                if len(train) < 5 or len(cal) < 2 or len(test) < 2:
                    continue
                split_seed = seed + int(well) * 101 + int(block)
                split_id = f"{well}:test{block}:cal{cal_block}"
                try:
                    pred, lower, upper, qhat = within.fit_interval(train, cal, test, target, method, alpha, split_seed)
                except Exception as exc:
                    FAILURE_ROWS.append(
                        {
                            "dataset": dataset_name,
                            "protocol": "within_well",
                            "method": method,
                            "method_label": local.METHOD_LABELS.get(method, method),
                            "target": target,
                            "seed": seed,
                            "split_id": split_id,
                            "error": repr(exc),
                        }
                    )
                    continue
                lower, upper = base.clip_interval(np.minimum(lower, upper), np.maximum(lower, upper))
                y = test[target].to_numpy(dtype=float)
                rec: dict[str, object] = {
                    "dataset": dataset_name,
                    "protocol": "within_well_depth_block",
                    "split_id": split_id,
                    "well": str(well),
                    "block": int(block),
                    "calibration_block": int(cal_block),
                    "method": method,
                    "method_label": local.METHOD_LABELS[method],
                    "target": target,
                    "seed": seed,
                    "n": int(len(test)),
                    "calibration_n": int(len(cal)),
                    "train_n": int(len(train)),
                    "qhat": float(qhat),
                }
                rec.update(base.point_metrics(y, pred))
                rec.update(base.interval_metrics(y, lower, upper, alpha))
                rows.append(rec)
    if not rows:
        return pd.DataFrame()
    split = pd.DataFrame(rows)
    agg = within.aggregate_metrics(split.assign(alpha=alpha, target_coverage=1.0 - alpha))
    agg["dataset"] = dataset_name
    agg["protocol"] = "within_well"
    agg["seed"] = seed
    return agg


def prepare_spwla(target: str, seed: int, max_per_well: int) -> pd.DataFrame:
    reset_spwla_features()
    return base.load_dataset(ROOT / "data" / "raw" / "train.csv", target, max_per_well=max_per_well, seed=seed)


def prepare_mendeley(max_per_well: int) -> pd.DataFrame:
    mendeley.configure_external_features()
    return mendeley.prepare_dataset(max_per_well=max_per_well, seed=42)


def write_report(summary: pd.DataFrame) -> None:
    lines = [
        "# Baseline Comparison Suite",
        "",
        "This suite uses the retained paper-facing baseline set.",
        "",
        "Methods:",
        "",
        "- WL-PCQR",
        "- Global RF-CQR",
        "- ExtraTrees-PCQR",
        "- Source-only RF-CQR (cross-well only)",
        "- NGBoost PI",
        "- HGB-CQR",
        "- LightGBM Quantile+CQR",
        "- CatBoost Quantile+CQR",
        "",
    ]
    for (dataset, protocol), group in summary.groupby(["dataset", "protocol"], sort=False):
        lines.extend(
            [
                f"## {dataset}: {protocol}",
                "",
                group[
                    [
                        "method_label",
                        "coverage_mean",
                        "p10_well_coverage_mean",
                        "worst_well_coverage_mean",
                        "width_mean",
                        "winkler_mean",
                        "rmse_mean",
                        "rank_winkler",
                    ]
                ].to_markdown(index=False, floatfmt=".4f"),
                "",
            ]
        )
    if FAILURE_ROWS:
        failures = pd.DataFrame(FAILURE_ROWS)
        failure_summary = (
            failures.groupby(["dataset", "protocol", "method_label"], dropna=False)
            .size()
            .reset_index(name="failed_splits")
            .sort_values(["dataset", "protocol", "method_label"])
        )
        lines.extend(
            [
                "## Failed Splits",
                "",
                "Some baselines can be numerically unstable on small target-well or depth-block fits. These failures are logged rather than silently replaced by another method.",
                "",
                failure_summary.to_markdown(index=False),
                "",
            ]
        )
    (OUT / "BASELINE_COMPARISON_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run paper-facing baseline comparisons on SPWLA and Mendeley.")
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--max-per-well", type=int, default=160)
    parser.add_argument("--local-calibration-frac", type=float, default=0.20)
    parser.add_argument("--blocks", type=int, default=5)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    failure_path = OUT / "baseline_comparison_failures.csv"
    if failure_path.exists():
        failure_path.unlink()

    raw_parts: list[pd.DataFrame] = []
    split_plan: list[tuple[str, str, str, int]] = []
    for target in SPWLA_TARGETS:
        for seed in args.seeds:
            split_plan.append(("SPWLA_PDDA_2021", "spwla", target, seed))
    for seed in args.seeds:
        split_plan.append(("Mendeley_GOM_DNS", "mendeley", "PhiT", seed))

    for dataset_name, dataset_kind, target, seed in split_plan:
        print(f"[dataset={dataset_name}] target={target} seed={seed}")
        if dataset_kind == "spwla":
            data = prepare_spwla(target, seed, args.max_per_well)
        else:
            data = prepare_mendeley(args.max_per_well)
        cross = run_target_cal_cross(data, dataset_name, target, seed, args.alpha, args.local_calibration_frac)
        within_agg = run_within(data, dataset_name, target, seed, args.alpha, args.blocks)
        raw_parts.append(cross)
        if not within_agg.empty:
            raw_parts.append(within_agg)

    raw = pd.concat(raw_parts, ignore_index=True)
    raw.to_csv(OUT / "baseline_comparison_raw.csv", index=False)
    if FAILURE_ROWS:
        pd.DataFrame(FAILURE_ROWS).to_csv(failure_path, index=False)
    summary = summarize(raw, ["dataset", "protocol", "method", "method_label"])
    summary.to_csv(OUT / "baseline_comparison_summary.csv", index=False)
    write_report(summary)
    print((OUT / "BASELINE_COMPARISON_REPORT.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()

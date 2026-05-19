from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import wlpcqr_core as base


ROOT = Path(__file__).resolve().parents[1]
EPS = 1e-8


METHODS = {
    "wl_pcqr": ("pcqr", "rf"),
    "extratrees_pcqr": ("clcqr", "et"),
    "hgb_cqr_signed": ("cqr_signed", "hgb"),
    "lightgbm_cqr_signed": ("cqr_signed", "lightgbm"),
    "catboost_cqr_signed": ("cqr_signed", "catboost"),
    "global_rf_cqr": ("cqr_signed", "rf"),
    "ngboost_cqr_signed": ("cqr_signed", "ngboost"),
}

METHOD_LABELS = {
    "wl_pcqr": "WL-PCQR",
    "extratrees_pcqr": "ExtraTrees-PCQR",
    "hgb_cqr_signed": "HGB-CQR",
    "lightgbm_cqr_signed": "LightGBM Quantile+CQR",
    "catboost_cqr_signed": "CatBoost Quantile+CQR",
    "global_rf_cqr": "Global RF-CQR",
    "ngboost_cqr_signed": "NGBoost PI",
}


def local_calibration_indices(n: int, frac: float, strategy: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if n < 2:
        raise ValueError("At least two target-well samples are required.")
    n_cal = int(round(n * frac))
    n_cal = min(max(n_cal, 1), n - 1)
    all_idx = np.arange(n)
    if strategy == "random":
        rng = np.random.default_rng(seed)
        cal_idx = np.sort(rng.choice(all_idx, size=n_cal, replace=False))
    elif strategy == "stratified":
        raw = np.linspace(0, n - 1, n_cal)
        cal_idx = np.unique(np.round(raw).astype(int))
        if len(cal_idx) < n_cal:
            remaining = np.setdiff1d(all_idx, cal_idx)
            need = n_cal - len(cal_idx)
            cal_idx = np.sort(np.concatenate([cal_idx, remaining[:need]]))
    elif strategy == "contiguous":
        start = max(0, (n - n_cal) // 2)
        cal_idx = np.arange(start, start + n_cal)
    else:
        raise ValueError(f"Unsupported local calibration strategy: {strategy}")
    test_idx = np.setdiff1d(all_idx, cal_idx)
    return cal_idx, test_idx


def fit_predict_variant(
    train: pd.DataFrame,
    eval_frame: pd.DataFrame,
    target: str,
    method: str,
    alpha: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    variant, family = METHODS[method]
    if variant == "cqr_signed":
        if family in {"rf", "et"}:
            pred, lower, upper = base.fit_predict_tree_quantiles(train, eval_frame, target, family, alpha, seed)
        elif family == "ngboost":
            pred, lower, upper = base.fit_predict_ngboost(train, eval_frame, target, alpha, seed)
        elif family == "hgb":
            lower, pred, upper = base.fit_predict_hgb_quantile_pair(train, eval_frame, target, alpha, seed)
        elif family == "lightgbm":
            lower, pred, upper = base.fit_predict_lightgbm_quantile_pair(train, eval_frame, target, alpha, seed)
        elif family == "catboost":
            lower, pred, upper = base.fit_predict_catboost_quantile_pair(train, eval_frame, target, alpha, seed)
        else:
            raise ValueError(method)
        return pred, lower, upper
    raise ValueError(method)


def fit_predict_tree_quantiles_with_leaves(
    train: pd.DataFrame,
    eval_frame: pd.DataFrame,
    target: str,
    family: str,
    alpha: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model = base.make_point_model(family, seed)
    model.fit(train[base.ACTIVE_FEATURES], train[target])
    pred = np.asarray(model.predict(eval_frame[base.ACTIVE_FEATURES]), dtype=float)
    x_eval = model.named_steps["simpleimputer"].transform(eval_frame[base.ACTIVE_FEATURES])
    forest = model.named_steps["extratreesregressor"] if family == "et" else model.named_steps["randomforestregressor"]
    tree_preds = np.asarray([tree.predict(x_eval) for tree in forest.estimators_], dtype=float)
    leaves = np.asarray([tree.apply(x_eval) for tree in forest.estimators_], dtype=np.int64).T
    lower = np.quantile(tree_preds, alpha / 2.0, axis=0)
    upper = np.quantile(tree_preds, 1.0 - alpha / 2.0, axis=0)
    return pred, np.minimum(lower, upper), np.maximum(lower, upper), leaves


def weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights >= 0.0)
    values = values[valid]
    weights = weights[valid]
    if values.size == 0:
        return 0.0
    if float(np.sum(weights)) <= EPS:
        weights = np.ones_like(values, dtype=float)
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cdf = np.cumsum(weights) / np.sum(weights)
    idx = int(np.searchsorted(cdf, quantile, side="left"))
    idx = min(max(idx, 0), len(values) - 1)
    return float(values[idx])


def empirical_quantile(values: np.ndarray, alpha: float) -> float:
    values = np.sort(np.asarray(values, dtype=float)[np.isfinite(values)])
    if values.size == 0:
        return 0.0
    k = int(np.ceil(values.size * (1.0 - alpha)))
    k = min(max(k, 1), values.size)
    return float(values[k - 1])


def mean_winkler(y: np.ndarray, lower: np.ndarray, upper: np.ndarray, alpha: float) -> float:
    width = upper - lower
    below = y < lower
    above = y > upper
    score = width.copy()
    score[below] += (2.0 / alpha) * (lower[below] - y[below])
    score[above] += (2.0 / alpha) * (y[above] - upper[above])
    return float(np.mean(score))


def wl_pcqr_path_interval(
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, str]:
    """Select a calibrated localization level from a fixed WL-PCQR path.

    The path has three nested levels: uniform CQR, proximity scale correction,
    and proximity location-scale correction. Selection uses only target-well
    calibration Winkler score subject to the nominal coverage constraint.
    """

    n_cal = len(y_cal)

    def global_candidate() -> tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        scores = np.maximum(lower_cal - y_cal, y_cal - upper_cal)
        q_all = base.conformal_quantile(scores, alpha)
        q_loo = np.empty(n_cal, dtype=float)
        for pos in range(n_cal):
            q_loo[pos] = base.conformal_quantile(scores[np.arange(n_cal) != pos], alpha)
        return (
            "uniform",
            pred_test,
            lower_cal - q_loo,
            upper_cal + q_loo,
            lower_test - q_all,
            upper_test + q_all,
            float(q_all),
        )

    def proximity_candidate(
        use_location: bool,
        empirical_wrapper: bool,
    ) -> tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        bias_cal = np.zeros(n_cal, dtype=float)
        bias_test = np.zeros(len(leaf_test), dtype=float)
        if use_location:
            residuals = y_cal - pred_cal
            for cal_pos, leaves in enumerate(leaf_cal):
                keep = np.arange(n_cal) != cal_pos
                weights = np.mean(leaf_cal[keep] == leaves[None, :], axis=1)
                bias_cal[cal_pos] = weighted_quantile(residuals[keep], weights, 0.5)
            for idx, leaves in enumerate(leaf_test):
                weights = np.mean(leaf_cal == leaves[None, :], axis=1)
                bias_test[idx] = weighted_quantile(residuals, weights, 0.5)

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
            cal_qhats[cal_pos] = weighted_quantile(scores[keep], weights, 1.0 - alpha)
        recal_scores = np.maximum.reduce(
            [
                shifted_lower_cal - cal_qhats - y_cal,
                y_cal - (shifted_upper_cal + cal_qhats),
                np.zeros_like(y_cal),
            ]
        )
        correction = (
            empirical_quantile(recal_scores, alpha)
            if empirical_wrapper
            else base.conformal_quantile(recal_scores, alpha)
        )

        qhats = np.empty(len(leaf_test), dtype=float)
        for idx, leaves in enumerate(leaf_test):
            weights = np.mean(leaf_cal == leaves[None, :], axis=1)
            qhats[idx] = weighted_quantile(scores, weights, 1.0 - alpha)
        name = "location-scale" if use_location else "scale"
        return (
            name,
            shifted_pred_test,
            shifted_lower_cal - cal_qhats - correction,
            shifted_upper_cal + cal_qhats + correction,
            shifted_lower_test - qhats - correction,
            shifted_upper_test + qhats + correction,
            float(np.mean(qhats) + correction) if qhats.size else float(correction),
        )

    candidates = [
        global_candidate(),
        proximity_candidate(use_location=False, empirical_wrapper=False),
        proximity_candidate(use_location=True, empirical_wrapper=True),
    ]
    target_coverage = 1.0 - alpha
    scored = []
    for name, pred_candidate, cal_lower, cal_upper, test_lower, test_upper, qhat in candidates:
        lo = np.minimum(cal_lower, cal_upper)
        hi = np.maximum(cal_lower, cal_upper)
        cal_lower, cal_upper = base.clip_interval(lo, hi)
        coverage = float(np.mean((y_cal >= cal_lower) & (y_cal <= cal_upper)))
        score = mean_winkler(y_cal, cal_lower, cal_upper, alpha)
        feasible = coverage + 1e-12 >= target_coverage
        scored.append((not feasible, score, -coverage, name, pred_candidate, test_lower, test_upper, qhat))
    scored.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
    _, _, _, selected, pred_out, lower, upper, qhat = scored[0]
    return pred_out, lower, upper, float(qhat), selected


def hide_stress_curves(frame: pd.DataFrame, stress_curves: list[str]) -> pd.DataFrame:
    if not stress_curves:
        return frame
    out = frame.copy()
    for curve in stress_curves:
        out[curve] = np.nan
    return out


def record_prediction_rows_local(
    frame: pd.DataFrame,
    target: str,
    protocol: str,
    split_id: str,
    method: str,
    seed: int,
    alpha: float,
    pred: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    clip_interval: bool,
) -> pd.DataFrame:
    lo = np.minimum(lower, upper)
    hi = np.maximum(lower, upper)
    lower, upper = lo, hi
    if clip_interval:
        lower, upper = base.clip_interval(lower, upper)
    out = frame[["WELLNUM", "DEPTH", target]].copy()
    out["protocol"] = protocol
    out["split_id"] = split_id
    out["method"] = method
    out["seed"] = seed
    out["alpha"] = alpha
    out["target_coverage"] = 1.0 - alpha
    out["pred"] = pred
    out["lower"] = lower
    out["upper"] = upper
    out["width"] = upper - lower
    out["covered"] = (out[target] >= out["lower"]) & (out[target] <= out["upper"])
    return out


def evaluate_local(
    data: pd.DataFrame,
    target: str,
    method: str,
    alpha: float,
    local_calibration_frac: float,
    strategy: str,
    seed: int,
    stress_curves: list[str] | None = None,
    clip_intervals: bool = True,
) -> tuple[list[dict[str, object]], list[pd.DataFrame]]:
    metric_rows: list[dict[str, object]] = []
    pred_parts: list[pd.DataFrame] = []
    stress_curves = stress_curves or []
    for test_well in sorted(data["WELLNUM"].unique()):
        train = data[data["WELLNUM"] != test_well].copy()
        target_well = data[data["WELLNUM"] == test_well].copy().sort_values("DEPTH").reset_index(drop=True)
        cal_idx, test_idx = local_calibration_indices(
            len(target_well),
            local_calibration_frac,
            strategy,
            seed + 7919 * int(test_well),
        )
        cal = target_well.iloc[cal_idx].copy().reset_index(drop=True)
        test = target_well.iloc[test_idx].copy().sort_values("DEPTH").reset_index(drop=True)
        eval_frame = pd.concat(
            [hide_stress_curves(cal, stress_curves), hide_stress_curves(test, stress_curves)],
            ignore_index=True,
        )
        variant, family = METHODS[method]
        leaf_all = None
        if variant in {"pcqr", "lcqr", "clcqr"}:
            pred_all, lower_all, upper_all, leaf_all = fit_predict_tree_quantiles_with_leaves(
                train,
                eval_frame,
                target,
                family,
                alpha,
                seed + 37 * int(test_well),
            )
        else:
            pred_all, lower_all, upper_all = fit_predict_variant(
                train,
                eval_frame,
                target,
                method,
                alpha,
                seed + 37 * int(test_well),
            )
        n_cal = len(cal)
        y_cal = cal[target].to_numpy(dtype=float)
        pred_test = pred_all[n_cal:]
        assert lower_all is not None and upper_all is not None
        lower_cal = lower_all[:n_cal]
        upper_cal = upper_all[:n_cal]
        lower_test = lower_all[n_cal:]
        upper_test = upper_all[n_cal:]
        selected_level = variant
        if variant == "cqr_signed":
            scores = np.maximum(lower_cal - y_cal, y_cal - upper_cal)
            qhat = base.conformal_quantile(scores, alpha)
            qhat_lower = np.nan
            qhat_upper = np.nan
            lower = lower_test - qhat
            upper = upper_test + qhat
        elif variant == "pcqr":
            assert leaf_all is not None
            pred_cal = pred_all[:n_cal]
            pred_test, lower, upper, qhat, selected_level = wl_pcqr_path_interval(
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
            qhat_lower = np.nan
            qhat_upper = np.nan
        elif variant in {"lcqr", "clcqr"}:
            assert leaf_all is not None
            leaf_cal = leaf_all[:n_cal]
            leaf_test = leaf_all[n_cal:]
            scores = np.maximum(lower_cal - y_cal, y_cal - upper_cal)
            correction = 0.0
            if variant == "clcqr":
                cal_qhats = np.empty(n_cal, dtype=float)
                for cal_pos, leaves in enumerate(leaf_cal):
                    keep = np.arange(n_cal) != cal_pos
                    weights = np.mean(leaf_cal[keep] == leaves[None, :], axis=1)
                    cal_qhats[cal_pos] = weighted_quantile(scores[keep], weights, 1.0 - alpha)
                recal_scores = np.maximum.reduce(
                    [
                        lower_cal - cal_qhats - y_cal,
                        y_cal - (upper_cal + cal_qhats),
                        np.zeros_like(y_cal),
                    ]
                )
                correction = max(0.0, base.conformal_quantile(recal_scores, alpha))
            qhats = np.empty(len(leaf_test), dtype=float)
            for idx, leaves in enumerate(leaf_test):
                weights = np.mean(leaf_cal == leaves[None, :], axis=1)
                qhats[idx] = weighted_quantile(scores, weights, 1.0 - alpha)
            qhat = float(np.mean(qhats) + correction) if qhats.size else float(correction)
            qhat_lower = np.nan
            qhat_upper = np.nan
            lower = lower_test - qhats - correction
            upper = upper_test + qhats + correction
        else:
            raise ValueError(f"Unsupported local variant: {variant}")
        lo = np.minimum(lower, upper)
        hi = np.maximum(lower, upper)
        lower, upper = lo, hi
        if clip_intervals:
            lower, upper = base.clip_interval(lower, upper)
        y = test[target].to_numpy(dtype=float)
        rec: dict[str, object] = {
            "protocol": "LOWO_LOCAL_CAL",
            "well": str(test_well),
            "method": method,
            "method_label": METHOD_LABELS[method],
            "target": target,
            "alpha": alpha,
            "target_coverage": 1.0 - alpha,
            "seed": seed,
            "local_calibration_frac": local_calibration_frac,
            "local_calibration_strategy": strategy,
            "stress_curves": ",".join(stress_curves) if stress_curves else "none",
            "clip_intervals": bool(clip_intervals),
            "calibration_n": int(n_cal),
            "n": int(len(test)),
            "qhat": float(qhat),
            "qhat_lower": float(qhat_lower) if np.isfinite(qhat_lower) else np.nan,
            "qhat_upper": float(qhat_upper) if np.isfinite(qhat_upper) else np.nan,
            "selected_level": selected_level,
        }
        rec.update(base.point_metrics(y, pred_test))
        rec.update(base.interval_metrics(y, lower, upper, alpha))
        metric_rows.append(rec)
        pred_frame = record_prediction_rows_local(
            test,
            target,
            "LOWO_LOCAL_CAL",
            f"LOWO_LOCAL_{test_well}",
            method,
            seed,
            alpha,
            pred_test,
            lower,
            upper,
            clip_intervals,
        )
        pred_frame["method_label"] = METHOD_LABELS[method]
        pred_frame["local_calibration_frac"] = local_calibration_frac
        pred_frame["local_calibration_strategy"] = strategy
        pred_frame["stress_curves"] = ",".join(stress_curves) if stress_curves else "none"
        pred_frame["clip_intervals"] = bool(clip_intervals)
        pred_parts.append(pred_frame)
        print(
            f"[LOWO_LOCAL_CAL] {method} well={test_well} n={len(test)} "
            f"cov={rec['coverage']:.4f} width={rec['width']:.4f} winkler={rec['winkler']:.4f}"
        )
    return metric_rows, pred_parts


def aggregate_metrics(split_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for method, group in split_df.groupby("method", sort=True):
        weights = group["n"].to_numpy(dtype=float)
        rec: dict[str, object] = {
            "method": method,
            "method_label": str(group["method_label"].iloc[0]),
            "target": str(group["target"].iloc[0]),
            "splits": int(len(group)),
            "n": int(group["n"].sum()),
        }
        for col in ["rmse", "mae", "r2", "corr", "coverage", "width", "width_std", "width_cv", "winkler"]:
            rec[col] = float(np.average(group[col].to_numpy(dtype=float), weights=weights))
        rec["macro_well_coverage"] = float(group["coverage"].mean())
        rec["p10_well_coverage"] = float(np.percentile(group["coverage"].to_numpy(dtype=float), 10))
        rec["worst_well_coverage"] = float(group["coverage"].min())
        rec["worst_well"] = str(group.sort_values("coverage").iloc[0]["well"])
        rec["macro_well_width"] = float(group["width"].mean())
        rec["macro_well_winkler"] = float(group["winkler"].mean())
        rows.append(rec)
    return pd.DataFrame(rows).sort_values(["winkler", "coverage"], ascending=[True, False])


def write_report(out_dir: Path, split_df: pd.DataFrame, agg_df: pd.DataFrame, args: argparse.Namespace) -> None:
    lines = [
        "# WL-PCQR Target-Well Interval Suite",
        "",
        f"Target: `{args.target}`",
        f"Alpha: `{args.alpha}`",
        f"Methods: `{', '.join(args.methods)}`",
        f"Max per well: `{args.max_per_well}`",
        f"Local calibration fraction: `{args.local_calibration_frac}`",
        f"Local calibration strategy: `{args.local_calibration_strategy}`",
        f"Excluded features: `{', '.join(args.exclude_features) if args.exclude_features else 'none'}`",
        f"Stress curves hidden in target calibration/test: `{', '.join(args.stress_curves) if args.stress_curves else 'none'}`",
        f"Clip intervals to [0, 1]: `{not args.no_clip_interval}`",
        "",
        "## Aggregate Metrics",
        "",
        agg_df.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Per-Well Metrics",
        "",
        split_df.sort_values(["method", "well"]).to_markdown(index=False, floatfmt=".6f"),
    ]
    (out_dir / "WLPCQR_TARGET_WELL_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run WL-PCQR and compact target-well interval baselines.")
    parser.add_argument("--data", default=str(ROOT / "data" / "raw" / "train.csv"))
    parser.add_argument("--out", default=str(ROOT / "analysis" / "wlpcqr_target_well_run"))
    parser.add_argument("--target", default="PHIF", choices=["PHIF", "SW", "VSH"])
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--max-per-well", type=int, default=160)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--methods", nargs="+", default=["global_rf_cqr", "wl_pcqr"])
    parser.add_argument("--local-calibration-frac", type=float, default=0.20)
    parser.add_argument("--local-calibration-strategy", choices=["random", "stratified", "contiguous"], default="random")
    parser.add_argument("--exclude-features", nargs="*", default=[])
    parser.add_argument(
        "--stress-curves",
        nargs="*",
        default=[],
        help="Feature curves to set to missing in target-well calibration/test only, e.g. DEN NEU or RDEP_LOG10 RMED_LOG10.",
    )
    parser.add_argument("--no-clip-interval", action="store_true", help="Do not clip prediction intervals to the physical [0, 1] target range.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    unknown_methods = sorted(set(args.methods) - set(METHODS))
    if unknown_methods:
        raise ValueError(f"Unknown methods: {unknown_methods}. Supported: {sorted(METHODS)}")
    unknown_excluded = sorted(set(args.exclude_features) - set(base.FEATURES))
    if unknown_excluded:
        raise ValueError(f"Unknown excluded features: {unknown_excluded}. Supported: {base.FEATURES}")
    unknown_stress = sorted(set(args.stress_curves) - set(base.FEATURES))
    if unknown_stress:
        raise ValueError(f"Unknown stress curves: {unknown_stress}. Supported: {base.FEATURES}")
    overlap = sorted(set(args.stress_curves) & set(args.exclude_features))
    if overlap:
        raise ValueError(f"Stress curves cannot also be excluded features: {overlap}")
    base.ACTIVE_FEATURES = [feature for feature in base.FEATURES if feature not in set(args.exclude_features)]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = base.load_dataset(Path(args.data), args.target, args.max_per_well, args.seed)

    rows: list[dict[str, object]] = []
    preds: list[pd.DataFrame] = []
    for method in args.methods:
        metric_rows, pred_parts = evaluate_local(
            data,
            args.target,
            method,
            args.alpha,
            args.local_calibration_frac,
            args.local_calibration_strategy,
            args.seed,
            args.stress_curves,
            not args.no_clip_interval,
        )
        rows.extend(metric_rows)
        preds.extend(pred_parts)

    split_df = pd.DataFrame(rows)
    agg_df = aggregate_metrics(split_df)
    split_df.to_csv(out_dir / f"{args.target.lower()}_wlpcqr_target_well_split_metrics.csv", index=False)
    agg_df.to_csv(out_dir / f"{args.target.lower()}_wlpcqr_target_well_aggregate_metrics.csv", index=False)
    if preds:
        pd.concat(preds, ignore_index=True).to_csv(out_dir / f"{args.target.lower()}_wlpcqr_target_well_predictions.csv", index=False)
    write_report(out_dir, split_df, agg_df, args)
    print(f"Wrote {out_dir}")
    print(agg_df.round(6).to_string(index=False))


if __name__ == "__main__":
    main()

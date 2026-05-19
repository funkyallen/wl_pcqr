from __future__ import annotations

import argparse
import contextlib
import io
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

import run_wlpcqr_methods as local
import wlpcqr_core as base


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "reviewer_required_diagnostics"
SPWLA_TARGETS = ["PHIF", "SW", "VSH"]
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


def reset_spwla_features() -> None:
    base.FEATURES = list(SPWLA_FEATURES)
    base.ACTIVE_FEATURES = [feature for feature in SPWLA_FEATURES if feature != "DEPTH"]


def cqr_scores(y: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    return np.maximum.reduce([lower - y, y - upper, np.zeros_like(y, dtype=float)])


def target_bound_audit() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(ROOT / "data" / "raw" / "train.csv").replace(-9999.0, np.nan)
    rows: list[dict[str, object]] = []
    by_well_rows: list[dict[str, object]] = []
    for target in SPWLA_TARGETS:
        valid = raw[target].dropna()
        outside = valid[(valid < 0.0) | (valid > 1.0)]
        rows.append(
            {
                "target": target,
                "valid_raw": int(len(valid)),
                "kept_0_1": int(((valid >= 0.0) & (valid <= 1.0)).sum()),
                "outside_0_1": int(len(outside)),
                "outside_pct": float(len(outside) / len(valid)) if len(valid) else 0.0,
                "min_raw": float(valid.min()) if len(valid) else np.nan,
                "max_raw": float(valid.max()) if len(valid) else np.nan,
            }
        )
        for well, group in raw[raw[target].notna()].groupby("WELLNUM", sort=True):
            values = group[target]
            by_well_rows.append(
                {
                    "target": target,
                    "well": int(well),
                    "valid_raw": int(len(values)),
                    "outside_0_1": int(((values < 0.0) | (values > 1.0)).sum()),
                }
            )
    summary = pd.DataFrame(rows)
    by_well = pd.DataFrame(by_well_rows)
    summary.to_csv(OUT / "spwla_target_bound_audit.csv", index=False)
    by_well.to_csv(OUT / "spwla_target_bound_by_well.csv", index=False)
    return summary, by_well


def mendeley_inventory() -> pd.DataFrame:
    path = ROOT / "data" / "external" / "mendeley_gom_dutch_north_sea" / "processed" / "mendeley_phi_external.csv"
    data = pd.read_csv(path)
    feature_cols = [c for c in data.columns if c not in {"WELLNUM", "WELL_NAME", "DOMAIN", "DEPTH", "PhiT"}]
    inv = (
        data.groupby(["WELLNUM", "WELL_NAME", "DOMAIN"], sort=True)
        .agg(
            samples=("PhiT", "size"),
            depth_min=("DEPTH", "min"),
            depth_max=("DEPTH", "max"),
            phit_min=("PhiT", "min"),
            phit_mean=("PhiT", "mean"),
            phit_max=("PhiT", "max"),
        )
        .reset_index()
    )
    for col in feature_cols:
        miss = data.groupby("WELLNUM", sort=True)[col].apply(lambda s: float(s.isna().mean())).reset_index(name=f"{col}_missing")
        inv = inv.merge(miss, on="WELLNUM", how="left")
    inv.to_csv(OUT / "mendeley_dataset_summary.csv", index=False)
    return inv


def summarize_with_ci(raw_path: Path) -> pd.DataFrame:
    raw = pd.read_csv(raw_path)
    raw = raw[(raw["dataset"] == "SPWLA_PDDA_2021") & (raw["protocol"] == "cross_well")].copy()
    metrics = ["coverage", "p10_well_coverage", "worst_well_coverage", "width", "winkler", "rmse"]
    rows: list[dict[str, object]] = []
    for method, group in raw.groupby("method_label", sort=False):
        rec: dict[str, object] = {"method_label": method, "n_target_seed": int(len(group))}
        for metric in metrics:
            values = group[metric].to_numpy(dtype=float)
            mean = float(np.mean(values))
            sd = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            half = 1.96 * sd / np.sqrt(max(len(values), 1))
            rec[f"{metric}_mean"] = mean
            rec[f"{metric}_sd"] = sd
            rec[f"{metric}_ci95_low"] = mean - half
            rec[f"{metric}_ci95_high"] = mean + half
        rows.append(rec)
    out = pd.DataFrame(rows).sort_values("winkler_mean")
    out["rank_winkler"] = out["winkler_mean"].rank(method="min", ascending=True).astype(int)
    out.to_csv(OUT / "spwla_cross_well_mean_sd_ci.csv", index=False)

    rank_rows: list[dict[str, object]] = []
    for (target, seed), group in raw.groupby(["target", "seed"], sort=True):
        ranked = group.sort_values("winkler")
        for rank, (_, row) in enumerate(ranked.iterrows(), start=1):
            rank_rows.append(
                {
                    "target": target,
                    "seed": int(seed),
                    "method_label": row["method_label"],
                    "winkler": float(row["winkler"]),
                    "coverage": float(row["coverage"]),
                    "worst_well_coverage": float(row["worst_well_coverage"]),
                    "rank_winkler": int(rank),
                }
            )
    ranks = pd.DataFrame(rank_rows)
    ranks.to_csv(OUT / "spwla_cross_well_target_seed_ranks.csv", index=False)
    return out


def interval_from_qrf(
    train: pd.DataFrame,
    eval_frame: pd.DataFrame,
    target: str,
    alpha: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model = base.make_point_model("rf", seed)
    model.fit(train[base.ACTIVE_FEATURES], train[target])
    pred = np.asarray(model.predict(eval_frame[base.ACTIVE_FEATURES]), dtype=float)
    imputer = model.named_steps["simpleimputer"]
    forest = model.named_steps["randomforestregressor"]
    x_train = imputer.transform(train[base.ACTIVE_FEATURES])
    x_eval = imputer.transform(eval_frame[base.ACTIVE_FEATURES])
    leaf_train = np.asarray([tree.apply(x_train) for tree in forest.estimators_], dtype=np.int64).T
    leaf_eval = np.asarray([tree.apply(x_eval) for tree in forest.estimators_], dtype=np.int64).T
    y_train = train[target].to_numpy(dtype=float)
    lower = np.empty(len(eval_frame), dtype=float)
    upper = np.empty(len(eval_frame), dtype=float)
    for i, leaves in enumerate(leaf_eval):
        weights = np.mean(leaf_train == leaves[None, :], axis=1)
        lower[i] = local.weighted_quantile(y_train, weights, alpha / 2.0)
        upper[i] = local.weighted_quantile(y_train, weights, 1.0 - alpha / 2.0)
    return pred, np.minimum(lower, upper), np.maximum(lower, upper)


def rf_tree_interval_with_scaled_features(
    train: pd.DataFrame,
    eval_frame: pd.DataFrame,
    target: str,
    alpha: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model = base.make_point_model("rf", seed)
    model.fit(train[base.ACTIVE_FEATURES], train[target])
    pred = np.asarray(model.predict(eval_frame[base.ACTIVE_FEATURES]), dtype=float)
    imputer = model.named_steps["simpleimputer"]
    forest = model.named_steps["randomforestregressor"]
    x_train = imputer.transform(train[base.ACTIVE_FEATURES])
    x_eval = imputer.transform(eval_frame[base.ACTIVE_FEATURES])
    scaler = StandardScaler().fit(x_train)
    x_eval_scaled = scaler.transform(x_eval)
    tree_preds = np.asarray([tree.predict(x_eval) for tree in forest.estimators_], dtype=float)
    lower = np.quantile(tree_preds, alpha / 2.0, axis=0)
    upper = np.quantile(tree_preds, 1.0 - alpha / 2.0, axis=0)
    return pred, np.minimum(lower, upper), np.maximum(lower, upper), x_eval_scaled


def gr_mondrian_groups(cal: pd.DataFrame, frame: pd.DataFrame) -> np.ndarray:
    values = cal["GR"].dropna().to_numpy(dtype=float)
    if values.size < 3:
        return np.zeros(len(frame), dtype=int)
    q1, q2 = np.quantile(values, [1.0 / 3.0, 2.0 / 3.0])
    raw = frame["GR"].to_numpy(dtype=float)
    groups = np.full(len(frame), -1, dtype=int)
    finite = np.isfinite(raw)
    groups[finite & (raw <= q1)] = 0
    groups[finite & (raw > q1) & (raw <= q2)] = 1
    groups[finite & (raw > q2)] = 2
    return groups


def evaluate_reviewer_baseline(
    data: pd.DataFrame,
    target: str,
    method: str,
    seed: int,
    alpha: float,
    frac: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for test_well in sorted(data["WELLNUM"].unique()):
        train = data[data["WELLNUM"] != test_well].copy()
        target_well = data[data["WELLNUM"] == test_well].copy().sort_values("DEPTH").reset_index(drop=True)
        cal_idx, test_idx = local.local_calibration_indices(len(target_well), frac, "stratified", seed + 7919 * int(test_well))
        cal = target_well.iloc[cal_idx].copy().reset_index(drop=True)
        test = target_well.iloc[test_idx].copy().sort_values("DEPTH").reset_index(drop=True)
        eval_frame = pd.concat([cal, test], ignore_index=True)
        n_cal = len(cal)
        y_cal = cal[target].to_numpy(dtype=float)
        if method == "qrf_cqr":
            pred_all, lower_all, upper_all = interval_from_qrf(train, eval_frame, target, alpha, seed + 37 * int(test_well))
            scores = cqr_scores(y_cal, lower_all[:n_cal], upper_all[:n_cal])
            qhat = base.conformal_quantile(scores, alpha)
            pred_test = pred_all[n_cal:]
            lower = lower_all[n_cal:] - qhat
            upper = upper_all[n_cal:] + qhat
        elif method == "knn_local_cqr":
            pred_all, lower_all, upper_all, x_scaled = rf_tree_interval_with_scaled_features(
                train, eval_frame, target, alpha, seed + 37 * int(test_well)
            )
            scores = cqr_scores(y_cal, lower_all[:n_cal], upper_all[:n_cal])
            cal_x = x_scaled[:n_cal]
            test_x = x_scaled[n_cal:]
            k = int(max(5, np.ceil(np.sqrt(n_cal))))
            qhats = np.empty(len(test), dtype=float)
            for i, x in enumerate(test_x):
                dist = np.linalg.norm(cal_x - x[None, :], axis=1)
                nn = np.argsort(dist)[: min(k, n_cal)]
                qhats[i] = base.conformal_quantile(scores[nn], alpha)
            pred_test = pred_all[n_cal:]
            lower = lower_all[n_cal:] - qhats
            upper = upper_all[n_cal:] + qhats
            qhat = float(np.mean(qhats))
        elif method == "gr_mondrian_cqr":
            pred_all, lower_all, upper_all = base.fit_predict_tree_quantiles(train, eval_frame, target, "rf", alpha, seed + 37 * int(test_well))
            scores = cqr_scores(y_cal, lower_all[:n_cal], upper_all[:n_cal])
            global_q = base.conformal_quantile(scores, alpha)
            cal_groups = gr_mondrian_groups(cal, cal)
            test_groups = gr_mondrian_groups(cal, test)
            q_by_group: dict[int, float] = {}
            for group_id in sorted(set(cal_groups.tolist())):
                mask = cal_groups == group_id
                q_by_group[group_id] = base.conformal_quantile(scores[mask], alpha) if int(mask.sum()) >= 5 else global_q
            qhats = np.asarray([q_by_group.get(int(g), global_q) for g in test_groups], dtype=float)
            pred_test = pred_all[n_cal:]
            lower = lower_all[n_cal:] - qhats
            upper = upper_all[n_cal:] + qhats
            qhat = float(np.mean(qhats))
        else:
            raise ValueError(method)

        lower, upper = base.clip_interval(np.minimum(lower, upper), np.maximum(lower, upper))
        y = test[target].to_numpy(dtype=float)
        rec: dict[str, object] = {
            "well": str(test_well),
            "method": method,
            "method_label": {
                "qrf_cqr": "RF-Leaf QRF+CQR",
                "knn_local_cqr": "kNN Local RF-CQR",
                "gr_mondrian_cqr": "GR-Mondrian RF-CQR",
            }[method],
            "target": target,
            "seed": seed,
            "n": int(len(test)),
            "calibration_n": int(n_cal),
            "qhat": float(qhat),
        }
        rec.update(base.point_metrics(y, pred_test))
        rec.update(base.interval_metrics(y, lower, upper, alpha))
        rows.append(rec)
    return rows


def summarize_aggregate(split_df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for group_keys, group in split_df.groupby(keys, sort=True):
        if not isinstance(group_keys, tuple):
            group_keys = (group_keys,)
        rec: dict[str, object] = dict(zip(keys, group_keys))
        rec["splits"] = int(len(group))
        rec["n"] = int(group["n"].sum())
        weights = group["n"].to_numpy(dtype=float)
        for col in ["rmse", "mae", "coverage", "width", "winkler"]:
            rec[col] = float(np.average(group[col].to_numpy(dtype=float), weights=weights))
        rec["macro_well_coverage"] = float(group["coverage"].mean())
        rec["p10_well_coverage"] = float(np.percentile(group["coverage"].to_numpy(dtype=float), 10))
        rec["worst_well_coverage"] = float(group["coverage"].min())
        rec["macro_well_width"] = float(group["width"].mean())
        rec["macro_well_winkler"] = float(group["winkler"].mean())
        rows.append(rec)
    out = pd.DataFrame(rows)
    if "winkler" in out.columns:
        rank_keys = [key for key in keys if key not in {"method", "method_label"}]
        if rank_keys:
            out["rank_winkler"] = out.groupby(rank_keys)["winkler"].rank(method="min", ascending=True).astype(int)
        else:
            out["rank_winkler"] = out["winkler"].rank(method="min", ascending=True).astype(int)
    return out.sort_values([key for key in keys if key in out.columns] + ["rank_winkler"])


def summarize_paper_level(split_df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    run_keys = [key for key in ["gap_steps", "target", "seed", "method", "method_label"] if key in split_df.columns]
    per_run = summarize_aggregate(split_df, run_keys)
    metrics = [
        "coverage",
        "p10_well_coverage",
        "worst_well_coverage",
        "width",
        "winkler",
        "rmse",
        "macro_well_coverage",
        "macro_well_winkler",
    ]
    available = [col for col in metrics if col in per_run.columns]
    out = per_run.groupby(keys, dropna=False)[available].mean().reset_index()
    rank_keys = [key for key in keys if key not in {"method", "method_label"}]
    if rank_keys:
        out["rank_winkler"] = out.groupby(rank_keys)["winkler"].rank(method="min", ascending=True).astype(int)
    else:
        out["rank_winkler"] = out["winkler"].rank(method="min", ascending=True).astype(int)
    sort_cols = rank_keys + ["rank_winkler"] if rank_keys else ["rank_winkler"]
    return out.sort_values(sort_cols).reset_index(drop=True)


def run_reviewer_baselines(args: argparse.Namespace) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for target in args.targets:
        for seed in args.seeds:
            data = base.load_dataset(ROOT / "data" / "raw" / "train.csv", target, args.max_per_well, seed)
            for method in ["qrf_cqr", "knn_local_cqr", "gr_mondrian_cqr"]:
                print(f"[reviewer-baseline] target={target} seed={seed} method={method}")
                rows.extend(evaluate_reviewer_baseline(data, target, method, seed, args.alpha, args.local_calibration_frac))
            for method in ["global_rf_cqr", "wl_pcqr"]:
                print(f"[reference-baseline] target={target} seed={seed} method={method}")
                with contextlib.redirect_stdout(io.StringIO()):
                    metric_rows, _ = local.evaluate_local(
                        data=data,
                        target=target,
                        method=method,
                        alpha=args.alpha,
                        local_calibration_frac=args.local_calibration_frac,
                        strategy="stratified",
                        seed=seed,
                        stress_curves=[],
                        clip_intervals=True,
                    )
                rows.extend(metric_rows)
    split = pd.DataFrame(rows)
    split.to_csv(OUT / "reviewer_nearest_baseline_split_metrics.csv", index=False)
    by_target = summarize_paper_level(split, ["target", "method", "method_label"])
    by_target.to_csv(OUT / "reviewer_nearest_baseline_by_target.csv", index=False)
    summary = summarize_paper_level(split, ["method", "method_label"])
    summary.to_csv(OUT / "reviewer_nearest_baseline_summary.csv", index=False)
    return summary


def stratified_gap_indices(n: int, frac: float, gap_steps: int) -> tuple[np.ndarray, np.ndarray]:
    cal_idx, test_idx = local.local_calibration_indices(n, frac, "stratified", 0)
    if gap_steps <= 0:
        return cal_idx, test_idx
    dist = np.min(np.abs(test_idx[:, None] - cal_idx[None, :]), axis=1)
    test_idx = test_idx[dist >= gap_steps]
    return cal_idx, test_idx


def evaluate_wlpcqr_with_indices(
    data: pd.DataFrame,
    target: str,
    seed: int,
    alpha: float,
    frac: float,
    gap_steps: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for test_well in sorted(data["WELLNUM"].unique()):
        train = data[data["WELLNUM"] != test_well].copy()
        target_well = data[data["WELLNUM"] == test_well].copy().sort_values("DEPTH").reset_index(drop=True)
        cal_idx, test_idx = stratified_gap_indices(len(target_well), frac, gap_steps)
        if len(test_idx) < 2:
            continue
        cal = target_well.iloc[cal_idx].copy().reset_index(drop=True)
        test = target_well.iloc[test_idx].copy().sort_values("DEPTH").reset_index(drop=True)
        eval_frame = pd.concat([cal, test], ignore_index=True)
        pred_all, lower_all, upper_all, leaves = local.fit_predict_tree_quantiles_with_leaves(
            train, eval_frame, target, "rf", alpha, seed + 37 * int(test_well)
        )
        n_cal = len(cal)
        pred_test, lower, upper, qhat, selected_level = local.wl_pcqr_path_interval(
            y_cal=cal[target].to_numpy(dtype=float),
            pred_cal=pred_all[:n_cal],
            pred_test=pred_all[n_cal:],
            lower_cal=lower_all[:n_cal],
            upper_cal=upper_all[:n_cal],
            lower_test=lower_all[n_cal:],
            upper_test=upper_all[n_cal:],
            leaf_cal=leaves[:n_cal],
            leaf_test=leaves[n_cal:],
            alpha=alpha,
        )
        lower, upper = base.clip_interval(np.minimum(lower, upper), np.maximum(lower, upper))
        y = test[target].to_numpy(dtype=float)
        rec: dict[str, object] = {
            "diagnostic": "depth_gap_stratified",
            "gap_steps": int(gap_steps),
            "well": str(test_well),
            "method": "wl_pcqr",
            "method_label": "WL-PCQR",
            "target": target,
            "seed": seed,
            "n": int(len(test)),
            "calibration_n": int(n_cal),
            "selected_level": selected_level,
            "qhat": float(qhat),
        }
        rec.update(base.point_metrics(y, pred_test))
        rec.update(base.interval_metrics(y, lower, upper, alpha))
        rows.append(rec)
    return rows


def run_depth_gap(args: argparse.Namespace) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for target in args.targets:
        for seed in args.seeds:
            data = base.load_dataset(ROOT / "data" / "raw" / "train.csv", target, args.max_per_well, seed)
            for gap in args.gap_steps:
                print(f"[depth-gap] target={target} seed={seed} gap={gap}")
                rows.extend(evaluate_wlpcqr_with_indices(data, target, seed, args.alpha, args.local_calibration_frac, gap))
    split = pd.DataFrame(rows)
    split.to_csv(OUT / "depth_gap_stratified_split_metrics.csv", index=False)
    by_target = summarize_paper_level(split, ["gap_steps", "target", "method", "method_label"])
    by_target.to_csv(OUT / "depth_gap_stratified_by_target.csv", index=False)
    summary = summarize_paper_level(split, ["gap_steps", "method", "method_label"])
    summary.to_csv(OUT / "depth_gap_stratified_summary.csv", index=False)
    return summary


def evaluate_proper_split(
    data: pd.DataFrame,
    target: str,
    seed: int,
    alpha: float,
    frac: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for test_well in sorted(data["WELLNUM"].unique()):
        train = data[data["WELLNUM"] != test_well].copy()
        target_well = data[data["WELLNUM"] == test_well].copy().sort_values("DEPTH").reset_index(drop=True)
        cal20_idx, test_idx = local.local_calibration_indices(len(target_well), frac, "stratified", seed + 7919 * int(test_well))
        cal20_idx = np.sort(cal20_idx)
        loc_idx = cal20_idx[::2]
        cov_idx = cal20_idx[1::2]
        if len(cov_idx) < 2 or len(loc_idx) < 2:
            continue
        loc = target_well.iloc[loc_idx].copy().reset_index(drop=True)
        cov = target_well.iloc[cov_idx].copy().reset_index(drop=True)
        test = target_well.iloc[test_idx].copy().sort_values("DEPTH").reset_index(drop=True)
        eval_frame = pd.concat([loc, cov, test], ignore_index=True)
        pred_all, lower_all, upper_all, leaves = local.fit_predict_tree_quantiles_with_leaves(
            train, eval_frame, target, "rf", alpha, seed + 37 * int(test_well)
        )
        n_loc = len(loc)
        n_cov = len(cov)
        pred_eval, lower_eval, upper_eval, q_path, selected_level = local.wl_pcqr_path_interval(
            y_cal=loc[target].to_numpy(dtype=float),
            pred_cal=pred_all[:n_loc],
            pred_test=pred_all[n_loc:],
            lower_cal=lower_all[:n_loc],
            upper_cal=upper_all[:n_loc],
            lower_test=lower_all[n_loc:],
            upper_test=upper_all[n_loc:],
            leaf_cal=leaves[:n_loc],
            leaf_test=leaves[n_loc:],
            alpha=alpha,
        )
        lower_cov = lower_eval[:n_cov]
        upper_cov = upper_eval[:n_cov]
        lower_test = lower_eval[n_cov:]
        upper_test = upper_eval[n_cov:]
        q_cov = base.conformal_quantile(cqr_scores(cov[target].to_numpy(dtype=float), lower_cov, upper_cov), alpha)
        lower = lower_test - q_cov
        upper = upper_test + q_cov
        lower, upper = base.clip_interval(np.minimum(lower, upper), np.maximum(lower, upper))
        y = test[target].to_numpy(dtype=float)
        rec: dict[str, object] = {
            "diagnostic": "proper_split_wrapper",
            "well": str(test_well),
            "method": "proper_split_wl_pcqr",
            "method_label": "Proper-split WL-PCQR",
            "target": target,
            "seed": seed,
            "n": int(len(test)),
            "selection_n": int(n_loc),
            "coverage_calibration_n": int(n_cov),
            "selected_level": selected_level,
            "q_path": float(q_path),
            "q_cov": float(q_cov),
        }
        rec.update(base.point_metrics(y, pred_eval[n_cov:]))
        rec.update(base.interval_metrics(y, lower, upper, alpha))
        rows.append(rec)
    return rows


def run_proper_split(args: argparse.Namespace) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for target in args.targets:
        for seed in args.seeds:
            data = base.load_dataset(ROOT / "data" / "raw" / "train.csv", target, args.max_per_well, seed)
            print(f"[proper-split] target={target} seed={seed}")
            rows.extend(evaluate_proper_split(data, target, seed, args.alpha, args.local_calibration_frac))
    split = pd.DataFrame(rows)
    split.to_csv(OUT / "proper_split_wrapper_split_metrics.csv", index=False)
    by_target = summarize_paper_level(split, ["target", "method", "method_label"])
    by_target.to_csv(OUT / "proper_split_wrapper_by_target.csv", index=False)
    summary = summarize_paper_level(split, ["method", "method_label"])
    summary.to_csv(OUT / "proper_split_wrapper_summary.csv", index=False)
    return summary


def write_report(
    bound_summary: pd.DataFrame,
    mendeley: pd.DataFrame,
    ci: pd.DataFrame,
    reviewer_baselines: pd.DataFrame,
    depth_gap: pd.DataFrame,
    proper_split: pd.DataFrame,
) -> None:
    lines = [
        "# Reviewer-Required Diagnostics",
        "",
        "This report records diagnostics added in response to the reviewer-style critique.",
        "",
        "## Target Bound Audit",
        "",
        bound_summary.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Mendeley Dataset Summary",
        "",
        mendeley.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## SPWLA Cross-Well Mean, SD, and Approximate 95% CI",
        "",
        ci[
            [
                "method_label",
                "coverage_mean",
                "coverage_sd",
                "p10_well_coverage_mean",
                "worst_well_coverage_mean",
                "width_mean",
                "winkler_mean",
                "winkler_sd",
                "winkler_ci95_low",
                "winkler_ci95_high",
                "rank_winkler",
            ]
        ].to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Reviewer-Requested Near-Neighbor Baselines",
        "",
        reviewer_baselines[
            ["method_label", "coverage", "p10_well_coverage", "worst_well_coverage", "width", "winkler", "rmse", "rank_winkler"]
        ].to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Depth-Gap Stratified Calibration",
        "",
        depth_gap[
            ["gap_steps", "coverage", "p10_well_coverage", "worst_well_coverage", "width", "winkler", "rmse"]
        ].to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Proper-Split Selection Bias Diagnostic",
        "",
        proper_split[
            ["method_label", "coverage", "p10_well_coverage", "worst_well_coverage", "width", "winkler", "rmse"]
        ].to_markdown(index=False, floatfmt=".4f"),
        "",
        "Interpretation: the new diagnostics are not new components of WL-PCQR. They are reviewer-risk controls for target-bound preprocessing, calibration representativeness, nearest-neighbor conformal alternatives, and selection-induced calibration bias.",
    ]
    (OUT / "REVIEWER_REQUIRED_DIAGNOSTICS.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run reviewer-requested WL-PCQR diagnostics.")
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--max-per-well", type=int, default=160)
    parser.add_argument("--local-calibration-frac", type=float, default=0.20)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 7, 123])
    parser.add_argument("--targets", nargs="+", default=SPWLA_TARGETS)
    parser.add_argument("--gap-steps", nargs="+", type=int, default=[0, 3])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    reset_spwla_features()
    bound_summary, _ = target_bound_audit()
    mendeley = mendeley_inventory()
    ci = summarize_with_ci(ROOT / "analysis" / "baseline_comparison" / "baseline_comparison_raw.csv")
    reviewer_baselines = run_reviewer_baselines(args)
    depth_gap = run_depth_gap(args)
    proper_split = run_proper_split(args)
    write_report(bound_summary, mendeley, ci, reviewer_baselines, depth_gap, proper_split)
    print((OUT / "REVIEWER_REQUIRED_DIAGNOSTICS.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import wlpcqr_core as base
import run_wlpcqr_methods as local


ROOT = Path(__file__).resolve().parents[1]


def adjacent_calibration_block(block: int, n_blocks: int) -> int:
    if block == 0:
        return 1
    if block == n_blocks - 1:
        return n_blocks - 2
    return block - 1


def fit_interval(
    train: pd.DataFrame,
    cal: pd.DataFrame,
    test: pd.DataFrame,
    target: str,
    method: str,
    alpha: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    variant, family = local.METHODS[method]
    eval_frame = pd.concat([cal, test], ignore_index=True)
    leaf_all = None
    if variant in {"pcqr", "lcqr", "clcqr"}:
        pred_all, lower_all, upper_all, leaf_all = local.fit_predict_tree_quantiles_with_leaves(
            train, eval_frame, target, family, alpha, seed
        )
    else:
        pred_all, lower_all, upper_all = local.fit_predict_variant(train, eval_frame, target, method, alpha, seed)

    n_cal = len(cal)
    y_cal = cal[target].to_numpy(dtype=float)
    pred_test = pred_all[n_cal:]

    if lower_all is None or upper_all is None:
        raise ValueError(f"Method {method} did not return quantile intervals.")

    lower_cal = lower_all[:n_cal]
    upper_cal = upper_all[:n_cal]
    lower_test = lower_all[n_cal:]
    upper_test = upper_all[n_cal:]

    if variant in {"pcqr", "cqr_signed"}:
        scores = np.maximum(lower_cal - y_cal, y_cal - upper_cal)
        qhat = base.conformal_quantile(scores, alpha)
        lower = lower_test - qhat
        upper = upper_test + qhat
        return pred_test, lower, upper, float(qhat)

    if variant in {"lcqr", "clcqr"}:
        if leaf_all is None:
            raise ValueError(f"Missing leaves for {method}.")
        leaf_cal = leaf_all[:n_cal]
        leaf_test = leaf_all[n_cal:]
        scores = np.maximum(lower_cal - y_cal, y_cal - upper_cal)
        correction = 0.0
        if variant == "clcqr":
            cal_qhats = np.empty(n_cal, dtype=float)
            for cal_pos, leaves in enumerate(leaf_cal):
                keep = np.arange(n_cal) != cal_pos
                weights = np.mean(leaf_cal[keep] == leaves[None, :], axis=1)
                cal_qhats[cal_pos] = local.weighted_quantile(scores[keep], weights, 1.0 - alpha)
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
            qhats[idx] = local.weighted_quantile(scores, weights, 1.0 - alpha)
        lower = lower_test - qhats - correction
        upper = upper_test + qhats + correction
        return pred_test, lower, upper, float(np.mean(qhats) + correction)

    raise ValueError(f"Unsupported within-well method variant for {method}: {variant}")


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
        rec["macro_well_coverage"] = float(group.groupby("well")["coverage"].mean().mean())
        rec["p10_well_coverage"] = float(np.percentile(group.groupby("well")["coverage"].mean().to_numpy(dtype=float), 10))
        rec["worst_well_coverage"] = float(group.groupby("well")["coverage"].mean().min())
        rec["macro_well_width"] = float(group.groupby("well")["width"].mean().mean())
        rec["macro_well_winkler"] = float(group.groupby("well")["winkler"].mean().mean())
        rows.append(rec)
    return pd.DataFrame(rows).sort_values(["winkler", "coverage"], ascending=[True, False])


def run(args: argparse.Namespace) -> None:
    unknown_methods = sorted(set(args.methods) - set(local.METHODS))
    if unknown_methods:
        raise ValueError(f"Unknown methods: {unknown_methods}. Supported: {sorted(local.METHODS)}")
    unknown_excluded = sorted(set(args.exclude_features) - set(base.FEATURES))
    if unknown_excluded:
        raise ValueError(f"Unknown excluded features: {unknown_excluded}. Supported: {base.FEATURES}")
    base.ACTIVE_FEATURES = [feature for feature in base.FEATURES if feature not in set(args.exclude_features)]
    data = base.load_dataset(Path(args.data), args.target, args.max_per_well, args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    preds: list[pd.DataFrame] = []
    for method in args.methods:
        for well, group in data.groupby("WELLNUM", sort=True):
            group = group.sort_values("DEPTH").reset_index(drop=True)
            blocks = base.assign_depth_blocks(len(group), args.blocks)
            for block in sorted(np.unique(blocks)):
                cal_block = adjacent_calibration_block(int(block), args.blocks)
                train = group[(blocks != block) & (blocks != cal_block)].copy().reset_index(drop=True)
                cal = group[blocks == cal_block].copy().reset_index(drop=True)
                test = group[blocks == block].copy().reset_index(drop=True)
                if len(train) < 5 or len(cal) < 2 or len(test) < 2:
                    continue
                split_seed = args.seed + int(well) * 101 + int(block)
                pred, lower, upper, qhat = fit_interval(train, cal, test, args.target, method, args.alpha, split_seed)
                lower, upper = base.clip_interval(np.minimum(lower, upper), np.maximum(lower, upper))
                y = test[args.target].to_numpy(dtype=float)
                rec: dict[str, object] = {
                    "protocol": f"Within-{args.blocks}Block-AdjacentCal",
                    "split_id": f"{well}:test{block}:cal{cal_block}",
                    "well": str(well),
                    "block": int(block),
                    "calibration_block": int(cal_block),
                    "method": method,
                    "method_label": local.METHOD_LABELS[method],
                    "target": args.target,
                    "alpha": args.alpha,
                    "target_coverage": 1.0 - args.alpha,
                    "seed": args.seed,
                    "n": int(len(test)),
                    "calibration_n": int(len(cal)),
                    "train_n": int(len(train)),
                    "qhat": float(qhat),
                }
                rec.update(base.point_metrics(y, pred))
                rec.update(base.interval_metrics(y, lower, upper, args.alpha))
                rows.append(rec)
                pred_frame = local.record_prediction_rows_local(
                    test,
                    args.target,
                    rec["protocol"],
                    rec["split_id"],
                    method,
                    args.seed,
                    args.alpha,
                    pred,
                    lower,
                    upper,
                    True,
                )
                pred_frame["method_label"] = local.METHOD_LABELS[method]
                pred_frame["calibration_block"] = int(cal_block)
                preds.append(pred_frame)
                print(
                    f"[WITHIN_ADJ] {method} target={args.target} well={well} block={block} "
                    f"cov={rec['coverage']:.4f} width={rec['width']:.4f} winkler={rec['winkler']:.4f}"
                )

    split_df = pd.DataFrame(rows)
    agg_df = aggregate_metrics(split_df)
    split_df.to_csv(out_dir / f"{args.target.lower()}_wlpcqr_within_split_metrics.csv", index=False)
    agg_df.to_csv(out_dir / f"{args.target.lower()}_wlpcqr_within_aggregate_metrics.csv", index=False)
    if preds:
        pd.concat(preds, ignore_index=True).to_csv(out_dir / f"{args.target.lower()}_wlpcqr_within_predictions.csv", index=False)
    report = [
        "# WL-PCQR Within-Well Depth-Block Results",
        "",
        f"Target: `{args.target}`",
        f"Methods: `{', '.join(args.methods)}`",
        f"Blocks: `{args.blocks}`",
        f"Excluded features: `{', '.join(args.exclude_features) if args.exclude_features else 'none'}`",
        "",
        "## Aggregate Metrics",
        "",
        agg_df.to_markdown(index=False, floatfmt=".6f"),
    ]
    (out_dir / "WLPCQR_WITHIN_DEPTH_BLOCK_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print(f"Wrote {out_dir}")
    print(agg_df.round(6).to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run WL-PCQR in within-well contiguous depth-block protocol.")
    parser.add_argument("--data", default=str(ROOT / "data" / "raw" / "train.csv"))
    parser.add_argument("--out", default=str(ROOT / "analysis" / "wlpcqr_within_depth_block_run"))
    parser.add_argument("--target", default="PHIF", choices=["PHIF", "SW", "VSH"])
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--max-per-well", type=int, default=160)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--methods", nargs="+", default=["wl_pcqr"])
    parser.add_argument("--blocks", type=int, default=5)
    parser.add_argument("--exclude-features", nargs="*", default=["DEPTH"])
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

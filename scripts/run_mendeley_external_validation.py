from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import run_wlpcqr_methods as local
import run_wlpcqr_within_depth_block as within
import wlpcqr_core as base


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "external" / "mendeley_gom_dutch_north_sea" / "raw"
PROCESSED_DIR = ROOT / "data" / "external" / "mendeley_gom_dutch_north_sea" / "processed"
OUT_DIR = ROOT / "analysis" / "external_mendeley"

TARGET = "PhiT"
FEATURES = ["DTC", "DTS", "GR", "NPHIL", "RESD_LOG10", "RHOB"]
METHODS = [
    "wl_pcqr",
    "global_rf_cqr",
    "extratrees_pcqr",
    "ngboost_cqr_signed",
    "hgb_cqr_signed",
    "lightgbm_cqr_signed",
    "catboost_cqr_signed",
]
METHOD_LABELS = {
    "wl_pcqr": "WL-PCQR",
    "global_rf_cqr": "Global RF-CQR",
    "extratrees_pcqr": "ExtraTrees-PCQR",
    "ngboost_cqr_signed": "NGBoost PI",
    "hgb_cqr_signed": "HGB-CQR",
    "lightgbm_cqr_signed": "LightGBM Quantile+CQR",
    "catboost_cqr_signed": "CatBoost Quantile+CQR",
}
SEEDS = [42, 7, 123]


def read_mendeley_txt(path: Path, wellnum: int, well_name: str, domain: str) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t", skiprows=[0, 2], engine="python")
    frame = frame.loc[:, ~frame.columns.astype(str).str.startswith("Unnamed")]
    if "DEPTH.1" in frame.columns:
        frame = frame.drop(columns=["DEPTH.1"])
    frame.columns = [str(col).strip() for col in frame.columns]
    frame = frame.replace([-9999, -9999.0, "", " "], np.nan)
    for col in frame.columns:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame["WELLNUM"] = int(wellnum)
    frame["WELL_NAME"] = well_name
    frame["DOMAIN"] = domain
    frame["RESD_LOG10"] = np.where(frame["RESD"] > 0, np.log10(frame["RESD"]), np.nan)
    keep = ["WELLNUM", "WELL_NAME", "DOMAIN", "DEPTH", *FEATURES, TARGET]
    frame = frame[[col for col in keep if col in frame.columns]].copy()
    frame = frame.dropna(subset=["DEPTH", TARGET]).sort_values("DEPTH").reset_index(drop=True)
    frame = frame[(frame[TARGET] >= 0.0) & (frame[TARGET] <= 1.0)].copy()
    return frame


def prepare_dataset(max_per_well: int, seed: int) -> pd.DataFrame:
    gulf = read_mendeley_txt(RAW_DIR / "Gulf of Mexico sand_dataset.txt", 0, "Gulf of Mexico lower Purple sand", "GOM")
    dutch = read_mendeley_txt(RAW_DIR / "Offshore Dutch North sea dataset.txt", 1, "Dutch North Sea Westphalian C", "DNS")
    data = pd.concat([gulf, dutch], ignore_index=True)
    if max_per_well and max_per_well > 0:
        rng = np.random.default_rng(seed)
        parts: list[pd.DataFrame] = []
        for _, group in data.groupby("WELLNUM", sort=True):
            if len(group) <= max_per_well:
                parts.append(group)
                continue
            idx = np.linspace(0, len(group) - 1, max_per_well).round().astype(int)
            idx = np.unique(idx)
            if len(idx) < max_per_well:
                pool = np.setdiff1d(np.arange(len(group)), idx)
                extra = rng.choice(pool, size=max_per_well - len(idx), replace=False)
                idx = np.sort(np.concatenate([idx, extra]))
            parts.append(group.iloc[idx])
        data = pd.concat(parts, ignore_index=True).sort_values(["WELLNUM", "DEPTH"]).reset_index(drop=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    data.to_csv(PROCESSED_DIR / "mendeley_phi_external.csv", index=False)
    return data


def configure_external_features() -> None:
    base.FEATURES = list(FEATURES)
    base.ACTIVE_FEATURES = list(FEATURES)


def summarize_mean_std(raw: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
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
    available = [col for col in metrics if col in raw.columns]
    grouped = raw.groupby(group_cols, dropna=False)[available]
    mean = grouped.mean().add_suffix("_mean")
    sd = grouped.std(ddof=1).fillna(0.0).add_suffix("_sd")
    out = pd.concat([mean, sd], axis=1).reset_index()
    sort_col = "winkler_mean" if "winkler_mean" in out.columns else None
    if sort_col:
        out = out.sort_values(sort_col).reset_index(drop=True)
        out["rank_winkler"] = out[sort_col].rank(method="min", ascending=True).astype(int)
    return out


def run_lowo(data: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    split_rows: list[pd.DataFrame] = []
    for seed in SEEDS:
        for method in METHODS:
            metric_rows, _ = local.evaluate_local(
                data=data,
                target=TARGET,
                method=method,
                alpha=args.alpha,
                local_calibration_frac=args.local_calibration_frac,
                strategy=args.local_calibration_strategy,
                seed=seed,
                stress_curves=[],
                clip_intervals=True,
            )
            split = pd.DataFrame(metric_rows)
            split["external_dataset"] = "Mendeley_GOM_DNS"
            split_rows.append(split)
            agg = local.aggregate_metrics(split)
            agg["seed"] = seed
            rows.append(agg)
    raw = pd.concat(rows, ignore_index=True)
    split_all = pd.concat(split_rows, ignore_index=True)
    raw.to_csv(OUT_DIR / "mendeley_lowo_raw.csv", index=False)
    split_all.to_csv(OUT_DIR / "mendeley_lowo_split_metrics.csv", index=False)
    summarize_mean_std(raw, ["method", "method_label", "target"]).to_csv(
        OUT_DIR / "mendeley_lowo_mean_std.csv", index=False
    )
    return raw


def run_within_depth_blocks(data: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for seed in SEEDS:
        for method in METHODS:
            for well, group in data.groupby("WELLNUM", sort=True):
                group = group.sort_values("DEPTH").reset_index(drop=True)
                blocks = base.assign_depth_blocks(len(group), args.blocks)
                for block in sorted(np.unique(blocks)):
                    cal_block = within.adjacent_calibration_block(int(block), args.blocks)
                    train = group[(blocks != block) & (blocks != cal_block)].copy().reset_index(drop=True)
                    cal = group[blocks == cal_block].copy().reset_index(drop=True)
                    test = group[blocks == block].copy().reset_index(drop=True)
                    if len(train) < 5 or len(cal) < 2 or len(test) < 2:
                        continue
                    split_seed = seed + int(well) * 101 + int(block)
                    pred, lower, upper, qhat = within.fit_interval(
                        train=train,
                        cal=cal,
                        test=test,
                        target=TARGET,
                        method=method,
                        alpha=args.alpha,
                        seed=split_seed,
                    )
                    lower, upper = base.clip_interval(np.minimum(lower, upper), np.maximum(lower, upper))
                    y = test[TARGET].to_numpy(dtype=float)
                    rec: dict[str, object] = {
                        "protocol": f"ExternalWithin-{args.blocks}Block-AdjacentCal",
                        "external_dataset": "Mendeley_GOM_DNS",
                        "split_id": f"{well}:test{block}:cal{cal_block}",
                        "well": str(well),
                        "well_name": str(group["WELL_NAME"].iloc[0]),
                        "domain": str(group["DOMAIN"].iloc[0]),
                        "block": int(block),
                        "calibration_block": int(cal_block),
                        "method": method,
                        "method_label": METHOD_LABELS[method],
                        "target": TARGET,
                        "alpha": args.alpha,
                        "target_coverage": 1.0 - args.alpha,
                        "seed": seed,
                        "n": int(len(test)),
                        "calibration_n": int(len(cal)),
                        "train_n": int(len(train)),
                        "qhat": float(qhat),
                    }
                    rec.update(base.point_metrics(y, pred))
                    rec.update(base.interval_metrics(y, lower, upper, args.alpha))
                    rows.append(rec)
    split = pd.DataFrame(rows)
    split.to_csv(OUT_DIR / "mendeley_within_split_metrics.csv", index=False)
    agg_parts: list[pd.DataFrame] = []
    for seed, seed_group in split.groupby("seed", sort=True):
        agg = within.aggregate_metrics(seed_group)
        agg["seed"] = seed
        agg_parts.append(agg)
    raw = pd.concat(agg_parts, ignore_index=True)
    raw.to_csv(OUT_DIR / "mendeley_within_raw.csv", index=False)
    summarize_mean_std(raw, ["method", "method_label", "target"]).to_csv(
        OUT_DIR / "mendeley_within_mean_std.csv", index=False
    )
    return raw


def write_report(data: pd.DataFrame, lowo_raw: pd.DataFrame, within_raw: pd.DataFrame) -> None:
    lowo_summary = pd.read_csv(OUT_DIR / "mendeley_lowo_mean_std.csv")
    within_summary = pd.read_csv(OUT_DIR / "mendeley_within_mean_std.csv")
    inventory = (
        data.groupby(["WELLNUM", "WELL_NAME", "DOMAIN"])
        .agg(samples=("PhiT", "size"), depth_min=("DEPTH", "min"), depth_max=("DEPTH", "max"), phit_mean=("PhiT", "mean"))
        .reset_index()
    )
    inventory.to_csv(OUT_DIR / "mendeley_dataset_inventory.csv", index=False)
    lines = [
        "# Mendeley Gulf of Mexico + Dutch North Sea External Validation",
        "",
        "Dataset DOI: `10.17632/sdv629nbjr.1`",
        "",
        "This is a porosity-only external validation because the available target is total porosity `PhiT`.",
        "The dataset contains two domains/wells, so the evidence should be framed as an external sanity validation rather than a large multi-well benchmark.",
        "",
        "## Dataset Inventory",
        "",
        inventory.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Two-Domain Leave-One-Well-Out With Target Calibration",
        "",
        lowo_summary.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Within-Domain Depth-Block Diagnostics",
        "",
        within_summary.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Interpretation",
        "",
        "- Use these results as external porosity validation, not as full external replication of SPWLA `PHIF/SW/VSH`.",
        "- Because one domain has only about 59 valid depth samples, lower-tail metrics are more informative than pooled means.",
        "- If WL-PCQR maintains stronger lower-tail coverage with competitive Winkler, it supports the paper's reliability-efficiency story on an independent porosity dataset.",
    ]
    (OUT_DIR / "MENDELEY_EXTERNAL_VALIDATION_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run WL-PCQR external validation on the Mendeley GOM/Dutch North Sea dataset.")
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--max-per-well", type=int, default=160)
    parser.add_argument("--local-calibration-frac", type=float, default=0.20)
    parser.add_argument("--local-calibration-strategy", choices=["stratified", "random", "contiguous"], default="stratified")
    parser.add_argument("--blocks", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    configure_external_features()
    data = prepare_dataset(max_per_well=args.max_per_well, seed=42)
    lowo_raw = run_lowo(data, args)
    within_raw = run_within_depth_blocks(data, args)
    write_report(data, lowo_raw, within_raw)
    print(f"Wrote {OUT_DIR}")
    print((OUT_DIR / "MENDELEY_EXTERNAL_VALIDATION_REPORT.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()

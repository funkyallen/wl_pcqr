from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import run_wlpcqr_methods as local
import wlpcqr_core as base
from paper_plot_style import NATURE_DOUBLE_WIDTH, apply_publication_style, mm_to_in


ROOT = Path(__file__).resolve().parents[1]


def interval_metrics(df: pd.DataFrame, target: str) -> dict[str, float]:
    y = df[target].to_numpy(dtype=float)
    lower = df["lower"].to_numpy(dtype=float)
    upper = df["upper"].to_numpy(dtype=float)
    return {
        "coverage": float(np.mean((y >= lower) & (y <= upper))),
        "width": float(np.mean(upper - lower)),
        "rmse": float(np.sqrt(np.mean((y - df["pred"].to_numpy(dtype=float)) ** 2))),
        "winkler": float(np.mean((upper - lower) + 20.0 * np.maximum(lower - y, 0.0) + 20.0 * np.maximum(y - upper, 0.0))),
    }


def load_case(path: Path, target: str, well: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[(df["method"] == "wl_pcqr") & (df["WELLNUM"].astype(int) == int(well))].copy()
    if df.empty:
        raise ValueError(f"No rows for method wl_pcqr well {well} in {path}")
    return df.sort_values("DEPTH").reset_index(drop=True)


def calibration_depths(target: str, well: int, seed: int, frac: float = 0.20) -> np.ndarray:
    data = base.load_dataset(ROOT / "data" / "raw" / "train.csv", target, max_per_well=160, seed=seed)
    target_well = data[data["WELLNUM"].astype(int) == int(well)].copy().sort_values("DEPTH").reset_index(drop=True)
    cal_idx, _ = local.local_calibration_indices(len(target_well), frac, "stratified", seed + 7919 * int(well))
    return target_well.iloc[cal_idx]["DEPTH"].to_numpy(dtype=float)


def plot_track(ax, df: pd.DataFrame, target: str, panel_label: str, color: str, cal_depths: np.ndarray) -> None:
    depth = df["DEPTH"].to_numpy(dtype=float)
    y = df[target].to_numpy(dtype=float)
    pred = df["pred"].to_numpy(dtype=float)
    lower = df["lower"].to_numpy(dtype=float)
    upper = df["upper"].to_numpy(dtype=float)
    covered = df["covered"].astype(bool).to_numpy()
    ax.fill_betweenx(depth, lower, upper, color=color, alpha=0.22, linewidth=0)
    ax.plot(pred, depth, color=color, linewidth=1.25, label="Prediction")
    ax.plot(y, depth, color="black", linewidth=1.0, label="Observed")
    if cal_depths.size:
        ax.scatter(
            np.full_like(cal_depths, 1.0, dtype=float),
            cal_depths,
            s=28,
            color="#4d4d4d",
            marker="|",
            linewidth=0.9,
            label="Calibration depth",
        )
    if (~covered).any():
        ax.scatter(y[~covered], depth[~covered], s=10, color="#b2182b", marker="x", linewidth=0.7, label="Missed")
    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel(target)
    ax.text(0.02, 0.98, panel_label, transform=ax.transAxes, ha="left", va="top", fontsize=7.0, weight="bold")
    ax.grid(axis="x", alpha=0.18)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot WL-PCQR hard-well depth track.")
    parser.add_argument("--target", default="SW", choices=["PHIF", "SW", "VSH"])
    parser.add_argument("--well", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--observed-dir", default=str(ROOT / "analysis" / "wlpcqr_stress_SW_seed42_observed_20260513"))
    parser.add_argument("--stress-dir", default=str(ROOT / "analysis" / "wlpcqr_stress_SW_seed42_rdep_rmed_hidden_20260513"))
    parser.add_argument("--stress-label", default="RDEP_LOG10 + RMED_LOG10 hidden")
    args = parser.parse_args()

    observed_path = Path(args.observed_dir) / f"{args.target.lower()}_wlpcqr_target_well_predictions.csv"
    stress_path = Path(args.stress_dir) / f"{args.target.lower()}_wlpcqr_target_well_predictions.csv"
    cached_dir = ROOT / "analysis" / "hard_well_depth_track"
    if not observed_path.exists():
        observed_path = cached_dir / "sw_well6_seed42_wlpcqr_observed_track.csv"
    if not stress_path.exists():
        stress_path = cached_dir / "sw_well6_seed42_wlpcqr_rdep_rmed_hidden_track.csv"
    observed = load_case(observed_path, args.target, args.well)
    stress = load_case(stress_path, args.target, args.well)
    cal_depths = calibration_depths(args.target, args.well, args.seed)
    observed_metrics = interval_metrics(observed, args.target)
    stress_metrics = interval_metrics(stress, args.target)

    out_data = ROOT / "analysis" / "hard_well_depth_track"
    out_data.mkdir(parents=True, exist_ok=True)
    observed.assign(setting="Observed logs").to_csv(out_data / "sw_well6_seed42_wlpcqr_observed_track.csv", index=False)
    stress.assign(setting=args.stress_label).to_csv(out_data / "sw_well6_seed42_wlpcqr_rdep_rmed_hidden_track.csv", index=False)

    apply_publication_style("nature")
    fig, axes = plt.subplots(1, 3, figsize=(NATURE_DOUBLE_WIDTH, mm_to_in(92)), sharey=True, gridspec_kw={"width_ratios": [1.0, 1.0, 0.72]})
    plot_track(
        axes[0],
        observed,
        args.target,
        f"a  Observed logs\nPICP={observed_metrics['coverage']:.3f}, width={observed_metrics['width']:.3f}",
        color=plt.cm.tab10.colors[0],
        cal_depths=cal_depths,
    )
    plot_track(
        axes[1],
        stress,
        args.target,
        f"b  {args.stress_label}\nPICP={stress_metrics['coverage']:.3f}, width={stress_metrics['width']:.3f}",
        color=plt.cm.tab10.colors[3],
        cal_depths=cal_depths,
    )
    depth = observed["DEPTH"].to_numpy(dtype=float)
    axes[2].plot(observed["width"].to_numpy(dtype=float), depth, color=plt.cm.tab10.colors[0], label="Observed logs")
    axes[2].plot(stress["width"].to_numpy(dtype=float), depth, color=plt.cm.tab10.colors[3], label=args.stress_label)
    axes[2].set_xlabel("Interval width")
    axes[2].text(0.02, 0.98, "c  Width by depth", transform=axes[2].transAxes, ha="left", va="top", fontsize=7.0, weight="bold")
    axes[2].set_xlim(0.0, 1.02)
    axes[2].grid(axis="x", alpha=0.18)

    axes[0].invert_yaxis()
    axes[0].set_ylabel("Depth")
    handles, labels = axes[0].get_legend_handles_labels()
    width_handles, width_labels = axes[2].get_legend_handles_labels()
    fig.legend(handles + width_handles, labels + width_labels, frameon=False, ncol=3, loc="lower center")
    fig.subplots_adjust(top=0.86, bottom=0.20, wspace=0.20)

    out_fig = ROOT / "outputs" / "figures" / "hard_well_depth_track" / "sw_well6_wlpcqr_rdep_rmed_hidden_track.svg"
    out_pdf = out_fig.with_suffix(".pdf")
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig)
    fig.savefig(out_pdf)
    fig.savefig(out_fig.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_fig}")
    print(f"Wrote {out_pdf}")
    print(f"Wrote {out_fig.with_suffix('.tiff')}")
    print(f"Wrote {out_data}")


if __name__ == "__main__":
    main()

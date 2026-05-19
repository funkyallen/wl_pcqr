from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from paper_plot_style import NATURE_COLORS, NATURE_DOUBLE_WIDTH, NATURE_SINGLE_WIDTH, apply_publication_style, mm_to_in


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "figures" / "paper_summary"

CI = ROOT / "analysis" / "reviewer_required_diagnostics" / "spwla_cross_well_mean_sd_ci.csv"
ABLATION = ROOT / "analysis" / "ablation_sensitivity" / "mechanism_ablation_summary.csv"
PATH_COUNTS = ROOT / "analysis" / "ablation_sensitivity" / "wlpcqr_selected_level_counts.csv"
BASELINE = ROOT / "analysis" / "baseline_comparison" / "baseline_comparison_summary.csv"
FULL_DEPTH = ROOT / "analysis" / "all_valid_weighted_diagnostic" / "all_valid_weighted_summary.csv"


COLORS = {
    "WL-PCQR": NATURE_COLORS["orange"],
    "Global RF-CQR": NATURE_COLORS["green"],
    "Source-only RF-CQR": NATURE_COLORS["gray"],
    "NGBoost PI": NATURE_COLORS["yellow"],
    "ExtraTrees-PCQR": NATURE_COLORS["pink"],
    "Uniform target-well CQR": NATURE_COLORS["blue"],
    "Raw proximity score": NATURE_COLORS["gray"],
    "Proximity scale recalibration": NATURE_COLORS["green"],
    "Proximity location-scale recalibration": NATURE_COLORS["pink"],
    "WL-PCQR selected path": NATURE_COLORS["orange"],
}


def save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.tiff", dpi=600, bbox_inches="tight")
    plt.close(fig)


def style_axis(ax: plt.Axes, grid_axis: str = "y") -> None:
    ax.grid(True, axis=grid_axis, color=NATURE_COLORS["light_gray"], linewidth=0.45)
    ax.set_axisbelow(True)
    ax.tick_params(length=2.5, width=0.6, color=NATURE_COLORS["black"])
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_linewidth(0.6)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.12, 1.05, label, transform=ax.transAxes, ha="left", va="top", fontsize=8.2, weight="bold")


def plot_winkler_ci() -> None:
    df = pd.read_csv(CI).sort_values("rank_winkler").head(5).copy()
    df = df.sort_values("winkler_mean", ascending=True)
    labels = df["method_label"].tolist()
    y = np.arange(len(df))
    xerr = np.vstack(
        [
            df["winkler_mean"] - df["winkler_ci95_low"],
            df["winkler_ci95_high"] - df["winkler_mean"],
        ]
    )
    fig, ax = plt.subplots(figsize=(NATURE_DOUBLE_WIDTH, mm_to_in(75)))
    colors = [COLORS.get(label, "#999999") for label in labels]
    ax.barh(y, df["winkler_mean"], color=colors, alpha=0.90, edgecolor="black", linewidth=0.45)
    ax.errorbar(df["winkler_mean"], y, xerr=xerr, fmt="none", ecolor=NATURE_COLORS["black"], elinewidth=0.8, capsize=2.2)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Mean Winkler score with descriptive 95% CI")
    style_axis(ax, "x")
    save(fig, "spwla_winkler_ci")


def plot_path_ablation() -> None:
    df = pd.read_csv(ABLATION).copy()
    order = ["uniform_cqr", "raw_proximity", "proximity_scale", "location_scale", "wl_pcqr"]
    df["order"] = df["method"].map({name: idx for idx, name in enumerate(order)})
    df = df.sort_values("order")
    labels = [
        "Uniform\nCQR",
        "Raw\nproximity",
        "Scale\nlocal",
        "Location-\nscale",
        "Selected\nWL-PCQR",
    ]
    x = np.arange(len(df))
    fig, axes = plt.subplots(1, 2, figsize=(NATURE_DOUBLE_WIDTH, mm_to_in(72)), sharex=True)
    colors = [COLORS.get(label, "#999999") for label in df["method_label"]]
    axes[0].bar(x, df["winkler"], color=colors, edgecolor="black", linewidth=0.45)
    axes[0].set_ylabel("Winkler score")
    axes[1].bar(x, df["coverage"], color=colors, edgecolor="black", linewidth=0.45)
    axes[1].axhline(0.90, color=NATURE_COLORS["gray"], linestyle="--", linewidth=0.75)
    axes[1].set_ylabel("PICP")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=0)
        style_axis(ax)
    panel_label(axes[0], "a")
    panel_label(axes[1], "b")
    save(fig, "wlpcqr_path_ablation")


def plot_selected_level_counts() -> None:
    df = pd.read_csv(PATH_COUNTS)
    targets = ["PHIF", "SW", "VSH"]
    levels = ["uniform", "scale", "location-scale"]
    colors = {"uniform": NATURE_COLORS["blue"], "scale": NATURE_COLORS["green"], "location-scale": NATURE_COLORS["orange"]}
    pivot = df.pivot(index="target", columns="selected_level", values="count").reindex(targets).fillna(0)
    fig, ax = plt.subplots(figsize=(NATURE_SINGLE_WIDTH, mm_to_in(48)))
    left = np.zeros(len(targets))
    y = np.arange(len(targets))
    for level in levels:
        values = pivot[level].to_numpy(dtype=float) if level in pivot.columns else np.zeros(len(targets))
        ax.barh(y, values, left=left, label=level, color=colors[level], edgecolor="black", linewidth=0.45)
        left += values
    ax.set_yticks(y)
    ax.set_yticklabels(targets)
    ax.set_xlabel("Selected splits")
    ax.set_xlim(0, max(left) * 1.04)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.20), columnspacing=1.0, handlelength=1.2)
    style_axis(ax, "x")
    save(fig, "wlpcqr_selected_level_counts")


def plot_protocol_winkler_overview() -> None:
    df = pd.read_csv(BASELINE)
    keep_methods = ["WL-PCQR", "Global RF-CQR", "NGBoost PI", "Source-only RF-CQR"]
    df = df[df["method_label"].isin(keep_methods)].copy()
    protocol_order = [
        ("SPWLA_PDDA_2021", "cross_well", "SPWLA\ncross-well"),
        ("SPWLA_PDDA_2021", "within_well", "SPWLA\nwithin-well"),
        ("Mendeley_GOM_DNS", "cross_well", "Mendeley\ncross-well"),
        ("Mendeley_GOM_DNS", "within_well", "Mendeley\nwithin-well"),
    ]
    method_order = ["WL-PCQR", "Global RF-CQR", "NGBoost PI", "Source-only RF-CQR"]
    x = np.arange(len(protocol_order))
    width = 0.18
    fig, ax = plt.subplots(figsize=(NATURE_DOUBLE_WIDTH, mm_to_in(80)))
    for idx, method in enumerate(method_order):
        values = []
        for dataset, protocol, _ in protocol_order:
            row = df[(df["dataset"] == dataset) & (df["protocol"] == protocol) & (df["method_label"] == method)]
            values.append(float(row["winkler_mean"].iloc[0]) if len(row) else np.nan)
        ax.bar(
            x + (idx - 1.5) * width,
            values,
            width,
            label=method,
            color=COLORS.get(method, "#999999"),
            edgecolor="black",
            linewidth=0.35,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, _, label in protocol_order])
    ax.set_ylabel("Winkler score")
    ax.legend(frameon=False, ncol=2)
    style_axis(ax)
    save(fig, "protocol_winkler_overview")


def plot_full_depth_weighted() -> None:
    df = pd.read_csv(FULL_DEPTH).copy()
    df["label"] = df["method_label"].replace({"WL-PCQR": "WL-PCQR", "Global RF-CQR": "Global\nRF-CQR"})
    fig, axes = plt.subplots(1, 2, figsize=(NATURE_DOUBLE_WIDTH, mm_to_in(68)))
    colors = [COLORS.get(label.replace("\n", " "), "#999999") for label in df["method_label"]]
    x = np.arange(len(df))
    axes[0].bar(x, df["macro_well_winkler"], color=colors, edgecolor="black", linewidth=0.45)
    axes[0].set_ylabel("Score")
    axes[1].bar(x, df["macro_well_width"], color=colors, edgecolor="black", linewidth=0.45)
    axes[1].set_ylabel("Width")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(df["label"])
        style_axis(ax)
    panel_label(axes[0], "a")
    panel_label(axes[1], "b")
    save(fig, "all_valid_weighted_diagnostic")


def main() -> None:
    apply_publication_style("nature")
    plot_winkler_ci()
    plot_path_ablation()
    plot_selected_level_counts()
    plot_protocol_winkler_overview()
    plot_full_depth_weighted()
    print(f"Wrote additional figures to {OUT}")


if __name__ == "__main__":
    main()

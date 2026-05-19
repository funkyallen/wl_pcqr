from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from paper_plot_style import NATURE_COLORS, NATURE_DOUBLE_WIDTH, NATURE_SINGLE_WIDTH, apply_publication_style, mm_to_in


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "figures" / "paper_summary"
FINAL = ROOT / "analysis" / "baseline_comparison" / "baseline_comparison_summary.csv"
MISSING = ROOT / "analysis" / "final_results" / "missing_stress_delta.csv"
CAL = ROOT / "analysis" / "final_results" / "calibration_sensitivity_mean.csv"
CAL5 = ROOT / "analysis" / "submission_revision_diagnostics" / "calibration_5pct_summary.csv"

PALETTE = {
    "WL-PCQR": NATURE_COLORS["orange"],
    "Global RF-CQR": NATURE_COLORS["green"],
    "ExtraTrees-PCQR": NATURE_COLORS["pink"],
    "Source-only RF-CQR": NATURE_COLORS["gray"],
    "NGBoost PI": NATURE_COLORS["yellow"],
    "HGB-CQR": NATURE_COLORS["sky"],
    "LightGBM Quantile+CQR": "#6A3D9A",
    "CatBoost Quantile+CQR": "#B15928",
    "contiguous": NATURE_COLORS["pink"],
    "random": NATURE_COLORS["blue"],
    "stratified": NATURE_COLORS["orange"],
}


def style_axes(ax: plt.Axes) -> None:
    ax.grid(True, axis="y", color=NATURE_COLORS["light_gray"], linewidth=0.45)
    ax.set_axisbelow(True)
    ax.tick_params(length=2.5, width=0.6, color=NATURE_COLORS["black"])
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_linewidth(0.6)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.12, 1.05, label, transform=ax.transAxes, ha="left", va="top", fontsize=8.2, weight="bold")


def save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.tiff", dpi=600, bbox_inches="tight")
    plt.close(fig)


def method_flow() -> None:
    fig, ax = plt.subplots(figsize=(NATURE_DOUBLE_WIDTH, mm_to_in(54)))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    steps = [
        ("Source wells", "Fit RF tree ensemble"),
        ("Base interval", "Tree-prediction quantiles"),
        ("Target labels", "Representative calibration depths"),
        ("Calibration path", "Uniform to proximity levels"),
        ("Score selection", "Coverage-constrained Winkler"),
        ("WL-PCQR interval", "Bounded final PI"),
    ]
    xs = np.linspace(0.08, 0.92, len(steps))
    y = 0.55
    w, h = 0.128, 0.30
    for idx, (title, body) in enumerate(steps):
        color = "#D55E00" if idx == len(steps) - 1 else "#f5f5f5"
        edge = "#D55E00" if idx == len(steps) - 1 else "#4d4d4d"
        text_color = "white" if idx == len(steps) - 1 else "#222222"
        box = FancyBboxPatch(
            (xs[idx] - w / 2, y - h / 2),
            w,
            h,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            linewidth=0.85,
            edgecolor=edge,
            facecolor=color,
        )
        ax.add_patch(box)
        ax.text(xs[idx], y + 0.04, title, ha="center", va="center", fontsize=7.2, weight="bold", color=text_color)
        ax.text(xs[idx], y - 0.06, body, ha="center", va="center", fontsize=6.25, color=text_color, wrap=True)
        if idx < len(steps) - 1:
            ax.add_patch(
                FancyArrowPatch(
                    (xs[idx] + w / 2 + 0.01, y),
                    (xs[idx + 1] - w / 2 - 0.01, y),
                    arrowstyle="-|>",
                    mutation_scale=10,
                    linewidth=0.75,
                    color=NATURE_COLORS["gray"],
                )
            )
    ax.text(
        0.5,
        0.14,
        "Design principle: use calibration geometry to move along a fixed CQR path, from uniform correction to RF-proximity location-scale correction.",
        ha="center",
        va="center",
        fontsize=6.8,
        color="#333333",
    )
    save(fig, "method_flow")


def main_tradeoff() -> None:
    df = pd.read_csv(FINAL)
    df = df[(df["dataset"] == "SPWLA_PDDA_2021") & (df["protocol"] == "cross_well")].copy()
    df = df.sort_values("rank_winkler")
    fig, ax = plt.subplots(figsize=(NATURE_DOUBLE_WIDTH, mm_to_in(92)))
    label_offsets = {
        "WL-PCQR": (4, 2),
        "CatBoost Quantile+CQR": (5, 5),
        "HGB-CQR": (5, -11),
        "LightGBM Quantile+CQR": (5, 5),
    }
    annotation_labels = {
        "LightGBM Quantile+CQR": "LightGBM Q+CQR",
        "CatBoost Quantile+CQR": "CatBoost Q+CQR",
        "Source-only RF-CQR": "Source-only RF",
        "ExtraTrees-PCQR": "ExtraTrees",
    }
    for _, row in df.iterrows():
        method = row["method_label"]
        size = 460 * float(row["worst_well_coverage_mean"]) ** 2
        ax.scatter(
            row["width_mean"],
            row["winkler_mean"],
            s=size,
            color=PALETTE.get(method, "#999999"),
            edgecolor="black",
            linewidth=0.45,
            alpha=0.9,
            label=method,
        )
        ax.annotate(
            annotation_labels.get(method, method),
            (row["width_mean"], row["winkler_mean"]),
            xytext=label_offsets.get(method, (4, 3)),
            textcoords="offset points",
            fontsize=6.2,
        )
    ax.set_xlabel("Mean interval width")
    ax.set_ylabel("Winkler score")
    style_axes(ax)
    ax.text(
        0.98,
        0.04,
        "Marker area scales with worst-well PICP",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.4,
        color="#555555",
    )
    save(fig, "main_tradeoff")


def missing_stress() -> None:
    df = pd.read_csv(MISSING)
    label_map = {
        "den_neu_hidden": "DEN + NEU",
        "rdep_rmed_hidden": "RDEP_LOG10\n+ RMED_LOG10",
        "gr_hidden": "GR",
        "DEN_NEU": "DEN + NEU",
        "RDEP_LOG10_RMED_LOG10": "RDEP_LOG10\n+ RMED_LOG10",
        "GR": "GR",
    }
    labels = [f"{r.target}\n{label_map.get(r.stress_setting, str(r.stress_setting))}" for r in df.itertuples()]
    width_col = "width_delta" if "width_delta" in df.columns else "delta_width"
    winkler_col = "winkler_delta" if "winkler_delta" in df.columns else "delta_winkler"
    x = np.arange(len(df))
    width = 0.36
    fig, ax = plt.subplots(figsize=(NATURE_SINGLE_WIDTH, mm_to_in(67)))
    ax.bar(x - width / 2, df[width_col], width, label="Width increase", color=NATURE_COLORS["blue"])
    ax.bar(x + width / 2, df[winkler_col], width, label="Winkler increase", color=NATURE_COLORS["orange"])
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Delta relative to observed logs")
    ax.legend(frameon=False, loc="upper left")
    style_axes(ax)
    save(fig, "missing_stress_cost")


def calibration_sensitivity() -> None:
    df = pd.read_csv(CAL)
    if CAL5.exists():
        cal5 = pd.read_csv(CAL5).rename(
            columns={
                "coverage": "coverage",
                "p10_well_coverage": "p10_well_coverage",
                "worst_well_coverage": "worst_well_coverage",
                "width": "width",
                "winkler": "winkler",
            }
        )
        df = pd.concat([df, cal5[df.columns]], ignore_index=True)
        df = df.drop_duplicates(subset=["frac", "strategy"], keep="last")
    fig, axes = plt.subplots(1, 2, figsize=(NATURE_DOUBLE_WIDTH, mm_to_in(72)), sharex=True)
    for strategy, group in df.groupby("strategy", sort=False):
        group = group.sort_values("frac")
        color = PALETTE.get(strategy, "#999999")
        axes[0].plot(group["frac"] * 100, group["worst_well_coverage"], marker="o", label=strategy, color=color, linewidth=1.05)
        axes[1].plot(group["frac"] * 100, group["winkler"], marker="o", label=strategy, color=color, linewidth=1.05)
    axes[0].axhline(0.90, color=NATURE_COLORS["gray"], linestyle="--", linewidth=0.75)
    axes[0].set_ylabel("Worst-well PICP")
    axes[1].set_ylabel("Winkler score")
    for ax in axes:
        ax.set_xlabel("Target-well calibration budget (%)")
        style_axes(ax)
    panel_label(axes[0], "a")
    panel_label(axes[1], "b")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper center", ncol=3, bbox_to_anchor=(0.52, 1.02))
    fig.subplots_adjust(top=0.80, wspace=0.30)
    save(fig, "calibration_sensitivity")


def main() -> None:
    apply_publication_style("nature")
    method_flow()
    main_tradeoff()
    missing_stress()
    calibration_sensitivity()
    print(f"Wrote figures to {OUT}")


if __name__ == "__main__":
    main()

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from paper_plot_style import NATURE_COLORS, NATURE_DOUBLE_WIDTH, apply_publication_style, mm_to_in


ROOT = Path(__file__).resolve().parents[1]
IN_PATH = ROOT / "analysis" / "reviewer_required_diagnostics" / "reviewer_nearest_baseline_split_metrics.csv"
OUT_DIR = ROOT / "outputs" / "figures" / "paper_summary"


def plot_per_well_coverage_heatmap() -> None:
    data = pd.read_csv(IN_PATH)
    data = data[data["method"] == "wl_pcqr"].copy()
    summary = data.groupby(["target", "well"], sort=True)["coverage"].mean().reset_index()
    pivot = summary.pivot(index="target", columns="well", values="coverage").loc[["PHIF", "SW", "VSH"]]

    apply_publication_style("nature")
    fig, ax = plt.subplots(figsize=(NATURE_DOUBLE_WIDTH, mm_to_in(56)))
    values = pivot.to_numpy(dtype=float)
    im = ax.imshow(values, cmap="YlGnBu", vmin=0.65, vmax=1.0, aspect="auto")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([str(col) for col in pivot.columns])
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("Target well")
    ax.set_ylabel("Target")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(j, i, f"{values[i, j]:.2f}", ha="center", va="center", color="white" if values[i, j] < 0.76 else NATURE_COLORS["black"], fontsize=6.4)
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("PICP")
    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_svg = OUT_DIR / "wlpcqr_per_well_coverage_heatmap.svg"
    out_pdf = out_svg.with_suffix(".pdf")
    out_tiff = out_svg.with_suffix(".tiff")
    fig.savefig(out_svg)
    fig.savefig(out_pdf)
    fig.savefig(out_tiff, dpi=600, bbox_inches="tight")
    plt.close(fig)
    summary.to_csv(ROOT / "analysis" / "reviewer_required_diagnostics" / "wlpcqr_per_well_coverage_heatmap_values.csv", index=False)
    print(f"Wrote {out_svg}")
    print(f"Wrote {out_pdf}")
    print(f"Wrote {out_tiff}")


def main() -> None:
    plot_per_well_coverage_heatmap()


if __name__ == "__main__":
    main()

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from paper_plot_style import NATURE_COLORS, NATURE_DOUBLE_WIDTH, NATURE_SINGLE_WIDTH, apply_publication_style, mm_to_in


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "train.csv"
FIG_DIR = ROOT / "outputs" / "figures" / "dataset_audit"
TABLE_DIR = ROOT / "outputs" / "tables" / "dataset_audit"

TARGET_COLS = ["PHIF", "SW", "VSH"]
ID_COLS = ["WELLNUM", "DEPTH"]


def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    if path.suffix.lower() != ".pdf":
        plt.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.savefig(path.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close()


def load_train() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    return df.replace(-9999.0, np.nan)


def write_summary_tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    feature_cols = [c for c in df.columns if c not in ID_COLS + TARGET_COLS]
    summary_rows = []
    for well, group in df.groupby("WELLNUM", sort=True):
        row: dict[str, float | int] = {
            "WELLNUM": int(well),
            "samples": int(len(group)),
            "depth_min": float(group["DEPTH"].min()),
            "depth_max": float(group["DEPTH"].max()),
            "feature_missing_rate": float(group[feature_cols].isna().mean().mean()),
        }
        for target in TARGET_COLS:
            row[f"{target}_valid"] = int(group[target].notna().sum())
            row[f"{target}_missing_rate"] = float(group[target].isna().mean())
            row[f"{target}_mean"] = float(group[target].mean())
            row[f"{target}_std"] = float(group[target].std())
            row[f"{target}_p05"] = float(group[target].quantile(0.05))
            row[f"{target}_p50"] = float(group[target].quantile(0.50))
            row[f"{target}_p95"] = float(group[target].quantile(0.95))
        summary_rows.append(row)

    well_summary = pd.DataFrame(summary_rows)
    curve_missingness = df.groupby("WELLNUM", sort=True)[feature_cols].apply(lambda g: g.isna().mean())
    curve_missingness = curve_missingness.reset_index()

    well_summary.to_csv(TABLE_DIR / "well_summary.csv", index=False)
    curve_missingness.to_csv(TABLE_DIR / "curve_missingness_by_well.csv", index=False)
    return well_summary, curve_missingness


def plot_missingness_heatmap(curve_missingness: pd.DataFrame) -> None:
    wells = curve_missingness["WELLNUM"].astype(int).astype(str).to_numpy()
    feature_cols = [c for c in curve_missingness.columns if c != "WELLNUM"]
    data = curve_missingness[feature_cols].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(NATURE_DOUBLE_WIDTH, mm_to_in(74)))
    im = ax.imshow(data, aspect="auto", cmap="YlGnBu", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(feature_cols)))
    ax.set_xticklabels(feature_cols, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(wells)))
    ax.set_yticklabels(wells)
    ax.set_xlabel("Input log")
    ax.set_ylabel("Training well")
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Missing fraction")
    savefig(FIG_DIR / "missingness_by_well.svg")


def plot_target_distributions(df: pd.DataFrame) -> None:
    wells = sorted(df["WELLNUM"].dropna().unique())
    fig, axes = plt.subplots(1, len(TARGET_COLS), figsize=(NATURE_DOUBLE_WIDTH, mm_to_in(66)), sharex=True)
    for ax, target in zip(axes, TARGET_COLS, strict=True):
        values = [df.loc[df["WELLNUM"] == well, target].dropna().to_numpy() for well in wells]
        ax.boxplot(
            values,
            tick_labels=[str(int(w)) for w in wells],
            showfliers=False,
            widths=0.65,
            patch_artist=True,
            boxprops={"facecolor": "#F1F1F1", "edgecolor": NATURE_COLORS["black"], "linewidth": 0.55},
            medianprops={"color": NATURE_COLORS["orange"], "linewidth": 0.85},
            whiskerprops={"color": NATURE_COLORS["black"], "linewidth": 0.55},
            capprops={"color": NATURE_COLORS["black"], "linewidth": 0.55},
        )
        ax.text(0.02, 0.96, target, transform=ax.transAxes, ha="left", va="top", fontsize=7.4, weight="bold")
        ax.set_xlabel("Training well")
        ax.grid(axis="y", color=NATURE_COLORS["light_gray"], linewidth=0.45)
    axes[0].set_ylabel("Target value")
    savefig(FIG_DIR / "target_distribution_by_well.svg")


def plot_pca_shift(df: pd.DataFrame, max_per_well: int = 2000, seed: int = 42) -> None:
    rng = np.random.default_rng(seed)
    feature_cols = [c for c in df.columns if c not in ID_COLS + TARGET_COLS]
    samples = []
    for _, group in df.groupby("WELLNUM", sort=True):
        if len(group) > max_per_well:
            idx = rng.choice(group.index.to_numpy(), size=max_per_well, replace=False)
            samples.append(group.loc[idx])
        else:
            samples.append(group)
    sample_df = pd.concat(samples, ignore_index=True)

    x = sample_df[feature_cols].copy()
    x = x.fillna(x.median(numeric_only=True))
    x = StandardScaler().fit_transform(x)
    coords = PCA(n_components=2, random_state=seed).fit_transform(x)

    fig, ax = plt.subplots(figsize=(NATURE_SINGLE_WIDTH, mm_to_in(78)))
    wells = sorted(sample_df["WELLNUM"].unique())
    cmap = plt.get_cmap("tab10")
    for i, well in enumerate(wells):
        mask = sample_df["WELLNUM"].to_numpy() == well
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=4.8,
            alpha=0.32,
            label=str(int(well)),
            color=cmap(i % 10),
            edgecolors="none",
        )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(title="Well", ncol=3, markerscale=2, frameon=False, loc="best")
    ax.grid(color=NATURE_COLORS["light_gray"], linewidth=0.4)
    savefig(FIG_DIR / "pca_well_shift.svg")


def main() -> None:
    apply_publication_style("nature")
    df = load_train()
    well_summary, curve_missingness = write_summary_tables(df)
    plot_missingness_heatmap(curve_missingness)
    plot_target_distributions(df)
    plot_pca_shift(df)

    print(f"Wrote tables to {TABLE_DIR}")
    print(f"Wrote figures to {FIG_DIR}")
    print(
        well_summary[
            ["WELLNUM", "samples", "feature_missing_rate", "PHIF_valid", "PHIF_missing_rate", "PHIF_mean", "PHIF_std"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()

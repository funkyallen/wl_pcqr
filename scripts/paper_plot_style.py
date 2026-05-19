from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


def mm_to_in(mm: float) -> float:
    return mm / 25.4


NATURE_SINGLE_WIDTH = mm_to_in(89)
NATURE_DOUBLE_WIDTH = mm_to_in(183)

NATURE_COLORS = {
    "blue": "#0072B2",
    "orange": "#D55E00",
    "green": "#009E73",
    "pink": "#CC79A7",
    "yellow": "#E69F00",
    "sky": "#56B4E9",
    "black": "#222222",
    "gray": "#777777",
    "light_gray": "#E6E6E6",
}


STYLE_PRESETS = {
    "publication": {
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "lines.linewidth": 1.8,
        "lines.markersize": 4,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    },
    "nature": {
        "font.size": 7.2,
        "axes.labelsize": 7.2,
        "axes.titlesize": 7.8,
        "xtick.labelsize": 6.7,
        "ytick.labelsize": 6.7,
        "legend.fontsize": 6.7,
        "lines.linewidth": 1.05,
        "lines.markersize": 3.2,
        "patch.linewidth": 0.5,
        "figure.dpi": 300,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
}


def apply_publication_style(preset: str = "publication") -> None:
    mpl.rcParams.update(STYLE_PRESETS.get(preset, STYLE_PRESETS["publication"]))


def save_figure(fig, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)
    return output

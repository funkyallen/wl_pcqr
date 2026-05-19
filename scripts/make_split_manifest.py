from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "train.csv"
SPLIT_DIR = ROOT / "data" / "splits"
TARGET_COLS = ["PHIF", "SW", "VSH"]


def load_labeled() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH).replace(-9999.0, np.nan)
    return df.sort_values(["WELLNUM", "DEPTH"]).reset_index(drop=True)


def build_lowo_manifest(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    wells = sorted(int(w) for w in df["WELLNUM"].dropna().unique())
    for target in TARGET_COLS:
        target_df = df[df[target].notna()].copy()
        target_wells = sorted(int(w) for w in target_df["WELLNUM"].dropna().unique())
        for test_well in target_wells:
            source_wells = [w for w in target_wells if w != test_well]
            rows.append(
                {
                    "target": target,
                    "protocol": "LOWO",
                    "split_id": f"{target}_LOWO_{test_well}",
                    "test_well": test_well,
                    "source_wells": " ".join(map(str, source_wells)),
                    "all_labeled_wells": " ".join(map(str, wells)),
                    "test_samples": int((target_df["WELLNUM"] == test_well).sum()),
                    "source_samples": int((target_df["WELLNUM"] != test_well).sum()),
                    "inner_calibration": "leave_one_source_well_out",
                    "notes": "Main well-blocked protocol; calibration folds are source wells only.",
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_labeled()
    manifest = build_lowo_manifest(df)
    out_path = SPLIT_DIR / "lowo_manifest.csv"
    manifest.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")
    print(manifest.groupby("target")["split_id"].count().to_string())


if __name__ == "__main__":
    main()

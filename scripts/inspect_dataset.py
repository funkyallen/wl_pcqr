from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw"
DOCS_DIR = ROOT / "docs"


def summarize_csv(path: Path) -> dict[str, object]:
    df = pd.read_csv(path)
    target_cols = [col for col in ("PHIF", "SW", "VSH") if col in df.columns]
    return {
        "file": path.name,
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": ",".join(df.columns),
        "well_count": df["WELLNUM"].nunique() if "WELLNUM" in df.columns else None,
        "wells": ",".join(map(str, sorted(df["WELLNUM"].dropna().unique()))) if "WELLNUM" in df.columns else "",
        "target_columns": ",".join(target_cols),
        "sentinel_minus_9999": int((df == -9999.0).sum().sum()),
    }


def main() -> None:
    DOCS_DIR.mkdir(exist_ok=True)
    rows = [summarize_csv(DATA_DIR / name) for name in ("train.csv", "test.csv")]
    out = pd.DataFrame(rows)
    out_path = DOCS_DIR / "dataset_inventory.csv"
    out.to_csv(out_path, index=False, encoding="utf-8")
    print(out.to_string(index=False))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()

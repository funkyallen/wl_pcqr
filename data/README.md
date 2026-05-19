# Data

This folder records the public data expected by the WL-PCQR experiments. Raw third-party datasets are not redistributed in the clean code repository. Download the public files from their original sources and place them in the paths below before running the experiment scripts.

| File | Rows | Columns | SHA256 |
|---|---:|---:|---|
| `raw/train.csv` | 318,967 | 17 | `45FBE4E77D4429104EEB06AA5B2FCC5AFBB8B9EE10C430D3A5B94E94B4ABCED7` |
| `raw/test.csv` | 11,275 | 14 | `1DE58AB42E0024809F178BBBC631CE28E44F2ACF9EEAA583324B86497283CF75` |

Mendeley raw files used by the external porosity sanity check:

| File | Size bytes | SHA256 |
|---|---:|---|
| `external/mendeley_gom_dutch_north_sea/raw/Gulf of Mexico sand_dataset.txt` | 4,868 | `8106823038A2F3D20B564852558C4A06D2AA7351DBDF777B6065D9BF87E982EC` |
| `external/mendeley_gom_dutch_north_sea/raw/Offshore Dutch North sea dataset.txt` | 62,145 | `9645268448E25E49476AB54E3B25C0AAE02C9F267139C3129FBA65E3427620B7` |

Sources:

- SPWLA PDDA 2021: `https://github.com/pddasig/Machine-Learning-Competition-2021`
- Gulf of Mexico/Dutch North Sea porosity dataset: Mendeley Data, version 1, DOI `10.17632/sdv629nbjr.1`

Expected local paths:

- SPWLA CSV files: `data/raw/train.csv` and `data/raw/test.csv`
- Mendeley raw files: `data/external/mendeley_gom_dutch_north_sea/raw/`

The Mendeley processing script writes `data/external/mendeley_gom_dutch_north_sea/processed/mendeley_phi_external.csv`. The repository tracks the split manifest and aggregate result tables needed to audit the paper, but raw and processed public datasets are ignored by git and should be downloaded or regenerated locally.

# Road Risk Analysis

Accident risk modelling by road type and vehicle class, combining DfT STATS19 collision data,
DfT AADF traffic volumes, and National Highways MIDAS/WebTRIS sensor data.

**Current geography:** Yorkshire (pilot) → Great Britain  
**Time range:** 2015–2024 (pre/post COVID analysis)

---

## Quick Start

```bash
# 1. Clone and install
git clone <repo>
cd road-risk-analysis
pip install -e ".[dev]"

# 2. Download raw data — see docs/quarto/data-sources/
#    STATS19:  https://www.gov.uk/government/statistical-data-sets/road-safety-open-data
#    AADF:     https://roadtraffic.dft.gov.uk/downloads
#    MRDB:     https://www.data.gov.uk/dataset/major-road-database
#    WebTRIS:  automated via src/ingest_webtris.py

# 3. Run ingestion
python src/ingest_stats19.py data/raw/stats19
python src/ingest_aadf.py data/raw/aadf
python src/ingest_webtris.py --region yorkshire

# 4. Build features and train model
python src/features.py
python src/model.py

# 5. Launch app
streamlit run app/main.py
```

---

## Data Sources

| Source | Provider | Granularity | Coverage |
|---|---|---|---|
| STATS19 (collisions, vehicles, casualties) | DfT | Per incident | GB 1979– |
| AADF by direction | DfT | Road link / year | GB |
| Major Road Database (MRDB) | DfT / OS | Road geometry | GB |
| MIDAS / WebTRIS | National Highways | Site / 15-min | Motorways + A-roads |

---

## Repo Structure

```
road-risk-analysis/
├── src/                  # Core Python modules
├── notebooks/            # EDA and exploration (numbered)
├── app/                  # Streamlit interactive app
├── data/
│   ├── raw/              # Source files — never modified
│   ├── processed/        # Cleaned, joined datasets
│   └── features/         # ML-ready feature tables
├── tests/
├── docs/quarto/          # Quarto documentation site
├── reports/              # Output reports and figures
└── config/               # YAML config files
```

---

## Documentation

Built with [Quarto](https://quarto.org). To render locally:

```bash
cd docs/quarto
quarto preview
```

Published site: `docs/quarto/_site/` (rendered on push via GitHub Actions)

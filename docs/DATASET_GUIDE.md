# Dataset Guide

## Included in this repository
The repository already ships with:
- `data/traces/synthetic_train/` and `data/traces/synthetic_test/` for quick demos
- `data/traces/research/train/`, `val/`, and `test/` for stronger experiments and dashboard visuals

The research bundle is synthetic but deliberately diversified so the PPO policy sees a better mix of:
- stable broadband
- bursty broadband
- hotel Wi-Fi congestion
- conference crowding
- mobile commute patterns
- handoff dropouts
- home Wi-Fi interference
- mixed OOD-style scenarios

## Public datasets to use for dissertation-grade training
Recommended workflow:
1. Create the workspace:

```bash
python scripts/prepare_public_trace_workspace.py
```

2. Download the public trace sources into `data/public_sources/`.
3. Convert them into simple `bandwidth_mbps` CSV files.
4. Point your config to the converted train/val/test directories.

## Suggested source folders
- `data/public_sources/fcc_raw/`
- `data/public_sources/norway_raw/`
- `data/public_sources/pensieve_reference/`
- `data/public_sources/converted/`

## Project format for converted traces
Every training/evaluation trace should be saved as:

```csv
bandwidth_mbps
4.21
3.95
5.11
...
```

## Trace preparation principles
- Use **1-second granularity** for the project simulator.
- Keep bandwidth values in **Mbps**.
- Avoid empty or negative values.
- Split the final set into **train**, **val**, and **test** folders.
- Keep train/test disjoint.

## Practical dissertation advice
- Use the included `research/` traces for early development.
- Add public FCC + Norway style traces for the final report.
- Save a `*_catalog.csv` for every split so the dashboard can show dataset coverage.


## Extra helper scripts

### Convert Norway HSDPA raw logs
```bash
python scripts/convert_norway_logs.py <raw_norway_log_directory> --output-dir data/public_sources/converted/norway
```

### Normalize any existing Pensieve-style or public trace folder
```bash
python scripts/import_trace_directory.py <source_trace_folder> --output-dir data/public_sources/converted/imported --recursive
```

These helpers are useful when you already have public traces in mixed `.csv`, `.txt`, `.log`, or `.json` formats.

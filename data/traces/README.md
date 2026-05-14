# Trace Datasets

## Included datasets

### 1. Quick demo traces
- `synthetic_train/`
- `synthetic_test/`

Use these for smoke tests and very short PPO runs.

### 2. Research traces
- `research/train/`
- `research/val/`
- `research/test/`

Use these for the professional dashboard, result screenshots, and stronger dissertation experiments.

## Trace format
Every trace file stores one bandwidth sample per row in Mbps:

```csv
bandwidth_mbps
4.12
5.01
3.88
```

## Dashboard support
After training or evaluation, the project creates catalog files like:
- `results/<run>/train_trace_catalog.csv`
- `results/<run>/val_trace_catalog.csv`
- `results/<run>/eval_trace_catalog.csv`

These are used by the professional dashboard to visualize dataset coverage.

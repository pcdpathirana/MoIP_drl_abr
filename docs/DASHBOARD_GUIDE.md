# Dashboard Guide

The professional Streamlit dashboard is at `dashboard/app.py`.

## Tabs

### Executive overview
- project KPIs
- algorithm leaderboard
- four-layer architecture summary
- run configuration

### Training analytics
- PPO reward curves
- QoE trend during training
- loss / entropy monitoring

### Benchmarking
- algorithm comparison tables
- QoE bar charts
- bitrate vs rebuffer trade-off view

### Session explorer
- per-trace, per-algorithm playback timeline
- throughput vs chosen bitrate
- buffer level and rebuffering inspection

### Datasets
- trace catalogs
- scenario coverage charts
- difficulty map

### Deployment
- PyCharm → Git → Ubuntu VM guide
- tc/netem export example
- dataset and emulation docs in one place

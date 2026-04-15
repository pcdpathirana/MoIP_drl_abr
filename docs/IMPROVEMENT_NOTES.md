# Improvement Notes

## Major development improvements delivered

### Training data quality
- Added a stronger synthetic research dataset.
- Added dataset catalogs for dashboard analytics.
- Added a public-data workspace for FCC / Norway / Pensieve-style trace preparation.

### PPO training correctness
- Fixed the multi-trace training problem by resampling traces every episode.
- Added richer training metrics: QoE, rebuffer, bitrate trends.

### Simulation realism
- Added variable-size chunk simulation.
- Added request RTT modeling.
- Added per-segment session exports.

### Professional visualization
- Replaced the simple CSV viewer dashboard with a professional multi-tab Streamlit UI.
- Added KPI cards, benchmarking, session explorer, dataset intelligence, and deployment guides.

### Dissertation/report alignment
This version better supports your chapters on:
- Project Description
- Requirements Analysis
- System Architecture
- Implementation
- Experimental Setup
- Results and Discussion

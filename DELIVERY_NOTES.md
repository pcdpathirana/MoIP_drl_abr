# Delivery Notes

This improved delivery focuses on the user feedback that the first development should be stronger and that the project should ship with better data support and a more professional GUI.

## Included improvements
- professional Streamlit dashboard
- richer synthetic research dataset
- session-level benchmarking exports
- fixed multi-trace PPO training behavior
- variable-size chunk simulator
- tc/netem replay exporter
- stronger configs for PyCharm and Ubuntu VM use

## Recommended starting point
Use:
```bash
python scripts/create_sample_data.py
python scripts/train_ppo.py --config configs/professional_demo.yaml
python scripts/evaluate_all.py --config configs/professional_demo.yaml
streamlit run dashboard/app.py
```


## Viva real-time demonstration upgrade (Live Simulation)

Main development changes:
- Added a **Live Simulation (Viva)** page to the Streamlit dashboard for real-time, interactive demonstration.
- Implemented **side-by-side playback panels** (two players) for direct comparison.
- Left panel uses the **trained PPO (DRL) policy** (fallback heuristic if checkpoint missing).
- Right panel allows switching baseline algorithms: **Rate-Based**, **BOLA**, **FESTIVE**.
- Added interactive **bandwidth slider** (and optional delay/loss controls) so evaluators can instantly see behavior changes.
- Added optional **tc/netem integration** to apply bandwidth conditions on an interface (requires sudo).
- Live charts: throughput, selected bitrate, buffer level, rebuffer events, and running QoE per panel.
- Bundled a short sample MP4 for the demo and improved the viva-friendly explanations in the UI.

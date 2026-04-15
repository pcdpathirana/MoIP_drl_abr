# Viva Live Demonstration (Real-time ABR)

This project includes a **Live Simulation (Viva)** page in the Streamlit dashboard so you can demonstrate
real-time ABR behavior without waiting for long training runs.

## What the demo shows
- **Two side-by-side players**:
  - **Player A:** Trained DRL policy (PPO)
  - **Player B:** Selectable baseline (Rate-Based / BOLA / FESTIVE)
- A **bandwidth slider** (plus delay/loss controls) to instantly show how each algorithm responds.
- Live charts of **throughput**, **selected bitrate**, **buffer level**, **stalling**, and **cumulative QoE**.

## How to run
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
streamlit run dashboard/app.py
```

## Optional: Apply real network conditions (tc/netem)
You can apply the same conditions to a real interface (requires sudo).

1) Install tc if missing:
```bash
sudo apt-get update && sudo apt-get install -y iproute2
```

2) Run Streamlit (option A: run with sudo - for lab/demo machines):
```bash
sudo -E streamlit run dashboard/app.py
```

3) Or (option B) configure passwordless `tc` for the demo user (recommended only in a controlled lab VM).

## Tips for viva
- Start at **8–10 Mbps**, then drop to **1–2 Mbps** and observe:
  - DRL should reduce bitrate earlier / avoid stalling
  - baselines may oscillate or stall depending on settings
- Increase bandwidth again to show recovery behavior and switching smoothness.

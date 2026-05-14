# Professional DRL-ABR Project for PyCharm and Ubuntu VM

This repository is an improved **research-grade ABR project** for **Deep Reinforcement Learning-based Adaptive Bitrate (ABR) selection** in multimedia streaming over IP.

It is designed to satisfy the development direction in your Chapter 02 project description:
- **Network emulator layer** using `tc/netem` or Mininet-WiFi compatible replay workflows.
- **Video chunk simulator** with variable-size chunks and segment-level content complexity.
- **DRL agent** implemented with **PyTorch** and a **Gymnasium-style** environment.
- **QoE evaluator** for bitrate, buffering, smoothness, throughput, stall ratio, and total QoE.
- **Professional Streamlit dashboard** for experiments, data intelligence, and deployment support.

## What is improved in this version

### 1) Better training data support
- The project now includes a **richer synthetic research dataset** under `data/traces/research/`.
- Traces cover multiple realistic scenarios:
  - broadband stable
  - bursty broadband
  - hotel Wi-Fi contention
  - conference peak congestion
  - commuter mobile sessions
  - handoff / dropout events
  - home Wi-Fi interference
  - mixed out-of-distribution sessions
- Catalog CSV files are generated automatically for the dashboard.

### 2) Real dataset workflow
The repo now includes a **public trace workspace** and guides for working with:
- FCC broadband traces
- Norway HSDPA mobile traces
- Pensieve-compatible trace preparation

See:
- `docs/DATASET_GUIDE.md`
- `scripts/prepare_public_trace_workspace.py`
- `scripts/export_netem_profile.py`

### 3) Real training fix
The previous lightweight demo effectively trained on a single trace at a time because the same environment instance was reused for many episodes. This version fixes that by **resampling a new training trace every episode reset** through a `TraceRepository`.

### 4) Better simulator realism
- Fixed chunk sizes were replaced with **segment-varying chunk sizes**.
- Request latency (`request_rtt_ms`) is now modeled.
- Session logs capture chunk size, complexity, throughput, rebuffering, and reward per segment.

### 5) Professional dashboard
The Streamlit dashboard now includes:
- executive overview and KPI cards
- PPO learning analytics
- algorithm benchmarking
- per-session playback explorer
- dataset intelligence panels
- deployment / netem guide tab

## Repository layout

```text
configs/         experiment settings
Dashboard/       Streamlit GUI (dashboard/app.py)
data/            synthetic and research trace bundles
docs/            architecture, dataset, dashboard, and VM guides
results/         demo outputs, checkpoints, session logs
scripts/         training, evaluation, dataset generation, netem export
src/abr_rl/      main Python package
tests/           unit tests
```

## Quick start in PyCharm

### 1) Open the project
- Open **PyCharm**
- Choose **Open**
- Select this repository folder

### 2) Create the interpreter
Use **Python 3.10** or newer.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

### 3) Build the sample + research datasets
```bash
python scripts/create_sample_data.py
```

### 4) Train the professional demo
```bash
python scripts/train_ppo.py --config configs/professional_demo.yaml
```

### 5) Evaluate PPO and baselines
```bash
python scripts/evaluate_all.py --config configs/professional_demo.yaml
```

### 6) Run the dashboard
```bash
streamlit run dashboard/app.py
```

## Git workflow from PyCharm to Ubuntu VM

### Local machine
```bash
git init
git add .
git commit -m "Professional DRL-ABR project"
git branch -M main
git remote add origin <your-repository-url>
git push -u origin main
```

### Ubuntu VM
```bash
git clone https://github.com/nimalanjanapiyumal/DRL.git
cd DRL_ABR_PyCharm_VM_Repo
bash scripts/bootstrap_ubuntu_vm.sh
```

## Main commands

```bash
make setup
make sample-data
make research-data
make train
make evaluate
make dashboard
make test
```

## Configs included

- `configs/small_demo.yaml` — very small sanity-check run
- `configs/professional_demo.yaml` — better demo for dashboard and dissertation screenshots
- `configs/default.yaml` — standard research baseline config

## Key deliverables already included

- complete Python source code
- PPO training and evaluation pipeline
- BOLA-inspired, FESTIVE-inspired, and Rate-based baselines
- research dataset generator
- professional dashboard
- tc/netem export script
- PyCharm + Git + Ubuntu VM workflow guide
- evaluation summaries and session-log support

## Recommended report sections supported by this repo

This repository directly supports your dissertation/report chapters on:
- Project Description
- Research Gap
- Requirements Analysis
- System Design / Architecture
- Implementation
- Experimental Setup
- Results and Discussion

See `docs/IMPROVEMENT_NOTES.md` for the exact mapping.

## Viva Real-time Demonstration (Live Simulation)

Run the professional dashboard and open the **Live Simulation (Viva)** tab to demonstrate real-time ABR decisions with a bandwidth slider and side-by-side comparison (DRL vs baselines).

```bash
streamlit run dashboard/app.py
```

See `docs/VIVA_DEMO.md` for viva-ready steps.

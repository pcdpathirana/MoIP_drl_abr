# Project Overview

This project implements a **Deep Reinforcement Learning-based Adaptive Bitrate (ABR)** pipeline for multimedia streaming over IP. It is organized as a clean research repository so that the same codebase can be:

1. developed in **PyCharm**,
2. version-controlled with **Git**,
3. pushed to GitHub/GitLab,
4. cloned on an **Ubuntu VM**, and
5. executed for **training, evaluation, visualization, and network-emulation export**.

## Core subsystems

### 1. Network emulator layer
- `scripts/export_netem_profile.py`
- `src/abr_rl/emulation/netem.py`

This layer converts 1-second bandwidth traces into executable `tc/netem` shell scripts.

### 2. Video chunk simulator
- `src/abr_rl/sim/video_profile.py`
- `src/abr_rl/sim/abr_simulator.py`

This layer simulates chunk downloads, playback buffering, rebuffering events, and content-driven chunk size variation.

### 3. DRL agent
- `src/abr_rl/envs/abr_env.py`
- `src/abr_rl/rl/ppo.py`
- `src/abr_rl/rl/models.py`

This layer uses a PPO agent that observes throughput history, download times, current buffer, last bitrate, and next-segment size ladder.

### 4. QoE evaluator
- `src/abr_rl/metrics/qoe.py`
- `scripts/evaluate_all.py`

This layer produces total QoE, bitrate, rebuffering, switch count, switch magnitude, throughput, and stall ratio.

## Research strength of this version
- multi-scenario training traces
- true multi-trace episode sampling
- session-level playback exports
- dashboard-ready result files
- Ubuntu-friendly deployment scripts

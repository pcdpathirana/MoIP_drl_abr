#!/usr/bin/env bash
set -euo pipefail

python scripts/create_sample_data.py
python scripts/train_ppo.py --config configs/professional_demo.yaml
python scripts/evaluate_all.py --config configs/professional_demo.yaml

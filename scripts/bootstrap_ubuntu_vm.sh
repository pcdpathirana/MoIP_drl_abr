#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${1:-$(pwd)}"

echo "[1/4] Installing Ubuntu packages"
sudo apt update
sudo apt install -y git python3 python3-venv python3-pip build-essential iproute2

echo "[2/4] Creating virtual environment"
cd "$PROJECT_DIR"
python3 -m venv .venv
source .venv/bin/activate

echo "[3/4] Installing Python dependencies"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .

echo "[4/4] Generating datasets"
python scripts/create_sample_data.py

echo "Ubuntu VM bootstrap complete."
echo "Next steps:"
echo "  source .venv/bin/activate"
echo "  python scripts/train_ppo.py --config configs/professional_demo.yaml"
echo "  python scripts/evaluate_all.py --config configs/professional_demo.yaml"
echo "  streamlit run dashboard/app.py"

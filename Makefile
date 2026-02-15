PYTHON ?= python

setup:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e .

sample-data:
	$(PYTHON) scripts/create_sample_data.py

research-data:
	$(PYTHON) scripts/build_research_dataset.py

train:
	$(PYTHON) scripts/train_ppo.py --config configs/professional_demo.yaml

evaluate:
	$(PYTHON) scripts/evaluate_all.py --config configs/professional_demo.yaml

netem-export:
	$(PYTHON) scripts/export_netem_profile.py data/traces/research/test/hotel_wifi_01.csv --output results/netem/hotel_wifi.sh

test:
	pytest -q

dashboard:
	streamlit run dashboard/app.py

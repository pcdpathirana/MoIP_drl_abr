# PyCharm → Git → Ubuntu VM Workflow

## Local development in PyCharm
1. Open this repository in **PyCharm**.
2. Create a **Python 3.10** interpreter.
3. Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

4. Generate the datasets:

```bash
python scripts/create_sample_data.py
```

5. Train and evaluate:

```bash
python scripts/train_ppo.py --config configs/professional_demo.yaml
python scripts/evaluate_all.py --config configs/professional_demo.yaml
```

## Git push from local machine
```bash
git init
git add .
git commit -m "Professional DRL-ABR project"
git branch -M main
git remote add origin <your-repository-url>
git push -u origin main
```

## Clone on Ubuntu VM
```bash
git clone <your-repository-url>
cd DRL_ABR_PyCharm_VM_Repo
bash scripts/bootstrap_ubuntu_vm.sh
```

## Run the professional dashboard on the VM
```bash
source .venv/bin/activate
streamlit run dashboard/app.py
```

## Tip
If you want dissertation screenshots, use `configs/professional_demo.yaml` because it creates richer dashboard outputs and session logs than the tiny smoke-test config.

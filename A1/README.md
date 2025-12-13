# SAR-MARL PettingZoo

Porównanie zespołów heterogenicznych i homogenicznych w kooperacyjnym scenariuszu SAR (Search & Rescue) z PettingZoo + RLlib.

## Szybki start
```bash
python -m venv .venv && source .venv/bin/activate # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Trening zespołu homogenicznego (wszyscy identyczni "generalist")
python train_rllib.py --hetero 0 --grid 15 --agents 4 --victims 6 --rubble 0.18 --steps 400 --timesteps 400000

# Trening zespołu heterogenicznego (scout, support, medic)
python train_rllib.py --hetero 1 --grid 15 --agents 4 --victims 6 --rubble 0.18 --steps 400 --timesteps 400000

# Ewaluacja na checkpointcie
python evaluate.py --checkpoint runs/ppo_sar_pz_<DATA>/checkpoint_xxxx --episodes 50 --grid 15 --agents 4 --victims 6 --rubble 0.18 --steps 400 --hetero 1 --out results_hetero.csv
python evaluate.py --checkpoint runs/ppo_sar_pz_<DATA>/checkpoint_xxxx --episodes 50 --grid 15 --agents 4 --victims 6 --rubble 0.18 --steps 400 --hetero 0 --out results_homo.csv
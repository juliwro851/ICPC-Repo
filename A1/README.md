# README – Experiment A1: Uniformity with Both Global and Local Rewards

## Directory Structure

.
├── README.md            # Experiment description
├── train_rllib.py       # Training script (RLlib)
├── sar_env.py           # SAR environment definition           
└── utils.py            # Shared utility functions

---

## Overview / Usage

### Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Training
```bash
Homogeneous Team (Generalist Agents)
python train_rllib.py --hetero 0 --grid 20 --agents 4 --victims 10 --rubble 0.2 --steps 400 --timesteps 1000
Heterogeneous Team (Role-Specialized Agents)
python train_rllib.py --hetero 1 --grid 20 --agents 4 --victims 10 --rubble 0.2 --steps 400 --timesteps 1000
```
Training artifacts, including checkpoints, logs, and metrics, are stored in the 
runs/ directory.


## Shared Utilities (utils.py)
The utils.py module provides helper functions shared across all experiments and is not experiment-specific.

## Map Representation
The SAR environment uses integer-encoded cell types defined as constants:
CELL_EMPTY
CELL_WALL
CELL_RUBBLE
CELL_VICTIM


## GIF Export
By default, generated GIF files are stored in the gifs/ directory.
# SAR Multi-Agent Reinforcement Learning Experiments

## Overview
This repository contains a series of Search and Rescue (SAR) multi-agent reinforcement learning experiments designed to analyze how different factors affect agent coordination and rescue performance. The project investigates the impact of:

- **Reward structure** (local vs. global rewards)
- **Agent uniformity** (homogeneous vs. heterogeneous teams)
- **Role composition** (firefighter-to-rescuer ratios)
- **Observation scope** (agent visibility range)

Each experiment is self-contained in its own folder and differs from the baseline by exactly one experimental factor.

## Repository Structure
```
.
├── A1/                    # Experiment A1: Uniformity with local rewards
├── A2/                    # Experiment A2: Uniformity with global rewards
├── B1/                    # Experiment B: Role ratio 3:1 and 1:3 (firefighters:rescuers)
├── C/                     # Experiment C: Increased observation window (5×5 → 7×7)
├── D/                     # Experiment D: Additional experimental configuration
├── analysis.py            # Statistical analysis and visualization script
└── README.md             # This file
```

### Variants

Each experiment includes different team compositions:

- **homo**: Homogeneous team (all agents have identical capabilities)
- **hetero**: Heterogeneous team (agents have specialized roles)
- **strazak**: Firefighter-focused configuration (used in B1)
- **ratownicy**: Rescuer-focused configuration (used in B2)

## Cross-Experiment Analysis

### Data Collection
Results from all experiments should be collected in the root directory following this naming convention:
```
results_<variant>_<iteration>_<experiment>.csv
episode_stats_<variant>_<iteration>_<experiment>.csv
```

**Example filenames:**
- `results_homo_1_A1.csv` - Homogeneous team, iteration 1, experiment A1
- `results_hetero_2_A2.csv` - Heterogeneous team, iteration 2, experiment A2
- `episode_stats_strazak_1_B1.csv` - Firefighters, iteration 1, experiment B1
- `episode_stats_ratownicy_3_B2.csv` - Rescuers, iteration 3, experiment B2

### Running Cross-Experiment Analysis

Once all experimental data is collected in the root directory:
```bash
python analysis.py
```

This script will:
1. **Load all experimental data** from CSV files
2. **Generate comprehensive plots** including:
   - Line plots: reward progression, rescue performance, episode length
   - Histograms: action distributions, collision frequencies
   - Scatter plots: relationships between metrics
3. **Perform statistical tests** (t-tests, Mann-Whitney U tests)
4. **Calculate effect sizes** (Cohen's d)
5. **Export results** to `./statistics/` directory

### Output Structure

The analysis generates the following outputs in `./statistics/`:
```
statistics/
├── summary_report.txt                          # Overview of all loaded data
├── plot_Average_Reward_per_Episode.pdf        # Training reward curves
├── plot_Number_of_Rescued_Victims.pdf         # Rescue performance
├── plot_Average_Episode_Length.pdf            # Episode duration trends
├── stats_Average_Reward_per_Episode.csv       # Descriptive statistics
├── stats_Number_of_Rescued_Victims.csv        # Rescue statistics
├── statistical_tests_Average_Reward.csv       # Pairwise comparisons
└── ...                                        # Additional plots and statistics
```
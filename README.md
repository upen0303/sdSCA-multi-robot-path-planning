# qlsdSCA: Q-Learning based Multi-Strategy Self-Adaptive Differential Sine-Cosine Algorithm for Multi-Robot Path Planning

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![Status](https://img.shields.io/badge/Status-Active%20Research-green)
![Institution](https://img.shields.io/badge/IIIT-Nagpur-orange)

## Overview

This repository contains the implementation of **qlsdSCA**, a novel
metaheuristic algorithm proposed as an improvement over sdSCA
(Akay & Yildirim, 2023) for online multi-robot path planning.

### Research Contribution

This work proposes two key improvements over the base sdSCA algorithm:

1. **Q-Learning Strategy Selector** — Replaces passive roulette wheel
   selection with an active reinforcement learning agent that learns
   which update strategy works best in each optimization state

2. **Adaptive F and CR Parameters** — Each strategy maintains its own
   memory of successful scale factor (F) and crossover rate (CR) values,
   enabling self-tuning behavior

### Algorithm Evolution

SCA (Mirjalili, 2016)
↓ single strategy
sdSCA (Akay & Yildirim, 2023)
↓ multi-strategy + roulette wheel
qlsdSCA (Our Contribution, 2024)
↓ multi-strategy + Q-Learning + adaptive F/CR

---

## Base Paper

> Akay, R., & Yildirim, M.Y. (2023). Multi-strategy and self-adaptive
> differential sine-cosine algorithm for multi-robot path planning.
> _Expert Systems With Applications_, 232, 120849.
> https://doi.org/10.1016/j.eswa.2023.120849

---

## Repository Structure

sdSCA-multi-robot-path-planning/
│
├── algorithms/
│ ├── sca.py ← Basic SCA (Mirjalili, 2016)
│ ├── sdsca.py ← sdSCA reproduction (Akay & Yildirim, 2023)
│ └── qlsdsca.py ← Proposed qlsdSCA (Our contribution)
│
├── path_planning/
│ ├── robot.py ← Robot class (Equations 8, 9)
│ ├── obstacle.py ← Static and Dynamic obstacles
│ ├── fitness.py ← Fitness functions F1+F2+F3+F4
│ ├── environment.py ← Full simulation loop (Algorithm 3)
│ └── scenarios.py ← 3 test scenarios from paper
│
├── experiments/
│ └── run_scenarios.py ← Run all experiments
│
├── analysis/
│ └── statistical_tests.py ← Wilcoxon test and metrics
│
├── visualization/
│ └── plot_paths.py ← Path and result plotting
│
├── results/
│ └── scenarios/ ← Saved CSV results
│
└── notebooks/
└── sdSCA_experiments.ipynb ← Google Colab experiments

---

## Key Results (Scenario 1 — 30 Runs)

| Metric      | SCA   | sdSCA | qlsdSCA | qlsdSCA vs sdSCA |
| ----------- | ----- | ----- | ------- | ---------------- |
| APDE (cm)   | 85.6  | 14.2  | 9.6     | **+32% ✅**      |
| AUGD (cm)   | 20247 | 16495 | 16167   | **+2% ✅**       |
| Total Steps | 499   | 410   | 403     | **+1.7% ✅**     |

### Chain of Improvement (APDE)

SCA → sdSCA → qlsdSCA
85.6 → 14.2 → 9.6
(83% ↓) (32% ↓)

---

## Test Scenarios

| Scenario | Environment | Robots | Static Obs | Dynamic Obs |
| -------- | ----------- | ------ | ---------- | ----------- |
| 1        | 100×100 cm  | 6      | 7          | 3           |
| 2        | 100×100 cm  | 7      | 7          | 3           |
| 3        | 200×200 cm  | 12     | 14         | 6           |

---

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/sdSCA-multi-robot-path-planning.git
cd sdSCA-multi-robot-path-planning

# Install dependencies
pip install numpy matplotlib scipy pandas seaborn jupyter
```

---

## Quick Start

```python
# Run basic SCA vs sdSCA vs qlsdSCA comparison
python test_qlsdsca.py

# Run path planning experiments
python experiments/run_scenarios.py

# Run statistical analysis
python analysis/statistical_tests.py
```

---

## Algorithm Details

### Q-Learning Agent

States : 9 (3 iteration stages × 3 improvement levels)
Actions : 4 (one per strategy)
Reward : +1.0 improvement | 0.0 no change | -0.1 worse
Update : Q(s,a) ← Q(s,a) + α[R + γ·max Q(s',a') - Q(s,a)]

### 4 Update Strategies

S1: Original SCA — Equation (2) — guided exploitation
S2: DE/rand/1 — Equation (4) — pure exploration
S3: DE/current-to-best/1 — Equation (5) — balanced
S4: DE/current-to-rand/1 — Equation (6) — most exploratory

### Fitness Function

Fit = F1 + F2 + F3 + F4
F1 = Shortest distance
F2 = Static obstacle avoidance (penalty ε = 10⁵)
F3 = Dynamic obstacle avoidance (penalty ε = 10⁵)
F4 = Inter-robot collision avoidance (penalty ε = 10⁵)

---

## Requirements

Python 3.11+
NumPy 1.24+
Matplotlib 3.7+
SciPy 1.11+
Pandas 2.0+

---

## Research Details

Institution : Indian Institute of Information Technology, Nagpur
Department : Computer Science and Engineering
Student : Upendra Prabhakar (BT22CSE122)
Supervisor : Dr. Kaushilendra Sharma
Year : 2026
Status : Active Research — Experiments Running
Target : Expert Systems With Applications (Elsevier)

---

## Acknowledgement

This work builds upon the sdSCA algorithm proposed by:

- Akay, R., & Yildirim, M.Y. (2023) — Erciyes University, Turkey

Original MATLAB implementation available at:
https://codeocean.com/capsule/2404110/tree/v1

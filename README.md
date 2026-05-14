Python implementation of sdSCA algorithm based on the paper:
"Multi-strategy and self-adaptive differential sine-cosine
algorithm for multi-robot path planning"
by Akay & Yildirim (2023)

qlsdSCA for Multi-Robot Path Planning
Overview

This project implements and extends the research paper:

“Multi-Strategy Self-Adaptive Differential Sine Cosine Algorithm for Multi-Robot Path Planning”

The project focuses on solving the multi-robot path planning problem using metaheuristic optimization algorithms.

The implementation includes:

Basic Sine Cosine Algorithm (SCA)
Multi-Strategy Differential SCA (sdSCA)
Proposed Q-Learning based sdSCA (qlsdSCA)

The objective is to generate:

shortest paths
collision-free paths
safe robot navigation
efficient multi-robot coordination

while avoiding:

static obstacles
dynamic obstacles
robot-to-robot collisions
Main Contribution

This work extends the original sdSCA algorithm by introducing:

Q-Learning based Adaptive Strategy Selection

Instead of selecting strategies randomly or using fixed probabilities, the proposed qlsdSCA algorithm uses reinforcement learning to:

learn which strategy performs best
adapt during optimization
improve convergence
improve exploration and exploitation balance

Additional improvements:

Adaptive parameter control (F and CR)
Multiple benchmark function testing
Multi-scenario robot path planning experiments
Statistical analysis using Wilcoxon/Mann-Whitney tests
Project Structure
sdSCA-multi-robot-path-planning/
│
├── algorithms/
│ ├── sca.py
│ ├── sdsca.py
│ ├── qlsdsca.py
│ ├── qlearning.py
│ └── adaptive_parameters.py
│
├── path_planning/
│ ├── robot.py
│ ├── obstacle.py
│ ├── environment.py
│ ├── fitness.py
│ └── scenarios.py
│
├── tests/
│ ├── test_sca.py
│ ├── test_sdsca.py
│ ├── test_qlsdsca.py
│ ├── test_environment.py
│ ├── test_fitness.py
│ ├── test_robot.py
│ ├── test_obstacle.py
│ └── test_visualization.py
│
├── experiments/
│ └── run_scenarios.py
│
├── visualization/
│ └── plot_paths.py
│
├── results/
│ ├── scenarios/
│ └── analysis/
│
└── README.md
Algorithms Implemented

1. SCA — Sine Cosine Algorithm

Basic optimization algorithm using sine and cosine mathematical operators.

Main limitation:

uses only one update strategy
may suffer from premature convergence 2. sdSCA — Multi-Strategy Self-Adaptive Differential SCA

Improved version of SCA.

Features:

multiple update strategies
strategy pool
adaptive strategy selection
differential evolution concepts

Strategies used:

S1 → SCA strategy
S2 → DE/rand strategy
S3 → DE/best strategy
S4 → mixed differential strategy 3. qlsdSCA — Proposed Algorithm

Further improvement over sdSCA.

Features:

Q-learning based strategy selection
adaptive F and CR parameters
intelligent exploration/exploitation control
reinforcement learning integration

The algorithm learns which strategy works best during optimization.

Multi-Robot Path Planning

The project simulates robots moving inside a 2D environment.

Environment includes:

multiple robots
static obstacles
dynamic obstacles
goal locations

Robots must:

avoid collisions
avoid obstacles
minimize travel distance
reach goals safely
Fitness Function

The optimization problem is formulated as:

F = F1 + F2 + F3 + F4

Where:

F1 → shortest path objective
F2 → static obstacle avoidance
F3 → dynamic obstacle avoidance
F4 → robot collision avoidance

Lower fitness indicates better solutions.

Scenarios

Three simulation scenarios are implemented.

Scenario 1
Environment: 100 × 100
6 robots
7 static obstacles
3 dynamic obstacles
Scenario 2
Environment: 100 × 100
7 robots
mixed obstacle shapes
Scenario 3
Environment: 200 × 200
12 robots
14 static obstacles
6 dynamic obstacles
Performance Metrics

The algorithms are evaluated using:

APDE → Average Path Deviation Error
AUGD → Average Unreached Goal Distance
Total Fitness
Required Steps
Total Distance
AET → Average Execution Time
Statistical Analysis

To ensure scientifically valid results, statistical tests are performed:

Wilcoxon Signed-Rank Test
Mann-Whitney U Test

The analysis verifies whether improvements are statistically significant.

Visualization

The project generates:

robot path visualizations
convergence curves
step comparison bar graphs
obstacle maps
Technologies Used
Python
NumPy
SciPy
Matplotlib
Pandas
How to Run
Clone Repository
git clone https://github.com/upen0303/sdSCA-multi-robot-path-planning.git
cd sdSCA-multi-robot-path-planning
Install Dependencies
pip install numpy scipy matplotlib pandas
Run Basic Tests
python tests/test_sca.py
python tests/test_sdsca.py
python tests/test_qlsdsca.py
Run Path Planning Experiments
python experiments/run_scenarios.py
Run Statistical Analysis
python analysis/statistical_tests.py
Example Outputs

The implementation generates:

optimized robot paths
convergence graphs
statistical comparison tables
CSV result files
Future Work

Possible future improvements:

ROS integration
3D path planning
real robot implementation
deep reinforcement learning
dynamic strategy generation
Author

Upendra Prabhakar

B.Tech CSE

Research Area:

Optimization Algorithms
Multi-Robot Path Planning
Reinforcement Learning
Swarm Intelligence
Reference

Original Paper:

Multi-Strategy Self-Adaptive Differential Sine Cosine Algorithm for Multi-Robot Path Planning

Published in: Expert Systems with Applications

I updated and structured your README professionally for a research/project repository.

It now includes:

project overview
your contribution
algorithm explanations
folder structure
scenarios
fitness function
statistical analysis
technologies
how to run
future work
research focus

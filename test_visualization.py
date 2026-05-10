"""
Test visualization — run small simulation and plot results.
"""
import numpy as np
import sys
sys.path.append('.')

from path_planning.scenarios      import get_scenario
from algorithms.sdsca             import sdSCA
from visualization.plot_paths     import (plot_scenario_paths,
                                          plot_fitness_curves,
                                          plot_steps_bar)

print("=" * 55)
print("  Visualization Test — Scenario 1 (quick run)")
print("=" * 55)

# ── Setup scenario 1 ───────────────────────────────────────
env = get_scenario(1)

# ── Set algorithm ──────────────────────────────────────────
NR        = len(env.robots)
D         = NR * 2
lb        = np.tile([1.0, 0.0],        NR)
ub        = np.tile([1.5, 2*np.pi],    NR)

algorithm = sdSCA(
    population_size = 30,
    max_iterations  = 50,    # small for quick test
    dim             = D,
    lower_bound     = lb,
    upper_bound     = ub,
    a  = 2,
    F  = 0.8,
    CR = 0.95
)
env.set_algorithm(algorithm)
env.max_steps = 50   # limit steps for quick test

# ── Run simulation ─────────────────────────────────────────
print("\nRunning quick simulation...")
results = env.run(verbose=True)

# ── Plot paths ─────────────────────────────────────────────
print("\nGenerating plots...")

plot_scenario_paths(
    env,
    algorithm_name  = 'sdSCA',
    scenario_number = 1,
    save_path       = 'plots/paths/scenario1_sdsca.png',
    show            = False    # set True to display
)

# ── Plot fake comparison bar chart ─────────────────────────
# Using paper values from Table 11 for demonstration
steps_comparison = {
    'SCA'  : [138, 142, 66, 135, 88, 115],
    'sdSCA': [88,  86,  35, 84,  46, 67 ],
}

plot_steps_bar(
    steps_comparison,
    scenario_number = 1,
    save_path       = 'plots/paths/scenario1_steps_bar.png',
    show            = False
)

# ── Plot fake convergence curves ───────────────────────────
# Using step fitness history for demonstration
fake_curves = [
    results['total_fitness'] *
    np.exp(-np.linspace(0, 3, 50)),   # SCA curve
    results['total_fitness'] *
    np.exp(-np.linspace(0, 5, 50)),   # sdSCA curve
]

plot_fitness_curves(
    fake_curves,
    algorithm_names = ['SCA', 'sdSCA'],
    scenario_number = 1,
    save_path       = 'plots/paths/scenario1_fitness.png',
    show            = False
)

print("\n[ Results ]")
print(f"  Total steps    : {results['total_steps']}")
print(f"  Total distance : {results['total_distance']:.2f} cm")
print(f"  APDE           : {results['APDE']:.2f} cm")
print(f"  Execution time : {results['AET']:.2f} s")
print(f"\n  Plots saved to: plots/paths/")

print("\n" + "=" * 55)
print("  ✅ Visualization working correctly!")
print("=" * 55)
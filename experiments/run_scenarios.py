"""
Run Path Planning Experiments — All 3 Scenarios
================================================
Based on Section 4.4 of the paper.

This script:
1. Runs each scenario with sdSCA and SCA
2. Runs each experiment 30 times (as per paper)
3. Calculates average metrics (APDE, AUGD, AET, steps, distance)
4. Saves results to CSV files
5. Prints comparison tables like Tables 11-19 in paper

Runtime warning:
----------------
Running 30 times × 3 scenarios × 2 algorithms is slow.
For testing, set NUM_RUNS = 3 first.
For paper results, set NUM_RUNS = 30.

Author: Your Name
Date: 2024
"""

import numpy as np
import pandas as pd
import time
import sys
import os
sys.path.append('.')

from path_planning.scenarios   import get_scenario
from algorithms.sca            import SCA
from algorithms.sdsca          import sdSCA


# ══════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════

NUM_RUNS       = 3      # set to 30 for paper results
MAX_STEPS      = 300    # max simulation steps per run
OPT_ITERATIONS = 100    # optimizer iterations per step
POPULATION     = 30     # population size

# ── Per scenario max steps ─────────────────────────────────
SCENARIO_MAX_STEPS = {
    1: 300,    # 6 robots  — 300 enough
    2: 400,    # 7 robots  — needs more
    3: 800,    # 12 robots — needs much more
}
OPT_ITERATIONS = 100    # optimizer iterations per step
POPULATION     = 30     # population size

# Algorithms to compare
# Add more here later (SFS, AOA, WOA, HHO)
ALGORITHMS = ['SCA', 'sdSCA', 'qlsdSCA']

# Scenarios to run
SCENARIOS = [1, 2, 3]


# ══════════════════════════════════════════════════════════════
# ALGORITHM FACTORY
# ══════════════════════════════════════════════════════════════

def create_algorithm(name, dim, lower_bound, upper_bound):
    """
    Create algorithm object by name.

    Parameters:
    -----------
    name        : str — 'SCA' or 'sdSCA'
    dim         : int — problem dimension
    lower_bound : np.array
    upper_bound : np.array

    Returns:
    --------
    algorithm object
    """
    if name == 'SCA':
        return SCA(
            population_size = POPULATION,
            max_iterations  = OPT_ITERATIONS,
            dim             = dim,
            lower_bound     = lower_bound,
            upper_bound     = upper_bound,
            a               = 2
        )
    elif name == 'sdSCA':
        return sdSCA(
            population_size = POPULATION,
            max_iterations  = OPT_ITERATIONS,
            dim             = dim,
            lower_bound     = lower_bound,
            upper_bound     = upper_bound,
            a               = 2,
            F               = 0.8,
            CR              = 0.95
        )
    elif name == 'qlsdSCA':
        from algorithms.qlsdsca import qlsdSCA
        return qlsdSCA(
            population_size = POPULATION,
            max_iterations  = OPT_ITERATIONS,
            dim             = dim,
            lower_bound     = lower_bound,
            upper_bound     = upper_bound,
            a               = 2,
            F               = 0.8,
            CR              = 0.95,
            alpha           = 0.3,
            gamma           = 0.95,
            epsilon         = 1.0
        )
    else:
        raise ValueError(f"Unknown algorithm: {name}")


# ══════════════════════════════════════════════════════════════
# SINGLE RUN
# ══════════════════════════════════════════════════════════════

def run_single(scenario_number, algorithm_name, verbose=False):
    """
    Run one simulation — one scenario, one algorithm, one time.

    Parameters:
    -----------
    scenario_number : int — 1, 2, or 3
    algorithm_name  : str — 'SCA' or 'sdSCA'
    verbose         : bool — print step details

    Returns:
    --------
    dict : results from this single run
    """
    # Create fresh environment
    env = get_scenario(scenario_number)
    env.max_steps = SCENARIO_MAX_STEPS[scenario_number]

    # Build bounds
    NR = len(env.robots)
    D  = NR * 2
    lb = np.tile([1.0, 0.0],     NR)
    ub = np.tile([1.5, 2*np.pi], NR)

    # Create algorithm
    algorithm = create_algorithm(
        algorithm_name, D, lb, ub
    )
    env.set_algorithm(algorithm)

    # Run simulation
    results = env.run(verbose=verbose)

    # Add per-robot info
    results['steps_per_robot']     = [
        r.steps_taken for r in env.robots
    ]
    results['distances_per_robot'] = [
        r.total_distance for r in env.robots
    ]
    results['all_reached_goal']    = all(
        r.reached_goal for r in env.robots
    )

    return results


# ══════════════════════════════════════════════════════════════
# MULTIPLE RUNS — AVERAGE RESULTS
# ══════════════════════════════════════════════════════════════

def run_multiple(scenario_number,
                 algorithm_name,
                 num_runs=NUM_RUNS):
    """
    Run simulation multiple times and average results.
    Paper runs each experiment 30 times.

    Parameters:
    -----------
    scenario_number : int
    algorithm_name  : str
    num_runs        : int — number of independent runs

    Returns:
    --------
    dict : averaged results across all runs
    """
    print(f"\n  Running {algorithm_name} on "
          f"Scenario {scenario_number} "
          f"({num_runs} runs)...")

    # Storage for all runs
    all_steps        = []
    all_distances    = []
    all_apde         = []
    all_augd         = []
    all_fitness      = []
    all_aet          = []
    all_steps_robot  = []
    all_dist_robot   = []

    for run in range(1, num_runs + 1):

        print(f"    Run {run:2d}/{num_runs}...",
              end=' ', flush=True)

        run_start = time.time()
        results   = run_single(scenario_number,
                               algorithm_name,
                               verbose=False)
        run_time  = time.time() - run_start

        # Collect metrics
        all_steps.append(results['total_steps'])
        all_distances.append(results['total_distance'])
        all_apde.append(results['APDE'])
        all_augd.append(results['AUGD'])
        all_fitness.append(results['total_fitness'])
        all_aet.append(results['AET'])
        all_steps_robot.append(results['steps_per_robot'])
        all_dist_robot.append(results['distances_per_robot'])

        reached = results['all_reached_goal']
        print(f"Steps={results['total_steps']:4d} | "
              f"APDE={results['APDE']:8.2f} | "
              f"Time={run_time:.1f}s | "
              f"All reached={reached}")

    # ── Calculate averages ─────────────────────────────────
    NR = len(all_steps_robot[0])

    avg_results = {
        'algorithm'       : algorithm_name,
        'scenario'        : scenario_number,
        'num_runs'        : num_runs,

        # Average total metrics
        'avg_total_steps'    : np.mean(all_steps),
        'avg_total_distance' : np.mean(all_distances),
        'avg_APDE'           : np.mean(all_apde),
        'avg_AUGD'           : np.mean(all_augd),
        'avg_total_fitness'  : np.mean(all_fitness),
        'avg_AET'            : np.mean(all_aet),

        # Standard deviations
        'std_APDE'           : np.std(all_apde),
        'std_AUGD'           : np.std(all_augd),
        'std_fitness'        : np.std(all_fitness),

        # Per robot averages
        'avg_steps_per_robot' : [
            np.mean([run[i] for run in all_steps_robot])
            for i in range(NR)
        ],
        'avg_dist_per_robot'  : [
            np.mean([run[i] for run in all_dist_robot])
            for i in range(NR)
        ],

        # Raw data for further analysis
        'raw_apde'    : all_apde,
        'raw_augd'    : all_augd,
        'raw_fitness' : all_fitness,
        'raw_aet'     : all_aet,
    }

    return avg_results


# ══════════════════════════════════════════════════════════════
# PRINT RESULTS TABLE
# ══════════════════════════════════════════════════════════════

def print_results_table(all_algo_results, scenario_number):
    """
    Print comparison table for all algorithms.
    Extended to support 3 algorithms.
    """
    num_runs   = list(all_algo_results.values())[0]['num_runs']
    algo_names = list(all_algo_results.keys())

    print(f"\n{'=' * 75}")
    print(f"  SCENARIO {scenario_number} RESULTS "
          f"(averaged over {num_runs} runs)")
    print(f"{'=' * 75}")

    # ── Per robot steps ────────────────────────────────────
    NR = len(list(
        all_algo_results.values()
    )[0]['avg_steps_per_robot'])

    print(f"\n  Average Required Steps per Robot:")
    header = f"  {'Robot':<12}"
    for name in algo_names:
        header += f" {name:>10}"
    print(header)
    print(f"  {'-' * (12 + 11 * len(algo_names))}")

    for i in range(NR):
        row = f"  Robot #{i+1:<6}"
        for name in algo_names:
            val = all_algo_results[name][
                'avg_steps_per_robot'
            ][i]
            row += f" {val:>10.1f}"
        print(row)

    # Totals row
    row = f"  {'Total':<12}"
    for name in algo_names:
        val = all_algo_results[name]['avg_total_steps']
        row += f" {val:>10.1f}"
    print(f"  {'-' * (12 + 11 * len(algo_names))}")
    print(row)

    # ── Summary metrics ────────────────────────────────────
    print(f"\n  Summary Metrics:")
    header = f"  {'Metric':<20}"
    for name in algo_names:
        header += f" {name:>12}"
    print(header)
    print(f"  {'-' * (20 + 13 * len(algo_names))}")

    metrics = [
        ('APDE (cm)',    'avg_APDE'),
        ('AUGD (cm)',    'avg_AUGD'),
        ('Total Fitness','avg_total_fitness'),
        ('AET (s)',      'avg_AET'),
    ]

    for label, key in metrics:
        row    = f"  {label:<20}"
        values = []

        for name in algo_names:
            val = all_algo_results[name][key]
            values.append(val)
            row += f" {val:>12.2f}"

        # Best value indicator
        best_val = min(values)
        best_idx = values.index(best_val)
        row     += f"  ← {algo_names[best_idx]} best"

        print(row)

    # ── qlsdSCA vs sdSCA improvement ──────────────────────
    if 'qlsdSCA' in all_algo_results and \
       'sdSCA'   in all_algo_results:

        print(f"\n  qlsdSCA improvements over sdSCA:")
        print(f"  {'-' * 40}")

        for label, key in metrics:
            sdsca_val = all_algo_results['sdSCA'][key]
            ql_val    = all_algo_results['qlsdSCA'][key]

            if abs(sdsca_val) > 1e-10:
                imp = ((sdsca_val - ql_val) /
                        abs(sdsca_val)) * 100
                symbol = '✅' if imp > 0 else '❌'
                print(f"  {label:<20} "
                      f"{imp:>+8.2f}% {symbol}")

    print(f"{'=' * 75}")


# ══════════════════════════════════════════════════════════════
# SAVE RESULTS
# ══════════════════════════════════════════════════════════════

def save_results(results, scenario_number, algorithm_name):
    """
    Save results to CSV file for later analysis.

    Parameters:
    -----------
    results         : dict
    scenario_number : int
    algorithm_name  : str
    """
    os.makedirs('results/scenarios', exist_ok=True)

    filepath = (f'results/scenarios/'
                f'scenario{scenario_number}_'
                f'{algorithm_name}.csv')

    df = pd.DataFrame({
        'run'         : range(1, results['num_runs'] + 1),
        'APDE'        : results['raw_apde'],
        'AUGD'        : results['raw_augd'],
        'total_fitness': results['raw_fitness'],
        'AET'         : results['raw_aet'],
    })

    df.to_csv(filepath, index=False)
    print(f"  Results saved: {filepath}")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    """Main function"""

    print("=" * 75)
    print("  qlsdSCA vs sdSCA vs SCA — Path Planning")
    print(f"  Runs: {NUM_RUNS} | "
          f"Max steps: {MAX_STEPS} | "
          f"Opt iterations: {OPT_ITERATIONS}")
    print("=" * 75)

    total_start = time.time()
    all_results = {}

    for scenario_num in SCENARIOS:

        print(f"\n{'#' * 75}")
        print(f"  SCENARIO {scenario_num}")
        print(f"{'#' * 75}")

        scenario_results = {}

        for algo_name in ALGORITHMS:
            results = run_multiple(
                scenario_num,
                algo_name,
                num_runs=NUM_RUNS
            )
            scenario_results[algo_name] = results
            save_results(results, scenario_num, algo_name)

        # Print comparison table with all algorithms
        print_results_table(scenario_results, scenario_num)
        all_results[scenario_num] = scenario_results

    total_time = time.time() - total_start
    print(f"\n{'=' * 75}")
    print(f"  All experiments complete!")
    print(f"  Total time: {total_time/60:.1f} minutes")
    print(f"  Results in: results/scenarios/")
    print(f"{'=' * 75}")

    return all_results


if __name__ == '__main__':
    main()
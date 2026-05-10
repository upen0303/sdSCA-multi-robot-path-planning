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
ALGORITHMS = ['SCA', 'sdSCA']

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

def print_results_table(results_sca, results_sdsca,
                        scenario_number):
    """
    Print comparison table like Tables 11-19 in paper.

    Parameters:
    -----------
    results_sca   : dict — averaged SCA results
    results_sdsca : dict — averaged sdSCA results
    scenario_number : int
    """
    print(f"\n{'=' * 65}")
    print(f"  SCENARIO {scenario_number} RESULTS "
          f"(averaged over {results_sca['num_runs']} runs)")
    print(f"{'=' * 65}")

    # ── Per robot steps table ──────────────────────────────
    NR = len(results_sca['avg_steps_per_robot'])
    print(f"\n  Average Required Steps per Robot:")
    print(f"  {'Robot':<12} {'SCA':>10} {'sdSCA':>10} "
          f"{'Improvement':>12}")
    print(f"  {'-'*46}")

    for i in range(NR):
        sca_steps   = results_sca['avg_steps_per_robot'][i]
        sdsca_steps = results_sdsca['avg_steps_per_robot'][i]
        imp = ((sca_steps - sdsca_steps) /
               max(sca_steps, 1e-10)) * 100
        print(f"  Robot #{i+1:<7} "
              f"{sca_steps:>10.1f} "
              f"{sdsca_steps:>10.1f} "
              f"{imp:>11.1f}%")

    # Totals
    sca_total   = results_sca['avg_total_steps']
    sdsca_total = results_sdsca['avg_total_steps']
    imp_total   = ((sca_total - sdsca_total) /
                   max(sca_total, 1e-10)) * 100
    print(f"  {'-'*46}")
    print(f"  {'Total':<12} "
          f"{sca_total:>10.1f} "
          f"{sdsca_total:>10.1f} "
          f"{imp_total:>11.1f}%")

    # ── Per robot distance table ───────────────────────────
    print(f"\n  Average Traveled Distance (cm) per Robot:")
    print(f"  {'Robot':<12} {'SCA':>10} {'sdSCA':>10} "
          f"{'Improvement':>12}")
    print(f"  {'-'*46}")

    for i in range(NR):
        sca_dist   = results_sca['avg_dist_per_robot'][i]
        sdsca_dist = results_sdsca['avg_dist_per_robot'][i]
        imp = ((sca_dist - sdsca_dist) /
               max(sca_dist, 1e-10)) * 100
        print(f"  Robot #{i+1:<7} "
              f"{sca_dist:>10.2f} "
              f"{sdsca_dist:>10.2f} "
              f"{imp:>11.1f}%")

    sca_dist_total   = results_sca['avg_total_distance']
    sdsca_dist_total = results_sdsca['avg_total_distance']
    imp_dist = ((sca_dist_total - sdsca_dist_total) /
                max(sca_dist_total, 1e-10)) * 100
    print(f"  {'-'*46}")
    print(f"  {'Total':<12} "
          f"{sca_dist_total:>10.2f} "
          f"{sdsca_dist_total:>10.2f} "
          f"{imp_dist:>11.1f}%")

    # ── Summary metrics table ──────────────────────────────
    print(f"\n  Summary Metrics:")
    print(f"  {'Metric':<20} {'SCA':>12} {'sdSCA':>12} "
          f"{'Improvement':>12}")
    print(f"  {'-'*58}")

    metrics = [
        ('APDE (cm)',
         results_sca['avg_APDE'],
         results_sdsca['avg_APDE']),
        ('AUGD (cm)',
         results_sca['avg_AUGD'],
         results_sdsca['avg_AUGD']),
        ('Total Fitness',
         results_sca['avg_total_fitness'],
         results_sdsca['avg_total_fitness']),
        ('AET (s)',
         results_sca['avg_AET'],
         results_sdsca['avg_AET']),
    ]

    for name, sca_val, sdsca_val in metrics:
        imp = ((sca_val - sdsca_val) /
               max(abs(sca_val), 1e-10)) * 100
        winner = "✅" if imp > 0 else "❌"
        print(f"  {name:<20} "
              f"{sca_val:>12.2f} "
              f"{sdsca_val:>12.2f} "
              f"{imp:>11.1f}% {winner}")

    print(f"{'=' * 65}")


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
    """
    Main function — run all scenarios and print results.
    """
    print("=" * 65)
    print("  sdSCA vs SCA — Path Planning Experiments")
    print(f"  Runs per experiment : {NUM_RUNS}")
    print(f"  Max steps           : {MAX_STEPS}")
    print(f"  Optimizer iterations: {OPT_ITERATIONS}")
    print("=" * 65)

    total_start = time.time()

    # Store all results
    all_results = {}

    for scenario_num in SCENARIOS:

        print(f"\n{'#' * 65}")
        print(f"  SCENARIO {scenario_num}")
        print(f"{'#' * 65}")

        scenario_results = {}

        for algo_name in ALGORITHMS:

            results = run_multiple(
                scenario_num,
                algo_name,
                num_runs=NUM_RUNS
            )

            scenario_results[algo_name] = results

            # Save to CSV
            save_results(results, scenario_num, algo_name)

        # Print comparison table
        print_results_table(
            scenario_results['SCA'],
            scenario_results['sdSCA'],
            scenario_num
        )

        all_results[scenario_num] = scenario_results

    # Final summary
    total_time = time.time() - total_start
    print(f"\n{'=' * 65}")
    print(f"  All experiments complete!")
    print(f"  Total time: {total_time/60:.1f} minutes")
    print(f"  Results saved in: results/scenarios/")
    print(f"{'=' * 65}")

    return all_results


if __name__ == '__main__':
    main()
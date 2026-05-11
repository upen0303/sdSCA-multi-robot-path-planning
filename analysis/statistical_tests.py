"""
Statistical Analysis for sdSCA vs SCA Comparison
==================================================
Based on standard practices in metaheuristic research papers.

Why statistical tests?
----------------------
Running an algorithm once and saying "it's better" is NOT
scientifically valid. Results vary due to randomness.
We need to PROVE the improvement is statistically significant
and not just due to luck.

Tests used:
-----------
1. Wilcoxon Rank-Sum Test
   - Non-parametric test (doesn't assume normal distribution)
   - Standard in metaheuristic comparison papers
   - p-value < 0.05 means improvement is statistically significant

2. Mean and Standard Deviation
   - Shows average performance and consistency
   - Low std = algorithm is stable (like paper's box plots)

3. Improvement Rate
   - Percentage improvement of sdSCA over SCA
   - Matches Tables 7-10 in paper

Author: Your Name
Date: 2024
"""

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, mannwhitneyu
import os
import sys
sys.path.append('.')


# ══════════════════════════════════════════════════════════════
# LOAD RESULTS
# ══════════════════════════════════════════════════════════════

def load_results(scenario_number, algorithm_name):
    """
    Load saved CSV results for one scenario and algorithm.

    Parameters:
    -----------
    scenario_number : int — 1, 2, or 3
    algorithm_name  : str — 'SCA' or 'sdSCA'

    Returns:
    --------
    pd.DataFrame : results dataframe
    """
    filepath = (f'results/scenarios/'
                f'scenario{scenario_number}_'
                f'{algorithm_name}.csv')

    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Results file not found: {filepath}\n"
            f"Run experiments/run_scenarios.py first!"
        )

    return pd.read_csv(filepath)


# ══════════════════════════════════════════════════════════════
# DESCRIPTIVE STATISTICS
# ══════════════════════════════════════════════════════════════

def descriptive_stats(data, algorithm_name, metric_name):
    """
    Calculate descriptive statistics for one metric.

    Parameters:
    -----------
    data           : list or np.array — raw values
    algorithm_name : str
    metric_name    : str

    Returns:
    --------
    dict : mean, std, min, max, median
    """
    data = np.array(data)

    stats = {
        'algorithm' : algorithm_name,
        'metric'    : metric_name,
        'mean'      : np.mean(data),
        'std'       : np.std(data),
        'min'       : np.min(data),
        'max'       : np.max(data),
        'median'    : np.median(data),
    }

    return stats


# ══════════════════════════════════════════════════════════════
# WILCOXON TEST
# ══════════════════════════════════════════════════════════════

def wilcoxon_test(data_sca, data_sdsca, metric_name):
    """
    Perform Wilcoxon Signed-Rank Test.

    Used when comparing two related samples (same scenario,
    same number of runs, paired comparison).

    H0 (null hypothesis)     : no difference between algorithms
    H1 (alternate hypothesis): sdSCA is significantly better

    p-value < 0.05 → reject H0 → sdSCA significantly better
    p-value ≥ 0.05 → cannot reject H0 → no significant difference

    Parameters:
    -----------
    data_sca   : list — SCA results across runs
    data_sdsca : list — sdSCA results across runs
    metric_name: str

    Returns:
    --------
    dict : test results
    """
    data_sca   = np.array(data_sca)
    data_sdsca = np.array(data_sdsca)

    # Need at least 10 samples for reliable Wilcoxon test
    # With only 3 runs, use Mann-Whitney U test instead
    if len(data_sca) < 10:
        # Mann-Whitney U — works with small samples
        stat, p_value = mannwhitneyu(
            data_sdsca, data_sca,
            alternative='less'   # sdSCA < SCA (lower is better)
        )
        test_name = 'Mann-Whitney U'
    else:
        # Wilcoxon signed-rank — standard for 30 runs
        try:
            stat, p_value = wilcoxon(
                data_sdsca, data_sca,
                alternative='less'
            )
            test_name = 'Wilcoxon'
        except ValueError:
            # All differences are zero — algorithms equal
            stat, p_value = 0, 1.0
            test_name = 'Wilcoxon'

    # Determine significance
    if p_value < 0.01:
        significance = '**'    # highly significant
        verdict      = 'sdSCA significantly better (p<0.01)'
    elif p_value < 0.05:
        significance = '*'     # significant
        verdict      = 'sdSCA significantly better (p<0.05)'
    else:
        significance = 'ns'    # not significant
        verdict      = 'No significant difference'

    return {
        'metric'       : metric_name,
        'test'         : test_name,
        'statistic'    : stat,
        'p_value'      : p_value,
        'significance' : significance,
        'verdict'      : verdict,
    }


# ══════════════════════════════════════════════════════════════
# IMPROVEMENT RATE
# ══════════════════════════════════════════════════════════════

def improvement_rate(mean_sca, mean_sdsca):
    """
    Calculate percentage improvement of sdSCA over SCA.
    Matches improvement rate formula used in paper Tables 7-10.

    improvement = (SCA - sdSCA) / SCA * 100

    Positive = sdSCA better
    Negative = SCA better

    Parameters:
    -----------
    mean_sca   : float
    mean_sdsca : float

    Returns:
    --------
    float : improvement percentage
    """
    if abs(mean_sca) < 1e-10:
        return 0.0

    return ((mean_sca - mean_sdsca) / abs(mean_sca)) * 100


# ══════════════════════════════════════════════════════════════
# FULL ANALYSIS FOR ONE SCENARIO
# ══════════════════════════════════════════════════════════════

def analyze_scenario(scenario_number):
    """
    Complete statistical analysis for one scenario.
    Compares SCA vs sdSCA on all metrics.

    Parameters:
    -----------
    scenario_number : int — 1, 2, or 3

    Returns:
    --------
    dict : complete analysis results
    """
    print(f"\n{'=' * 65}")
    print(f"  STATISTICAL ANALYSIS — SCENARIO {scenario_number}")
    print(f"{'=' * 65}")

    # Load results
    try:
        df_sca   = load_results(scenario_number, 'SCA')
        df_sdsca = load_results(scenario_number, 'sdSCA')
    except FileNotFoundError as e:
        print(f"  ❌ Error: {e}")
        return None

    num_runs = len(df_sca)
    print(f"  Runs per algorithm: {num_runs}")

    # Metrics to analyze
    metrics = ['APDE', 'AUGD', 'total_fitness', 'AET']

    all_results = {}

    for metric in metrics:

        sca_data   = df_sca[metric].values
        sdsca_data = df_sdsca[metric].values

        # Descriptive statistics
        stats_sca   = descriptive_stats(
            sca_data,   'SCA',   metric
        )
        stats_sdsca = descriptive_stats(
            sdsca_data, 'sdSCA', metric
        )

        # Statistical test
        test_result = wilcoxon_test(
            sca_data, sdsca_data, metric
        )

        # Improvement rate
        imp = improvement_rate(
            stats_sca['mean'],
            stats_sdsca['mean']
        )

        all_results[metric] = {
            'sca'         : stats_sca,
            'sdsca'       : stats_sdsca,
            'test'        : test_result,
            'improvement' : imp,
        }

        # Print results
        print(f"\n  [ {metric} ]")
        print(f"  {'Algorithm':<10} "
              f"{'Mean':>12} "
              f"{'Std':>12} "
              f"{'Min':>12} "
              f"{'Max':>12}")
        print(f"  {'-' * 60}")
        print(f"  {'SCA':<10} "
              f"{stats_sca['mean']:>12.4f} "
              f"{stats_sca['std']:>12.4f} "
              f"{stats_sca['min']:>12.4f} "
              f"{stats_sca['max']:>12.4f}")
        print(f"  {'sdSCA':<10} "
              f"{stats_sdsca['mean']:>12.4f} "
              f"{stats_sdsca['std']:>12.4f} "
              f"{stats_sdsca['min']:>12.4f} "
              f"{stats_sdsca['max']:>12.4f}")
        print(f"\n  Improvement    : {imp:+.2f}%"
              f"  {'✅' if imp > 0 else '❌'}")
        print(f"  Test           : "
              f"{test_result['test']}")
        print(f"  p-value        : "
              f"{test_result['p_value']:.4f} "
              f"[{test_result['significance']}]")
        print(f"  Verdict        : "
              f"{test_result['verdict']}")

    print(f"\n{'=' * 65}")
    return all_results


# ══════════════════════════════════════════════════════════════
# SUMMARY TABLE ACROSS ALL SCENARIOS
# ══════════════════════════════════════════════════════════════

def print_summary_table(all_scenario_results):
    """
    Print final summary table across all scenarios.
    Similar to what would appear in a research paper.

    Parameters:
    -----------
    all_scenario_results : dict
                           keys   = scenario numbers
                           values = analyze_scenario() results
    """
    print(f"\n{'=' * 75}")
    print(f"  SUMMARY TABLE — sdSCA vs SCA Improvement Rates")
    print(f"  (Positive = sdSCA better, Negative = SCA better)")
    print(f"{'=' * 75}")
    print(f"  {'Metric':<18} "
          f"{'Scenario 1':>14} "
          f"{'Scenario 2':>14} "
          f"{'Scenario 3':>14} "
          f"{'Average':>10}")
    print(f"  {'-' * 73}")

    metrics = ['APDE', 'AUGD', 'total_fitness', 'AET']
    metric_labels = {
        'APDE'         : 'APDE (cm)',
        'AUGD'         : 'AUGD (cm)',
        'total_fitness': 'Total Fitness',
        'AET'          : 'AET (s)',
    }

    for metric in metrics:
        improvements = []
        row = f"  {metric_labels[metric]:<18}"

        for scenario in [1, 2, 3]:
            if (all_scenario_results.get(scenario) and
                    metric in all_scenario_results[scenario]):
                imp = all_scenario_results[scenario][metric][
                    'improvement'
                ]
                improvements.append(imp)
                symbol = '✅' if imp > 0 else '❌'
                row += f" {imp:>12.1f}%{symbol}"
            else:
                row += f" {'N/A':>14}"

        if improvements:
            avg_imp = np.mean(improvements)
            symbol  = '✅' if avg_imp > 0 else '❌'
            row    += f" {avg_imp:>8.1f}%{symbol}"

        print(row)

    print(f"{'=' * 75}")

    # Significance summary
    print(f"\n  Statistical Significance Summary:")
    print(f"  ** = p<0.01 (highly significant)")
    print(f"  *  = p<0.05 (significant)")
    print(f"  ns = not significant")
    print(f"\n  {'Metric':<18} "
          f"{'Scenario 1':>14} "
          f"{'Scenario 2':>14} "
          f"{'Scenario 3':>14}")
    print(f"  {'-' * 62}")

    for metric in metrics:
        row = f"  {metric_labels[metric]:<18}"

        for scenario in [1, 2, 3]:
            if (all_scenario_results.get(scenario) and
                    metric in all_scenario_results[scenario]):
                sig = all_scenario_results[scenario][metric][
                    'test'
                ]['significance']
                row += f" {sig:>14}"
            else:
                row += f" {'N/A':>14}"

        print(row)

    print(f"{'=' * 75}")


# ══════════════════════════════════════════════════════════════
# SAVE STATISTICAL RESULTS
# ══════════════════════════════════════════════════════════════

def save_statistical_results(all_scenario_results):
    """
    Save complete statistical analysis to CSV.
    Useful for including in research paper tables.

    Parameters:
    -----------
    all_scenario_results : dict
    """
    os.makedirs('results/analysis', exist_ok=True)

    rows = []

    for scenario, results in all_scenario_results.items():
        if results is None:
            continue

        for metric, data in results.items():
            rows.append({
                'scenario'       : scenario,
                'metric'         : metric,
                'sca_mean'       : data['sca']['mean'],
                'sca_std'        : data['sca']['std'],
                'sdsca_mean'     : data['sdsca']['mean'],
                'sdsca_std'      : data['sdsca']['std'],
                'improvement_pct': data['improvement'],
                'p_value'        : data['test']['p_value'],
                'significance'   : data['test']['significance'],
                'verdict'        : data['test']['verdict'],
            })

    df = pd.DataFrame(rows)
    filepath = 'results/analysis/statistical_analysis.csv'
    df.to_csv(filepath, index=False)
    print(f"\n  Statistical results saved: {filepath}")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    """Run complete statistical analysis for all scenarios."""

    print("=" * 65)
    print("  Statistical Analysis — sdSCA vs SCA")
    print("=" * 65)
    print("  Note: Run experiments/run_scenarios.py first")
    print("        to generate results CSV files")

    all_scenario_results = {}

    for scenario in [1, 2, 3]:
        results = analyze_scenario(scenario)
        all_scenario_results[scenario] = results

    # Print summary table
    print_summary_table(all_scenario_results)

    # Save results
    save_statistical_results(all_scenario_results)

    print("\n✅ Statistical analysis complete!")
    print("   Results ready for research paper!")


if __name__ == '__main__':
    main()
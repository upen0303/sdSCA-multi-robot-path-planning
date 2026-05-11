"""
Compare SCA vs sdSCA vs qlsdSCA.

Strategy:
─────────
1. Benchmark functions — show qlsdSCA competitive
2. Focus on functions similar to CEC2015 paper uses
3. Main contribution proven in path planning scenarios

Author: Your Name
Date  : 2024
"""
import numpy as np
import sys
sys.path.append('.')

from algorithms.sca     import SCA
from algorithms.sdsca   import sdSCA
from algorithms.qlsdsca import qlsdSCA


# ── Test Functions ─────────────────────────────────────────

def rastrigin(x):
    """Multimodal — similar to f4 in CEC2015"""
    n = len(x)
    return 10*n + np.sum(x**2 - 10*np.cos(2*np.pi*x))

def rosenbrock(x):
    """Valley shaped — tests exploitation"""
    return np.sum(100*(x[1:]-x[:-1]**2)**2 + (1-x[:-1])**2)

def ackley(x):
    """Multimodal — similar to f3 in CEC2015"""
    n  = len(x)
    s1 = np.sum(x**2)
    s2 = np.sum(np.cos(2*np.pi*x))
    return (-20*np.exp(-0.2*np.sqrt(s1/n))
            - np.exp(s2/n) + 20 + np.e)

def griewank(x):
    """
    Multimodal with many widespread local optima.
    Good test for multi-strategy algorithms.
    Similar difficulty to CEC2015 hybrid functions.
    """
    n    = len(x)
    sum_ = np.sum(x**2) / 4000
    prod = np.prod(np.cos(x / np.sqrt(np.arange(1, n+1))))
    return sum_ - prod + 1

def levy(x):
    """
    Multimodal — complex landscape.
    Tests balance of exploration and exploitation.
    """
    w  = 1 + (x - 1) / 4
    w1 = w[0]
    wd = w[-1]
    term1 = np.sin(np.pi * w1)**2
    term2 = np.sum(
        (w[:-1]-1)**2 * (1 + 10*np.sin(np.pi*w[:-1]+1)**2)
    )
    term3 = (wd-1)**2 * (1 + np.sin(2*np.pi*wd)**2)
    return term1 + term2 + term3


# ── Functions with correct bounds ──────────────────────────
functions = [
    {"name": "Rastrigin", "func": rastrigin,
     "lb": -5.12, "ub": 5.12},
    {"name": "Rosenbrock", "func": rosenbrock,
     "lb": -30,   "ub": 30},
    {"name": "Griewank  ", "func": griewank,
     "lb": -600,  "ub": 600},
    {"name": "Levy      ", "func": levy,
     "lb": -10,   "ub": 10},
    {"name": "Ackley    ", "func": ackley,
     "lb": -32,   "ub": 32},
]

# ── Parameters ─────────────────────────────────────────────
DIM      = 30
ITERS    = 1000
POP_SIZE = 30
RUNS     = 5     # set to 30 for paper

# ── Results storage ────────────────────────────────────────
summary = {
    'sca_wins'   : 0,
    'sdsca_wins' : 0,
    'ql_wins'    : 0,
    'ties'       : 0,
}

print("=" * 72)
print("  SCA vs sdSCA vs qlsdSCA — Benchmark Comparison")
print(f"  D={DIM} | Iterations={ITERS} | Runs={RUNS}")
print("=" * 72)

for fn in functions:

    name = fn["name"]
    func = fn["func"]
    lb   = fn["lb"]
    ub   = fn["ub"]

    sca_results   = []
    sdsca_results = []
    ql_results    = []

    print(f"\n  Running {name}...", end=' ', flush=True)

    for run in range(RUNS):

        # SCA
        sca = SCA(POP_SIZE, ITERS, DIM, lb, ub, a=2)
        _, best, _ = sca.optimize(func)
        sca_results.append(best)

        # sdSCA
        sd = sdSCA(POP_SIZE, ITERS, DIM, lb, ub,
                   a=2, F=0.8, CR=0.95)
        _, best, _ = sd.optimize(func)
        sdsca_results.append(best)

        # qlsdSCA
        ql = qlsdSCA(POP_SIZE, ITERS, DIM, lb, ub,
                     a       = 2,
                     F       = 0.8,
                     CR      = 0.95,
                     alpha   = 0.3,
                     gamma   = 0.95,
                     epsilon = 1.0)
        _, best, _ = ql.optimize(func)
        ql_results.append(best)

        print(f"{run+1}", end=' ', flush=True)

    print()

    # ── Calculate statistics ───────────────────────────────
    sca_mean   = np.mean(sca_results)
    sdsca_mean = np.mean(sdsca_results)
    ql_mean    = np.mean(ql_results)

    sca_std    = np.std(sca_results)
    sdsca_std  = np.std(sdsca_results)
    ql_std     = np.std(ql_results)

    eps = 1e-10

    # qlsdSCA vs sdSCA improvement
    if abs(sdsca_mean) > eps:
        imp_vs_sdsca = (
            (sdsca_mean - ql_mean) / abs(sdsca_mean)
        ) * 100
    else:
        imp_vs_sdsca = 0.0

    # qlsdSCA vs SCA improvement
    if abs(sca_mean) > eps:
        imp_vs_sca = (
            (sca_mean - ql_mean) / abs(sca_mean)
        ) * 100
    else:
        imp_vs_sca = 0.0

    # Determine winner
    best_mean = min(sca_mean, sdsca_mean, ql_mean)
    threshold = 1e-6   # tie if within threshold

    if abs(ql_mean - best_mean) < threshold:
        if (abs(sca_mean - best_mean) < threshold and
                abs(sdsca_mean - best_mean) < threshold):
            winner = "All Tie 🤝"
            summary['ties'] += 1
        elif ql_mean <= sdsca_mean and ql_mean <= sca_mean:
            winner = "qlsdSCA 🏆"
            summary['ql_wins'] += 1
        else:
            winner = "Tie 🤝"
            summary['ties'] += 1
    elif ql_mean == best_mean:
        winner = "qlsdSCA 🏆"
        summary['ql_wins'] += 1
    elif sdsca_mean == best_mean:
        winner = "sdSCA ✅"
        summary['sdsca_wins'] += 1
    else:
        winner = "SCA"
        summary['sca_wins'] += 1

    # ── Print results ──────────────────────────────────────
    print(f"\n  [ {name} ]")
    print(f"  {'Algorithm':<12} {'Mean':>15} "
          f"{'Std':>12} {'Rank':>6}")
    print(f"  {'-'*48}")

    # Rank algorithms
    ranked = sorted([
        ('SCA',     sca_mean),
        ('sdSCA',   sdsca_mean),
        ('qlsdSCA', ql_mean),
    ], key=lambda x: x[1])

    ranks = {name: i+1 for i, (name, _) in enumerate(ranked)}

    print(f"  {'SCA':<12} {sca_mean:>15.6f} "
          f"{sca_std:>12.6f} {ranks['SCA']:>6}")
    print(f"  {'sdSCA':<12} {sdsca_mean:>15.6f} "
          f"{sdsca_std:>12.6f} {ranks['sdSCA']:>6}")
    print(f"  {'qlsdSCA':<12} {ql_mean:>15.6f} "
          f"{ql_std:>12.6f} {ranks['qlsdSCA']:>6}")
    print(f"\n  Winner              : {winner}")
    print(f"  qlsdSCA vs sdSCA    : {imp_vs_sdsca:+.2f}%")
    print(f"  qlsdSCA vs SCA      : {imp_vs_sca:+.2f}%")

# ── Summary ────────────────────────────────────────────────
print(f"\n{'=' * 72}")
print(f"  OVERALL SUMMARY ({RUNS} runs each)")
print(f"{'=' * 72}")
print(f"  SCA    wins : {summary['sca_wins']}/{len(functions)}")
print(f"  sdSCA  wins : {summary['sdsca_wins']}/{len(functions)}")
print(f"  qlsdSCA wins: {summary['ql_wins']}/{len(functions)}")
print(f"  Ties        : {summary['ties']}/{len(functions)}")

# ── Q-Table Analysis ───────────────────────────────────────
print(f"\n{'=' * 72}")
print(f"  Q-Table Analysis (last qlsdSCA run — {name})")
print(f"  Shows what strategy agent prefers in each state")
print(f"{'=' * 72}")

state_names = [
    'early+stuck', 'early+slow ', 'early+fast ',
    'mid+stuck  ', 'mid+slow   ', 'mid+fast   ',
    'late+stuck ', 'late+slow  ', 'late+fast  ',
]
strategy_names = ['S1(SCA)', 'S2(rand)', 'S3(best)', 'S4(rand2)']

print(f"\n  {'State':<14} {'S1(SCA)':>9} "
      f"{'S2(rand)':>9} "
      f"{'S3(best)':>9} "
      f"{'S4(rand2)':>10} "
      f"{'Best':>10}")
print(f"  {'-' * 64}")

for s in range(9):
    q_vals    = ql.agent.Q[s]
    best_idx  = np.argmax(q_vals)
    preferred = strategy_names[best_idx]
    print(f"  {state_names[s]:<14} "
          f"{q_vals[0]:>9.4f} "
          f"{q_vals[1]:>9.4f} "
          f"{q_vals[2]:>9.4f} "
          f"{q_vals[3]:>10.4f} "
          f"{preferred:>10}")

# ── Adaptive parameters analysis ───────────────────────────
print(f"\n  Adaptive F and CR (last run means):")
print(f"  {'Strategy':<12} {'Mean F':>10} {'Mean CR':>10}")
print(f"  {'-'*34}")
for stat in ql.adaptive_params.get_stats():
    print(f"  {'S'+str(stat['strategy']):<12} "
          f"{stat['mean_F']:>10.4f} "
          f"{stat['mean_CR']:>10.4f}")

print(f"\n  Final epsilon : {ql.agent.epsilon:.4f}")
print(f"  Total rewards : {len(ql.agent.reward_history)}")
if ql.agent.reward_history:
    print(f"  Mean reward   : "
          f"{np.mean(ql.agent.reward_history):.4f}")
    print(f"  Positive rewards: "
          f"{sum(1 for r in ql.agent.reward_history if r > 0)}"
          f"/{len(ql.agent.reward_history)}")

print("=" * 72)
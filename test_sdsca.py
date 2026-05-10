"""
Compare Basic SCA vs sdSCA on multiple functions.
==================================================
Tests both algorithms on 4 standard benchmark functions
with correct bounds for each function.

Author: Your Name
Date: 2024
"""

import numpy as np
import sys
sys.path.append('.')

from algorithms.sca   import SCA
from algorithms.sdsca import sdSCA


# ── Test Functions ─────────────────────────────────────────────────────────────

def sphere_function(x):
    """
    Simple unimodal function.
    Minimum = 0 at x=[0,0,...,0]
    Bounds: [-100, 100]
    """
    return np.sum(x ** 2)


def rastrigin_function(x):
    """
    Complex multimodal — many local optima.
    Minimum = 0 at x=[0,0,...,0]
    Similar to f4 in CEC2015 (paper Table 2)
    Bounds: [-5.12, 5.12]
    """
    n = len(x)
    return 10 * n + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))


def rosenbrock_function(x):
    """
    Valley-shaped function — hard to optimize.
    Minimum = 0 at x=[1,1,...,1]
    Tests exploitation ability.
    Bounds: [-30, 30]
    """
    return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)


def ackley_function(x):
    """
    Many local optima with one global optimum.
    Similar to f3 in CEC2015.
    Minimum = 0 at x=[0,0,...,0]
    Bounds: [-32, 32]
    """
    n    = len(x)
    sum1 = np.sum(x**2)
    sum2 = np.sum(np.cos(2 * np.pi * x))
    return (-20 * np.exp(-0.2 * np.sqrt(sum1 / n))
            - np.exp(sum2 / n)
            + 20 + np.e)


# ── Test Functions List with Correct Bounds ────────────────────────────────────

functions = [
    {
        "name" : "Sphere    ",
        "func" : sphere_function,
        "lb"   : -100,
        "ub"   : 100,
    },
    {
        "name" : "Rastrigin ",
        "func" : rastrigin_function,
        "lb"   : -5.12,    # correct bound for Rastrigin
        "ub"   : 5.12,
    },
    {
        "name" : "Rosenbrock",
        "func" : rosenbrock_function,
        "lb"   : -30,
        "ub"   : 30,
    },
    {
        "name" : "Ackley    ",
        "func" : ackley_function,
        "lb"   : -32,
        "ub"   : 32,
    },
]


# ── Run Comparison ─────────────────────────────────────────────────────────────

print("=" * 65)
print("  SCA vs sdSCA Comparison (D=30, 1000 iterations)")
print("=" * 65)
print(f"  {'Function':<12} {'SCA':>15} {'sdSCA':>15} {'Winner':>12}")
print("-" * 65)

for fn in functions:

    name = fn["name"]
    func = fn["func"]
    lb   = fn["lb"]
    ub   = fn["ub"]

    # Build params with correct bounds for this function
    params = dict(
        population_size = 30,
        max_iterations  = 1000,
        dim             = 30,
        lower_bound     = lb,
        upper_bound     = ub,
    )

    # ── Run Basic SCA ──────────────────────────────────────
    sca = SCA(**params, a=2)
    _, sca_best, _ = sca.optimize(func)

    # ── Run sdSCA ──────────────────────────────────────────
    sdsca = sdSCA(**params, a=2, F=0.8, CR=0.95)
    _, sdsca_best, sdsca_curve = sdsca.optimize(func)

    # ── Determine Winner Safely ────────────────────────────
    eps = 1e-10   # tiny number to avoid division by zero

    if abs(sca_best) < eps and abs(sdsca_best) < eps:
        winner      = "Tie"
        improvement = "Both≈0"

    elif abs(sca_best) < eps and sdsca_best > eps:
        winner      = "SCA ✅"
        improvement = "SCA=0"

    elif abs(sdsca_best) < eps and sca_best > eps:
        winner      = "sdSCA ✅"
        improvement = "sdSCA=0"

    else:
        pct = ((sca_best - sdsca_best) / abs(sca_best)) * 100
        if pct > 1.0:
            winner      = "sdSCA ✅"
            improvement = f"+{pct:.1f}%"
        elif pct < -1.0:
            winner      = "SCA ✅"
            improvement = f"{pct:.1f}%"
        else:
            winner      = "Tie"
            improvement = f"{abs(pct):.1f}%"

    print(f"  {name:<12} {sca_best:>15.6f} "
          f"{sdsca_best:>15.6f} "
          f"{winner:>12}  ({improvement})")

# ── Final Summary ──────────────────────────────────────────────────────────────

print("=" * 65)
print("\nFinal Strategy Probabilities (sdSCA last run):")
print(f"  S1(SCA)      = {sdsca.probabilities[0]:.3f}")
print(f"  S2(DE/rand)  = {sdsca.probabilities[1]:.3f}")
print(f"  S3(DE/best)  = {sdsca.probabilities[2]:.3f}")
print(f"  S4(DE/rand2) = {sdsca.probabilities[3]:.3f}")
print("=" * 65)
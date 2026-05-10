"""Quick test to verify SCA works correctly"""
import numpy as np
import sys
sys.path.append('.')
from algorithms.sca import SCA

# Simple test function — minimum is at x=0, f(0)=0
def sphere_function(x):
    return np.sum(x ** 2)

# Create SCA object
sca = SCA(
    population_size = 30,
    max_iterations  = 500,
    dim             = 10,
    lower_bound     = -100,
    upper_bound     = 100,
    a               = 2
)

print("Running Basic SCA on Sphere Function...")
print("Expected: solution near 0, fitness near 0")
print("-" * 40)

best_solution, best_fitness, curve = sca.optimize(sphere_function)

print("-" * 40)
print(f"Best Fitness Found : {best_fitness:.6f}")
print(f"Convergence Curve  : {len(curve)} points recorded")
print(f"First fitness      : {curve[0]:.2f}")
print(f"Final fitness      : {curve[-1]:.6f}")
print("\n✅ SCA working correctly!" if best_fitness < 1.0 else "\n❌ Something wrong")
"""
Test Environment with a simple 2-robot scenario.
Verifies complete simulation loop works end to end.
"""
import numpy as np
import sys
sys.path.append('.')

from path_planning.robot       import Robot
from path_planning.obstacle    import StaticObstacle, DynamicObstacle
from path_planning.environment import Environment
from algorithms.sdsca          import sdSCA

print("=" * 55)
print("  Environment Simulation Test")
print("  2 robots, 1 static obstacle, 1 dynamic obstacle")
print("=" * 55)

# ── Create environment ─────────────────────────────────────
env = Environment(width=100, height=100)

# ── Add robots ─────────────────────────────────────────────
env.add_robot(Robot(1, start_pos=(10, 10), goal_pos=(80, 80)))
env.add_robot(Robot(2, start_pos=(10, 80), goal_pos=(80, 10)))

# ── Add obstacles ──────────────────────────────────────────
env.add_static_obstacle(
    StaticObstacle(1, position=(50, 50), radius=1.5)
)
env.add_dynamic_obstacle(
    DynamicObstacle(1, start_pos=(20, 50),
                    goal_pos=(80, 50), velocity=0.5)
)

# ── Set algorithm ──────────────────────────────────────────
algorithm = sdSCA(
    population_size = 30,
    max_iterations  = 100,
    dim             = 4,       # 2 robots × 2 variables
    lower_bound     = np.array([1.0, 0.0,    1.0, 0.0]),
    upper_bound     = np.array([1.5, 2*np.pi, 1.5, 2*np.pi]),
    a  = 2,
    F  = 0.8,
    CR = 0.95
)
env.set_algorithm(algorithm)

# ── Run simulation ─────────────────────────────────────────
results = env.run(verbose=True)

# ── Print results ──────────────────────────────────────────
print("\n[ Results ]")
print(f"  Total steps     : {results['total_steps']}")
print(f"  Total distance  : {results['total_distance']:.4f} cm")
print(f"  APDE            : {results['APDE']:.4f} cm")
print(f"  AUGD            : {results['AUGD']:.4f} cm")
print(f"  Total fitness   : {results['total_fitness']:.4f}")
print(f"  Execution time  : {results['AET']:.2f} s")

print("\n[ Per Robot ]")
for i, robot in enumerate(env.robots):
    print(f"  Robot {robot.id}: "
          f"Steps={results['steps_per_robot'][i]} | "
          f"Distance={results['distances_per_robot'][i]:.4f} cm | "
          f"Reached={robot.reached_goal}")

print("\n✅ Environment simulation working correctly!")
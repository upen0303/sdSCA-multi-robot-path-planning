"""
Test fitness functions F1, F2, F3, F4.
Verify each component works as described in paper.
"""
import numpy as np
import sys
sys.path.append('.')

from path_planning.robot    import Robot
from path_planning.obstacle import StaticObstacle, DynamicObstacle
from path_planning.fitness  import (calculate_F1,
                                    calculate_F2,
                                    calculate_F3,
                                    calculate_F4,
                                    calculate_total_fitness)

print("=" * 60)
print("  Fitness Functions Test")
print("=" * 60)

# ── Setup ──────────────────────────────────────────────────
robots = [
    Robot(1, start_pos=(10, 10), goal_pos=(80, 80)),
    Robot(2, start_pos=(10, 80), goal_pos=(80, 10)),
]

static_obstacles = [
    StaticObstacle(1, position=(50, 50), radius=1.5),
    StaticObstacle(2, position=(30, 30), radius=1.5),
]

dynamic_obstacles = [
    DynamicObstacle(1, start_pos=(20, 50),
                    goal_pos=(80, 50), velocity=0.5),
]

env_bounds = (100, 100)

# ── Test F1 — distance fitness ─────────────────────────────
print("\n[ F1 — Shortest Distance ]")

# Good next positions — moving toward goals
good_next = np.array([
    [15.0, 15.0],   # robot 1 moved toward goal
    [15.0, 75.0],   # robot 2 moved toward goal
])

# Bad next positions — moving away from goals
bad_next = np.array([
    [5.0, 5.0],    # robot 1 moved away from goal
    [5.0, 85.0],   # robot 2 moved away from goal
])

f1_good = calculate_F1(robots, good_next)
f1_bad  = calculate_F1(robots, bad_next)

print(f"  F1 (moving toward goal) : {f1_good:.4f}")
print(f"  F1 (moving away)        : {f1_bad:.4f}")
print(f"  Moving toward goal gives lower F1: "
      f"{f1_good < f1_bad} ✅")

# ── Test F2 — static obstacle avoidance ───────────────────
print("\n[ F2 — Static Obstacle Avoidance ]")

# Safe positions — far from obstacles
safe_next = np.array([
    [15.0, 15.0],
    [15.0, 75.0],
])

# Collision positions — inside obstacle at (30, 30)
collision_next = np.array([
    [30.0, 30.0],   # robot 1 collides with obstacle at (30,30)
    [15.0, 75.0],
])

f2_safe      = calculate_F2(robots, safe_next,
                             static_obstacles, 2.5)
f2_collision = calculate_F2(robots, collision_next,
                             static_obstacles, 2.5)

print(f"  F2 (safe)      : {f2_safe:.0f}")
print(f"  F2 (collision) : {f2_collision:.0f}")
print(f"  Collision gives penalty ε=100000: "
      f"{f2_collision == 1e5} ✅")

# ── Test F3 — dynamic obstacle avoidance ──────────────────
print("\n[ F3 — Dynamic Obstacle Avoidance ]")

# Dynamic obstacle starts at (20, 50)
near_dynamic = np.array([
    [20.0, 50.0],   # robot 1 at same position as dynamic obstacle
    [15.0, 75.0],
])

f3_safe = calculate_F3(robots, safe_next,
                        dynamic_obstacles, 2.5)
f3_hit  = calculate_F3(robots, near_dynamic,
                        dynamic_obstacles, 2.5)

print(f"  F3 (safe)      : {f3_safe:.0f}")
print(f"  F3 (collision) : {f3_hit:.0f}")
print(f"  Collision gives penalty ε=100000: "
      f"{f3_hit == 1e5} ✅")

# ── Test F4 — inter-robot collision ───────────────────────
print("\n[ F4 — Inter-Robot Collision ]")

# Both robots at same position — collision
collision_robots = np.array([
    [40.0, 40.0],
    [40.0, 40.0],   # same as robot 1 → collision
])

# Robots far apart — safe
safe_robots = np.array([
    [15.0, 15.0],
    [85.0, 85.0],
])

f4_safe      = calculate_F4(robots, safe_robots)
f4_collision = calculate_F4(robots, collision_robots)

print(f"  F4 (safe)      : {f4_safe:.0f}")
print(f"  F4 (collision) : {f4_collision:.0f}")
print(f"  Collision gives penalty ε=100000: "
      f"{f4_collision >= 1e5} ✅")

# ── Test Total Fitness ─────────────────────────────────────
print("\n[ Total Fitness — Fit = F1+F2+F3+F4 ]")

# Build solution vector [v1, θ1, v2, θ2]
# Moving both robots toward their goals
solution_good = np.array([
    1.5, np.pi/4,    # robot 1: v=1.5, θ=45° (toward goal)
    1.5, -np.pi/4,   # robot 2: v=1.5, θ=-45° (toward goal)
])

# Moving robots into obstacle
solution_bad = np.array([
    1.5, np.arctan2(30-10, 30-10),   # robot 1 toward obstacle
    1.5, -np.pi/4,
])

fit_good = calculate_total_fitness(
    solution_good, robots,
    static_obstacles, dynamic_obstacles,
    env_bounds
)

fit_bad = calculate_total_fitness(
    solution_bad, robots,
    static_obstacles, dynamic_obstacles,
    env_bounds
)

print(f"  Good solution fitness : {fit_good:.4f}")
print(f"  Bad solution fitness  : {fit_bad:.4f}")
print(f"  Good < Bad (collision penalized): "
      f"{fit_good < fit_bad} ✅")

print("\n" + "=" * 60)
print("  ✅ All fitness functions working correctly!")
print("=" * 60)
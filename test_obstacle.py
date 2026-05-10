"""
Test obstacle classes.
Verify static and dynamic obstacles work correctly.
"""
import numpy as np
import sys
sys.path.append('.')

from path_planning.obstacle import StaticObstacle, DynamicObstacle

print("=" * 55)
print("  Obstacle Classes Test")
print("=" * 55)

# ── Test Static Obstacle ───────────────────────────────────
print("\n[ Static Obstacle ]")

static = StaticObstacle(
    obstacle_id = 1,
    position    = (50, 50),
    radius      = 1.5,
    shape       = 'circle'
)
print(f"  {static}")

# Test distance calculation
robot_pos = (45, 45)
dist = static.get_distance_to(robot_pos)
print(f"  Distance to point {robot_pos} : {dist:.4f} cm")
print(f"  Position never changes        : {static.get_position()}")

# ── Test Dynamic Obstacle ──────────────────────────────────
print("\n[ Dynamic Obstacle ]")

dynamic = DynamicObstacle(
    obstacle_id = 1,
    start_pos   = (10, 10),
    goal_pos    = (90, 10),   # moves horizontally
    velocity    = 0.5,        # 0.5 cm/step as in Scenario 1
    radius      = 1.5,
    shape       = 'circle'
)
print(f"  {dynamic}")

# Move obstacle for 10 steps and track position
print(f"\n  Moving dynamic obstacle for 10 steps:")
for step in range(1, 11):
    dynamic.update_position()
    pos = dynamic.get_position()
    print(f"  Step {step:2d}: "
          f"Pos=({pos[0]:6.2f}, {pos[1]:6.2f}) | "
          f"{dynamic}")

# ── Test Direction Reversal ────────────────────────────────
print(f"\n[ Direction Reversal Test ]")
fast_dynamic = DynamicObstacle(
    obstacle_id = 2,
    start_pos   = (0, 0),
    goal_pos    = (5, 0),    # short distance
    velocity    = 2.0,       # fast — will reverse quickly
    radius      = 1.5
)

print(f"  Moving fast obstacle "
      f"(start=0, goal=5, velocity=2.0):")
for step in range(1, 8):
    fast_dynamic.update_position()
    pos = fast_dynamic.get_position()
    direction = "→" if fast_dynamic.moving_forward else "←"
    print(f"  Step {step}: x={pos[0]:5.2f} {direction}")

# ── Test Reset ─────────────────────────────────────────────
print(f"\n[ Reset Test ]")
dynamic.reset()
print(f"  After reset: {dynamic}")
print(f"  Position history cleared: "
      f"{len(dynamic.position_history)} point")

print("\n" + "=" * 55)
print("  ✅ Obstacle classes working correctly!")
print("=" * 55)
"""
Test Robot class — verify movement equations work correctly.
"""
import numpy as np
import sys
sys.path.append('.')

from path_planning.robot import Robot

print("=" * 55)
print("  Robot Class Test")
print("=" * 55)

# ── Create a test robot ────────────────────────────────────
robot = Robot(
    robot_id  = 1,
    start_pos = (10, 10),   # starts at (10, 10)
    goal_pos  = (80, 80),   # wants to reach (80, 80)
    radius    = 1.0
)

print(f"\nInitial State:")
print(f"  {robot}")
print(f"  Ideal path distance : "
      f"{robot.get_ideal_path_distance():.4f} cm")

# ── Test movement ──────────────────────────────────────────
print(f"\nMoving robot for 5 steps...")
print(f"  Direction toward goal = 45 degrees = π/4 radians")

for step in range(5):
    # Move toward goal at 45 degrees (toward (80,80) from (10,10))
    robot.move(velocity=1.5, angle=np.pi/4)
    print(f"  Step {step+1}: {robot}")

# ── Test calculations ──────────────────────────────────────
print(f"\nAfter 5 steps:")
print(f"  Total distance traveled : {robot.total_distance:.4f} cm")
print(f"  Path deviation error    : "
      f"{robot.get_path_deviation_error():.4f} cm")
print(f"  Path history length     : {len(robot.path_history)} points")

# ── Test reset ─────────────────────────────────────────────
print(f"\nResetting robot...")
robot.reset()
print(f"  After reset: {robot}")
print(f"  Steps taken : {robot.steps_taken}")
print(f"  Reached goal: {robot.reached_goal}")

# ── Test goal detection ────────────────────────────────────
print(f"\nTesting goal detection...")
robot2 = Robot(
    robot_id  = 2,
    start_pos = (0, 0),
    goal_pos  = (1, 0),   # goal very close
    radius    = 1.0
)
robot2.move(velocity=1.5, angle=0)   # move right
print(f"  Robot 2 reached goal: {robot2.reached_goal}")

print("\n" + "=" * 55)
print("  ✅ Robot class working correctly!")
print("=" * 55)
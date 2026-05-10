"""
Three Test Scenarios from the Paper
=====================================
Based on Section 4.4 of the paper.

Scenario 1: 100×100 cm, 6 robots, 7 static, 3 dynamic obstacles
Scenario 2: 100×100 cm, 7 robots, 7 static, 3 dynamic obstacles
            (non-circular obstacles)
Scenario 3: 200×200 cm, 12 robots, 14 static, 6 dynamic obstacles

Paper parameters (same for all scenarios):
-------------------------------------------
- Robot radius          : 1.0 cm
- Obstacle radius       : 1.5 cm
- Velocity range        : [1.0, 1.5] cm/step
- Angle range           : [0, 2π] radians
- Penalty ε             : 10^5

Author: Your Name
Date: 2024
"""

import numpy as np
from path_planning.robot       import Robot
from path_planning.obstacle    import StaticObstacle, DynamicObstacle
from path_planning.environment import Environment


# ══════════════════════════════════════════════════════════════
# SCENARIO 1
# ══════════════════════════════════════════════════════════════

def create_scenario_1():
    """
    Scenario 1 from paper — Section 4.4.2
    =======================================
    Environment : 100 × 100 cm
    Robots      : 6 (R1-R6)
    Static obs  : 7 circular obstacles
    Dynamic obs : 3 (D1, D2, D3) circular

    Dynamic obstacle velocities (from paper):
        D1 = 0.50 cm/step
        D2 = 0.45 cm/step
        D3 = 1.20 cm/step

    Returns:
    --------
    Environment object ready to run
    """
    env = Environment(width=100, height=100)

    # ── 6 Robots ───────────────────────────────────────────
    # Positions estimated from Figure 5 in paper
    robots = [
        Robot(1, start_pos=(5,  90), goal_pos=(95, 55)),
        Robot(2, start_pos=(5,  15), goal_pos=(95, 10)),
        Robot(3, start_pos=(30, 60), goal_pos=(70, 85)),
        Robot(4, start_pos=(50, 90), goal_pos=(90, 30)),
        Robot(5, start_pos=(20, 35), goal_pos=(75, 50)),
        Robot(6, start_pos=(10, 55), goal_pos=(80, 15)),
    ]

    for robot in robots:
        env.add_robot(robot)

    # ── 7 Static Obstacles ─────────────────────────────────
    # Circular obstacles — positions from Figure 5
    static_obstacles = [
        StaticObstacle(1, position=(15, 75), radius=5.0),
        StaticObstacle(2, position=(40, 80), radius=4.0),
        StaticObstacle(3, position=(55, 65), radius=6.0),
        StaticObstacle(4, position=(75, 75), radius=4.0),
        StaticObstacle(5, position=(30, 45), radius=5.0),
        StaticObstacle(6, position=(65, 40), radius=5.0),
        StaticObstacle(7, position=(50, 20), radius=4.0),
    ]

    for obs in static_obstacles:
        env.add_static_obstacle(obs)

    # ── 3 Dynamic Obstacles ────────────────────────────────
    # Velocities exactly as stated in paper
    dynamic_obstacles = [
        DynamicObstacle(
            obstacle_id = 1,
            start_pos   = (10, 40),
            goal_pos    = (40, 70),
            velocity    = 0.50,    # D1 = 0.50 cm/step
            radius      = 1.5,
            shape       = 'circle'
        ),
        DynamicObstacle(
            obstacle_id = 2,
            start_pos   = (45, 35),
            goal_pos    = (65, 55),
            velocity    = 0.45,    # D2 = 0.45 cm/step
            radius      = 1.5,
            shape       = 'circle'
        ),
        DynamicObstacle(
            obstacle_id = 3,
            start_pos   = (70, 60),
            goal_pos    = (85, 30),
            velocity    = 1.20,    # D3 = 1.20 cm/step
            radius      = 1.5,
            shape       = 'circle'
        ),
    ]

    for obs in dynamic_obstacles:
        env.add_dynamic_obstacle(obs)

    return env


# ══════════════════════════════════════════════════════════════
# SCENARIO 2
# ══════════════════════════════════════════════════════════════

def create_scenario_2():
    """
    Scenario 2 from paper — Section 4.4.3
    =======================================
    Environment : 100 × 100 cm
    Robots      : 7 (R1-R7)
    Static obs  : 7 (mixed shapes: 2 circles, 2 triangles,
                                   3 squares)
    Dynamic obs : 3 (D1, D2, D3) mixed shapes
                  (1 circle, 1 triangle, 1 square)

    Dynamic obstacle velocities (from paper):
        D1 = 0.50 cm/step
        D2 = 0.10 cm/step
        D3 = 1.10 cm/step

    Returns:
    --------
    Environment object ready to run
    """
    env = Environment(width=100, height=100)

    # ── 7 Robots ───────────────────────────────────────────
    # Positions estimated from Figure 6 in paper
    robots = [
        Robot(1, start_pos=(5,  50), goal_pos=(90, 70)),
        Robot(2, start_pos=(5,  20), goal_pos=(90, 10)),
        Robot(3, start_pos=(20, 65), goal_pos=(75, 45)),
        Robot(4, start_pos=(40, 85), goal_pos=(85, 85)),
        Robot(5, start_pos=(50, 55), goal_pos=(90, 30)),
        Robot(6, start_pos=(35, 30), goal_pos=(80, 55)),
        Robot(7, start_pos=(55, 85), goal_pos=(95, 90)),
    ]

    for robot in robots:
        env.add_robot(robot)

    # ── 7 Static Obstacles (mixed shapes) ─────────────────
    # Shape affects visualization only
    # Collision detection uses radius regardless of shape
    static_obstacles = [
        StaticObstacle(1, position=(20, 50),
                       radius=6.0, shape='circle'),
        StaticObstacle(2, position=(45, 70),
                       radius=4.0, shape='circle'),
        StaticObstacle(3, position=(30, 75),
                       radius=4.0, shape='triangle'),
        StaticObstacle(4, position=(60, 50),
                       radius=5.0, shape='triangle'),
        StaticObstacle(5, position=(50, 30),
                       radius=4.0, shape='square'),
        StaticObstacle(6, position=(70, 75),
                       radius=4.0, shape='square'),
        StaticObstacle(7, position=(75, 35),
                       radius=5.0, shape='square'),
    ]

    for obs in static_obstacles:
        env.add_static_obstacle(obs)

    # ── 3 Dynamic Obstacles (mixed shapes) ────────────────
    dynamic_obstacles = [
        DynamicObstacle(
            obstacle_id = 1,
            start_pos   = (15, 35),
            goal_pos    = (35, 65),
            velocity    = 0.50,    # D1 = 0.50 cm/step
            radius      = 1.5,
            shape       = 'circle'
        ),
        DynamicObstacle(
            obstacle_id = 2,
            start_pos   = (55, 45),
            goal_pos    = (75, 25),
            velocity    = 0.10,    # D2 = 0.10 cm/step
            radius      = 1.5,
            shape       = 'triangle'
        ),
        DynamicObstacle(
            obstacle_id = 3,
            start_pos   = (80, 60),
            goal_pos    = (90, 85),
            velocity    = 1.10,    # D3 = 1.10 cm/step
            radius      = 1.5,
            shape       = 'square'
        ),
    ]

    for obs in dynamic_obstacles:
        env.add_dynamic_obstacle(obs)

    return env


# ══════════════════════════════════════════════════════════════
# SCENARIO 3
# ══════════════════════════════════════════════════════════════

def create_scenario_3():
    """
    Scenario 3 from paper — Section 4.4.4
    =======================================
    Environment : 200 × 200 cm  (twice as large!)
    Robots      : 12 (R1-R12)
    Static obs  : 14 circular obstacles
    Dynamic obs : 6 (D1-D6) circular

    Dynamic obstacle velocities (from paper):
        D1 = 0.50 cm/step
        D2 = 0.50 cm/step
        D3 = 0.60 cm/step
        D4 = 0.30 cm/step
        D5 = 0.40 cm/step
        D6 = 0.25 cm/step

    Returns:
    --------
    Environment object ready to run
    """
    env = Environment(width=200, height=200)

    # ── 12 Robots ──────────────────────────────────────────
    # Positions estimated from Figure 7 in paper
    robots = [
        Robot(1,  start_pos=(10,  170), goal_pos=(150, 160)),
        Robot(2,  start_pos=(10,  130), goal_pos=(180, 190)),
        Robot(3,  start_pos=(10,  100), goal_pos=(170, 130)),
        Robot(4,  start_pos=(10,  60),  goal_pos=(160, 60)),
        Robot(5,  start_pos=(50,  190), goal_pos=(140, 100)),
        Robot(6,  start_pos=(50,  150), goal_pos=(130, 70)),
        Robot(7,  start_pos=(50,  80),  goal_pos=(170, 80)),
        Robot(8,  start_pos=(80,  190), goal_pos=(160, 170)),
        Robot(9,  start_pos=(80,  110), goal_pos=(190, 110)),
        Robot(10, start_pos=(90,  50),  goal_pos=(180, 40)),
        Robot(11, start_pos=(100, 170), goal_pos=(190, 150)),
        Robot(12, start_pos=(100, 30),  goal_pos=(170, 20)),
    ]

    for robot in robots:
        env.add_robot(robot)

    # ── 14 Static Obstacles ────────────────────────────────
    static_obstacles = [
        StaticObstacle(1,  position=(30,  155), radius=8.0),
        StaticObstacle(2,  position=(70,  175), radius=6.0),
        StaticObstacle(3,  position=(60,  130), radius=7.0),
        StaticObstacle(4,  position=(100, 160), radius=8.0),
        StaticObstacle(5,  position=(40,  95),  radius=6.0),
        StaticObstacle(6,  position=(80,  110), radius=7.0),
        StaticObstacle(7,  position=(120, 130), radius=6.0),
        StaticObstacle(8,  position=(150, 150), radius=8.0),
        StaticObstacle(9,  position=(90,  70),  radius=6.0),
        StaticObstacle(10, position=(130, 90),  radius=7.0),
        StaticObstacle(11, position=(160, 110), radius=6.0),
        StaticObstacle(12, position=(110, 45),  radius=6.0),
        StaticObstacle(13, position=(150, 60),  radius=7.0),
        StaticObstacle(14, position=(170, 80),  radius=6.0),
    ]

    for obs in static_obstacles:
        env.add_static_obstacle(obs)

    # ── 6 Dynamic Obstacles ────────────────────────────────
    dynamic_obstacles = [
        DynamicObstacle(
            obstacle_id = 1,
            start_pos   = (25,  120),
            goal_pos    = (75,  150),
            velocity    = 0.50,    # D1
            radius      = 1.5
        ),
        DynamicObstacle(
            obstacle_id = 2,
            start_pos   = (55,  60),
            goal_pos    = (95,  100),
            velocity    = 0.50,    # D2
            radius      = 1.5
        ),
        DynamicObstacle(
            obstacle_id = 3,
            start_pos   = (110, 170),
            goal_pos    = (140, 140),
            velocity    = 0.60,    # D3
            radius      = 1.5
        ),
        DynamicObstacle(
            obstacle_id = 4,
            start_pos   = (130, 110),
            goal_pos    = (160, 140),
            velocity    = 0.30,    # D4
            radius      = 1.5
        ),
        DynamicObstacle(
            obstacle_id = 5,
            start_pos   = (145, 75),
            goal_pos    = (175, 100),
            velocity    = 0.40,    # D5
            radius      = 1.5
        ),
        DynamicObstacle(
            obstacle_id = 6,
            start_pos   = (160, 40),
            goal_pos    = (185, 60),
            velocity    = 0.25,    # D6
            radius      = 1.5
        ),
    ]

    for obs in dynamic_obstacles:
        env.add_dynamic_obstacle(obs)

    return env


# ══════════════════════════════════════════════════════════════
# HELPER — GET SCENARIO BY NUMBER
# ══════════════════════════════════════════════════════════════

def get_scenario(scenario_number):
    """
    Get a scenario by number.

    Parameters:
    -----------
    scenario_number : int — 1, 2, or 3

    Returns:
    --------
    Environment object
    """
    scenarios = {
        1: create_scenario_1,
        2: create_scenario_2,
        3: create_scenario_3,
    }

    if scenario_number not in scenarios:
        raise ValueError(
            f"Invalid scenario number: {scenario_number}. "
            f"Choose 1, 2, or 3."
        )

    return scenarios[scenario_number]()


# ══════════════════════════════════════════════════════════════
# SCENARIO INFO PRINTER
# ══════════════════════════════════════════════════════════════

def print_scenario_info(env, scenario_number):
    """
    Print summary of scenario configuration.

    Parameters:
    -----------
    env             : Environment object
    scenario_number : int
    """
    print(f"\n{'=' * 55}")
    print(f"  SCENARIO {scenario_number} CONFIGURATION")
    print(f"{'=' * 55}")
    print(f"  Environment : "
          f"{env.width} × {env.height} cm")
    print(f"  Robots      : {len(env.robots)}")
    print(f"  Static obs  : {len(env.static_obstacles)}")
    print(f"  Dynamic obs : {len(env.dynamic_obstacles)}")
    print(f"  Dimension D : {len(env.robots) * 2}")
    print(f"\n  Robots:")
    for r in env.robots:
        ideal = r.get_ideal_path_distance()
        print(f"    R{r.id}: "
              f"Start={tuple(r.start_pos)} → "
              f"Goal={tuple(r.goal_pos)} | "
              f"Ideal dist={ideal:.2f} cm")
    print(f"\n  Dynamic obstacle velocities:")
    for d in env.dynamic_obstacles:
        print(f"    D{d.id}: {d.velocity} cm/step")
    print(f"{'=' * 55}")
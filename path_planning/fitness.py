"""
Fitness Functions for Multi-Robot Path Planning
================================================
Based on Section 3.3 — Equations (10) to (20) of the paper.

Main fitness function:
    Fit = F1 + F2 + F3 + F4        — Equation (20)

Where:
    F1 = Shortest distance          — Equations (11, 12, 13)
    F2 = Avoiding static obstacles  — Equations (14, 15)
    F3 = Avoiding dynamic obstacles — Equations (16, 17)
    F4 = Avoiding other robots      — Equations (18, 19)

Key concept:
    - Lower fitness = better solution
    - Collision = massive penalty (ε = 100,000)
    - No collision = only distance cost

Paper parameters:
    - Penalty ε        = 10^5 = 100,000
    - Security distance = robot radius + obstacle radius
                        = 1.0 + 1.5 = 2.5 cm

Author: Your Name
Date: 2024
"""

import numpy as np


# ══════════════════════════════════════════════════════════════
# INDIVIDUAL FITNESS COMPONENTS
# ══════════════════════════════════════════════════════════════

def calculate_F1(robots, next_positions):
    """
    F1: Shortest Distance Fitness
    ==============================
    Encourages robots to move efficiently toward their goals.
    
    Equation (11): F1 = Σ (fi + gi)
    
    Where:
        fi = distance moved in this step    — Equation (12)
           = ||next_pos - current_pos||
        
        gi = remaining distance to goal     — Equation (13)
           = ||next_pos - goal_pos||
    
    Parameters:
    -----------
    robots         : list of Robot objects
    next_positions : np.array, shape (NR, 2)
                     predicted next positions from optimization
    
    Returns:
    --------
    float : F1 value (lower = better)
    """
    F1 = 0.0

    for i, robot in enumerate(robots):

        if robot.reached_goal:
            continue   # skip robots already at goal

        next_pos = next_positions[i]

        # fi — distance moved this step (Equation 12)
        fi = np.linalg.norm(next_pos - robot.current_pos)

        # gi — remaining distance to goal (Equation 13)
        gi = np.linalg.norm(next_pos - robot.goal_pos)

        F1 += fi + gi

    return F1


def calculate_F2(robots, next_positions, static_obstacles,
                 security_distance, penalty=1e5):
    """
    F2: Static Obstacle Avoidance Fitness
    =======================================
    Applies massive penalty if any robot gets too close
    to any static obstacle.
    
    Equation (14):
        F2 = ε  if d_s_i <= security_distance
        F2 = 0  if d_s_i >  security_distance
    
    Equation (15):
        d_s_i = Σ_j ||next_pos_i - static_obstacle_j||
    
    Parameters:
    -----------
    robots             : list of Robot objects
    next_positions     : np.array, shape (NR, 2)
    static_obstacles   : list of StaticObstacle objects
    security_distance  : float — minimum safe distance in cm
    penalty            : float — ε value = 100,000
    
    Returns:
    --------
    float : F2 value (0 if safe, ε if collision)
    """
    F2 = 0.0

    for i, robot in enumerate(robots):

        if robot.reached_goal:
            continue

        next_pos = next_positions[i]

        # Check distance to every static obstacle
        for obstacle in static_obstacles:

            dist = np.linalg.norm(
                next_pos - obstacle.get_position()
            )

            # Security distance = robot radius + obstacle radius
            min_safe = robot.radius + obstacle.radius

            if dist <= min_safe:
                # Collision detected — apply penalty
                F2 += penalty
                break   # one collision per robot is enough
                        # no need to check remaining obstacles

    return F2


def calculate_F3(robots, next_positions, dynamic_obstacles,
                 security_distance, penalty=1e5):
    """
    F3: Dynamic Obstacle Avoidance Fitness
    ========================================
    Same structure as F2 but for moving obstacles.
    
    Equation (16):
        F3 = ε  if d_d_i <= security_distance
        F3 = 0  if d_d_i >  security_distance
    
    Equation (17):
        d_d_i = Σ_j ||next_pos_i - dynamic_obstacle_j||
    
    Key difference from F2:
        Dynamic obstacles change position every step,
        so we use their CURRENT position at this step.
    
    Parameters:
    -----------
    robots             : list of Robot objects
    next_positions     : np.array, shape (NR, 2)
    dynamic_obstacles  : list of DynamicObstacle objects
    security_distance  : float
    penalty            : float — ε = 100,000
    
    Returns:
    --------
    float : F3 value
    """
    F3 = 0.0

    for i, robot in enumerate(robots):

        if robot.reached_goal:
            continue

        next_pos = next_positions[i]

        # Check distance to every dynamic obstacle
        for obstacle in dynamic_obstacles:

            dist = np.linalg.norm(
                next_pos - obstacle.get_position()
            )

            # Security distance = robot radius + obstacle radius
            min_safe = robot.radius + obstacle.radius

            if dist <= min_safe:
                F3 += penalty
                break

    return F3


def calculate_F4(robots, next_positions, penalty=1e5):
    """
    F4: Inter-Robot Collision Avoidance Fitness
    =============================================
    Applies penalty if any two robots get too close
    to each other.
    
    Equation (18):
        F4 = ε  if d_o_i <= security_distance
        F4 = 0  if d_o_i >  security_distance
    
    Equation (19):
        d_o_i = Σ_j ||next_pos_i - next_pos_j||
                for all j ≠ i
    
    Parameters:
    -----------
    robots         : list of Robot objects
    next_positions : np.array, shape (NR, 2)
    penalty        : float — ε = 100,000
    
    Returns:
    --------
    float : F4 value
    """
    F4    = 0.0
    NR    = len(robots)

    for i in range(NR):

        if robots[i].reached_goal:
            continue

        for j in range(NR):

            if i == j:
                continue   # skip self

            if robots[j].reached_goal:
                continue

            # Distance between robot i and robot j
            dist = np.linalg.norm(
                next_positions[i] - next_positions[j]
            )

            # Minimum safe distance = sum of both radii
            min_safe = robots[i].radius + robots[j].radius

            if dist <= min_safe:
                F4 += penalty
                break   # one collision per robot enough

    return F4


# ══════════════════════════════════════════════════════════════
# MAIN FITNESS FUNCTION
# ══════════════════════════════════════════════════════════════

def calculate_total_fitness(solution,
                            robots,
                            static_obstacles,
                            dynamic_obstacles,
                            env_bounds,
                            penalty = 1e5):
    """
    Main fitness function — combines F1+F2+F3+F4.
    Equation (20): Fit = F1 + F2 + F3 + F4
    
    This function is called by sdSCA optimizer at each
    iteration to evaluate each candidate solution.
    
    How the solution vector maps to robot movements:
    ------------------------------------------------
    solution = [v1, θ1, v2, θ2, v3, θ3, ...]
    
    For NR robots, solution has length NR * 2:
        solution[0] = velocity of robot 1
        solution[1] = angle    of robot 1
        solution[2] = velocity of robot 2
        solution[3] = angle    of robot 2
        ...etc
    
    Parameters:
    -----------
    solution          : np.array, shape (NR*2,)
                        optimization variable vector
    robots            : list of Robot objects
    static_obstacles  : list of StaticObstacle objects
    dynamic_obstacles : list of DynamicObstacle objects
    env_bounds        : tuple (width, height) of environment
    penalty           : float — ε = 100,000
    
    Returns:
    --------
    float : total fitness value (lower = better)
    """

    NR = len(robots)

    # ── Decode solution into next positions ────────────────
    # Extract velocity and angle for each robot
    # and calculate predicted next positions
    next_positions = np.zeros((NR, 2))

    for i, robot in enumerate(robots):

        if robot.reached_goal:
            # Robot already at goal — keep at goal position
            next_positions[i] = robot.goal_pos
            continue

        # Extract this robot's velocity and angle from solution
        # solution is structured as [v1, θ1, v2, θ2, ...]
        velocity = solution[i * 2]        # v_i
        angle    = solution[i * 2 + 1]    # θ_i

        # Calculate predicted next position
        next_positions[i] = robot.calculate_next_position(
            velocity, angle
        )

    # ── Check environment bounds ───────────────────────────
    # Robots must stay within environment
    width, height = env_bounds
    for i in range(NR):
        x, y = next_positions[i]
        if x < 0 or x > width or y < 0 or y > height:
            # Out of bounds — apply penalty
            return penalty * NR   # large enough to reject

    # ── Calculate each fitness component ──────────────────
    security_distance = 2.5   # robot(1.0) + obstacle(1.5)

    F1 = calculate_F1(
        robots, next_positions
    )

    F2 = calculate_F2(
        robots, next_positions,
        static_obstacles, security_distance, penalty
    )

    F3 = calculate_F3(
        robots, next_positions,
        dynamic_obstacles, security_distance, penalty
    )

    F4 = calculate_F4(
        robots, next_positions, penalty
    )

    # ── Total fitness (Equation 20) ────────────────────────
    total = F1 + F2 + F3 + F4

    return total
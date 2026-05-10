"""
Robot Class for Multi-Robot Path Planning Simulation
=====================================================
Based on Section 3.3 of the paper:
"Formulation of online path planning"

Each robot:
- Has a current position, start position, goal position
- Moves using velocity and radial position (angle)
- Is circular with radius = 1 cm (as per paper)
- Updates position using Equations (8) and (9)

Equations:
----------
x_next = x_current + v * cos(θ)   — Equation (8)
y_next = y_current + v * sin(θ)   — Equation (9)

Author: Your Name
Date: 2024
"""

import numpy as np

class Robot:
    """
    Represents a single circular mobile robot.
    
    Parameters:
    -----------
    robot_id : int
        Unique identifier for this robot (1, 2, 3, ...)
    start_pos : tuple or list
        Starting position (x, y) in cm
    goal_pos : tuple or list
        Goal position (x, y) in cm
    radius : float
        Physical radius of robot in cm (default = 1.0 as per paper)
    """

    def __init__(self,
                 robot_id,
                 start_pos,
                 goal_pos,
                 radius = 1.0):

        self.id     = robot_id
        self.radius = radius

        # ── Positions ──────────────────────────────────────────
        # Convert to numpy arrays for easy math
        self.start_pos   = np.array(start_pos,  dtype=float)
        self.goal_pos    = np.array(goal_pos,   dtype=float)

        # Current position starts at start position
        self.current_pos = np.array(start_pos,  dtype=float)

        # Next position (calculated during optimization)
        self.next_pos    = np.array(start_pos,  dtype=float)

        # ── Motion Parameters ──────────────────────────────────
        # These are the optimization variables
        # sdSCA finds best v and θ at each step
        self.velocity = 0.0    # v in paper, range [1, 1.5] cm/step
        self.angle    = 0.0    # θ in paper, range [0, 2π] radians

        # ── Tracking ───────────────────────────────────────────
        # Store complete path history for visualization
        self.path_history = [self.current_pos.copy()]

        # Count steps taken
        self.steps_taken  = 0

        # Track distance traveled
        self.total_distance = 0.0

        # Whether robot has reached its goal
        self.reached_goal   = False

    def calculate_next_position(self, velocity, angle):
        """
        Calculate next position using robot motion equations.
        
        Equation (8): x_next = x_current + v * cos(θ)
        Equation (9): y_next = y_current + v * sin(θ)
        
        Parameters:
        -----------
        velocity : float — how far to move (v)
        angle    : float — which direction to move (θ) in radians
        
        Returns:
        --------
        next_pos : np.array — predicted next position [x, y]
        """
        x_next = self.current_pos[0] + velocity * np.cos(angle)
        y_next = self.current_pos[1] + velocity * np.sin(angle)

        return np.array([x_next, y_next])

    def move(self, velocity, angle):
        """
        Actually move the robot to its next position.
        Called after optimization finds best v and θ.
        
        Parameters:
        -----------
        velocity : float — optimized velocity
        angle    : float — optimized angle
        """
        if self.reached_goal:
            return   # Don't move if already at goal

        # Store motion parameters
        self.velocity = velocity
        self.angle    = angle

        # Calculate and apply next position
        self.next_pos    = self.calculate_next_position(velocity, angle)
        self.current_pos = self.next_pos.copy()

        # Update tracking
        self.path_history.append(self.current_pos.copy())
        self.steps_taken  += 1

        # Calculate distance moved this step
        if len(self.path_history) >= 2:
            step_distance = np.linalg.norm(
                self.path_history[-1] - self.path_history[-2]
            )
            self.total_distance += step_distance

        # Check if goal reached
        self.check_goal_reached()

    def check_goal_reached(self, tolerance=1.5):
        """
        Check if robot has reached its goal position.
        
        Parameters:
        -----------
        tolerance : float
            Distance threshold to consider goal reached.
            Set to 1.5 cm (slightly larger than robot radius)
        """
        distance_to_goal = self.get_distance_to_goal()

        if distance_to_goal <= tolerance:
            self.reached_goal   = True
            self.current_pos    = self.goal_pos.copy()
            self.path_history.append(self.current_pos.copy())

    def get_distance_to_goal(self):
        """
        Calculate Euclidean distance from current position to goal.
        Used in fitness function F1 (Equation 13).
        
        Returns:
        --------
        float : distance in cm
        """
        return np.linalg.norm(self.current_pos - self.goal_pos)

    def get_ideal_path_distance(self):
        """
        Calculate straight line distance from start to goal.
        This is the IDEAL path — used to calculate APDE.
        
        Equation (23): PDE = TraveledDistance - IdealDistance
        
        Returns:
        --------
        float : ideal straight line distance in cm
        """
        return np.linalg.norm(self.start_pos - self.goal_pos)

    def get_path_deviation_error(self):
        """
        Calculate how much actual path deviated from ideal path.
        Used in APDE metric (Equation 23).
        
        Returns:
        --------
        float : deviation in cm (0 = perfect straight line)
        """
        return self.total_distance - self.get_ideal_path_distance()

    def reset(self):
        """
        Reset robot to initial state.
        Used between multiple simulation runs.
        """
        self.current_pos    = self.start_pos.copy()
        self.next_pos       = self.start_pos.copy()
        self.velocity       = 0.0
        self.angle          = 0.0
        self.path_history   = [self.start_pos.copy()]
        self.steps_taken    = 0
        self.total_distance = 0.0
        self.reached_goal   = False

    def __repr__(self):
        """String representation for easy debugging"""
        return (f"Robot {self.id} | "
                f"Pos: ({self.current_pos[0]:.2f}, "
                f"{self.current_pos[1]:.2f}) | "
                f"Goal: ({self.goal_pos[0]:.2f}, "
                f"{self.goal_pos[1]:.2f}) | "
                f"Distance to goal: {self.get_distance_to_goal():.2f} cm | "
                f"Steps: {self.steps_taken} | "
                f"Reached: {self.reached_goal}")
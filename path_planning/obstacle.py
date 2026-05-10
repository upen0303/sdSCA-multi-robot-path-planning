"""
Obstacle Classes for Multi-Robot Path Planning Simulation
=========================================================
Based on Section 3.3 of the paper.

Two types of obstacles:
-----------------------
1. StaticObstacle  — Fixed position, never moves
2. DynamicObstacle — Moves linearly between two points
                     at constant velocity

Dynamic obstacle movement:
--------------------------
Equation (21): x_next = x + v_d * cos(α)
Equation (22): y_next = y + v_d * sin(α)

Paper settings:
---------------
- Radius of obstacles : 1.5 cm
- Radius of robots    : 1.0 cm
- Security distance   : sum of both radii = 2.5 cm

Author: Your Name
Date: 2024
"""

import numpy as np


# ══════════════════════════════════════════════════════════════
# STATIC OBSTACLE
# ══════════════════════════════════════════════════════════════

class StaticObstacle:
    """
    A fixed obstacle that never moves.
    Can be circular or any shape (represented by position + radius).
    
    Parameters:
    -----------
    obstacle_id : int
        Unique identifier
    position : tuple or list
        Fixed position (x, y) in cm
    radius : float
        Size of obstacle in cm
    shape : str
        'circle', 'square', 'triangle' (for visualization only)
        Collision detection always uses radius regardless of shape
    """

    def __init__(self,
                 obstacle_id,
                 position,
                 radius = 1.5,
                 shape  = 'circle'):

        self.id       = obstacle_id
        self.position = np.array(position, dtype=float)
        self.radius   = radius
        self.shape    = shape

        # Static obstacles never move
        self.is_static = True

    def get_position(self):
        """
        Return current position.
        Same every call since static obstacle never moves.
        """
        return self.position.copy()

    def get_distance_to(self, point):
        """
        Calculate Euclidean distance from obstacle center to a point.
        Used in fitness functions F2 (Equation 15).
        
        Parameters:
        -----------
        point : np.array or tuple — [x, y] coordinates
        
        Returns:
        --------
        float : distance in cm
        """
        point = np.array(point)
        return np.linalg.norm(self.position - point)

    def __repr__(self):
        return (f"StaticObstacle {self.id} | "
                f"Pos: ({self.position[0]:.2f}, "
                f"{self.position[1]:.2f}) | "
                f"Radius: {self.radius} | "
                f"Shape: {self.shape}")


# ══════════════════════════════════════════════════════════════
# DYNAMIC OBSTACLE
# ══════════════════════════════════════════════════════════════

class DynamicObstacle:
    """
    A moving obstacle that travels linearly between
    two fixed points at constant velocity.
    
    Movement equations from paper:
    Equation (21): x_next = x + v_d * cos(α)
    Equation (22): y_next = y + v_d * sin(α)
    
    When obstacle reaches its goal point, it reverses
    direction and moves back to start point. This creates
    the back-and-forth linear movement described in paper.
    
    Parameters:
    -----------
    obstacle_id : int
        Unique identifier (D1, D2, D3, etc.)
    start_pos : tuple
        Starting position (x, y) in cm
    goal_pos : tuple
        Goal position (x, y) in cm
        Obstacle moves back and forth between start and goal
    velocity : float
        Speed of movement in cm/step
        Paper uses: 0.5, 0.45, 1.2 cm/step for Scenario 1
    radius : float
        Size of obstacle = 1.5 cm as per paper
    shape : str
        'circle', 'square', 'triangle' for visualization
    """

    def __init__(self,
                 obstacle_id,
                 start_pos,
                 goal_pos,
                 velocity,
                 radius = 1.5,
                 shape  = 'circle'):

        self.id       = obstacle_id
        self.radius   = radius
        self.shape    = shape
        self.velocity = velocity

        # Positions
        self.start_pos   = np.array(start_pos, dtype=float)
        self.goal_pos    = np.array(goal_pos,  dtype=float)
        self.current_pos = np.array(start_pos, dtype=float)

        # Calculate initial angle toward goal
        # α = angle from current position toward goal
        self.angle = self._calculate_angle_to_goal()

        # Track position history for visualization
        self.position_history = [self.current_pos.copy()]

        # Flag to track movement direction
        # True  = moving toward goal
        # False = moving back toward start
        self.moving_forward = True

        self.is_static = False

    def _calculate_angle_to(self, target):
        """
        Calculate angle from current position to target.
        Uses arctan2 for correct quadrant handling.
        
        Parameters:
        -----------
        target : np.array — target position [x, y]
        
        Returns:
        --------
        float : angle in radians
        """
        diff = target - self.current_pos
        return np.arctan2(diff[1], diff[0])

    def _calculate_angle_to_goal(self):
        """Calculate angle from start toward goal position"""
        diff = self.goal_pos - self.start_pos
        return np.arctan2(diff[1], diff[0])

    def update_position(self):
        """
        Move obstacle one step using Equations (21) and (22).
        
        Equation (21): x_next = x + v_d * cos(α)
        Equation (22): y_next = y + v_d * sin(α)
        
        When obstacle reaches goal, it reverses direction.
        This creates back-and-forth linear movement.
        """
        # Calculate next position using paper equations
        x_next = self.current_pos[0] + self.velocity * np.cos(self.angle)
        y_next = self.current_pos[1] + self.velocity * np.sin(self.angle)
        next_pos = np.array([x_next, y_next])

        # Check if obstacle has reached its current target
        # (either goal or start depending on direction)
        current_target = (self.goal_pos
                         if self.moving_forward
                         else self.start_pos)

        distance_to_target = np.linalg.norm(
            next_pos - current_target
        )

        if distance_to_target <= self.velocity:
            # Reached target — reverse direction
            self.current_pos    = current_target.copy()
            self.moving_forward = not self.moving_forward

            # Recalculate angle for new direction
            new_target   = (self.goal_pos
                           if self.moving_forward
                           else self.start_pos)
            self.angle   = self._calculate_angle_to(new_target)
        else:
            # Normal movement
            self.current_pos = next_pos

        # Save position history
        self.position_history.append(self.current_pos.copy())

    def get_position(self):
        """Return current position"""
        return self.current_pos.copy()

    def get_distance_to(self, point):
        """
        Calculate Euclidean distance from obstacle to a point.
        Used in fitness function F3 (Equation 17).
        
        Parameters:
        -----------
        point : np.array or tuple
        
        Returns:
        --------
        float : distance in cm
        """
        point = np.array(point)
        return np.linalg.norm(self.current_pos - point)

    def reset(self):
        """Reset obstacle to initial state"""
        self.current_pos      = self.start_pos.copy()
        self.angle            = self._calculate_angle_to_goal()
        self.moving_forward   = True
        self.position_history = [self.current_pos.copy()]

    def __repr__(self):
        direction = "→ goal" if self.moving_forward else "← start"
        return (f"DynamicObstacle {self.id} | "
                f"Pos: ({self.current_pos[0]:.2f}, "
                f"{self.current_pos[1]:.2f}) | "
                f"Velocity: {self.velocity} cm/step | "
                f"Direction: {direction}")
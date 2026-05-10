"""
Simulation Environment for Multi-Robot Path Planning
=====================================================
Based on Algorithm 3 (Pseudo code of path planning algorithm)
from the paper — Section 3.3.

This class:
1. Holds all robots, static obstacles, dynamic obstacles
2. Runs the optimization at each step using sdSCA
3. Moves robots and obstacles each step
4. Tracks all metrics (APDE, AUGD, total fitness, AET)
5. Stores complete simulation history for visualization

Simulation loop (Algorithm 3):
-------------------------------
WHILE robot with max distance hasn't reached goal:
    1. Call sdSCA to find best (v, θ) for all robots
    2. Move each robot to its next position
    3. Update dynamic obstacle positions
    4. Calculate metrics
    5. Repeat

Author: Your Name
Date: 2024
"""

import numpy as np
import time


class Environment:
    """
    Complete simulation environment for multi-robot path planning.

    Parameters:
    -----------
    width  : float — environment width  in cm
    height : float — environment height in cm
    """

    def __init__(self, width, height):

        self.width  = width
        self.height = height

        # ── Entities ───────────────────────────────────────────
        self.robots            = []
        self.static_obstacles  = []
        self.dynamic_obstacles = []

        # ── Optimization Parameters ────────────────────────────
        # Velocity range [vmin, vmax] — as per paper
        self.v_min = 1.0    # cm/step
        self.v_max = 1.5    # cm/step

        # Angle range [θmin, θmax] — full circle
        self.theta_min = 0.0
        self.theta_max = 2 * np.pi

        # Penalty value ε = 10^5
        self.penalty = 1e5

        # ── Simulation State ───────────────────────────────────
        self.current_step   = 0
        self.max_steps      = 500    # safety limit
        self.simulation_done = False

        # ── Metrics Tracking ───────────────────────────────────
        # Store fitness at each step (for AUGD calculation)
        self.step_fitness_history = []

        # Store distance of each robot to goal at each step
        # Shape: steps × NR
        self.goal_distances_history = []

        # Execution time
        self.execution_time = 0.0

        # ── Algorithm ──────────────────────────────────────────
        self.algorithm = None   # set via set_algorithm()

    # ══════════════════════════════════════════════════════════
    # SETUP METHODS
    # ══════════════════════════════════════════════════════════

    def add_robot(self, robot):
        """Add a robot to the environment"""
        self.robots.append(robot)

    def add_static_obstacle(self, obstacle):
        """Add a static obstacle to the environment"""
        self.static_obstacles.append(obstacle)

    def add_dynamic_obstacle(self, obstacle):
        """Add a dynamic obstacle to the environment"""
        self.dynamic_obstacles.append(obstacle)

    def set_algorithm(self, algorithm):
        """
        Set the optimization algorithm to use.
        Can be SCA or sdSCA — both have same optimize() interface.
        """
        self.algorithm = algorithm

    # ══════════════════════════════════════════════════════════
    # OPTIMIZATION SETUP
    # ══════════════════════════════════════════════════════════

    def _build_fitness_function(self):
        """
        Build the fitness function for current step.
        
        This wraps calculate_total_fitness() so it only
        takes a solution vector — matching what sdSCA expects.
        
        Returns:
        --------
        callable : fitness_function(solution) → float
        """
        from path_planning.fitness import calculate_total_fitness

        # Capture current state in closure
        robots            = self.robots
        static_obstacles  = self.static_obstacles
        dynamic_obstacles = self.dynamic_obstacles
        env_bounds        = (self.width, self.height)
        penalty           = self.penalty

        def fitness_function(solution):
            return calculate_total_fitness(
                solution,
                robots,
                static_obstacles,
                dynamic_obstacles,
                env_bounds,
                penalty
            )

        return fitness_function

    def _get_optimization_bounds(self):
        """
        Build lower and upper bounds for optimization.
        
        Solution structure: [v1, θ1, v2, θ2, ..., vNR, θNR]
        
        For each robot:
            velocity ∈ [v_min, v_max] = [1.0, 1.5]
            angle    ∈ [0,    2π]
        
        Returns:
        --------
        lower_bound : np.array, shape (NR*2,)
        upper_bound : np.array, shape (NR*2,)
        """
        NR = len(self.robots)
        D  = NR * 2   # dimension = NR × 2

        lower_bound = np.zeros(D)
        upper_bound = np.zeros(D)

        for i in range(NR):
            lower_bound[i * 2]     = self.v_min       # velocity min
            upper_bound[i * 2]     = self.v_max        # velocity max
            lower_bound[i * 2 + 1] = self.theta_min   # angle min
            upper_bound[i * 2 + 1] = self.theta_max   # angle max

        return lower_bound, upper_bound

    # ══════════════════════════════════════════════════════════
    # SIMULATION STEP
    # ══════════════════════════════════════════════════════════

    def _run_optimization_step(self):
        """
        Run one optimization step using the algorithm.
        Finds best (v, θ) for all robots at current step.

        This corresponds to line 11 in Algorithm 3:
        (v_i, θ_i) ← call sdSCA(current positions)

        Returns:
        --------
        best_solution : np.array — best [v1,θ1,v2,θ2,...] found
        best_fitness  : float    — fitness of best solution
        """
        NR = len(self.robots)
        D  = NR * 2

        # Get bounds for this step
        lower_bound, upper_bound = self._get_optimization_bounds()

        # Reset algorithm for fresh optimization
        self.algorithm.PS   = 30
        self.algorithm.T    = 100     # iterations per step
                                      # paper uses max 1000 evaluations
                                      # 30 population × ~33 iterations
                                      # ≈ 1000 evaluations
        self.algorithm.D    = D
        self.algorithm.Xmin = lower_bound
        self.algorithm.Xmax = upper_bound
        self.algorithm.reset()

        # Build fitness function for current state
        fitness_fn = self._build_fitness_function()

        # Run optimization
        best_solution, best_fitness, _ = \
            self.algorithm.optimize(fitness_fn)

        return best_solution, best_fitness

    def _move_robots(self, best_solution):
        """
        Move all robots using optimized velocities and angles.
        Lines 12-17 in Algorithm 3.

        Parameters:
        -----------
        best_solution : np.array — [v1, θ1, v2, θ2, ...]
        """
        for i, robot in enumerate(self.robots):
            if robot.reached_goal:
                continue

            velocity = best_solution[i * 2]
            angle    = best_solution[i * 2 + 1]
            robot.move(velocity, angle)

    def _update_dynamic_obstacles(self):
        """
        Move all dynamic obstacles one step.
        Lines 18-20 in Algorithm 3.
        Uses Equations (21) and (22).
        """
        for obstacle in self.dynamic_obstacles:
            obstacle.update_position()

    def _record_metrics(self, step_fitness):
        """
        Record metrics for this step.
        Used later to calculate APDE, AUGD, AET.

        Parameters:
        -----------
        step_fitness : float — best fitness found this step
        """
        # Record step fitness
        self.step_fitness_history.append(step_fitness)

        # Record distance of each robot to its goal
        distances = []
        for robot in self.robots:
            distances.append(robot.get_distance_to_goal())
        self.goal_distances_history.append(distances)

    def _check_simulation_done(self):
        """
        Check if simulation is complete.
        Line 10 in Algorithm 3:
        WHILE robot with maximum distance hasn't reached goal.

        Simulation ends when the robot that started
        farthest from its goal has reached it.
        """
        # Find robot with maximum initial distance
        max_dist_robot = max(
            self.robots,
            key=lambda r: np.linalg.norm(
                r.start_pos - r.goal_pos
            )
        )

        return max_dist_robot.reached_goal

    # ══════════════════════════════════════════════════════════
    # MAIN SIMULATION LOOP
    # ══════════════════════════════════════════════════════════

    def run(self, verbose=True):
        """
        Run complete path planning simulation.
        Implements Algorithm 3 from paper.

        Parameters:
        -----------
        verbose : bool — print progress every 10 steps

        Returns:
        --------
        results : dict — all metrics and history
        """
        if self.algorithm is None:
            raise ValueError(
                "No algorithm set! "
                "Call set_algorithm() before run()"
            )

        if len(self.robots) == 0:
            raise ValueError("No robots in environment!")

        # Record start time for AET calculation
        start_time = time.time()

        if verbose:
            NR = len(self.robots)
            NS = len(self.static_obstacles)
            ND = len(self.dynamic_obstacles)
            print(f"\n  Starting simulation...")
            print(f"  Robots: {NR} | "
                  f"Static: {NS} | "
                  f"Dynamic: {ND}")
            print(f"  Environment: "
                  f"{self.width}×{self.height} cm")
            print("-" * 50)

        # ── Initial distances (line 4-8 in Algorithm 3) ───────
        initial_distances = []
        for robot in self.robots:
            dist = robot.get_distance_to_goal()
            initial_distances.append(dist)
            if verbose:
                print(f"  Robot {robot.id}: "
                      f"Start={tuple(robot.start_pos)} "
                      f"Goal={tuple(robot.goal_pos)} "
                      f"Distance={dist:.2f} cm")

        if verbose:
            print("-" * 50)

        # ── Main simulation loop ───────────────────────────────
        # Line 10 in Algorithm 3: WHILE not done
        while (not self._check_simulation_done() and
               self.current_step < self.max_steps):

            self.current_step += 1

            # Line 11: call algorithm to find best v, θ
            best_solution, best_fitness = \
                self._run_optimization_step()

            # Lines 12-17: move robots
            self._move_robots(best_solution)

            # Lines 18-20: update dynamic obstacles
            self._update_dynamic_obstacles()

            # Record metrics
            self._record_metrics(best_fitness)

            # Print progress
            if verbose and self.current_step % 10 == 0:
                reached = sum(
                    1 for r in self.robots if r.reached_goal
                )
                print(f"  Step {self.current_step:4d} | "
                      f"Fitness: {best_fitness:10.4f} | "
                      f"Robots at goal: "
                      f"{reached}/{len(self.robots)}")

        # Record total execution time
        self.execution_time = time.time() - start_time
        self.simulation_done = True

        if verbose:
            print("-" * 50)
            print(f"  Simulation complete!")
            print(f"  Total steps     : {self.current_step}")
            print(f"  Execution time  : "
                  f"{self.execution_time:.2f} seconds")

        # Return all results
        return self._collect_results()

    # ══════════════════════════════════════════════════════════
    # RESULTS COLLECTION
    # ══════════════════════════════════════════════════════════

    def _collect_results(self):
        """
        Collect and calculate all paper metrics.
        
        Returns:
        --------
        dict with:
            steps_per_robot    : steps each robot took
            total_steps        : sum of all steps
            distances_per_robot: distance each robot traveled
            total_distance     : sum of all distances
            APDE               : Average Path Deviation Error
            AUGD               : Average Untraveled Goal Distance
            total_fitness      : sum of step fitnesses
            AET                : Average Execution Time
            path_histories     : complete paths for visualization
        """
        results = {}

        # ── Steps per robot ────────────────────────────────────
        results['steps_per_robot'] = [
            r.steps_taken for r in self.robots
        ]
        results['total_steps'] = sum(results['steps_per_robot'])

        # ── Distances per robot ────────────────────────────────
        results['distances_per_robot'] = [
            r.total_distance for r in self.robots
        ]
        results['total_distance'] = sum(
            results['distances_per_robot']
        )

        # ── APDE — Average Path Deviation Error ───────────────
        # Equation (23): PDE = Σ(TraveledDist - IdealDist)
        pde = sum(
            r.get_path_deviation_error() for r in self.robots
        )
        results['APDE'] = pde   # single run value
                                # averaged over 30 runs outside

        # ── AUGD — Average Untraveled Goal Distance ────────────
        # Equation (26): UGD = Σ_steps Σ_robots H_i_j
        ugd = 0.0
        for step_distances in self.goal_distances_history:
            ugd += sum(step_distances)
        results['AUGD'] = ugd   # single run value

        # ── Total fitness ──────────────────────────────────────
        results['total_fitness'] = sum(self.step_fitness_history)

        # ── Execution time ─────────────────────────────────────
        results['AET'] = self.execution_time

        # ── Path histories for visualization ───────────────────
        results['path_histories'] = [
            r.path_history for r in self.robots
        ]

        # ── Simulation steps ───────────────────────────────────
        results['simulation_steps'] = self.current_step

        return results

    def reset(self):
        """
        Reset environment for a new run.
        Used when running 30 independent experiments.
        """
        # Reset all robots
        for robot in self.robots:
            robot.reset()

        # Reset all dynamic obstacles
        for obstacle in self.dynamic_obstacles:
            obstacle.reset()

        # Reset simulation state
        self.current_step         = 0
        self.simulation_done      = False
        self.step_fitness_history = []
        self.goal_distances_history = []
        self.execution_time       = 0.0

        # Reset algorithm
        if self.algorithm is not None:
            self.algorithm.reset()
"""
Basic Sine-Cosine Algorithm (SCA)
==================================
Based on: Mirjalili, S. (2016). SCA: a sine cosine algorithm 
for solving optimization problems. Knowledge-Based Systems, 96, 120-133.

This is the BASE algorithm that sdSCA improves upon.
Every weakness here is what sdSCA fixes.

Author: Upendra Prabhakar
Date: 2024
"""

import numpy as np

class SCA:
    """
    Sine Cosine Algorithm for minimization problems.
    
    Parameters:
    -----------
    population_size : int
        Number of candidate solutions (PS in paper) = 30
    max_iterations : int  
        Maximum number of iterations (T in paper)
    dim : int
        Problem dimension (D in paper)
    lower_bound : float or np.array
        Minimum boundary of search space (Xmin in paper)
    upper_bound : float or np.array
        Maximum boundary of search space (Xmax in paper)
    a : float
        Constant for r1 calculation (Equation 3) = 2
    """

    def __init__(self, 
                 population_size, 
                 max_iterations, 
                 dim, 
                 lower_bound, 
                 upper_bound, 
                 a=2):
        
        # Store all parameters as class variables
        self.PS  = population_size   # PS in paper
        self.T   = max_iterations    # T in paper
        self.D   = dim               # D in paper
        self.Xmin = lower_bound      # Xmin in paper
        self.Xmax = upper_bound      # Xmax in paper
        self.a   = a                 # constant = 2

        # These will be filled during optimization
        self.population  = None      # All candidate solutions
        self.fitness     = None      # Fitness of each solution
        self.Xbest       = None      # Best solution found
        self.Fbest       = np.inf    # Best fitness (inf = worst possible start)
        
        # Track progress over iterations (for plotting convergence curve)
        self.convergence_curve = []

    def initialize_population(self):
        """
        Generate initial random population.
        Equation (1) from paper:
        Xi = Xmin + (Xmax - Xmin) * rand(1, D)
        """
        # Create PS solutions, each with D dimensions
        # np.random.rand gives random numbers between 0 and 1
        self.population = self.Xmin + (self.Xmax - self.Xmin) * \
                          np.random.rand(self.PS, self.D)
        
        # Initialize fitness array with infinity for all solutions
        self.fitness = np.full(self.PS, np.inf)

    def calculate_r1(self, current_iteration):
        """
        Calculate r1 - controls step size (decreases over time).
        Equation (3) from paper:
        r1 = a - t * (a/T)
        
        At start (t=0):   r1 = 2 - 0 = 2.0  (large steps, explore)
        At end   (t=T):   r1 = 2 - 2 = 0.0  (tiny steps, exploit)
        """
        r1 = self.a - current_iteration * (self.a / self.T)
        return r1
    
    def update_solution(self, solution, r1):
        """
        Update a single solution using SCA update rule.
        Equation (2) from paper:
        
        If r4 < 0.5:
            X(t+1) = X(t) + r1*sin(r2) * |r3*Xbest - X(t)|
        Else:
            X(t+1) = X(t) + r1*cos(r2) * |r3*Xbest - X(t)|
        """
        # Generate random numbers as defined in paper
        r2 = np.random.uniform(0, 2 * np.pi)  # range [0, 2π]
        r3 = np.random.uniform(0, 2)           # range [0, 2]
        r4 = np.random.uniform(0, 1)           # range [0, 1]

        # Distance between best solution and current solution
        # Absolute value ensures it's always positive
        distance = np.abs(r3 * self.Xbest - solution)

        # Equation (2) — sine or cosine branch
        if r4 < 0.5:
            # Sine branch
            new_solution = solution + r1 * np.sin(r2) * distance
        else:
            # Cosine branch
            new_solution = solution + r1 * np.cos(r2) * distance

        return new_solution
    def check_bounds(self, solution):
        """
        Make sure solution stays within [Xmin, Xmax].
        If a value goes out of bounds, clip it back.
        
        Example:
        Xmin=-100, Xmax=100
        If solution value = 150 → clip to 100
        If solution value = -200 → clip to -100
        """
        solution = np.clip(solution, self.Xmin, self.Xmax)
        return solution

    def optimize(self, fitness_function):
        """
        Main SCA optimization loop.
        Follows Algorithm 1 (pseudo code) from paper exactly.
        
        Parameters:
        -----------
        fitness_function : callable
            The function we want to minimize.
            Takes a solution array, returns a single number.
        
        Returns:
        --------
        Xbest : np.array — Best solution found
        Fbest : float   — Best fitness value found
        convergence_curve : list — Fitness history for plotting
        """

        # ── INITIALIZATION PHASE ──────────────────────────────
        self.initialize_population()

        # Evaluate initial population (lines 4-9 in Algorithm 1)
        for i in range(self.PS):
            self.fitness[i] = fitness_function(self.population[i])
            
            # Track best solution
            if self.fitness[i] < self.Fbest:
                self.Fbest = self.fitness[i]
                self.Xbest = self.population[i].copy()
                # .copy() is IMPORTANT — without it, Xbest would
                # just point to population[i] and change when it changes

        # Save initial best for convergence curve
        self.convergence_curve.append(self.Fbest)

        # ── ITERATIVE OPTIMIZATION PHASE ─────────────────────
        # Lines 10-21 in Algorithm 1
        for t in range(1, self.T + 1):

            # Calculate r1 for this iteration (decreases over time)
            r1 = self.calculate_r1(t)

            # Update each solution in population
            for i in range(self.PS):

                # Generate new solution using update equation
                new_solution = self.update_solution(
                    self.population[i], r1
                )

                # Keep within bounds
                new_solution = self.check_bounds(new_solution)

                # Evaluate new solution
                new_fitness = fitness_function(new_solution)

                # Accept new solution only if it's better
                # (Greedy selection — standard in metaheuristics)
                if new_fitness < self.fitness[i]:
                    self.population[i] = new_solution
                    self.fitness[i]    = new_fitness

                    # Update global best if needed
                    if new_fitness < self.Fbest:
                        self.Fbest = new_fitness
                        self.Xbest = new_solution.copy()

            # Save best fitness of this iteration
            self.convergence_curve.append(self.Fbest)

            # Print progress every 100 iterations
            if t % 100 == 0:
                print(f"  Iteration {t}/{self.T} | "
                      f"Best Fitness: {self.Fbest:.6f}")

        return self.Xbest, self.Fbest, self.convergence_curve

        def reset(self):
            """
            Reset algorithm to initial state.
            Used when running multiple independent experiments (30 runs).
            """
            self.population        = None
            self.fitness           = None
            self.Xbest             = None
            self.Fbest             = np.inf
            self.convergence_curve = []
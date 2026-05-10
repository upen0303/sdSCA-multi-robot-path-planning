"""
Multi-Strategy Self-Adaptive Differential Sine-Cosine Algorithm (sdSCA)
========================================================================
Based on: Akay, R., & Yildirim, M.Y. (2023). Multi-strategy and 
self-adaptive differential sine-cosine algorithm for multi-robot 
path planning. Expert Systems With Applications, 232, 120849.

This is the PROPOSED algorithm that improves basic SCA by:
1. Adding 3 differential evolution strategies to SCA's original strategy
2. Self-adaptive roulette wheel selection — better strategies chosen more often
3. Removing dependency on single update strategy

Strategies:
-----------
Strategy 1: Original SCA          — Equation (2)
Strategy 2: DE/rand/1             — Equation (4)
Strategy 3: DE/current-to-best/1  — Equation (5)
Strategy 4: DE/current-to-rand/1  — Equation (6)

Author: Your Name
Date: 2024
"""

import numpy as np

class sdSCA:
    """
    Multi-Strategy Self-Adaptive Differential Sine-Cosine Algorithm.
    
    Parameters:
    -----------
    population_size : int
        Number of candidate solutions (PS in paper) = 30
    max_iterations : int
        Maximum number of iterations (T in paper)
    dim : int
        Problem dimension (D in paper)
    lower_bound : float or np.array
        Minimum boundary of search space (Xmin)
    upper_bound : float or np.array
        Maximum boundary of search space (Xmax)
    a : float
        Constant for r1 calculation = 2
    F : float
        Scale factor for DE strategies = 0.8
    CR : float
        Crossover rate for DE strategies = 0.95
    """

    # Number of strategies in the pool (fixed at 4 as per paper)
    NS = 4

    def __init__(self,
                 population_size,
                 max_iterations,
                 dim,
                 lower_bound,
                 upper_bound,
                 a  = 2,
                 F  = 0.8,
                 CR = 0.95):

        # ── Algorithm Parameters ───────────────────────────────
        self.PS   = population_size
        self.T    = max_iterations
        self.D    = dim
        self.Xmin = lower_bound
        self.Xmax = upper_bound
        self.a    = a       # for r1 calculation (Equation 3)
        self.F    = F       # scale factor for DE strategies
        self.CR   = CR      # crossover rate for DE strategies

        # ── Population ─────────────────────────────────────────
        self.population = None   # shape: (PS, D)
        self.fitness    = None   # shape: (PS,)

        # ── Best Solution Tracking ─────────────────────────────
        self.Xbest = None
        self.Fbest = np.inf

        # ── Strategy Pool (THE KEY DIFFERENCE FROM BASIC SCA) ──
        # Probabilities of each strategy being selected
        # Initially equal — 25% each (1/NS)
        self.probabilities = np.full(self.NS, 1.0 / self.NS)

        # Selection counters — how many times each strategy
        # produced a better solution
        # Start at 1 not 0 — prevents immediate starvation
        self.counters = np.ones(self.NS, dtype=float)

        # Strategy assigned to each solution in population
        # Shape: (PS,) — each solution has one strategy
        self.strategy_assignments = np.zeros(self.PS, dtype=int)

        # Accumulated counters — never reset to zero
        # This prevents strategy starvation
        self.counters = np.ones(self.NS)  # Start at 1 not 0
        # Starting at 1 gives each strategy a fair base count
        # Prevents division by zero and early starvation

        # ── Result Tracking ────────────────────────────────────
        self.convergence_curve = []

        # Track strategy usage over time (for analysis/plotting)
        self.strategy_history = []

    def initialize_population(self):
        """
        Generate initial random population.
        Same as basic SCA — Equation (1).
        """
        self.population = self.Xmin + \
                         (self.Xmax - self.Xmin) * \
                          np.random.rand(self.PS, self.D)

        self.fitness = np.full(self.PS, np.inf)

    def roulette_wheel_selection(self):
        """
        Select a strategy index based on current probabilities.
        Higher probability = more likely to be selected.
        
        Think of it like a spinning wheel:
        ┌─────────────────────────────────┐
        │  S1: 40% │ S2: 30% │S3:20%│S4:10%│
        └─────────────────────────────────┘
        Spin the wheel → lands on section proportional to its size.
        
        Returns:
        --------
        int : index of selected strategy (0, 1, 2, or 3)
        """
        # np.random.choice selects from [0,1,2,3]
        # with probability = self.probabilities
        return np.random.choice(self.NS, p=self.probabilities)

    def assign_strategies(self):
        """
        Assign a strategy to each solution in population
        using roulette wheel selection.
        Called once at initialization and after each iteration.
        """
        for i in range(self.PS):
            self.strategy_assignments[i] = \
                self.roulette_wheel_selection()
            
    def update_probabilities(self):
        """
        Update strategy selection probabilities.
        Equation (7) from paper with minimum floor added.
        
        Minimum floor ensures no strategy ever dies completely.
        This maintains diversity in strategy selection.
        """
        # Minimum probability any strategy can have
        # Ensures all strategies always get some chance
        MIN_PROB = 0.05   # 5% minimum for each strategy

        total = np.sum(self.counters)

        if total > 0:
            # Calculate raw probabilities from counters
            raw_probs = self.counters / total

            # Apply minimum floor
            # Any strategy below MIN_PROB gets bumped up to MIN_PROB
            floored_probs = np.maximum(raw_probs, MIN_PROB)

            # Renormalize so all probabilities sum to 1.0
            self.probabilities = floored_probs / np.sum(floored_probs)
        else:
            # No improvements happened — reset to equal
            self.probabilities = np.full(self.NS, 1.0 / self.NS)

        # Verify probabilities sum to 1 (safety check)
        assert abs(np.sum(self.probabilities) - 1.0) < 1e-9, \
               "Probabilities must sum to 1!"

        # Save history for analysis
        self.strategy_history.append(self.probabilities.copy())

    def strategy_1_sca(self, solution, current_iteration):
        """
        Strategy 1: Original SCA update rule.
        Equation (2) from paper.
        Moves toward best solution using sine or cosine.
        """
        # r1 decreases over time (Equation 3)
        r1 = self.a - current_iteration * (self.a / self.T)
        r2 = np.random.uniform(0, 2 * np.pi)
        r3 = np.random.uniform(0, 2)
        r4 = np.random.uniform(0, 1)

        distance = np.abs(r3 * self.Xbest - solution)

        if r4 < 0.5:
            new_solution = solution + r1 * np.sin(r2) * distance
        else:
            new_solution = solution + r1 * np.cos(r2) * distance

        return new_solution

    def strategy_2_de_rand_1(self, solution):
        """
        Strategy 2: DE/rand/1
        Equation (4) from paper.
        
        Combines THREE random solutions.
        Does NOT use Xbest → good for EXPLORATION.
        Has crossover condition (CR).
        
        If rand < CR:
            X_new = X_R1 + F * (X_R2 - X_R3)
        Else:
            X_new = X_current (no change)
        """
        # Select 3 random DIFFERENT solutions from population
        # We need indices that are all different from each other
        indices = list(range(self.PS))
        R1, R2, R3 = np.random.choice(indices, 3, replace=False)

        X_R1 = self.population[R1]
        X_R2 = self.population[R2]
        X_R3 = self.population[R3]

        # Apply crossover condition dimension by dimension
        new_solution = solution.copy()
        for d in range(self.D):
            if np.random.rand() < self.CR:
                new_solution[d] = X_R1[d] + self.F * (X_R2[d] - X_R3[d])
            # else: keep original value (no change for this dimension)

        return new_solution

    def strategy_3_de_current_to_best_1(self, solution):
        """
        Strategy 3: DE/current-to-best/1
        Equation (5) from paper.
        
        Uses BOTH best solution AND random solutions.
        Balances exploration and exploitation.
        
        If rand < CR:
            X_new = X_current + F*(Xbest - X_current + X_R1 - X_R2)
        Else:
            X_new = X_current
        """
        # Select 2 random different solutions
        indices = list(range(self.PS))
        R1, R2 = np.random.choice(indices, 2, replace=False)

        X_R1 = self.population[R1]
        X_R2 = self.population[R2]

        new_solution = solution.copy()
        for d in range(self.D):
            if np.random.rand() < self.CR:
                new_solution[d] = solution[d] + self.F * (
                    self.Xbest[d] - solution[d] +
                    X_R1[d] - X_R2[d]
                )

        return new_solution

    def strategy_4_de_current_to_rand_1(self, solution):
        """
        Strategy 4: DE/current-to-rand/1
        Equation (6) from paper.
        
        NO crossover condition — ALWAYS updates.
        NO Xbest used — most EXPLORATORY strategy.
        Best for escaping local optima.
        
        X_new = X_current + L*(X_R1 - X_current) + F*(X_R2 - X_R3)
        """
        indices = list(range(self.PS))
        R1, R2, R3 = np.random.choice(indices, 3, replace=False)

        X_R1 = self.population[R1]
        X_R2 = self.population[R2]
        X_R3 = self.population[R3]

        # L is random number in [0,1]
        L = np.random.rand()

        new_solution = solution + \
                       L * (X_R1 - solution) + \
                       self.F * (X_R2 - X_R3)

        return new_solution
    
    def apply_strategy(self, solution_index, current_iteration):
        """
        Apply the assigned strategy to solution at solution_index.
        Acts as a router — sends solution to correct strategy function.
        """
        solution = self.population[solution_index]
        strategy = self.strategy_assignments[solution_index]

        if strategy == 0:
            return self.strategy_1_sca(solution, current_iteration)
        elif strategy == 1:
            return self.strategy_2_de_rand_1(solution)
        elif strategy == 2:
            return self.strategy_3_de_current_to_best_1(solution)
        elif strategy == 3:
            return self.strategy_4_de_current_to_rand_1(solution)

    def check_bounds(self, solution):
        """Keep solution within [Xmin, Xmax]"""
        return np.clip(solution, self.Xmin, self.Xmax)
    
    def optimize(self, fitness_function):
        """
        Main sdSCA optimization loop.
        Follows Algorithm 2 (pseudo code) from paper exactly.
        
        Parameters:
        -----------
        fitness_function : callable
            Function to minimize.
        
        Returns:
        --------
        Xbest, Fbest, convergence_curve
        """

        # ── INITIALIZATION (Lines 1-11 in Algorithm 2) ────────
        self.initialize_population()

        # Evaluate initial population and find best
        for i in range(self.PS):
            self.fitness[i] = fitness_function(self.population[i])

            if self.fitness[i] < self.Fbest:
                self.Fbest = self.fitness[i]
                self.Xbest = self.population[i].copy()

        # Assign initial strategies via roulette wheel
        # At start all probabilities equal so truly random
        self.assign_strategies()
        self.convergence_curve.append(self.Fbest)

        # ── MAIN LOOP (Lines 12-41 in Algorithm 2) ────────────
        for t in range(1, self.T + 1):

            # Decay counters — recent performance matters more
            # 0.8 decay means old counts fade gradually
            # Starting from ones() ensures no strategy starves
            self.counters = self.counters * 0.8

            for i in range(self.PS):

                # Apply this solution's assigned strategy
                new_solution = self.apply_strategy(i, t)

                # Keep within bounds
                new_solution = self.check_bounds(new_solution)

                # Evaluate new solution
                new_fitness = fitness_function(new_solution)

                # Greedy selection (Lines 27-35 in Algorithm 2)
                if new_fitness < self.fitness[i]:

                    # Accept better solution
                    self.population[i] = new_solution
                    self.fitness[i]    = new_fitness

                    # Increment counter for the strategy that worked
                    used_strategy = self.strategy_assignments[i]
                    self.counters[used_strategy] += 1

                    # Update global best
                    if new_fitness < self.Fbest:
                        self.Fbest = new_fitness
                        self.Xbest = new_solution.copy()

            # Update probabilities based on this iteration's counters
            # Equation (7) — Line 37 in Algorithm 2
            self.update_probabilities()

            # Reassign strategies for next iteration
            # Line 38-40 in Algorithm 2
            self.assign_strategies()

            # Track convergence
            self.convergence_curve.append(self.Fbest)

            # Progress report
            if t % 100 == 0:
                print(f"  Iter {t}/{self.T} | "
                      f"Best: {self.Fbest:.6f} | "
                      f"Strategy probs: "
                      f"S1={self.probabilities[0]:.2f} "
                      f"S2={self.probabilities[1]:.2f} "
                      f"S3={self.probabilities[2]:.2f} "
                      f"S4={self.probabilities[3]:.2f}")

        return self.Xbest, self.Fbest, self.convergence_curve

    def reset(self):
        """Reset for multiple independent runs"""
        self.population           = None
        self.fitness              = None
        self.Xbest                = None
        self.Fbest                = np.inf
        self.probabilities        = np.full(self.NS, 1.0 / self.NS)
        self.counters             = np.ones(self.NS, dtype=float)
        self.strategy_assignments = np.zeros(self.PS, dtype=int)
        self.convergence_curve    = []
        self.strategy_history     = []

    
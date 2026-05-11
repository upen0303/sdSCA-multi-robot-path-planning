"""
Q-Learning based Multi-Strategy Self-Adaptive Differential
Sine-Cosine Algorithm (qlsdSCA)
===========================================================
Your original research contribution.

Improvements over sdSCA (Akay & Yildirim, 2023):
──────────────────────────────────────────────────
1. Q-Learning Strategy Selector
   - Replaces passive roulette wheel selection
   - Agent actively LEARNS which strategy works
     best in each optimization STATE
   - Uses temporal difference learning (TD-learning)

2. Adaptive F and CR per Strategy (Option 4)
   - F and CR are no longer fixed at 0.8 and 0.95
   - Each strategy maintains its own F and CR
   - Parameters adapt based on success rate
   - Successful F/CR values are remembered and reused

Algorithm name : qlsdSCA
Full name      : Q-Learning based Multi-Strategy
                 Self-Adaptive Differential
                 Sine-Cosine Algorithm

Author: Your Name
Date  : 2024
"""

import numpy as np

class QLearningAgent:
    """
    Q-Learning agent for strategy selection.

    Learns which strategy (action) works best
    in each optimization state.

    State definition:
    -----------------
    State = (iteration_stage, improvement_level)

    iteration_stage:
        0 = early  (t < T/3)
        1 = middle (T/3 <= t < 2T/3)
        2 = late   (t >= 2T/3)

    improvement_level:
        0 = stuck  (improvement < 1%)
        1 = slow   (1% <= improvement < 10%)
        2 = fast   (improvement >= 10%)

    Total states = 3 × 3 = 9

    Actions = 4 strategies (0, 1, 2, 3)

    Parameters:
    -----------
    n_states  : int   — number of states = 9
    n_actions : int   — number of strategies = 4
    alpha     : float — learning rate (0.1 to 0.5)
                        how fast agent updates beliefs
    gamma     : float — discount factor (0.9 to 0.99)
                        importance of future rewards
    epsilon   : float — exploration rate (starts at 1.0)
                        probability of random action
    epsilon_min : float — minimum exploration (0.05)
    epsilon_decay : float — how fast exploration decreases
    """

    def __init__(self,
                 n_states      = 9,
                 n_actions     = 4,
                 alpha         = 0.1,
                 gamma         = 0.9,
                 epsilon       = 1.0,
                 epsilon_min   = 0.05,
                 epsilon_decay = 0.98):

        self.n_states      = n_states
        self.n_actions     = n_actions
        self.alpha         = alpha
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay

        # ── Q-Table ────────────────────────────────────────
        # Shape: (n_states, n_actions) = (9, 4)
        # Initialize with small random values
        # Not zeros — avoids tie-breaking issues early on
        # Rows = states, Cols = [S1, S2, S3, S4]
        self.Q = np.array([
            # S1      S2      S3      S4
        [0.00,   0.02,   0.01,   0.02],  # early+stuck → explore
        [0.00,   0.01,   0.01,   0.02],  # early+slow  → explore
        [0.01,   0.02,   0.01,   0.01],  # early+fast  → any
        [0.00,   0.01,   0.02,   0.01],  # mid+stuck   → DE/best
        [0.01,   0.01,   0.02,   0.01],  # mid+slow    → DE/best
        [0.01,   0.01,   0.02,   0.01],  # mid+fast    → DE/best
        [0.02,   0.01,   0.01,   0.01],  # late+stuck  → SCA
        [0.02,   0.01,   0.01,   0.01],  # late+slow   → SCA
        [0.02,   0.00,   0.01,   0.00],  # late+fast   → SCA
        ])

        # ── Tracking ───────────────────────────────────────
        # Remember last state and action for Q-update
        self.last_state  = None
        self.last_action = None

        # Track Q-table evolution for analysis
        self.q_history   = []

        # Track rewards received
        self.reward_history = []

    def get_state(self, current_iteration,
                  max_iterations, improvement_rate):
        """
        Convert optimization progress into a state index.

        Parameters:
        -----------
        current_iteration : int   — current iteration t
        max_iterations    : int   — maximum iterations T
        improvement_rate  : float — recent improvement %

        Returns:
        --------
        int : state index (0 to 8)
        """
        # ── Iteration stage ────────────────────────────────
        progress = current_iteration / max(max_iterations, 1)

        if progress < 0.33:
            iter_stage = 0      # early exploration phase
        elif progress < 0.67:
            iter_stage = 1      # middle balanced phase
        else:
            iter_stage = 2      # late exploitation phase

        # ── Improvement level ──────────────────────────────
        imp = abs(improvement_rate)

        if imp < 1.0:
            imp_level = 0       # stuck — little improvement
        elif imp < 10.0:
            imp_level = 1       # slow improvement
        else:
            imp_level = 2       # fast improvement

        # ── Combine into single state index ────────────────
        # state = iter_stage * 3 + imp_level
        # Gives unique index 0-8 for each combination
        state = iter_stage * 3 + imp_level

        return state

    def select_action(self, state):
        """
        Select strategy using epsilon-greedy policy.

        Epsilon-greedy:
        ───────────────
        With probability epsilon    → random action (explore)
        With probability 1-epsilon  → best known action (exploit)

        Early training: epsilon ≈ 1.0 → mostly random (explore)
        Late training:  epsilon ≈ 0.05 → mostly best (exploit)

        Parameters:
        -----------
        state : int — current state index

        Returns:
        --------
        int : selected action (strategy index 0-3)
        """
        if np.random.rand() < self.epsilon:
            # Explore — random strategy
            action = np.random.randint(self.n_actions)
        else:
            # Exploit — best known strategy for this state
            action = np.argmax(self.Q[state])

        # Remember for Q-update
        self.last_state  = state
        self.last_action = action

        return action

    def calculate_reward(self, old_fitness, new_fitness):
        """
        Calculate reward based on fitness improvement.

        Reward design:
        ──────────────
        Better solution → positive reward
        Same solution   → zero reward
        Worse solution  → negative reward (penalize)

        Normalized by old fitness for scale independence.

        Parameters:
        -----------
        old_fitness : float — fitness before strategy applied
        new_fitness : float — fitness after strategy applied

        Returns:
        --------
        float : reward value
        --------

        Improved reward function.
        Simpler and more stable than previous version.
    
        +1.0 = clear improvement
        0.0 = no change
        -0.1 = got worse (small penalty, not harsh
        """

        if old_fitness == np.inf and new_fitness == np.inf:
            return 0.0

        if new_fitness < old_fitness:
            # Improved — positive reward
            if old_fitness == np.inf or old_fitness == 0:
                return 1.0
            # Scale by relative improvement
            rel_imp = (old_fitness - new_fitness) / abs(old_fitness)
            reward  = min(rel_imp * 10, 1.0)
            return max(reward, 0.1)   # at least 0.1 for any improvement

        elif new_fitness > old_fitness:
            # Got worse — small negative reward
            # Not too harsh — agent should still explore
            return -0.1

        else:
            # No change
            return 0.0

    def update(self, new_state, reward):
        """
        Update Q-table using Bellman equation.

        Q-Learning update rule:
        ────────────────────────
        Q(s,a) ← Q(s,a) + α × [R + γ × max(Q(s',a')) - Q(s,a)]

        Where:
        s  = last state
        a  = last action taken
        R  = reward received
        s' = new state after action
        α  = learning rate
        γ  = discount factor

        Parameters:
        -----------
        new_state : int   — state after action was taken
        reward    : float — reward received
        """
        if self.last_state is None:
            return   # nothing to update yet

        s  = self.last_state
        a  = self.last_action
        s_ = new_state

        # Bellman equation
        old_q   = self.Q[s, a]
        max_q_  = np.max(self.Q[s_])
        new_q   = old_q + self.alpha * (
            reward + self.gamma * max_q_ - old_q
        )
        self.Q[s, a] = new_q

        # Track reward
        self.reward_history.append(reward)

    def decay_epsilon(self):
        """
        Reduce epsilon over time.
        Agent explores less as it learns more.

        epsilon = max(epsilon × decay, epsilon_min)
        """
        self.epsilon = max(
            self.epsilon * self.epsilon_decay,
            self.epsilon_min
        )

    def get_best_strategy_per_state(self):
        """
        Return best strategy for each state.
        Useful for analyzing what agent learned.

        Returns:
        --------
        np.array : best action for each state
        """
        return np.argmax(self.Q, axis=1)

    def reset(self):
        """Reset agent for new run"""
        self.Q = np.random.uniform(
            low  = -0.01,
            high =  0.01,
            size = (self.n_states, self.n_actions)
        )
        self.epsilon        = 1.0
        self.last_state     = None
        self.last_action    = None
        self.q_history      = []
        self.reward_history = []

class AdaptiveParameters:
    """
    Adaptive F and CR for each strategy.
    Option 4 improvement.

    Instead of fixed F=0.8 and CR=0.95 for all strategies,
    each strategy learns its own best F and CR values.

    Mechanism:
    ──────────
    - Keep memory of F and CR values that produced improvements
    - New F and CR sampled from distribution centered on
      successful past values
    - If no history yet, use default values
    - Memory fades over time (recent success matters more)

    Based on SHADE algorithm concept (Zhang & Sanderson, 2009)
    adapted per-strategy.

    Parameters:
    -----------
    n_strategies : int   — number of strategies = 4
    memory_size  : int   — how many past values to remember
    F_init       : float — initial scale factor = 0.8
    CR_init      : float — initial crossover rate = 0.95
    """

    def __init__(self,
                 n_strategies = 4,
                 memory_size  = 10,
                 F_init       = 0.8,
                 CR_init      = 0.95):

        self.NS          = n_strategies
        self.memory_size = memory_size
        self.F_init      = F_init
        self.CR_init     = CR_init

        # Memory of successful F and CR per strategy
        # Each strategy has a list of successful values
        self.F_memory  = [
            [F_init]  * memory_size
            for _ in range(n_strategies)
        ]
        self.CR_memory = [
            [CR_init] * memory_size
            for _ in range(n_strategies)
        ]

        # Memory pointer (circular buffer)
        self.memory_ptr = [0] * n_strategies

        # Current F and CR per strategy
        self.current_F  = [F_init]  * n_strategies
        self.current_CR = [CR_init] * n_strategies

    def sample_parameters(self, strategy_idx):
        """
        Sample F and CR for a given strategy.
        Simplified and stabilized version.
        
        Uses truncated normal distribution for both F and CR.
        More stable than Cauchy for small memory sizes.
        """
        F_mean  = np.mean(self.F_memory[strategy_idx])
        CR_mean = np.mean(self.CR_memory[strategy_idx])

        # Sample F — normal distribution, small std
        F  = np.random.normal(F_mean, 0.05)
        F  = np.clip(F, 0.4, 1.0)   # tighter range

        # Sample CR — normal distribution, small std
        CR = np.random.normal(CR_mean, 0.05)
        CR = np.clip(CR, 0.5, 1.0)   # tighter range

        self.current_F[strategy_idx]  = F
        self.current_CR[strategy_idx] = CR

        return F, CR

    def update_memory(self, strategy_idx, F, CR, success):
        """
        Update memory if strategy was successful.

        Only successful F and CR values are stored.
        Failed values are forgotten immediately.

        Parameters:
        -----------
        strategy_idx : int   — which strategy
        F            : float — F value that was used
        CR           : float — CR value that was used
        success      : bool  — did it improve solution?
        """
        if success:
            ptr = self.memory_ptr[strategy_idx]
            self.F_memory[strategy_idx][ptr]  = F
            self.CR_memory[strategy_idx][ptr] = CR

            # Advance circular buffer pointer
            self.memory_ptr[strategy_idx] = (
                (ptr + 1) % self.memory_size
            )

    def get_stats(self):
        """
        Return current mean F and CR per strategy.
        Used for analysis and printing.
        """
        stats = []
        for i in range(self.NS):
            stats.append({
                'strategy' : i + 1,
                'mean_F'   : np.mean(self.F_memory[i]),
                'mean_CR'  : np.mean(self.CR_memory[i]),
            })
        return stats

    def reset(self):
        """Reset to initial state"""
        self.F_memory  = [
            [self.F_init]  * self.memory_size
            for _ in range(self.NS)
        ]
        self.CR_memory = [
            [self.CR_init] * self.memory_size
            for _ in range(self.NS)
        ]
        self.memory_ptr = [0] * self.NS
        self.current_F  = [self.F_init]  * self.NS
        self.current_CR = [self.CR_init] * self.NS

class qlsdSCA:
    """
    Q-Learning based Multi-Strategy Self-Adaptive
    Differential Sine-Cosine Algorithm.

    Your original research contribution.

    Differences from sdSCA:
    ───────────────────────
    1. QLearningAgent replaces roulette wheel
    2. AdaptiveParameters replaces fixed F and CR
    3. Each solution uses Q-agent selected strategy
    4. Each strategy uses its own adaptive F and CR

    Parameters:
    -----------
    population_size : int   — PS = 30
    max_iterations  : int   — T
    dim             : int   — D
    lower_bound     : float — Xmin
    upper_bound     : float — Xmax
    a               : float — SCA constant = 2
    F               : float — initial F = 0.8
    CR              : float — initial CR = 0.95
    alpha           : float — Q-learning rate = 0.1
    gamma           : float — Q discount factor = 0.9
    epsilon         : float — initial exploration = 1.0
    """

    NS = 4   # number of strategies

    def __init__(self,
                 population_size,
                 max_iterations,
                 dim,
                 lower_bound,
                 upper_bound,
                 a       = 2,
                 F       = 0.8,
                 CR      = 0.95,
                 alpha   = 0.1,
                 gamma   = 0.9,
                 epsilon = 1.0):

        # ── Standard Parameters ────────────────────────────
        self.PS   = population_size
        self.T    = max_iterations
        self.D    = dim
        self.Xmin = lower_bound
        self.Xmax = upper_bound
        self.a    = a

        # ── Population ─────────────────────────────────────
        self.population = None
        self.fitness    = None
        self.Xbest      = None
        self.Fbest      = np.inf

        # ── Q-Learning Agent (Option 1) ────────────────────
        self.agent = QLearningAgent(
            n_states      = 9,
            n_actions     = self.NS,
            alpha         = alpha,
            gamma         = gamma,
            epsilon       = epsilon,
            epsilon_min   = 0.05,
            epsilon_decay = 0.995
        )

        # ── Adaptive Parameters (Option 4) ─────────────────
        self.adaptive_params = AdaptiveParameters(
            n_strategies = self.NS,
            memory_size  = 20,
            F_init       = F,
            CR_init      = CR
        )

        # ── Strategy assignments per solution ──────────────
        self.strategy_assignments = np.zeros(
            self.PS, dtype=int
        )

        # ── Tracking ───────────────────────────────────────
        self.convergence_curve  = []
        self.strategy_history   = []
        self.improvement_rate   = 0.0
        self.prev_best_fitness  = np.inf

    def initialize_population(self):
        """Generate initial random population — Equation (1)"""
        self.population = self.Xmin + \
                         (self.Xmax - self.Xmin) * \
                          np.random.rand(self.PS, self.D)
        self.fitness = np.full(self.PS, np.inf)

    def check_bounds(self, solution):
        """Keep solution within [Xmin, Xmax]"""
        return np.clip(solution, self.Xmin, self.Xmax)

    # ── 4 Update Strategies ────────────────────────────────

    def strategy_1_sca(self, solution, t, F, CR):
        """
        Original SCA — Equation (2).
        CR controls probability of updating each dimension.
        F scales the step size additively.
        """
        r1 = self.a - t * (self.a / self.T)
        r2 = np.random.uniform(0, 2 * np.pi)
        r3 = np.random.uniform(0, 2)
        r4 = np.random.uniform(0, 1)

        distance = np.abs(r3 * self.Xbest - solution)
        new_solution = solution.copy()

        for d in range(self.D):
            if np.random.rand() < CR:
                if r4 < 0.5:
                    new_solution[d] = (solution[d] +
                        r1 * F * np.sin(r2) * distance[d])
                else:
                    new_solution[d] = (solution[d] +
                        r1 * F * np.cos(r2) * distance[d])

        return new_solution

    def strategy_2_de_rand_1(self, solution, F, CR):
        """DE/rand/1 — Equation (4) with adaptive F, CR"""
        indices    = list(range(self.PS))
        R1, R2, R3 = np.random.choice(
            indices, 3, replace=False
        )

        new_solution = solution.copy()
        for d in range(self.D):
            if np.random.rand() < CR:
                new_solution[d] = (
                    self.population[R1][d] +
                    F * (self.population[R2][d] -
                         self.population[R3][d])
                )
        return new_solution

    def strategy_3_de_current_to_best_1(self, solution,
                                         F, CR):
        """DE/current-to-best/1 — Equation (5) adaptive F,CR"""
        indices = list(range(self.PS))
        R1, R2  = np.random.choice(indices, 2, replace=False)

        new_solution = solution.copy()
        for d in range(self.D):
            if np.random.rand() < CR:
                new_solution[d] = solution[d] + F * (
                    self.Xbest[d]          - solution[d] +
                    self.population[R1][d] -
                    self.population[R2][d]
                )
        return new_solution

    def strategy_4_de_current_to_rand_1(self, solution,
                                          F, CR):
        """DE/current-to-rand/1 — Equation (6) adaptive F,CR"""
        indices    = list(range(self.PS))
        R1, R2, R3 = np.random.choice(
            indices, 3, replace=False
        )

        L = np.random.rand()
        return (solution +
                L * (self.population[R1] - solution) +
                F * (self.population[R2] -
                     self.population[R3]))

    def apply_strategy(self, solution_idx, strategy_idx,
                       t, F, CR):
        """Apply selected strategy with given F and CR"""
        solution = self.population[solution_idx]

        if strategy_idx == 0:
            return self.strategy_1_sca(solution, t, F, CR)
        elif strategy_idx == 1:
            return self.strategy_2_de_rand_1(
                solution, F, CR
            )
        elif strategy_idx == 2:
            return self.strategy_3_de_current_to_best_1(
                solution, F, CR
            )
        elif strategy_idx == 3:
            return self.strategy_4_de_current_to_rand_1(
                solution, F, CR
            )

    def _calculate_improvement_rate(self):
        """
        Calculate recent improvement rate.
        Used as part of state for Q-agent.
        """
        if self.prev_best_fitness == np.inf:
            return 0.0
        if self.prev_best_fitness == 0:
            return 0.0

        imp = ((self.prev_best_fitness - self.Fbest) /
                abs(self.prev_best_fitness)) * 100

        return imp

    def optimize(self, fitness_function, verbose=False):
        """
        Main qlsdSCA optimization loop.

        Key differences from sdSCA.optimize():
        1. Q-agent selects strategy per solution
        2. Adaptive F and CR per strategy
        3. Q-table updated after each solution evaluation
        4. Epsilon decays over iterations

        Parameters:
        -----------
        fitness_function : callable
        verbose          : bool

        Returns:
        --------
        Xbest, Fbest, convergence_curve
        """

        # ── INITIALIZATION ─────────────────────────────────
        self.initialize_population()

        # Evaluate initial population
        for i in range(self.PS):
            self.fitness[i] = fitness_function(
                self.population[i]
            )
            if self.fitness[i] < self.Fbest:
                self.Fbest = self.fitness[i]
                self.Xbest = self.population[i].copy()

        self.prev_best_fitness = self.Fbest
        self.convergence_curve.append(self.Fbest)

        # Get initial state for Q-agent
        current_state = self.agent.get_state(
            0, self.T, 0.0
        )

        # ── MAIN LOOP ──────────────────────────────────────
        for t in range(1, self.T + 1):

            # Track best at start of iteration
            iter_best_start = self.Fbest

            for i in range(self.PS):

                # ── Q-agent selects strategy ───────────────
                strategy_idx = self.agent.select_action(
                    current_state
                )

                # ── Adaptive F and CR for this strategy ────
                F, CR = self.adaptive_params.sample_parameters(
                    strategy_idx
                )

                # ── Apply strategy ─────────────────────────
                new_solution = self.apply_strategy(
                    i, strategy_idx, t, F, CR
                )
                new_solution = self.check_bounds(new_solution)

                # ── Evaluate ───────────────────────────────
                old_fitness = self.fitness[i]
                new_fitness = fitness_function(new_solution)

                # ── Calculate reward for Q-agent ───────────
                reward = self.agent.calculate_reward(
                    old_fitness, new_fitness
                )

                # ── Greedy selection ───────────────────────
                success = new_fitness < old_fitness
                if success:
                    self.population[i] = new_solution
                    self.fitness[i]    = new_fitness

                    if new_fitness < self.Fbest:
                        self.Fbest = new_fitness
                        self.Xbest = new_solution.copy()

                # ── Update adaptive parameters ─────────────
                self.adaptive_params.update_memory(
                    strategy_idx, F, CR, success
                )

                # ── Update Q-table ─────────────────────────
                # Calculate new state after action
                imp_rate = self._calculate_improvement_rate()
                new_state = self.agent.get_state(
                    t, self.T, imp_rate
                )
                self.agent.update(new_state, reward)

                # Current state for next solution
                current_state = new_state

            # ── End of iteration ───────────────────────────

            # Decay epsilon — explore less over time
            self.agent.decay_epsilon()

            # Update improvement tracking
            self.improvement_rate = (
                self._calculate_improvement_rate()
            )
            self.prev_best_fitness = self.Fbest

            # Track convergence
            self.convergence_curve.append(self.Fbest)

            # Track strategy preferences
            best_per_state = (
                self.agent.get_best_strategy_per_state()
            )
            self.strategy_history.append(
                best_per_state.copy()
            )

            if verbose and t % 100 == 0:
                param_stats = self.adaptive_params.get_stats()
                print(f"  Iter {t}/{self.T} | "
                      f"Best: {self.Fbest:.6f} | "
                      f"ε={self.agent.epsilon:.3f} | "
                      f"F=[" +
                      ", ".join(
                          f"{s['mean_F']:.2f}"
                          for s in param_stats
                      ) + "] | CR=[" +
                      ", ".join(
                          f"{s['mean_CR']:.2f}"
                          for s in param_stats
                      ) + "]")

        return self.Xbest, self.Fbest, self.convergence_curve

    def reset(self):
        """Reset for multiple independent runs"""
        self.population           = None
        self.fitness              = None
        self.Xbest                = None
        self.Fbest                = np.inf
        self.prev_best_fitness    = np.inf
        self.improvement_rate     = 0.0
        self.convergence_curve    = []
        self.strategy_history     = []
        self.strategy_assignments = np.zeros(
            self.PS, dtype=int
        )
        self.agent.reset()
        self.adaptive_params.reset()                
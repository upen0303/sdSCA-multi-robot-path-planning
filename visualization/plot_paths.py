"""
Path Visualization for Multi-Robot Path Planning
=================================================
Generates figures similar to Figure 9, 12, 15 in the paper.

Each plot shows:
- Robot start positions (colored circles)
- Robot goal positions (colored crosses)
- Robot traveled paths (colored lines)
- Ideal straight line paths (colored dashed lines)
- Static obstacles (gray filled shapes)
- Dynamic obstacle paths (black dash-dotted lines)
- Dynamic obstacle start positions (black circles)
- Dynamic obstacle goal positions (black squares)

Author: Your Name
Date: 2024
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle, Rectangle, FancyArrowPatch
import os


# ── Color scheme for robots (matches paper style) ──────────
ROBOT_COLORS = [
    '#e41a1c',   # red
    '#377eb8',   # blue
    '#4daf4a',   # green
    '#984ea3',   # purple
    '#ff7f00',   # orange
    '#a65628',   # brown
    '#f781bf',   # pink
    '#999999',   # gray
    '#1b9e77',   # teal
    '#d95f02',   # dark orange
    '#7570b3',   # medium purple
    '#e7298a',   # magenta
]


def plot_scenario_paths(env,
                        algorithm_name,
                        scenario_number,
                        save_path=None,
                        show=True):
    """
    Plot complete simulation results for one scenario.
    Similar to Figures 9, 12, 15 in the paper.

    Parameters:
    -----------
    env            : Environment object (after simulation)
    algorithm_name : str — e.g. 'sdSCA', 'SCA', 'WOA'
    scenario_number: int — 1, 2, or 3
    save_path      : str — file path to save figure
                     None = don't save
    show           : bool — display figure
    """

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    # ── Plot environment boundary ──────────────────────────
    ax.set_xlim(0, env.width)
    ax.set_ylim(0, env.height)
    ax.set_aspect('equal')
    ax.set_facecolor('#f8f8f8')

    # Border
    border = plt.Rectangle(
        (0, 0), env.width, env.height,
        fill=False, edgecolor='black', linewidth=2
    )
    ax.add_patch(border)

    # ── Plot static obstacles ──────────────────────────────
    for obs in env.static_obstacles:
        pos    = obs.get_position()
        shape  = obs.shape
        radius = obs.radius

        if shape == 'circle':
            circle = Circle(
                pos, radius,
                color='#888888',
                alpha=0.7,
                zorder=3
            )
            ax.add_patch(circle)

        elif shape == 'square':
            square = Rectangle(
                (pos[0] - radius, pos[1] - radius),
                radius * 2, radius * 2,
                color='#888888',
                alpha=0.7,
                zorder=3
            )
            ax.add_patch(square)

        elif shape == 'triangle':
            triangle = plt.Polygon(
                [
                    [pos[0],          pos[1] + radius],
                    [pos[0] - radius, pos[1] - radius],
                    [pos[0] + radius, pos[1] - radius],
                ],
                color='#888888',
                alpha=0.7,
                zorder=3
            )
            ax.add_patch(triangle)

    # ── Plot dynamic obstacles ─────────────────────────────
    for obs in env.dynamic_obstacles:

        start = obs.start_pos
        goal  = obs.goal_pos

        # Path line (black dash-dotted)
        ax.plot(
            [start[0], goal[0]],
            [start[1], goal[1]],
            color='black',
            linestyle='-.',
            linewidth=1.5,
            zorder=4,
            label='_nolegend_'
        )

        # Start position (black circle)
        dyn_circle = Circle(
            obs.current_pos, obs.radius,
            color='black',
            alpha=0.8,
            zorder=5
        )
        ax.add_patch(dyn_circle)

        # Label (D1, D2, D3...)
        ax.annotate(
            f'D{obs.id}',
            obs.start_pos,
            fontsize=8,
            color='white',
            ha='center',
            va='center',
            fontweight='bold',
            zorder=6
        )

        # Goal position (black square marker)
        ax.plot(
            goal[0], goal[1],
            marker='s',
            color='black',
            markersize=8,
            zorder=5
        )

    # ── Plot robots ────────────────────────────────────────
    for i, robot in enumerate(env.robots):

        color = ROBOT_COLORS[i % len(ROBOT_COLORS)]

        # ── Ideal path (dashed line) ───────────────────────
        ax.plot(
            [robot.start_pos[0], robot.goal_pos[0]],
            [robot.start_pos[1], robot.goal_pos[1]],
            color=color,
            linestyle='--',
            linewidth=1.0,
            alpha=0.5,
            zorder=2
        )

        # ── Actual traveled path ───────────────────────────
        path = np.array(robot.path_history)
        if len(path) > 1:
            ax.plot(
                path[:, 0], path[:, 1],
                color=color,
                linewidth=1.5,
                alpha=0.9,
                zorder=6
            )

        # ── Start position (filled circle) ─────────────────
        start_circle = Circle(
            robot.start_pos, robot.radius * 1.5,
            color=color,
            alpha=1.0,
            zorder=7
        )
        ax.add_patch(start_circle)

        # Robot label
        ax.annotate(
            f'R{robot.id}',
            robot.start_pos,
            fontsize=7,
            color='white',
            ha='center',
            va='center',
            fontweight='bold',
            zorder=8
        )

        # ── Goal position (cross marker) ───────────────────
        ax.plot(
            robot.goal_pos[0],
            robot.goal_pos[1],
            marker='x',
            color=color,
            markersize=10,
            markeredgewidth=2,
            zorder=7
        )

    # ── Title and labels ───────────────────────────────────
    NR = len(env.robots)
    NS = len(env.static_obstacles)
    ND = len(env.dynamic_obstacles)

    ax.set_title(
        f'Scenario {scenario_number} — {algorithm_name}\n'
        f'({NR} robots, {NS} static, {ND} dynamic obstacles)',
        fontsize=12,
        fontweight='bold',
        pad=15
    )
    ax.set_xlabel('X position (cm)', fontsize=10)
    ax.set_ylabel('Y position (cm)', fontsize=10)

    # ── Legend ─────────────────────────────────────────────
    legend_elements = [
        mpatches.Patch(color='#888888', alpha=0.7,
                       label='Static obstacle'),
        mpatches.Patch(color='black',   alpha=0.8,
                       label='Dynamic obstacle'),
        plt.Line2D([0], [0], color='gray', linestyle='--',
                   label='Ideal path'),
        plt.Line2D([0], [0], color='gray', linestyle='-',
                   label='Actual path'),
        plt.Line2D([0], [0], marker='x', color='gray',
                   linestyle='None', markersize=8,
                   label='Goal position'),
    ]
    ax.legend(
        handles=legend_elements,
        loc='upper right',
        fontsize=8,
        framealpha=0.9
    )

    ax.grid(True, alpha=0.3, linestyle=':')
    plt.tight_layout()

    # ── Save figure ────────────────────────────────────────
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Figure saved: {save_path}")

    if show:
        plt.show()

    plt.close()


def plot_fitness_curves(convergence_data,
                        algorithm_names,
                        scenario_number,
                        save_path=None,
                        show=True):
    """
    Plot convergence curves for multiple algorithms.
    Similar to Figures 11, 14, 17 in the paper.

    Parameters:
    -----------
    convergence_data : list of lists
                       each inner list = fitness curve
                       of one algorithm
    algorithm_names  : list of str
    scenario_number  : int
    save_path        : str or None
    show             : bool
    """

    fig, ax = plt.subplots(figsize=(10, 6))

    colors    = ['#e41a1c', '#377eb8', '#4daf4a',
                 '#984ea3', '#ff7f00', '#a65628']
    linestyle = ['-', '--', '-.', ':', '-', '--']

    for i, (curve, name) in enumerate(
            zip(convergence_data, algorithm_names)):

        ax.plot(
            curve,
            color     = colors[i % len(colors)],
            linestyle = linestyle[i % len(linestyle)],
            linewidth = 2,
            label     = name
        )

    ax.set_title(
        f'Average Fitness Curves — Scenario {scenario_number}',
        fontsize=13,
        fontweight='bold'
    )
    ax.set_xlabel('Number of Steps', fontsize=11)
    ax.set_ylabel('Average Fitness',  fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Figure saved: {save_path}")

    if show:
        plt.show()

    plt.close()


def plot_steps_bar(results_dict,
                   scenario_number,
                   save_path=None,
                   show=True):
    """
    Plot bar graph of required steps per robot.
    Similar to Figures 10, 13, 16 in the paper.

    Parameters:
    -----------
    results_dict : dict
                   keys   = algorithm names
                   values = list of steps per robot
    scenario_number : int
    save_path    : str or None
    show         : bool
    """

    algorithms = list(results_dict.keys())
    NR         = len(list(results_dict.values())[0])
    robot_ids  = [f'Robot #{i+1}' for i in range(NR)]

    x     = np.arange(NR)
    width = 0.8 / len(algorithms)

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = ['#e41a1c', '#377eb8', '#4daf4a',
              '#984ea3', '#ff7f00', '#a65628']

    for i, (algo, steps) in enumerate(results_dict.items()):
        offset = (i - len(algorithms)/2 + 0.5) * width
        bars   = ax.bar(
            x + offset, steps,
            width,
            label     = algo,
            color     = colors[i % len(colors)],
            alpha     = 0.85,
            edgecolor = 'white',
            linewidth = 0.5
        )

    ax.set_title(
        f'Required Steps per Robot — Scenario {scenario_number}',
        fontsize=13,
        fontweight='bold'
    )
    ax.set_xlabel('Robot',          fontsize=11)
    ax.set_ylabel('Required Steps', fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(robot_ids, fontsize=9)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Figure saved: {save_path}")

    if show:
        plt.show()

    plt.close()
import torch
import numpy as np
import matplotlib.pyplot as plt

def plot_trajectories(x, T, obstacles_data, num_obstacles, save=False, filename=''):
    """
    Plots the robot's trajectory and multiple obstacles.

    Args:
        x (torch.Tensor): The robot's trajectory of shape (T, state_dim).
        T (int): The time horizon to plot.
        obstacles_data (torch.Tensor): A tensor of shape (num_obstacles, 3)
                                       where each row is [center_x, center_y, radius].
        num_obstacles (int): The number of obstacles.
        save (bool): Whether to save the plot.
        filename (str): The filename for the saved plot.
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot trajectory
    ax.plot(x[:T, 0].detach().cpu(), x[:T, 1].detach().cpu(), color='tab:blue', linewidth=1.5, label='Robot Path')
    ax.plot(x[0, 0].detach().cpu(), x[0, 1].detach().cpu(), 'go', markersize=10, label='Start')
    ax.plot(0, 0, 'rx', markersize=12, markeredgewidth=3, label='Goal (Origin)')

    # Plot obstacles
    for i in range(num_obstacles):
        center = obstacles_data[i, 0:2].cpu()
        radius = obstacles_data[i, 2].cpu()
        circle = plt.Circle((center[0], center[1]), radius, color='k', fill=True, alpha=0.3, zorder=10)
        ax.add_patch(circle)

    ax.set_title('Robot Trajectory with Multiple Obstacles')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.grid(True, linestyle=':')
    ax.set_aspect('equal', adjustable='box')
    ax.legend()

    if save:
        plt.savefig(filename + '.png', format='png')
    plt.show()


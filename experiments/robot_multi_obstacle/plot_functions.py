import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def plot_multi_obstacle_performance(
        loss_fn: 'RobotsLossMultiObstacle',
        ctl, sys,
        scenario: dict,
        horizon: int,
        resolution=75,
        bounds=(-6, 6),
        vmax_percentile=99.5,
        save=False,
        filename='multi_obstacle_performance.png'
):
    """
    Generates a performance visualization with the corrected torch.max call.
    """
    print("\n--- Generating Multi-Obstacle Performance Visualization (Corrected) ---")

    fig, ax = plt.subplots(figsize=(10, 10))
    ctl.eval()

    # Extract scenario details
    obstacle_centers = scenario['centers']
    obstacle_radii = scenario['radii']  # This is a 1D tensor of shape (3,)
    start_point = scenario['start_point']
    num_obstacles = obstacle_centers.shape[0]

    # --- 1. Static Cost Heatmap Calculation ---
    print("Calculating static cost field...")
    x_coords = np.linspace(bounds[0], bounds[1], resolution)
    y_coords = np.linspace(bounds[0], bounds[1], resolution)
    xv, yv = np.meshgrid(x_coords, y_coords)
    grid_points = torch.tensor(np.stack([xv, yv], axis=-1).reshape(-1, 2), dtype=torch.float32)

    # --- Multi-Obstacle Cost ---
    dist_vectors = grid_points.unsqueeze(1) - obstacle_centers.unsqueeze(0)
    dist_centers = torch.norm(dist_vectors, dim=-1)
    dist_edge = dist_centers - (loss_fn.radius_robot + obstacle_radii.unsqueeze(0))

    barrier_cost_per_obs = loss_fn.alpha_barrier / torch.clamp(dist_edge, min=1e-4)
    corridor_cost_per_obs = loss_fn.alpha_corridor * (dist_edge - loss_fn.d_safe) ** 2

    cost_obst = (barrier_cost_per_obs + corridor_cost_per_obs).sum(dim=1)

    # --- State Cost ---
    full_state_grid = torch.cat([grid_points, torch.zeros_like(grid_points)], dim=1)
    cost_x_unscaled = (full_state_grid.unsqueeze(1) @ loss_fn.Q @ full_state_grid.unsqueeze(-1)).squeeze()

    # ======================================================================
    # --- THE FIX IS HERE ---
    # `torch.max` on a 1D tensor returns a single scalar (0-D tensor).
    # We assign it directly to one variable instead of trying to unpack it.
    max_radius = torch.max(obstacle_radii)
    # ======================================================================

    q_scaling_factor = torch.exp(-loss_fn.alpha_q_scaling * max_radius)
    cost_x = q_scaling_factor * cost_x_unscaled

    total_static_cost = cost_x + cost_obst

    # ... (The rest of the function remains exactly the same) ...

    # Plotting the heatmap
    log_cost = torch.log(total_static_cost + 1e-6)
    Z = log_cost.view(resolution, resolution).detach().numpy()
    vmin = np.percentile(Z[np.isfinite(Z)], 0.5)
    vmax = np.percentile(Z[np.isfinite(Z)], vmax_percentile)
    contour = ax.contourf(xv, yv, Z, levels=100, cmap='magma_r', vmin=vmin, vmax=vmax)
    fig.colorbar(contour, ax=ax, label='Log(Static Cost Field)')

    # --- 2. Simulate and Overlay the Trajectory ---
    print("Simulating trajectory for the specific start point...")
    initial_data = torch.zeros(1, horizon, 4 + num_obstacles * 3)
    initial_data[0, 0, :4] = torch.cat([start_point, torch.zeros(2)])
    obstacle_info_flat = torch.cat([obstacle_centers, obstacle_radii.unsqueeze(1)], dim=1).flatten()
    initial_data[0, :, 4:] = obstacle_info_flat.repeat(horizon, 1)

    with torch.no_grad():
        x_log, _ = sys.rollout(ctl, initial_data, train=False)

    trajectory = x_log[0, :, :].detach().numpy()

    ax.plot(trajectory[:, 0], trajectory[:, 1], 'c--', label='Robot Path', lw=2.5)
    ax.plot(start_point[0], start_point[1], 'go', markersize=12, label='Start', markeredgecolor='k')

    # --- 3. Plot Environment and Final Touches ---
    ax.plot(0, 0, 'y*', markersize=18, label='Goal (Origin)', markeredgecolor='k')
    for i in range(num_obstacles):
        circle = plt.Circle(tuple(obstacle_centers[i].cpu()), obstacle_radii[i].cpu(), color='r', fill=True, alpha=0.6,
                            zorder=10)
        ax.add_patch(circle)
    ax.plot([], [], 'o', color='r', alpha=0.6, markersize=10, label='Obstacles')

    ax.set_title(f'Controller Performance in Narrow Passage')
    ax.set_xlabel('X Coordinate');
    ax.set_ylabel('Y Coordinate')
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linestyle=':')
    ax.legend(fontsize=12)

    if save:
        plt.savefig(filename, dpi=300)
    plt.show()
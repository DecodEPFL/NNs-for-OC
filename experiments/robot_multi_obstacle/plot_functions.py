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


def plot_multi_obstacle_value_landscape(
        loss_fn, ctl, sys,
        scenario_for_plot: dict,
        horizon: int,
        resolution: int = 50,
        bounds: tuple = (-6, 6),
        vmax_percentile: float = 99.0,
        save: bool = False,
        filename: str = 'multi_obstacle_value_landscape.png'
):
    """
    Generates the ultimate multi-obstacle analysis plot, coherent with the data pipeline.

    1.  Calculates and plots the "Value Landscape" (Cost-to-Go) heatmap.
    2.  Simulates and overlays a primary trajectory for the given scenario.
    3.  Overlays velocity vectors (quiver plot) to show dynamics.
    4.  Overlays a "threat-awareness" trail to show which obstacle the controller
        prioritizes at each moment.
    """
    print("\n--- Generating Comprehensive Multi-Obstacle Value Landscape ---")

    fig, ax = plt.subplots(figsize=(12, 12))
    ctl.eval()

    # --- Part 1: Value Landscape Heatmap ---
    print("Step 1/4: Calculating Value Landscape (this may take a while)...")

    # Grid of starting points for the heatmap
    x_coords = np.linspace(bounds[0], bounds[1], resolution)
    y_coords = np.linspace(bounds[0], bounds[1], resolution)
    xv, yv = np.meshgrid(x_coords, y_coords)
    grid_points = torch.tensor(np.stack([xv, yv], axis=-1).reshape(-1, 2), dtype=torch.float32)

    # Use the fixed obstacle configuration from the provided scenario
    obstacle_centers = scenario_for_plot['centers']
    obstacle_radii = scenario_for_plot['radii']
    num_obstacles = len(obstacle_radii)

    # Prepare static obstacle info for batch processing
    obstacle_info = torch.cat([obstacle_centers, obstacle_radii.unsqueeze(1)], dim=1).flatten()
    static_obstacle_data = obstacle_info.unsqueeze(0).repeat(horizon, 1)

    total_losses = []
    batch_size = 256
    for i in tqdm(range(0, len(grid_points), batch_size)):
        batch_starts = grid_points[i:i + batch_size]
        B = len(batch_starts)

        # Prepare the initial data tensor for the batch of simulations
        initial_data = torch.zeros(B, horizon, 4 + num_obstacles * 3)
        initial_data[:, 0, :2] = batch_starts
        initial_data[:, :, 4:] = static_obstacle_data.unsqueeze(0).repeat(B, 1, 1)

        with torch.no_grad():
            # Step 1: Simulate the trajectories for the current batch
            # We assume sys.rollout returns (robot_state_log, control_log, context_log)
            # The robot state log `x_log` should have shape (B, T, 4)
            x_log, u_log = sys.rollout(ctl, initial_data, train=False)

            B, T, _ = x_log.shape  # Get batch size and horizon from results

            # Prepare tensors for matrix multiplication
            x_batch = x_log.unsqueeze(-1)  # (B, T, 4, 1)
            u_batch = u_log.unsqueeze(-1)  # (B, T, 2, 1)

            # --- State Cost (loss_x) ---
            # Use the single max radius from the scenario for all samples in the batch
            max_r = torch.max(obstacle_radii)
            q_scale = torch.exp(-loss_fn.alpha_q_scaling * max_r)
            # Sum over time (T), keep batch dimension (B)
            xTQx = (x_batch.transpose(-1, -2) @ loss_fn.Q @ x_batch).sum(dim=1)
            loss_x = q_scale * xTQx

            # --- Control Cost (loss_u) ---
            uTRu = loss_fn.R * (u_batch.transpose(-1, -2) @ u_batch)
            loss_u = uTRu.sum(dim=1)  # Sum over time (T)

            # --- Obstacle Cost (loss_obst) ---
            # First, construct the `obs_data` tensor that the loss function expects.
            # Shape needs to be (B, num_obstacles, 3)
            obs_centers_batch = obstacle_centers.unsqueeze(0).repeat(B, 1, 1)
            obs_radii_batch = obstacle_radii.unsqueeze(0).repeat(B, 1).unsqueeze(-1)
            obs_data_batch = torch.cat([obs_centers_batch, obs_radii_batch], dim=-1)

            # Now, call the loss function's helper method with the correct arguments
            # This helper method should return a tensor of shape (B, 1, 1)
            loss_obst = loss_fn.f_loss_obst_multi(x_log, obs_data_batch)

            # --- Total Loss for Each Sample in the Batch ---
            # Squeeze to remove trailing dimensions, resulting in shape (B,)
            batch_total_loss = (loss_x + loss_u + loss_obst).squeeze()

            # Append the losses for this batch to our master list
            total_losses.append(batch_total_loss.cpu())

    all_total_losses = torch.cat(total_losses)

    # Plotting the heatmap
    log_losses = torch.log(all_total_losses + 1e-6)
    Z = log_losses.view(resolution, resolution).detach().numpy()
    vmin = np.percentile(Z[np.isfinite(Z)], 1)
    vmax = np.percentile(Z[np.isfinite(Z)], vmax_percentile)
    contour = ax.contourf(xv, yv, Z, levels=100, cmap='viridis_r', vmin=vmin, vmax=vmax)
    fig.colorbar(contour, ax=ax, label='Log(Total Trajectory Cost)')

    # --- Part 2: Primary Trajectory Simulation ---
    print("Step 2/4: Simulating primary trajectory...")
    primary_start_point = scenario_for_plot['start_point']
    primary_initial_data = torch.zeros(1, horizon, 4 + num_obstacles * 3)
    primary_initial_data[0, 0, :4] = torch.cat([primary_start_point, torch.zeros(2)])
    primary_initial_data[0, :, 4:] = static_obstacle_data

    with torch.no_grad():
        x_log_primary, _ = sys.rollout(ctl, primary_initial_data, train=False)

    trajectory = x_log_primary[0].detach().numpy()
    ax.plot(trajectory[:, 0], trajectory[:, 1], 'w--', label='Robot Path', lw=2.5, zorder=11)
    ax.plot(primary_start_point[0], primary_start_point[1], 'go', markersize=12, label='Start', markeredgecolor='k',
            zorder=12)

    # --- Part 3: Threat Awareness & Velocity Overlay ---
    print("Step 3/4: Analyzing trajectory details...")
    positions = trajectory[:, :2]
    velocities = trajectory[:, 2:]

    # Plot velocity vectors (quiver plot)
    skip = 25  # Plot every 25th vector to avoid clutter
    ax.quiver(positions[::skip, 0], positions[::skip, 1], velocities[::skip, 0], velocities[::skip, 1],
              color='white', scale=25, width=0.005, headwidth=4, alpha=0.9, zorder=11)

    # Plot threat awareness breadcrumbs
    threat_colors = ['cyan', 'lime', 'magenta']
    for i in range(len(positions)):
        pos = torch.from_numpy(positions[i]).float()
        dist_edge = torch.norm(pos - obstacle_centers, dim=-1) - (loss_fn.radius_robot + obstacle_radii)
        _, primary_threat_idx = torch.min(dist_edge, dim=0)
        color = threat_colors[primary_threat_idx.item() % len(threat_colors)]
        ax.plot(pos[0], pos[1], '.', color=color, markersize=5, alpha=0.7, zorder=10)

    # --- Part 4: Final Plot Formatting ---
    print("Step 4/4: Finalizing plot...")
    ax.plot(0, 0, 'y*', markersize=18, label='Goal (Origin)', markeredgecolor='k', zorder=12)
    for i in range(num_obstacles):
        circle = plt.Circle(tuple(obstacle_centers[i].cpu()), obstacle_radii[i].cpu(), color='r', fill=True, alpha=0.6,
                            zorder=9)
        ax.add_patch(circle)
        ax.text(obstacle_centers[i, 0], obstacle_centers[i, 1] + obstacle_radii[i] + 0.2, f'Obs #{i}', color='white',
                ha='center', weight='bold')

    # Custom legend for threat awareness
    for i in range(num_obstacles):
        ax.plot([], [], 'o', color=threat_colors[i % len(threat_colors)], label=f'Threat: Obs #{i}')

    ax.set_title(f'Multi-Obstacle Controller Performance Analysis')
    ax.set_xlabel('X Coordinate');
    ax.set_ylabel('Y Coordinate')
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linestyle=':')
    ax.legend(fontsize=11)

    if save:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
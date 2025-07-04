import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


# plt.rcParams['text.usetex'] = True


def plot_single_trajectory_on_ax(ax, x, T, obstacle_centers, obstacle_radius,
                                 color='tab:blue', label=None, show_start=True):
    """
    A helper function to draw a single robot trajectory and its corresponding
    obstacle onto a GIVEN matplotlib Axes object (ax).

    This is the building block for more complex visualizations.
    """
    # Plot the main trajectory
    ax.plot(
        x[:T + 1, 0].detach(), x[:T + 1, 1].detach(),
        color=color, linewidth=1.5, label=label
    )

    # Plot the starting point
    if show_start:
        ax.plot(
            x[0, 0].detach(), x[0, 1].detach(),
            color=color, marker='o', markersize=8, markeredgewidth=1.5,
            markerfacecolor='none'  # Make it an open circle for clarity
        )

    # Plot the obstacle
    if obstacle_centers is not None:
        # Note: obstacle_radius might be a tensor, so we get the item
        r = obstacle_radius.item() if torch.is_tensor(obstacle_radius) else obstacle_radius
        circle = plt.Circle((obstacle_centers[0, 0], obstacle_centers[0, 1]),
                            r, color=color, fill=False, linestyle='--',
                            alpha=0.8, zorder=10, lw=2
                            )
        ax.add_patch(circle)


def plot_trajectories(
        x, save=False, filename='', T=100, obst=False,
        dots=False, circles=False, radius_robot=1, f=5,
        obstacle_centers=None, obstacle_radius=None
):
    # fig = plt.figure(f)
    fig, ax = plt.subplots(figsize=(f, f))
    # plot obstacles

    colors = ['tab:blue', 'tab:orange']
    ax.plot(
        x[:T + 1, 0].detach(), x[:T + 1, 1].detach(),
        color=colors[0], linewidth=1
    )
    ax.plot(
        x[T:, 0].detach(), x[T:, 1].detach(),
        color='k', linewidth=0.1, linestyle='dotted', dashes=(3, 15)
    )
    ax.plot(
        x[0, 0].detach(), x[0, 1].detach(),
        color=colors[0], marker='8'
    )
    if dots:
        for j in range(T):
            ax.plot(
                x[j, 0].detach(), x[j, 1].detach(),
                color=colors[0], marker='o'
            )
    if circles:
        r = radius_robot
        circle = plt.Circle((x[T - 1, 0].detach(), x[T - 1, 1].detach()),
                            r, color=colors[0], alpha=0.5, zorder=10
                            )
        ax.add_patch(circle)
    if obstacle_centers is not None:
        r = obstacle_radius[0, 0]
        circle = plt.Circle((obstacle_centers[0, 0], obstacle_centers[0, 1]),
                            r, color='k', alpha=0.1, zorder=10
                            )
        ax.add_patch(circle)
    if save:
        plt.savefig(filename + '.pdf', format='pdf')
    plt.show()


def plot_traj_vs_time(t_end, x, u=None, save=False, filename=''):
    t = torch.linspace(0, t_end - 1, t_end)
    if u is not None:
        p = 3
    else:
        p = 2
    plt.figure(figsize=(4 * p, 4))
    plt.subplot(1, p, 1)
    plt.plot(t, x[:, 0].detach())
    plt.plot(t, x[:, 1].detach())
    plt.xlabel(r'$t$')
    plt.title(r'$p(t)$ - position')
    plt.subplot(1, p, 2)
    plt.plot(t, x[:, 2].detach())
    plt.plot(t, x[:, 3].detach())
    plt.xlabel(r'$t$')
    plt.title(r'$q(t)$ - velocity')
    if p == 3:
        plt.subplot(1, 3, 3)
        plt.plot(t, u[:, 0].detach())
        plt.plot(t, u[:, 1].detach())
        plt.xlabel(r'$t$')
        plt.title(r'$u(t)$')
    if save:
        plt.savefig(filename + '.pdf', format='pdf')
    else:
        plt.show()


# ==============================================================================
# NEW VISUALIZATION FUNCTIONS (TAILORED FOR YOUR CODE)
# ==============================================================================

def plot_radius_sweep(ctl, sys, start_point, center, radii_to_test, horizon):
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.viridis(np.linspace(0, 1, len(radii_to_test)))

    print("\n--- Generating Radius Sweep Plot ---")

    for i, radius in enumerate(radii_to_test):
        # 1. Create the data point for this simulation
        plot_data = torch.zeros(1, horizon, 7)  # state_dim=4, obs_dim=3
        plot_data[0, 0, 0:4] = torch.tensor([start_point[0], start_point[1], 0, 0])
        plot_data[0, 0, 4:7] = torch.cat((center, torch.tensor([radius])))

        # 2. Simulate the trajectory
        with torch.no_grad():
            x_log, _, _ = sys.rollout(ctl, plot_data, train=False)

        # 3. Plot using our new helper function
        plot_single_trajectory_on_ax(
            ax=ax, x=x_log[0], T=horizon,
            obstacle_centers=plot_data[0, 0, 4:6].unsqueeze(0),
            obstacle_radius=plot_data[0, 0, 6:7],
            color=colors[i], label=f'Radius = {radius:.2f}',
            show_start=False  # Only plot the single start point once later
        )

    # Formatting
    ax.plot(start_point[0], start_point[1], 'go', markersize=10, label='Start')
    ax.plot(0, 0, 'rx', markersize=12, markeredgewidth=3, label='Goal (Origin)')
    ax.set_title(f'Trajectory vs. Obstacle Radius')
    ax.set_xlabel('X Coordinate');
    ax.set_ylabel('Y Coordinate')
    ax.grid(True, linestyle=':');
    ax.set_aspect('equal', adjustable='box')
    ax.legend()
    fig.savefig('plot_radius_sweep.png')
    plt.show()


def plot_facet_grid(ctl, sys, start_points, radii_to_test, center, horizon):
    nrows = len(start_points)
    ncols = len(radii_to_test)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 4), sharex=True, sharey=True)
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, -1)
    elif ncols == 1:
        axes = axes.reshape(-1, 1)

    print("\n--- Generating Facet Grid Plot ---")

    for i, start_point in enumerate(start_points):
        for j, radius in enumerate(radii_to_test):
            ax = axes[i, j]

            # 1. Create data and simulate
            plot_data = torch.zeros(1, horizon, 7)
            plot_data[0, 0, 0:4] = torch.tensor([start_point[0], start_point[1], 0, 0])
            plot_data[0, 0, 4:7] = torch.cat((center, torch.tensor([radius])))

            with torch.no_grad():
                x_log, _, _ = sys.rollout(ctl, plot_data, train=False)

            # 2. Plot on the specific subplot using the helper
            plot_single_trajectory_on_ax(
                ax=ax, x=x_log[0], T=horizon,
                obstacle_centers=plot_data[0, 0, 4:6].unsqueeze(0),
                obstacle_radius=plot_data[0, 0, 6:7],
                color='tab:blue'  # Use a consistent color for the grid
            )
            # Add goal marker
            ax.plot(0, 0, 'rx', markersize=10)

            # 3. Set titles
            if i == 0: ax.set_title(f'Radius = {radius:.2f}', fontsize=12)
            if j == 0: ax.set_ylabel(f'Start ({start_point[0]:.1f}, {start_point[1]:.1f})', fontsize=12, rotation=90,
                                     labelpad=20)

            ax.grid(True, linestyle=':');
            ax.set_aspect('equal', adjustable='box')
            ax.set_xlim(-5, 5);
            ax.set_ylim(-5, 5)

    fig.suptitle('Controller Performance Matrix', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig('plot_facet_grid.png')
    plt.show()


def plot_loss_landscape(
        loss_fn, ctl, sys, start_point, center, radius, horizon,
        zoom_to_trajectory=True,
        zoom_padding=1.5,
        vmax_percentile=99.0  # Lowered percentile for more detail
):
    """
    A corrected version that properly zooms the color scale by applying the
    percentile cutoff to the log-transformed cost data.
    """
    print("\n--- Generating Correctly Zoomed Loss Landscape Heatmap Plot ---")

    fig, ax = plt.subplots(figsize=(10, 10))

    # --- 1. Grid and Cost Calculation (Identical to before) ---
    resolution = 200
    bounds = (-6, 6)
    x_coords = np.linspace(bounds[0], bounds[1], resolution)
    y_coords = np.linspace(bounds[0], bounds[1], resolution)
    xv, yv = np.meshgrid(x_coords, y_coords)
    robot_pos = torch.tensor(np.stack([xv, yv], axis=-1).reshape(-1, 2), dtype=torch.float32)

    Q = loss_fn.Q;
    radius_robot = loss_fn.radius_robot
    alpha_barrier = loss_fn.alpha_barrier;
    alpha_corridor = loss_fn.alpha_corridor
    d_safe = loss_fn.d_safe;
    alpha_q_scaling = loss_fn.alpha_q_scaling

    obstacle_centers = center.unsqueeze(0).repeat(robot_pos.shape[0], 1)
    obstacle_radii_tensor = torch.tensor(radius).repeat(robot_pos.shape[0])

    dist_center = torch.norm(robot_pos - obstacle_centers, dim=-1)
    dist_edge = dist_center - (radius_robot + obstacle_radii_tensor)

    barrier_cost = alpha_barrier / torch.clamp(dist_edge, min=1e-4)
    corridor_cost = alpha_corridor * (dist_edge - d_safe) ** 2
    cost_obst = barrier_cost + corridor_cost

    full_state = torch.cat([robot_pos, torch.zeros_like(robot_pos)], dim=1)
    cost_x_unscaled = (full_state.unsqueeze(1) @ Q @ full_state.unsqueeze(-1)).squeeze()
    q_scaling_factor = torch.exp(-torch.tensor(alpha_q_scaling) * radius)
    cost_x = q_scaling_factor * cost_x_unscaled

    total_cost = cost_x + cost_obst

    # --- 2. CORRECTED: Log Transform first, THEN calculate color scale ---

    # First, take the log of the total cost
    log_cost = torch.log(total_cost + 1e-6)
    Z = log_cost.view(resolution, resolution).detach().numpy()

    # Now, find the percentiles *on the log-transformed data*
    valid_log_costs = Z[np.isfinite(Z)]
    vmin_val = valid_log_costs.min()
    vmax_val = np.percentile(valid_log_costs, vmax_percentile)

    # --- 3. Plot the Heatmap with the correctly calculated color scale ---
    contour = ax.contourf(xv, yv, Z, levels=50, cmap='magma', vmin=vmin_val, vmax=vmax_val)
    fig.colorbar(contour, ax=ax, label='Log(Static Cost) (Zoomed Scale)')

    # --- 4. Simulate and Overlay Trajectory (Identical to before) ---
    plot_data = torch.zeros(1, horizon, 7)
    plot_data[0, 0, 0:4] = torch.tensor([start_point[0], start_point[1], 0, 0])
    plot_data[0, 0, 4:7] = torch.cat((center, torch.tensor([radius])))
    with torch.no_grad():
        x_log, _, _ = sys.rollout(ctl, plot_data, train=False)
    trajectory = x_log[0, :, :].detach().numpy()

    ax.plot(trajectory[:, 0], trajectory[:, 1], 'w.--', label='Actual Path', lw=2)
    ax.plot(start_point[0], start_point[1], 'go', markersize=10, label='Start')
    ax.plot(0, 0, 'c*', markersize=15, label='Goal (Origin)', markeredgecolor='k')

    obstacle_circle = plt.Circle(center, radius, color='w', fill=False, linestyle='--', lw=2)
    ax.add_artist(obstacle_circle)

    # --- 5. Spatial Zoom and Formatting (Identical to before) ---
    if zoom_to_trajectory:
        min_x = min(trajectory[:, 0].min(), 0);
        max_x = max(trajectory[:, 0].max(), start_point[0])
        min_y = min(trajectory[:, 1].min(), 0);
        max_y = max(trajectory[:, 1].max(), start_point[1])
        ax.set_xlim(min_x - zoom_padding, max_x + zoom_padding)
        ax.set_ylim(min_y - zoom_padding, max_y + zoom_padding)
    else:
        ax.set_xlim(bounds);
        ax.set_ylim(bounds)

    ax.set_title(f'Corrected Zoom Loss Landscape (Radius: {radius:.2f})')
    ax.set_xlabel('X Coordinate');
    ax.set_ylabel('Y Coordinate')
    ax.set_aspect('equal', adjustable='box')
    ax.legend()
    fig.savefig('plot_loss_landscape_zoomed_corrected.png')
    plt.show()


def plot_value_landscape(
    loss_fn, ctl, sys,
    center, radius, horizon,
    overlay_trajectories_from: list = None, # <-- NEW ARGUMENT
    resolution=100,
    bounds=(-.2, 2.1),
    vmax_percentile=99.0,
    batch_size=3200,
    output_filename="plot_value_landscape.png"
):
    """
    Generates a Value Landscape heatmap and overlays specific example trajectories.
    """
    print("\n--- Generating Value Landscape with Trajectory Overlays ---")

    fig, ax = plt.subplots(figsize=(10, 10))
    ctl.eval()

    # --- 1. Heatmap Calculation (Identical to before) ---
    x_coords = np.linspace(bounds[0], bounds[1], resolution)
    y_coords = np.linspace(bounds[0], bounds[1], resolution)
    xv, yv = np.meshgrid(x_coords, y_coords)
    start_points_grid = torch.tensor(np.stack([xv, yv], axis=-1).reshape(-1, 2), dtype=torch.float32)

    total_losses = []
    obstacle_info_static = torch.cat((center, torch.tensor([radius]))).unsqueeze(0).unsqueeze(0).repeat(1, horizon, 1)

    print(f"Simulating {len(start_points_grid)} trajectories for heatmap...")
    # (The batch processing loop is identical to the previous version)
    for i in tqdm(range(0, len(start_points_grid), batch_size)):
        batch_starts = start_points_grid[i:i + batch_size]
        # ... (rest of the batch processing logic is the same) ...
        # (It calculates `all_total_losses`)
        current_batch_size = len(batch_starts)
        batch_initial_data = torch.zeros(current_batch_size, horizon, 7)
        batch_initial_data[:, 0, 0:2] = batch_starts
        batch_initial_data[:, :, 4:7] = obstacle_info_static.repeat(current_batch_size, 1, 1)
        with torch.no_grad():
            x_log,_,  u_log = sys.rollout(ctl, batch_initial_data, train=False)
            S, T, _ = x_log.shape
            q_scaling = torch.exp(-torch.tensor(loss_fn.alpha_q_scaling) * radius)
            xTQx = x_log.unsqueeze(-2) @ loss_fn.Q @ x_log.unsqueeze(-1)
            loss_x = q_scaling * (xTQx.sum(dim=1) / T)
            uTRu = loss_fn.R * (u_log.unsqueeze(-2) @ u_log.unsqueeze(-1))
            loss_u = uTRu.sum(dim=1) / T
            obstacle_centers_batch = center.view(1, 1, 2).repeat(S, T, 1)
            obstacle_radii_batch = torch.tensor(radius).view(1, 1).repeat(S, T)
            robot_pos = x_log[:, :, 0:2]
            dist_center = torch.norm(robot_pos - obstacle_centers_batch, dim=-1)
            total_radius = loss_fn.radius_robot + obstacle_radii_batch
            dist_edge = dist_center - total_radius
            barrier_cost = loss_fn.alpha_barrier / torch.clamp(dist_edge, min=1e-4)
            corridor_cost = loss_fn.alpha_corridor * (dist_edge - loss_fn.d_safe)**2
            loss_obst = ((barrier_cost + corridor_cost).sum(dim=1) / T).view(S, 1, 1)
            batch_losses = (loss_x + loss_u + loss_obst).squeeze()
            total_losses.append(batch_losses.cpu())
    all_total_losses = torch.cat(total_losses)


    log_losses = torch.log(all_total_losses + 1e-6)
    Z = log_losses.view(resolution, resolution).detach().numpy()
    valid_log_losses = Z[np.isfinite(Z)]
    vmin_val = np.percentile(valid_log_losses, 1)
    vmax_val = np.percentile(valid_log_losses, vmax_percentile)
    contour = ax.contourf(xv, yv, Z, levels=100, cmap='magma_r', vmin=vmin_val, vmax=vmax_val)
    fig.colorbar(contour, ax=ax, label='Log(Average Trajectory Cost)')

    # --- 2. Overlay Key Static Elements ---
    ax.plot(0, 0, 'w*', markersize=15, label='Goal (Origin)', markeredgecolor='k')
    obstacle_circle = plt.Circle(center, radius, color='r', fill=True, alpha=0.4, label='Obstacle')
    ax.add_artist(obstacle_circle)

    # --- 3. NEW: Simulate and Plot Specific Trajectories ---
    if overlay_trajectories_from:
        print(f"Simulating and plotting {len(overlay_trajectories_from)} specific trajectories...")
        colors = plt.cm.cool(np.linspace(0, 1, len(overlay_trajectories_from)))
        for i, start_pos in enumerate(overlay_trajectories_from):
            start_pos = torch.tensor(start_pos, dtype=torch.float32)

            # Prepare data for this single simulation
            initial_data = torch.zeros(1, horizon, 7)
            initial_data[0, 0, 0:2] = start_pos
            initial_data[0, :, 4:7] = obstacle_info_static

            # Simulate
            with torch.no_grad():
                x_log, _, _ = sys.rollout(ctl, initial_data, train=False)

            trajectory = x_log[0, :, :2].detach().numpy()

            # Plot
            ax.plot(trajectory[:, 0], trajectory[:, 1], color=colors[i], linestyle='--', lw=2)
            ax.plot(start_pos[0], start_pos[1], 'o', color=colors[i], markersize=10, markeredgecolor='white', label=f'Path {i+1}')

    # --- 4. Final Formatting ---
    ax.set_title(f'Value Landscape with Trajectories (Radius: {radius:.2f})')
    ax.set_xlabel('X Coordinate'); ax.set_ylabel('Y Coordinate')
    ax.set_aspect('equal', adjustable='box')
    ax.legend()
    fig.savefig(output_filename)
    plt.show()
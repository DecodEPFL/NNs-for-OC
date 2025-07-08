import torch
from assistive_functions import to_tensor

class RobotsLoss_v2:
    """
    A revised loss function designed to make the controller highly sensitive to
    obstacle radius by implementing advanced cost-shaping strategies.

    New Strategies Implemented:
    1.  **Optimal Safety Corridor:** Instead of just punishing proximity, this cost
        creates a low-cost "valley" at a desired safe distance from the
        obstacle's edge. This incentivizes the robot to follow the contour
        of the obstacle, making its path naturally dependent on the radius.
    2.  **Radius-Modulated Aggressiveness:** The state regulation cost (x^T Q x)
        is scaled based on the obstacle's radius. For small, trivial obstacles,
        the cost of deviating from the path to the goal is high (encouraging
        aggression). For large, dangerous obstacles, this cost is reduced,
        allowing the robot to take wider, safer detours.
    """

    def __init__(
            self, Q, alpha_u=.2,
            # --- Obstacle Avoidance Hyperparameters ---
            alpha_barrier=10.0,  # (Replaces alpha_obst) Strength of the hard repulsive barrier.
            alpha_corridor=0,  # NEW: Strength of the "safety corridor" shaping cost.
            d_safe=0.15,  # NEW: Desired safety distance from the obstacle's edge.
            # --- Aggressiveness Modulation Hyperparameter ---
            alpha_q_scaling=1.5,  # NEW: How strongly the radius affects goal-seeking behavior.
            radius_robot=0.02
    ):
        """
        Args:
            Q: The state weighting matrix.
            alpha_u: The control cost weight.
            alpha_barrier: Weight for the barrier function that prevents collision.
            alpha_corridor: Weight for the corridor-shaping function.
            d_safe: The ideal distance to maintain from the obstacle's edge.
            alpha_q_scaling: Controls sensitivity of state cost to radius.
            radius_robot: The robot's own radius.
        """
        self.Q, self.R = to_tensor(Q), to_tensor(alpha_u)
        self.radius_robot = radius_robot

        # New hyperparameters for advanced cost shaping
        self.alpha_barrier = alpha_barrier
        self.alpha_corridor = alpha_corridor
        self.d_safe = d_safe
        self.alpha_q_scaling = alpha_q_scaling

    def forward(self, xs, us, circle, num_obstacles=1):
        """
        Compute loss using advanced cost shaping.

        Args:
            - xs: State trajectory tensor of shape (S, T, state_dim).
            - us: Control trajectory tensor of shape (S, T, in_dim).
            - circle: Dynamic obstacle info tensor, likely (S, T, obs_dim).
                      We assume obs_dim contains [center_x, center_y, radius] for one obstacle,
                      or flattened for multiple, e.g., [c1x, c1y, r1, c2x, c2y, r2, ...].
            - num_obstacles: The number of obstacles.
        """
        S, T, _ = xs.shape  # Samples, Time
        x_batch = xs.unsqueeze(-1)  # (S, T, state_dim, 1)
        u_batch = us.unsqueeze(-1)  # (S, T, in_dim, 1)

        # Extract Obstacle Radius for cost modulation
        # For simplicity with multiple obstacles, we'll use the radius of the first obstacle
        # for the scaling factor. A more advanced implementation could use the radius of the
        # closest obstacle at each time step.
        obstacle_radius = circle[:, 0, 2].view(S, 1, 1, 1) # Index 2 is the radius of the first obstacle

        # --- 1. Radius-Modulated State Cost ---
        q_scaling_factor = torch.exp(-self.alpha_q_scaling * obstacle_radius)
        xTQx = torch.matmul(x_batch.transpose(-1, -2), self.Q) @ x_batch
        loss_x_unscaled = xTQx.sum(dim=1) / T
        loss_x = q_scaling_factor * loss_x_unscaled

        # --- 2. Control Cost (Unchanged) ---
        uTRu = self.R * (u_batch.transpose(-1, -2) @ u_batch)
        loss_u = uTRu.sum(dim=1) / T

        # --- 3. Advanced Obstacle Avoidance Cost ---
        loss_obst = self.f_loss_obst_v2(xs, circle[:, 0, 4:], num_obstacles)

        # --- Total Loss ---
        loss_per_sample = loss_x + loss_u + loss_obst
        final_loss = torch.mean(loss_per_sample)

        return final_loss

    def f_loss_obst_v2(self, xs, obstacle_info, num_obstacles):
        """
        A revised obstacle loss with a smooth barrier and a safety corridor.
        This provides a continuous gradient that guides the robot.
        Handles multiple obstacles by finding the minimum distance to any of them.
        """
        S, T, _ = xs.shape
        robot_pos = xs[:, :, 0:2]  # (S, T, 2)

        # Reshape obstacle info to handle multiple obstacles
        # Input shape: (S, 3 * num_obstacles) -> (S, num_obstacles, 3)
        obstacle_data = obstacle_info.view(S, num_obstacles, 3)
        obstacle_centers = obstacle_data[:, :, 0:2].unsqueeze(1)  # (S, 1, num_obstacles, 2)
        obstacle_radii = obstacle_data[:, :, 2].unsqueeze(1)    # (S, 1, num_obstacles)

        # Expand robot position to calculate distances to all obstacles at once
        robot_pos_expanded = robot_pos.unsqueeze(2)  # (S, T, 1, 2)

        # Calculate distance from robot center to all obstacle centers
        dist_center_sq = torch.sum((robot_pos_expanded - obstacle_centers) ** 2, dim=-1)  # (S, T, num_obstacles)
        dist_center = torch.sqrt(dist_center_sq + 1e-6)  # Add epsilon for stability

        # Distance from robot edge to each obstacle edge
        total_radius = self.radius_robot + obstacle_radii # (S, 1, num_obstacles)
        dist_edge = dist_center - total_radius  # (S, T, num_obstacles)

        # Find the minimum distance to any obstacle at each time step
        min_dist_edge, _ = torch.min(dist_edge, dim=-1) # (S, T)

        # --- Cost Component 1: The Repulsive Barrier ---
        # The cost is based on the closest obstacle.
        barrier_cost = self.alpha_barrier / torch.clamp(min_dist_edge, min=1e-4)

        # --- Cost Component 2: The Optimal Safety Corridor ---
        # This cost is also based on the distance to the closest obstacle.
        corridor_cost = self.alpha_corridor * (min_dist_edge - self.d_safe) ** 2

        # Combine costs and average over the time horizon
        total_obst_cost = (barrier_cost + corridor_cost).sum(dim=1) / T  # (S,)

        return total_obst_cost.view(S, 1, 1)  # Reshape for broadcasting


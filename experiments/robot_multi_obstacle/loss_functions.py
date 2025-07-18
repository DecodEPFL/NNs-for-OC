import torch
from assistive_functions import to_tensor


class RobotsLossMultiObstacle:
    """
    An adapted loss function for a multi-obstacle scenario.

    It calculates the total cost by:
    1.  Summing the repulsive barrier/corridor costs from ALL obstacles.
    2.  Scaling the state regulation cost based on the MAXIMUM radius present
        in the scene, ensuring a conservative policy when any large obstacle exists.
    """

    def __init__(
            self, Q, alpha_u=.2,
            alpha_barrier=10.0,
            alpha_corridor=0.0,
            d_safe=0.15,
            alpha_q_scaling=1.5,
            radius_robot=0.02,
            num_obstacles=3
    ):
        """Initializes the loss with the same hyperparameters as before."""
        self.Q, self.R = to_tensor(Q), to_tensor(alpha_u)
        self.radius_robot = radius_robot
        self.num_obstacles = num_obstacles

        self.alpha_barrier = alpha_barrier
        self.alpha_corridor = alpha_corridor
        self.d_safe = d_safe
        self.alpha_q_scaling = alpha_q_scaling

    def forward(self, xs_log, us_log, initial_data_batch):
        """
        Compute loss for the multi-obstacle scenario.

        Args:
            - xs_log: Robot state trajectory tensor of shape (S, T, state_dim).
            - us_log: Control trajectory tensor of shape (S, T, in_dim).
            - initial_data_batch: The initial data tensor of shape (S, T, full_dim)
                                  used to extract static obstacle info.
        """
        S, T, _ = xs_log.shape
        x_batch = xs_log.unsqueeze(-1)
        u_batch = us_log.unsqueeze(-1)

        # --- Extract and Reshape Obstacle Data ---
        # Obstacle data is static, so we only need it from t=0.
        # Shape: (S, 9) -> (S, num_obstacles, 3)
        obs_data_flat = initial_data_batch[:, 0, 4:]
        obs_data = obs_data_flat.view(S, self.num_obstacles, 3)  # [cx, cy, r]

        # --- 1. Radius-Modulated State Cost ---
        # We define the "threat level" by the BIGGEST obstacle in the scene.
        all_radii = obs_data[:, :, 2]  # Shape: (S, 3)
        max_radius, _ = torch.max(all_radii, dim=1, keepdim=True)  # Shape: (S, 1)
        # Reshape for broadcasting: (S, 1, 1, 1)
        max_radius = max_radius.view(S, 1, 1, 1)

        q_scaling_factor = torch.exp(-self.alpha_q_scaling * max_radius)
        xTQx = torch.matmul(x_batch.transpose(-1, -2), self.Q) @ x_batch
        loss_x_unscaled = xTQx.sum(dim=1) / T
        loss_x = q_scaling_factor * loss_x_unscaled

        # --- 2. Control Cost (Unchanged Logic) ---
        uTRu = self.R * (u_batch.transpose(-1, -2) @ u_batch)
        loss_u = uTRu.sum(dim=1) / T

        # --- 3. Multi-Obstacle Avoidance Cost ---
        loss_obst = self.f_loss_obst_multi(xs_log, obs_data)

        # --- Total Loss ---
        loss_per_sample = loss_x + loss_u + loss_obst
        final_loss = torch.mean(loss_per_sample)

        return final_loss

    def f_loss_obst_multi(self, xs, obs_data):
        """
        Calculates the obstacle avoidance cost for multiple obstacles.
        The costs from all obstacles are summed together.
        """
        S, T, _ = xs.shape

        # --- Prepare Tensors for Broadcasting ---
        robot_pos = xs[:, :, 0:2].unsqueeze(2)  # Shape: (S, T, 1, 2)

        # Obstacle data needs to be repeated across the time dimension
        obs_centers = obs_data[:, :, 0:2].unsqueeze(1).repeat(1, T, 1, 1)  # Shape: (S, T, 3, 2)
        obs_radii = obs_data[:, :, 2].unsqueeze(1).unsqueeze(-1).repeat(1, T, 1, 1)  # Shape: (S, T, 3, 1)

        # --- Vectorized Distance Calculation ---
        # Calculate distance from robot to each of the 3 obstacle centers at every time step
        dist_center = torch.norm(robot_pos - obs_centers, dim=-1, keepdim=True)  # Shape: (S, T, 3, 1)

        # Calculate distance from robot edge to each obstacle edge
        total_radii = self.radius_robot + obs_radii
        dist_edge = dist_center - total_radii  # Shape: (S, T, 3, 1)

        # --- Vectorized Cost Calculation ---
        # Calculate barrier and corridor cost for each obstacle simultaneously
        barrier_cost = self.alpha_barrier / torch.clamp(dist_edge, min=1e-4)
        corridor_cost = self.alpha_corridor * (dist_edge - self.d_safe) ** 2

        # Shape of both cost tensors: (S, T, 3, 1)
        per_obstacle_cost = barrier_cost + corridor_cost

        # --- Aggregate Costs ---
        # 1. Sum the costs from all obstacles at each time step
        total_cost_at_each_timestep = per_obstacle_cost.sum(dim=2)  # Shape: (S, T, 1)

        # 2. Average this total cost over the time horizon
        avg_total_obst_cost = total_cost_at_each_timestep.sum(dim=1) / T  # Shape: (S, 1)

        return avg_total_obst_cost.view(S, 1, 1)  # Reshape for broadcasting
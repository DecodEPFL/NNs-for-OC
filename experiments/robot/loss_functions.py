import torch
from assistive_functions import to_tensor


class RobotsLoss:
    def __init__(
            self, Q, alpha_u=1,
            alpha_obst=None,
            radius_robot=0.04,
            obstacle_centers=None, obstacle_radius=None
    ):
        self.Q, self.R = to_tensor(Q), to_tensor(alpha_u)
        self.alpha_obst, self.radius_robot = alpha_obst, radius_robot
        # define obstacles
        if obstacle_centers is None:
            self.obstacle_centers = torch.tensor([[1., 0.5]])
        else:
            self.obstacle_centers = obstacle_centers
        self.n_obstacles = self.obstacle_centers.shape[0]
        if obstacle_radius is None:
            self.obstacle_radius = torch.tensor([[0.7]])
        else:
            self.obstacle_radius = obstacle_radius
        assert self.n_obstacles == self.obstacle_radius.shape[0]

    def forward(self, xs, us, circle):
        """
        Compute loss.

        Args:
            - xs: tensor of shape (S, T, state_dim)
            - us: tensor of shape (S, T, in_dim)

        Return:
            - loss of shape (1, 1).
        """
        # batch
        x_batch = xs.reshape(*xs.shape, 1)
        u_batch = us.reshape(*us.shape, 1)
        # loss states = 1/T sum_{t=1}^T (x_t)^T Q (x_t)
        xTQx = torch.matmul(
            torch.matmul(x_batch.transpose(-1, -2), self.Q),
            x_batch
        )  # shape = (S, T, 1, 1)
        loss_x = torch.sum(xTQx, 1) / x_batch.shape[1]  # average over the time horizon. shape = (S, 1, 1)
        # loss control actions = 1/T sum_{t=1}^T u_t^T R u_t
        uTRu = self.R * torch.matmul(
            u_batch.transpose(-1, -2),
            u_batch
        )  # shape = (S, T, 1, 1)
        loss_u = torch.sum(uTRu, 1) / x_batch.shape[1]  # average over the time horizon. shape = (S, 1, 1)
        # obstacle avoidance loss
        if self.alpha_obst is None:
            loss_obst = 0
        else:
            loss_obst = self.alpha_obst * self.f_loss_obst(x_batch, circle)  # shape = (S, 1, 1)
        # sum up all losses
        loss_val = loss_x + loss_u + loss_obst  # shape = (S, 1, 1)
        # average over the samples
        loss_val = torch.sum(loss_val, 0) / xs.shape[0]  # shape = (1, 1)
        return loss_val

    def f_loss_obst(self, x_batch, circle):
        """
        Obstacle avoidance loss.
        Args:
            - x_batched: tensor of shape (S, T, state_dim, 1)
                concatenated states of all agents on the third dimension.
        Return:
            - collision avoidance loss of shape (1, 1).
        """
        min_sec_dist = 1 * (self.radius_robot + circle[:, 1, -1])
        # compute pairwise distances
        distance_sq = self.get_pairwise_distance_sq(x_batch, circle)  # shape = (S, T, n_agents, n_agents)
        # compute and sum up loss when two agents are too close
        loss_obs = (1 / (distance_sq + 1e-3) * (distance_sq.detach() < (min_sec_dist ** 2))).sum(
            (-1, -2)) / 2  # shape = (S, T)
        # average over time steps
        loss_obs = loss_obs.sum(1) / loss_obs.shape[1]
        # reshape to S,1,1
        loss_obs = loss_obs.reshape(-1, 1, 1)
        return loss_obs

    def get_pairwise_distance_sq(self, x_batch, circle):
        """
        Squared distance between robot and obstacle.
        Args:
            - x_batched: tensor of shape (S, T, state_dim, 1)
                concatenated states of all agents on the third dimension.
        Return:
            - matrix of shape (S, T, 1, n_obstacles) of squared pairwise distances.
        """
        # collision avoidance:
        x_robot = x_batch[:, :, 0:1, :]  # shape = (S, T, 1, 1)
        y_robot = x_batch[:, :, 1:2, :]  # shape = (S, T, 1, 1)
        deltaqx = x_robot.repeat(1, 1, 1, self.n_obstacles) - circle[:, :, 4:5].unsqueeze(2)  # shape = (S, T,
        # 1, n_obstacles)
        deltaqy = y_robot.repeat(1, 1, 1, self.n_obstacles) - circle[:, :, 5:6].unsqueeze(2)  # shape = (S, T,
        # 1, n_obstacles)
        distance_sq = deltaqx ** 2 + deltaqy ** 2  # shape = (S, T, 1, n_obstacles)
        return distance_sq


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

    def forward(self, xs, us, circle):
        """
        Compute loss using advanced cost shaping.

        Args:
            - xs: State trajectory tensor of shape (S, T, state_dim).
            - us: Control trajectory tensor of shape (S, T, in_dim).
            - circle: Dynamic obstacle info tensor, likely (S, T, obs_dim).
                      We assume obs_dim contains [center_x, center_y, radius].
        """
        S, T, _ = xs.shape  # Samples, Time
        x_batch = xs.unsqueeze(-1)  # (S, T, state_dim, 1)
        u_batch = us.unsqueeze(-1)  # (S, T, in_dim, 1)

        # Extract Obstacle Radius for cost modulation
        # Assuming radius is the last feature and consistent across time. Shape: (S,) -> (S, 1, 1, 1)
        obstacle_radius = circle[:, 0, -1].view(S, 1, 1, 1)

        # --- 1. Radius-Modulated State Cost ---
        q_scaling_factor = torch.exp(-self.alpha_q_scaling * obstacle_radius)
        xTQx = torch.matmul(x_batch.transpose(-1, -2), self.Q) @ x_batch
        loss_x_unscaled = xTQx.sum(dim=1) / T
        loss_x = q_scaling_factor * loss_x_unscaled
        #loss_x = loss_x_unscaled

        # --- 2. Control Cost (Unchanged) ---
        uTRu = self.R * (u_batch.transpose(-1, -2) @ u_batch)
        loss_u = uTRu.sum(dim=1) / T

        # --- 3. Advanced Obstacle Avoidance Cost ---
        loss_obst = self.f_loss_obst_v2(xs, circle)

        # --- Total Loss ---
        # Combine the losses for each sample in the batch.
        # The shape of each loss component is (S, 1, 1).
        loss_per_sample = loss_x + loss_u + loss_obst

        # *** THE FIX IS HERE ***
        # Instead of sum/divide/squeeze, we use torch.mean().
        # This robustly calculates the average over all samples in the batch,
        # guaranteeing a single scalar output.
        final_loss = torch.mean(loss_per_sample)

        return final_loss

    def f_loss_obst_v2(self, xs, circle):
        """
        A revised obstacle loss with a smooth barrier and a safety corridor.
        This provides a continuous gradient that guides the robot.
        """
        S, T, _ = xs.shape
        robot_pos = xs[:, :, 0:2]  # (S, T, 2)

        # Assuming circle format is (S, T, [cx, cy, r])
        obstacle_centers = circle[:, :, 0:2]  # (S, T, 2)
        obstacle_radius = circle[:, :, -1]  # (S, T)

        # Calculate distance from robot center to obstacle center
        dist_center_sq = torch.sum((robot_pos - obstacle_centers) ** 2, dim=-1)  # (S, T)
        dist_center = torch.sqrt(dist_center_sq + 1e-6)  # Add epsilon for stability

        # THE KEY FEATURE: Distance from robot *edge* to obstacle *edge*
        total_radius = self.radius_robot + obstacle_radius
        dist_edge = dist_center - total_radius  # (S, T)

        # --- Cost Component 1: The Repulsive Barrier ---
        # This cost explodes as the robot gets very close to the obstacle.
        # It's a smooth function, providing a gradient long before collision.
        # We use clamp to prevent division by zero or negative values if a collision occurs.
        barrier_cost = self.alpha_barrier / torch.clamp(dist_edge, min=1e-4)

        # --- Cost Component 2: The Optimal Safety Corridor ---
        # This cost is a quadratic valley with its minimum at d_safe.
        # It punishes the robot for being too close OR unnecessarily far.
        corridor_cost = self.alpha_corridor * (dist_edge - self.d_safe) ** 2

        # Combine costs and average over the time horizon
        total_obst_cost = (barrier_cost + corridor_cost).sum(dim=1) / T  # (S,)

        return total_obst_cost.view(S, 1, 1)  # Reshape for broadcasting

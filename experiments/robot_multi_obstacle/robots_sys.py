import torch
from plants.robots.base_robots_sys import BaseRobotsSystem
import torch.nn.functional as F # Need this for one_hot

class RobotsSystemMultiObstacle(BaseRobotsSystem):
    def __init__(self, xbar: torch.Tensor, linear_plant: bool, num_obstacles: int, x_init=None, u_init=None,
                 k: float = 1.0):
        super().__init__(xbar, linear_plant, x_init, u_init, k)
        self.num_obstacles = num_obstacles

    import torch.nn.functional as F  # Need this for one_hot

    def _rollout_step(self, t, x, u, data, controller):
        """
        Performs a single rollout step with a corrected, robust, and simplified
        method for feature engineering.
        """
        # --- Step 1: Evolve System Dynamics ---
        x_next = self.forward(t=t, x=x, u=u, w=data[:, t:t + 1, 0:4])

        # ======================================================================
        # --- START: CORRECTED AND SIMPLIFIED FEATURE ENGINEERING ---
        # ======================================================================

        # --- Step 2: Extract Cleanly Shaped Tensors ---
        robot_pos = x_next[:, :, 0:2]  # Shape: (S, 1, 2)
        robot_vel = x_next[:, :, 2:4]  # Shape: (S, 1, 2)

        obs_data_flat = data[:, 0, 4:]  # Shape: (S, 9)
        obs_data = obs_data_flat.view(-1, self.num_obstacles, 3)
        obs_centers = obs_data[:, :, 0:2]  # Shape: (S, 3, 2)
        obs_radii = obs_data[:, :, 2]  # Shape: (S, 3)

        # --- Step 3: Calculate Relative Features with Simple Broadcasting ---
        vector_to_goal = -robot_pos  # Shape: (S, 1, 2)

        # Broadcasting `obs_centers` (S, 3, 2) - `robot_pos` (S, 1, 2)
        # results in `vectors_to_centers` of shape (S, 3, 2). This is clean and efficient.
        vectors_to_centers = obs_centers - robot_pos
        dist_to_centers = torch.norm(vectors_to_centers, dim=-1)  # Shape: (S, 3)

        dist_to_edges = dist_to_centers - obs_radii  # Shape: (S, 3)

        # --- Step 4: Identify the Primary Threat (Now Correct) ---
        # `torch.min` on `dist_to_edges` (S, 3) along dim=1 finds the min among the 3 obstacles.
        # `idx_primary_threat` will have shape (S,) with values in {0, 1, 2}.
        min_dist_to_edge, idx_primary_threat = torch.min(dist_to_edges, dim=1)

        # --- Step 5: Select the Primary Threat Vector using One-Hot `bmm` ---
        # This will now work correctly because `idx_primary_threat` has the right values.
        one_hot_selector = F.one_hot(idx_primary_threat, num_classes=self.num_obstacles).float()

        # Reshape for bmm: (S, 1, 3) @ (S, 3, 2) -> (S, 1, 2)
        vector_to_primary_threat = torch.bmm(one_hot_selector.unsqueeze(1), vectors_to_centers)

        # ======================================================================
        # --- END: CORRECTED AND SIMPLIFIED FEATURE ENGINEERING ---
        # ======================================================================

        # --- Step 6: Assemble the Final Context Vector `w_controller` ---
        w_controller = torch.cat([
            robot_vel.squeeze(1),
            vector_to_goal.squeeze(1),
            min_dist_to_edge.unsqueeze(1),
            vector_to_primary_threat.squeeze(1)
        ], dim=1)

        w_controller = w_controller.unsqueeze(1)

        # --- Step 7: Get Control Input ---
        u_next = controller(t, x_next.detach(), w_controller.detach())

        return x_next, u_next

        # In your RobotsSystemMultiObstacle class

    def rollout(self, controller, data, train=False):
        """
        Rollout the system with a clean, standardized return signature.
        """
        controller.reset()
        batch_size, T, _ = data.shape
        x = data[:, 0:1, :4].clone()
        u = self.u_init.detach().clone().repeat(batch_size, 1, 1)

        x_log = torch.zeros(batch_size, T, self.state_dim, device=data.device)
        u_log = torch.zeros(batch_size, T, self.in_dim, device=data.device)

        # Store initial state and control action placeholder
        x_log[:, 0:1, :] = x
        u_log[:, 0:1, :] = u  # Log the initial u

        context = torch.no_grad() if not train else torch.enable_grad()
        with context:
            for t in range(T - 1):
                x_next, u_next = self._rollout_step(t, x, u, data, controller)

                x_log[:, t + 1:t + 2, :] = x_next
                # Log the control action that *led to* the next state
                u_log[:, t:t + 1, :] = u

                x = x_next
                u = u_next

            # Log the final action
            u_log[:, -1:, :] = u

        controller.reset()

        return x_log, u_log

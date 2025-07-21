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
        Performs a single rollout step using a "Sorted Threat" feature vector.
        This provides the controller with complete, ordered information about all obstacles.
        """
        # --- Step 1: Evolve System Dynamics ---
        x_next = self.forward(t=t, x=x, u=u, w=data[:, t:t + 1, 0:4])

        # --- Step 2: Extract Cleanly Shaped Tensors ---
        robot_pos = x_next[:, :, 0:2]  # (S, 1, 2)
        robot_vel = x_next[:, :, 2:4]  # (S, 1, 2)

        obs_data_flat = data[:, 0, 4:] # (S, 9) assuming 3 obstacles
        obs_data = obs_data_flat.view(-1, self.num_obstacles, 3) # (S, 3, 3)
        obs_centers = obs_data[:, :, 0:2] # (S, 3, 2)
        obs_radii = obs_data[:, :, 2]     # (S, 3)

        # --- Step 3: Calculate Relative Features for ALL Obstacles ---
        vector_to_goal = -robot_pos # (S, 1, 2)
        vectors_to_centers = obs_centers - robot_pos # (S, 3, 2)
        dist_to_centers = torch.norm(vectors_to_centers, dim=-1) # (S, 3)
        dist_to_edges = dist_to_centers - obs_radii # (S, 3)

        # --- Step 4: Sort Obstacles by Threat Level (distance to edge) ---
        # `torch.sort` returns sorted values and the indices of the original elements
        sorted_dist_to_edges, sorted_indices = torch.sort(dist_to_edges, dim=1)

        # --- Step 5: Use `gather` to Reorder Other Tensors Based on Sorted Indices ---
        # We need to expand sorted_indices to match the shape of `vectors_to_centers`
        # sorted_indices shape: (S, 3) -> (S, 3, 1) -> (S, 3, 2)
        sorted_indices_expanded = sorted_indices.unsqueeze(-1).expand(-1, -1, 2)

        # `torch.gather` selects elements along dim=1 using the sorted indices
        sorted_vectors_to_centers = torch.gather(vectors_to_centers, 1, sorted_indices_expanded)

        # --- Step 6: Assemble the Final, Rich Context Vector ---
        # Flatten the sorted obstacle information.
        # sorted_dist_to_edges is (S, 3) -> we need (S, 3*1=3)
        # sorted_vectors_to_centers is (S, 3, 2) -> we need (S, 3*2=6)

        w_controller = torch.cat([
            robot_vel.squeeze(1),               # Shape: (S, 2)
            vector_to_goal.squeeze(1),          # Shape: (S, 2)
            sorted_dist_to_edges,               # Shape: (S, 3)
            sorted_vectors_to_centers.view(x.shape[0], -1) # Shape: (S, 6)
        ], dim=1)

        w_controller = w_controller.unsqueeze(1) # (S, 1, 13)

        # --- Step 7: Get Control Input ---
        u_next = controller(t, x_next.detach(), w_controller.detach())

        return x_next, u_next


    def rollout(self, controller, data, train=False):
        """
        Rollout the system dynamics using the provided controller and data.
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

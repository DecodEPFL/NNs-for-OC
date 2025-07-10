import torch
from plants.robots.base_robots_sys import BaseRobotsSystem


class RobotsSystem(BaseRobotsSystem):
    def __init__(self, xbar: torch.Tensor, linear_plant: bool, x_init=None, u_init=None, k: float = 1.0):
        super().__init__(xbar, linear_plant, x_init, u_init, k)

    def _rollout_step(self, t, x, u, data, controller):
        """
        Helper function to perform a single step of the rollout.
        """
        # Forward pass through the system
        x = self.forward(t=t, x=x, u=u, w=data[:, t:t + 1, 0:4])

        # Extract obstacle information from data
        obstacle_centers = data[:, 0:1, 4:6]
        obstacle_radii = data[:, 0:1, 6:7]

        # Calculate distance to obstacle
        delta_p = x[:, :, 0:2] - obstacle_centers
        distance_sq = delta_p.pow(2).sum(dim=-1, keepdim=True)
        distance = torch.sqrt(distance_sq)
        distance_to_circle_surface = distance - obstacle_radii

        # Prepare input for the controller
        i2 = data[:, 0:1, :].clone()
        i2[:, :, 0:4] = x.detach()
        i2[:, :, 6:7] = distance_to_circle_surface.detach()

        # Get control input
        u = controller(t, x, i2)
        return x, u

    # simulation
    def rollout(self, controller, data, train=False):
        """
        rollout REN for rollouts of the process noise
        Args:
            - data: sequence of disturbance samples of shape (batch_size, T, state_dim).
        Return:
            - x_log of shape (batch_size, T, state_dim)
            - u_log of shape (batch_size, T, in_dim)
        """

        # init
        controller.reset()
        batch_size, T, _ = data.shape
        x = self.x_init.detach().clone().repeat(batch_size, 1, 1)
        u = self.u_init.detach().clone().repeat(batch_size, 1, 1)

        x_log = torch.zeros(batch_size, T, self.state_dim, device=data.device)
        u_log = torch.zeros(batch_size, T, self.in_dim, device=data.device)

        context = torch.no_grad() if not train else torch.enable_grad()
        with context:
            for t in range(T):
                x, u = self._rollout_step(t, x, u, data, controller)
                x_log[:, t:t + 1, :] = x
                u_log[:, t:t + 1, :] = u

        controller.reset()

        return x_log, None, u_log


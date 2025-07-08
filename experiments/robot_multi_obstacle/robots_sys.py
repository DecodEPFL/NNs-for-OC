import torch
from plants.robots.base_robots_sys import BaseRobotsSystem

class RobotsSystem(BaseRobotsSystem):
    def __init__(self, xbar: torch.Tensor, linear_plant: bool, x_init=None, u_init=None, k: float = 1.0):
        super().__init__(xbar, linear_plant, x_init, u_init, k)

    def _rollout_step(self, t, x, u, data, controller, num_obstacles):
        x = self.forward(t=t, x=x, u=u, w=data[:, t:t + 1, 0:4])
        i2 = data[:, 0:1, :].clone()
        i2[:, :, 0:4] = x.detach()
        u = controller(t, x, i2)
        return x, u

    def rollout(self, controller, data, train=False, num_obstacles=1):
        controller.reset()
        batch_size, T, data_dim = data.shape
        x = self.x_init.detach().clone().repeat(batch_size, 1, 1)
        u = self.u_init.detach().clone().repeat(batch_size, 1, 1)

        x_log = torch.zeros(batch_size, T, self.state_dim, device=data.device)
        u_log = torch.zeros(batch_size, T, self.in_dim, device=data.device)

        context = torch.no_grad() if not train else torch.enable_grad()
        with context:
            for t in range(T):
                x, u = self._rollout_step(t, x, u, data, controller, num_obstacles)
                x_log[:, t:t + 1, :] = x
                u_log[:, t:t + 1, :] = u

        controller.reset()

        return x_log, None, u_log

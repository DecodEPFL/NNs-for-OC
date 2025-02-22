import torch


class TankSystem(torch.nn.Module):
    def __init__(self, xbar : torch.Tensor, x_init=None, u_init=None, a: float = 0.1):
        """
        Args:
            xbar:           Concatenated nominal equilibrium point of all agents.
            x_init:         Concatenated initial point of all agents. Default to xbar when None.
            u_init:         Initial input to the plant. Defaults to zero when None.
            a (float):      Gain of the pre-stabilizing controller (acts as a spring constant).
        """
        super().__init__()

        # initial state
        self.xbar = xbar.reshape(1, -1)  # shape = (1, state_dim)
        self.x_init = self.xbar.detach().clone() if x_init is None else x_init.reshape(1, -1)   # shape = (1, state_dim)
        self.state_dim = 1
        self.in_dim = 1
        if u_init is None:
            u_init = torch.zeros(1, self.in_dim)
        else:
            u_init.reshape(1, -1)  # shape = (1, in_dim)
        self.register_buffer('u_init', u_init)
        # check dimensions
        assert self.xbar.shape[1] == self.state_dim and self.x_init.shape[1] == self.state_dim
        assert self.u_init.shape[1] == self.in_dim

        self.h = 0.1
        self.a = a
        self.b = 0.5

    def dynamics(self, x, u):
        f = 1 / (self.b + x) * (-self.a * torch.sqrt(x) + u + self.a * torch.sqrt(self.xbar))
        return f

    def noiseless_forward(self, t, x: torch.Tensor, u: torch.Tensor):
        """
        forward of the plant without the process noise.

        Args:
            - x (torch.Tensor): plant's state at t. shape = (batch_size, 1, state_dim)
            - u (torch.Tensor): plant's input at t. shape = (batch_size, 1, in_dim)

        Returns:
            next state of the noise-free dynamics.
        """
        x = x.view(-1, 1, self.state_dim)
        u = u.view(-1, 1, self.in_dim)

        # Calculate x_dot
        f = self.dynamics(x, u)
        # Calculate x^+ using forward Euler integration scheme
        x_ = x + self.h * f
        return x_    # shape = (batch_size, 1, state_dim)

    def forward(self, t, x, u, w):
        """
        forward of the plant with the process noise.
        Args:
            - t (int):          current time step
            - x (torch.Tensor): plant's state at t. shape = (batch_size, 1, state_dim)
            - u (torch.Tensor): plant's input at t. shape = (batch_size, 1, in_dim)
            - w (torch.Tensor): process noise at t. shape = (batch_size, 1, state_dim)
        Returns:
            next state.
        """
        u = torch.relu(u + self.a * torch.sqrt(self.xbar)) - self.a * torch.sqrt(self.xbar)
        x_plus = self.noiseless_forward(t, x, u) + w.view(-1, 1, self.state_dim)
        return torch.relu(x_plus), x_plus

    # simulation
    def rollout(self, controller, data, train=False):
        """
        rollout of the closed-loop (of system and controller) for rollouts of the process noise
        Args:
            - data: sequence of disturbance samples of shape (batch_size, T, state_dim).
        Return:
            - x_log of shape (batch_size, T, state_dim)
            - u_log of shape (batch_size, T, in_dim)
        """

        # init
        controller.reset()
        x = self.x_init.detach().clone().repeat(data.shape[0], 1, 1)
        u = self.u_init.detach().clone().repeat(data.shape[0], 1, 1)

        # Simulate
        if train:
            x_log, x_nonfilter_log, u_log = self._closed_loop_sim(x, u, data, controller)
        else:
            with torch.no_grad():
                x_log, x_nonfilter_log, u_log = self._closed_loop_sim(x, u, data, controller)
        controller.reset()

        return x_log, x_nonfilter_log, u_log

    def _closed_loop_sim(self, x, u, data, controller):
        x_log, x_nonfilter_log, u_log = x, x, u
        for t in range(data.shape[1]):
            x, x_nonfilter = self.forward(t=t, x=x, u=u, w=data[:, t:t + 1, :])  # shape = (batch_size, 1, state_dim)
            u = controller(t, x)                                       # shape = (batch_size, 1, in_dim)
            if t > 0:
                x_log = torch.cat((x_log, x), 1)
                x_nonfilter_log = torch.cat((x_nonfilter_log, x), 1)
                u_log = torch.cat((u_log, u), 1)
        return x_log, x_nonfilter_log, u_log

import torch
import torch.nn.functional as F

class BaseRobotsSystem(torch.nn.Module):
    def __init__(self, xbar: torch.Tensor, linear_plant: bool, x_init=None, u_init=None, k: float = 1.0):
        """
        Args:
            xbar:           Concatenated nominal equilibrium point of all agents.
            linear_plant:   If True, a linearized model of the system is used.
                            Otherwise, the model is nonlinear due to the dependence of friction on the speed.
            x_init:         Concatenated initial point of all agents. Default to xbar when None.
            u_init:         Initial input to the plant. Defaults to zero when None.
            k (float):      Gain of the pre-stabilizing controller (acts as a spring constant).
        """
        super().__init__()

        self.linear_plant = linear_plant

        # initial state
        self.register_buffer('xbar', xbar.reshape(1, -1))  # shape = (1, state_dim)
        x_init = self.xbar.detach().clone() if x_init is None else x_init.reshape(1, -1)  # shape = (1, state_dim)
        self.register_buffer('x_init', x_init)
        if u_init is None:
            u_init = torch.zeros(1, int(self.xbar.shape[1] / 2))
        else:
            u_init.reshape(1, -1)  # shape = (1, in_dim)
        self.register_buffer('u_init', u_init)
        # check dimensions
        self.state_dim = 4
        self.in_dim = 2
        assert self.xbar.shape[1] == self.state_dim and self.x_init.shape[1] == self.state_dim
        assert self.u_init.shape[1] == self.in_dim

        self.h = 0.05
        self.mass = 1.0
        self.k = k
        self.b = 1.0
        self.b2 = None if self.linear_plant else 0.1
        m = self.mass
        self.B = torch.tensor([[0, 0],
                               [0., 0],
                               [1 / m, 0],
                               [0, 1 / m]]) * self.h

        _A1 = torch.eye(4)
        _A2 = torch.cat((torch.cat((torch.zeros(2, 2),
                                    torch.eye(2)
                                    ), dim=1),
                         torch.cat((torch.diag(torch.tensor([-self.k / self.mass, -self.k / self.mass])),
                                    torch.diag(torch.tensor([-self.b / self.mass, -self.b / self.mass]))
                                    ), dim=1),
                         ), dim=0)
        self.A_lin = _A1 + self.h * _A2

        self.mask = torch.tensor([[0, 0], [1, 1]])

    def A_nonlin(self, x):
        assert not self.linear_plant
        A3 = torch.norm(
            x.view(-1, 2, 2) * self.mask, dim=-1, keepdim=True
        )  # shape = (batch_size, 2, 1)
        A3 = torch.kron(
            A3, torch.ones(2, 1, device=A3.device)
        )  # shape = (batch_size, 4, 1)
        A3 = -self.b2 / self.mass * torch.diag_embed(
            A3.squeeze(dim=-1), offset=0, dim1=-2, dim2=-1
        )  # shape = (batch_size, 4, 4)
        A = self.A_lin + self.h * A3
        return A  # shape = (batch_size, 4, 4)

    def noiseless_forward(self, t, x: torch.Tensor, u: torch.Tensor):
        x = x.view(-1, 1, self.state_dim)
        u = u.view(-1, 1, self.in_dim)
        if self.linear_plant:
            f = F.linear(x - self.xbar, self.A_lin) + F.linear(u, self.B) + self.xbar
        else:
            f = torch.bmm(x - self.xbar, self.A_nonlin(x).transpose(1, 2)) + F.linear(u, self.B) + self.xbar
        return f

    def forward(self, t, x, u, w):
        return self.noiseless_forward(t, x, u) + w.view(-1, 1, self.state_dim)


import torch
from plants.tanks.tank_sys import TankSystem


class TankLoss:
    def __init__(self, Q, R, alpha_smooth=0, alpha_u_pos=0, alpha_barrier=0, xbar=None, u_bar=None, x_max=None, sys=None):
        assert (not hasattr(Q, "__len__")) or (len(Q.shape) == 2 and Q.shape[0] == Q.shape[1])  # int or square matrix
        assert (not hasattr(R, "__len__")) or (len(R.shape) == 2 and R.shape[0] == R.shape[1])  # int or square matrix
        self.Q, self.R = Q, R
        if not hasattr(self.Q, "__len__"):
            self.Q = torch.tensor([[self.Q]])
        if not hasattr(self.R, "__len__"):
            self.R = torch.tensor([[self.R]])
        self.alpha_smooth = alpha_smooth
        self.alpha_u_pos = alpha_u_pos
        self.alpha_barrier = alpha_barrier
        self.x_max = x_max

        if xbar is not None:
            self.xbar = xbar.reshape(self.Q.shape[0], 1)
        else:
            self.xbar = torch.zeros(self.Q.shape[0], 1)
        if u_bar is not None:
            self.u_bar = u_bar.reshape(self.R.shape[0], 1)
        else:
            self.u_bar = torch.zeros(self.R.shape[0], 1)

        if sys is not None:
            self.sys = sys
        else:
            self.sys = None


    def forward(self, xs, us):
        """
        Compute loss for tank system.
        Args:
            - xs: tensor of shape (S, T, state_dim)
            - us: tensor of shape (S, T, in_dim)
        Return:
            - loss of shape (1, 1).
        """
        # batch
        x_batch = xs.reshape(*xs.shape, 1)  # TODO: remove 3rd dim
        u_batch = us.reshape(*us.shape, 1)
        loss_x,loss_u = self.loss_quadratic(x_batch, u_batch)
        loss_u_smooth = self.loss_smooth_input(u_batch)
        loss_u_pos = self.loss_positive_input(u_batch)
        loss_barrier_max, loss_barrier_min = self.loss_barrier_function(x_batch, u_batch, self.sys)
        # sum up all losses
        loss_val = loss_x + loss_u + loss_u_smooth + loss_u_pos + loss_barrier_max + loss_barrier_min  # shape = (S, 1, 1)
        # average over the samples
        loss_val = torch.sum(loss_val, 0)/xs.shape[0]                   # shape = (1, 1)
        return loss_val

    def loss_quadratic(self, x_batch, u_batch):
        # loss states = 1/T sum_{t=1}^T (x_t-xbar)^T Q (x_t-xbar)
        x_batch_centered = x_batch - self.xbar
        xTQx = torch.matmul(
            torch.matmul(x_batch_centered.transpose(-1, -2), self.Q),
            x_batch_centered
        )  # shape = (S, T, 1, 1)
        loss_x = torch.sum(xTQx, 1) / x_batch.shape[1]  # average over the time horizon. shape = (S, 1, 1)
        # loss control actions = 1/T sum_{t=1}^T u_t^T R u_t
        uTRu = torch.matmul(
            torch.matmul(u_batch.transpose(-1, -2), self.R),
            u_batch
        )  # shape = (S, T, 1, 1)
        loss_u = torch.sum(uTRu, 1) / x_batch.shape[1]  # average over the time horizon. shape = (S, 1, 1)
        return loss_x, loss_u

    def loss_smooth_input(self, u_batch):
        loss_u_smooth = 0
        if self.alpha_smooth > 0:
            u_plus = u_batch[:, 1:, :, :]
            u_minus = u_batch[:, :-1, :, :]
            loss_u_smooth = self.alpha_smooth * torch.sum(torch.abs(u_plus - u_minus), 1) / u_batch.shape[1]
        return loss_u_smooth

    def loss_positive_input(self, u_batch):
        loss_u_pos = 0
        if self.alpha_u_pos > 0:
            u_tot = u_batch + self.u_bar
            loss_u_pos = self.alpha_u_pos * torch.sum(torch.relu(-u_tot), 1) / u_batch.shape[1]
        return loss_u_pos

    def loss_barrier_function(self, x_batch, u_batch, sys):
        loss_barrier_max = 0
        loss_barrier_min = 0
        gamma = 0.2
        alpha = 1
        if self.alpha_barrier > 0:

            _, x_next = sys.forward(t=0, x=x_batch[0,:,:,:], u=u_batch[0,:,:,:], w=torch.zeros_like(x_batch[0,:,:,:]))

            h_next_max = alpha * (self.x_max - x_next)
            h_max = alpha * (self.x_max - x_batch[0,:,:,:])
            h_next_min = alpha * x_next
            h_min = alpha * x_batch[0,:,:,:]

            loss_barrier_max = (self.alpha_barrier * torch.relu((1 - gamma) * h_max - h_next_max)).sum()
            loss_barrier_min = (self.alpha_barrier * torch.relu((1 - gamma) * h_min - h_next_min)).sum()

        return loss_barrier_max, loss_barrier_min

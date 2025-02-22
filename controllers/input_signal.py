import torch
import torch.nn as nn



class InputController(nn.Module):

    def __init__(self, u):
        super().__init__()
        self.u = u

    def reset(self):
        pass

    def forward(self, t, input_t: torch.Tensor):
        return self.u[:, t:t+1, :]  # (batch_size, 1, self.dim_out)

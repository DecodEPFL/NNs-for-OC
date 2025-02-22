import torch
import torch.nn as nn


class ZeroController(nn.Module):

    def __init__(self, dim_out):
        super().__init__()
        self.dim_out = dim_out

    def reset(self):
        pass

    def forward(self, t, input_t: torch.Tensor):
        batch_size = input_t.shape[0]
        return torch.zeros(batch_size, 1, self.dim_out)
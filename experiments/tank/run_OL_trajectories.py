import matplotlib.pyplot as plt
import torch

from arg_parser import argument_parser, print_args
from plants.tanks import TankSystem
from controllers.zero_controller import ZeroController
from controllers.input_signal import InputController

# ----- Overwriting arguments -----
args = argument_parser()
args.horizon = 200
args.xbar = 0.2

# ------------ 2. Plant (with prestabilizing controller) ------------
xbar = torch.Tensor([args.xbar])
sys = TankSystem(xbar=xbar)

# Simulate the plant:
zero_ctl = ZeroController(dim_out=sys.in_dim)
x_inits = [
    torch.tensor([[0.]]),
    torch.tensor([[0.1]]),
    torch.tensor([[0.2]]),
    torch.tensor([[0.3]]),
    torch.tensor([[0.4]]),
]
# x_logs = torch.empty(0, args.horizon, 1)
plt.figure()
for x_init in x_inits:
    sys.x_init = x_init
    x_plot, _, _ = sys.rollout(zero_ctl, torch.zeros(1, args.horizon, sys.state_dim))
    plt.plot(x_plot[0, :, 0])
plt.xlabel(r'$k$')
plt.ylabel(r'$x_k$')
plt.show()
plt.close()

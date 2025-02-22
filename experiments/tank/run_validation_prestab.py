import matplotlib.pyplot as plt
import torch

from arg_parser import argument_parser, print_args
from plants.tanks import TankSystem


# ----- Overwriting arguments -----
args = argument_parser()
# args.horizon = 200
args.xbar = 0.2

# ------------ 2. Plant (with prestabilizing controller) ------------
xbar = torch.Tensor([args.xbar])
sys = TankSystem(xbar=xbar)

# Validate the asymptotic stability of the system:
# We plot x vs x_dot
# Generate vector of x:
x = torch.linspace(0, 1, 100).view(-1, 1, sys.state_dim)
u = torch.zeros_like(x)
x_dot = sys.dynamics(x, u)
plt.plot(x.squeeze(),x_dot.squeeze())
plt.plot([sys.xbar.squeeze(), sys.xbar.squeeze()], [torch.min(x_dot), torch.max(x_dot)], '--', label=r'$\bar{x}$')
plt.plot([0,1], [0,0], 'gray', linewidth=0.5)
plt.grid(linestyle='--', linewidth=0.5)
plt.xlabel('$x$')
plt.ylabel('$\dot{x}$')
plt.legend()
plt.show()
plt.close()

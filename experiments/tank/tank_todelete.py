import os
import logging

import matplotlib.pyplot as plt
import torch
import time
import copy
from datetime import datetime
from torch.utils.data import DataLoader

from arg_parser import argument_parser, print_args
from plants.tanks import TankSystem, TankDataset
from controllers.PB_controller import PerfBoostController
from controllers.zero_controller import ZeroController
from plot_functions import plot_traj_vs_time
from loss_functions import TankLoss

# ----- Overwriting arguments -----
args = argument_parser()
args.horizon = 200
args.epochs = 30  # 5000
args.lr = 5e-2  # SSM
# args.lr = 5e-3  # REN
args.num_rollouts = 8
args.log_epoch = 1  # args.epochs//10 if args.epochs//10 > 0 else 1
args.nn_type = "SSM"
args.dim_internal = 2
args.dim_nl = 2
args.non_linearity = "coupling_layers"
args.batch_size = 2
args.Q = 1e3
args.R = 2.5e3
args.alpha_smooth = 1e3*0
t_ext = args.horizon * 4

args.std_init_plant = 0.05

args.xbar = 0.2

dataset = TankDataset(random_seed=args.random_seed, horizon=args.horizon, std_ini=args.std_init_plant)
# divide to train and test
train_data, test_data = dataset.get_data(num_train_samples=args.num_rollouts, num_test_samples=500)
# data for plots
plot_data = torch.zeros(1, t_ext, train_data.shape[-1])
# batch the data
train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

# ------------ 2. Plant (with prestabilizing controller) ------------
xbar = torch.Tensor([args.xbar])
plant_input_init = None     # all zero
# plant_state_init = None     # same as xbar
plant_state_init = torch.Tensor([0.1])
sys = TankSystem(xbar=xbar,
                 x_init=plant_state_init,
                 u_init=plant_input_init,
                 )

# Validate the asymptotic stability of the system:
# We plot x vs x_dot
# Generate vector of x:
x = torch.linspace(0, 1, 100).view(-1, 1, sys.state_dim)
x_dot = sys.dynamics(x, torch.zeros_like(x))
plt.plot(x.squeeze(),x_dot.squeeze())
plt.plot([sys.xbar.squeeze(), sys.xbar.squeeze()], [torch.min(x_dot), torch.max(x_dot)], '--', label=r'$\bar{x}$')
# plt.plot([0,1], [0,0], '--')
plt.grid(linestyle='--', linewidth=0.5)
plt.xlabel('$x$')
plt.ylabel('$\dot{x}$')
plt.legend()
plt.savefig("figures/x_xdot.pdf", format='pdf')
plt.show()
plt.close()

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
    x_log, _, _ = sys.rollout(zero_ctl, plot_data)
    # x_logs = torch.cat((x_logs, x_log[:, :args.horizon, :]))
    plt.plot(x_log[0, :args.horizon, 0])
plt.xlabel(r'$k$')
plt.ylabel(r'$x_k$')
plt.savefig("figures/x_init.pdf", format='pdf')
plt.show()
plt.close()
sys.x_init = plant_state_init

# ------------ 3. Controller ------------
ctl = PerfBoostController(noiseless_forward=sys.noiseless_forward,
                          input_init=sys.x_init,
                          output_init=sys.u_init,
                          nn_type=args.nn_type,
                          non_linearity=args.non_linearity,
                          dim_internal=args.dim_internal,
                          dim_nl=args.dim_nl,
                          initialization_std=args.cont_init_std,
                          )

# Count numer of parameters
total_n_params = sum(p.numel() for p in ctl.parameters() if p.requires_grad)
print("[INFO] Number of parameters: %i" % total_n_params)

# plot closed-loop trajectories before training the controller
x_log, _, u_log = sys.rollout(ctl, plot_data)
plot_traj_vs_time(args.horizon, x_log[0, :args.horizon, :], u_log[0, :args.horizon, :], save=False)

# ------------ 4. Loss ------------
loss_fn = TankLoss(Q=args.Q, R=args.R, alpha_smooth=args.alpha_smooth, xbar=xbar, u_bar=sys.a * torch.sqrt(sys.xbar))
loss = loss_fn.forward(x_log[0, :args.horizon, :].unsqueeze(0), u_log[0, :args.horizon, :].unsqueeze(0))
print("Hola, esta es mi loss: ", loss)

# ------------ 5. Optimizer ------------
valid_data = train_data      # use the entire train data for validation
assert not (valid_data is None and args.return_best)
optimizer = torch.optim.Adam(ctl.parameters(), lr=args.lr)

# ------------ 6. Training ------------
print('------------ Begin training ------------')
best_valid_loss = 1e6
best_params = ctl.state_dict()
loss = 1e6
t = time.time()
for epoch in range(1+args.epochs):
    # iterate over all data batches
    for train_data_batch in train_dataloader:
        optimizer.zero_grad()
        # simulate over horizon steps
        x_log, _, u_log = sys.rollout(
            controller=ctl, data=train_data_batch, train=True,
        )
        # loss of this rollout
        loss = loss_fn.forward(x_log, u_log)
        # take a step
        loss.backward()
        optimizer.step()

    # print info
    if epoch % args.log_epoch == 0:
        msg = 'Epoch: %i --- train loss: %.2f' % (epoch, loss)

        if args.return_best:
            # rollout the current controller on the valid data
            with torch.no_grad():
                x_log_valid, _, u_log_valid = sys.rollout(
                    controller=ctl, data=valid_data, train=False,
                )
                # loss of the valid data
                loss_valid = loss_fn.forward(x_log_valid, u_log_valid)
            msg += ' ---||--- validation loss: %.2f' % (loss_valid.item())
            # compare with the best valid loss
            if loss_valid.item() < best_valid_loss:
                best_valid_loss = loss_valid.item()
                best_params = copy.deepcopy(ctl.state_dict())
                msg += ' (best so far)'
        duration = time.time() - t
        msg += ' ---||--- time: %.0f s' % duration
        print(msg)
        # plot trajectory
        x_log, _, u_log = sys.rollout(ctl, plot_data)
        plot_traj_vs_time(args.horizon, x_log[0, :args.horizon, :], u_log[0, :args.horizon, :], save=False)
        # plot_traj_vs_time(x_log_valid.shape[1], x_log_valid[0, :, :], u_log[0, :, :], save=False)
        t = time.time()

# set to best seen during training
if args.return_best:
    ctl.load_state_dict(best_params)
    # ctl.set_parameters_as_vector(best_params)

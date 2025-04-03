import torch
from torch.utils.data import DataLoader

from experiments.robot.arg_parser import argument_parser, print_args
from plants.robots import RobotsSystem, RobotsDataset
from plot_functions import plot_trajectories, plot_traj_vs_time
from controllers.input_signal import InputController


# ----- Overwriting arguments -----
args = argument_parser()
args.spring_const = 0
args.batch_size = 1

# ------------ 1. Dataset ------------
dataset = RobotsDataset(random_seed=args.random_seed, horizon=args.horizon, std_ini=args.std_init_plant)
# divide to train and test
train_data, test_data = dataset.get_data(num_train_samples=args.num_rollouts, num_test_samples=500)
# data for plots
t_ext = args.horizon
plot_data = torch.zeros(1, t_ext, train_data.shape[-1])
plot_data[:, 0, :] = (dataset.x0.detach() - dataset.xbar)
# batch the data
train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

# ------------ 2. Plant ------------
plant_input_init = None     # all zero
plant_state_init = None     # same as xbar
sys = RobotsSystem(xbar=dataset.xbar,
                   x_init=plant_state_init,
                   u_init=plant_input_init,
                   linear_plant=args.linearize_plant,
                   k=args.spring_const
                   )

# Plot OL trajectories
u1 = torch.cat([torch.ones(1, 20, sys.in_dim),
                torch.zeros(1, 80, sys.in_dim)], dim=1)
u2 = torch.cat([torch.sin(torch.arange(0,10,0.5)).view(1, -1, 1).repeat(1, 1, sys.in_dim),
                torch.zeros(1,80, sys.in_dim) ], dim=1)
u1[:,:,1] = -u1[:,:,1]
u2[:,:,1] = -u2[:,:,1]
ctl1 = InputController(u1)
ctl2 = InputController(u2)

x_log1, _, u_log1 = sys.rollout(ctl1, plot_data)
x_log2, _, u_log2 = sys.rollout(ctl2, plot_data)
plot_traj_vs_time(t_ext, x_log1[0, :, :], u_log1[0, :, :])
plot_traj_vs_time(t_ext, x_log2[0, :, :], u_log2[0, :, :])
# plot_traj_vs_time(t_ext, x_log1[0, :, :], u_log1[0, :, :], save=True, filename="OL_robot1")
# plot_traj_vs_time(t_ext, x_log2[0, :, :], u_log2[0, :, :], save=True, filename="OL_robot2")

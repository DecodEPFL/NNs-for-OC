import torch
from torch.utils.data import DataLoader

from experiments.robot.arg_parser import argument_parser, print_args
from plants.robots import RobotsSystem, RobotsDataset
from plot_functions import plot_trajectories, plot_traj_vs_time
from controllers.zero_controller import ZeroController


# ----- Overwriting arguments -----
args = argument_parser()
args.batch_size = 1

# ------------ 1. Dataset ------------
dataset = RobotsDataset(random_seed=args.random_seed, horizon=args.horizon, std_ini=args.std_init_plant)
# divide to train and test
train_data, test_data = dataset.get_data(num_train_samples=args.num_rollouts, num_test_samples=500)
# data for plots
t_ext = args.horizon * 5
plot_data = 0.01 * torch.randn(1, t_ext, train_data.shape[-1])
plot_data[:, 0, :] = (dataset.x0.detach())
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

# Plot CL trajectories using only the base controller
ctl = ZeroController(sys.in_dim)

x_log, _, u_log = sys.rollout(ctl, plot_data)
plot_traj_vs_time(t_ext, x_log[0, :, :])
# plot_traj_vs_time(t_ext, x_log[0, :, :], save=True, filename="base_robot")

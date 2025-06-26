import torch
import time
import copy
from torch.utils.data import DataLoader
from experiments.robot.arg_parser import argument_parser, print_args
from plants.robots import RobotsSystem, RobotsDataset
from plants.robots.robots_dataset import RobotsDatasetMulti, RobotsDatasetMultiCircle
from plot_functions import plot_trajectories, plot_traj_vs_time
from controllers.PB_controller import PerfBoostController
from loss_functions import RobotsLoss
import os
import logging
import math
from datetime import datetime
from torch.utils.data import DataLoader
from arg_parser import argument_parser, print_args
from argparse import Namespace
from loss_functions import RobotsLoss, RobotsLoss_v2
from assistive_functions import WrapLogger
from controllers.architectures import DWNConfig
import random

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ----- Overwriting arguments -----  # TODO: remove and put it in argsparse

cfg = {
    "n_u": 1,
    "n_y": 3,
    "d_model": 10,
    "d_state": 14,
    "n_layers": 1,
    "ff": "MLP",  # GLU | MLP | LMLP
    "max_phase": math.pi / 50,
    "r_min": 0.7,
    "r_max": 0.98,
    "gamma": False,
    "trainable": True,
    "gain": 2.4
}
cfg = Namespace(**cfg)

#torch.set_num_threads(10)

# Build model
config = DWNConfig(d_model=cfg.d_model, d_state=cfg.d_state, n_layers=cfg.n_layers, ff=cfg.ff, rmin=cfg.r_min,
                   rmax=cfg.r_max, max_phase=cfg.max_phase, gamma=cfg.gamma, trainable=cfg.trainable, gain=cfg.gain)

# ----- Overwriting arguments -----
args = argument_parser()
args.epochs = 600
# args.lr = 1e-3
args.num_rollouts = 150
args.log_epoch = args.epochs // 10 if args.epochs // 10 > 0 else 1
args.nn_type = "MI"
args.non_linearity = "coupling_layers"
args.batch_size = 75
args.config = config
args.horizon = 180
#args.alpha_u=50

# ----- SET UP LOGGER -----
now = datetime.now().strftime("%m_%d_%H_%M_%S")
save_path = os.path.join(BASE_DIR, 'experiments', 'minimal_example', 'saved_results')
save_folder = os.path.join(save_path, 'perf_boost_' + args.nn_type + '_' + now)
os.makedirs(save_folder)
logging.basicConfig(filename=os.path.join(save_folder, 'log'), format='%(asctime)s %(message)s', filemode='w')
logger = logging.getLogger('perf_boost_' + args.nn_type + '_')
logger.setLevel(logging.DEBUG)
logger = WrapLogger(logger)

# ----- parse and set experiment arguments -----
msg = print_args(args)
logger.info(msg)
torch.manual_seed(2)

# ------------ 1. Dataset ------------
dataset = RobotsDatasetMultiCircle(random_seed=2, horizon=args.horizon, std_ini=args.std_init_plant)
# divide to train and test
train_data, test_data = dataset.get_data(num_train_samples=args.num_rollouts, num_test_samples=500)

# data for plots


t_ext = args.horizon
plot_data = torch.zeros(1, t_ext, train_data.shape[-1])
plot_data[:, 0, 0:7] = train_data[0, 0, :]
# batch the data
train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

# ------------ 2. Plant ------------
plant_input_init = None  # all zero
plant_state_init = None  # same as xbar
sys = RobotsSystem(xbar=dataset.xbar,
                   x_init=plant_state_init,
                   u_init=plant_input_init,
                   linear_plant=args.linearize_plant,
                   k=args.spring_const
                   )

# ------------ 3. Controller ------------
ctl = PerfBoostController(noiseless_forward=sys.noiseless_forward,
                          input_init=sys.x_init,
                          output_init=sys.u_init,
                          nn_type=args.nn_type,
                          non_linearity=args.non_linearity,
                          dim_internal=args.dim_internal,
                          dim_nl=args.dim_nl,
                          config=args.config,
                          dim_in2=7,
                          initialization_std=args.cont_init_std,
                          )
# plot closed-loop trajectories before training the controller

x_log, _, u_log = sys.rollout(ctl, plot_data)
plot_trajectories(x_log[0, :, :], T=t_ext, obstacle_radius=plot_data[:, 0, 6:7], obstacle_centers=plot_data[:, 0, 4:6])
plot_traj_vs_time(t_ext, x_log[0, :, :], u_log[0, :, :])
total_n_params = sum(p.numel() for p in ctl.parameters() if p.requires_grad)
logger.info("[INFO] Number of parameters: %i" % total_n_params)

# ------------ 4. Loss ------------
Q = torch.eye(4) * 100
loss_fn = RobotsLoss_v2(
    Q=Q, alpha_u=args.alpha_u
)

# ------------ 5. Optimizer ------------
valid_data = train_data  # use the entire train data for validation
assert not (valid_data is None and args.return_best)
optimizer = torch.optim.Adam(ctl.parameters(), lr=args.lr)

# ------------ 6. Training ------------
print('------------ Begin training ------------')
best_valid_loss = 1e6
best_params = ctl.state_dict()  # ctl.get_parameters_as_vector()
loss = 1e6
t = time.time()
for epoch in range(1 + args.epochs):
    # iterate over all data batches
    for train_data_batch in train_dataloader:
        optimizer.zero_grad()
        # simulate over horizon steps
        x_log, _, u_log = sys.rollout(
            controller=ctl, data=train_data_batch, train=True,
        )
        # loss of this rollout
        circle = train_data_batch[:, :, 4:7].detach().clone()
        loss = loss_fn.forward(x_log, u_log, circle=circle)
        # take a step
        loss.backward()
        # Clip the gradients to a maximum norm (e.g., 1.0) before the optimizer step.
        torch.nn.utils.clip_grad_norm_(ctl.parameters(), max_norm=1.0)
        #      for p in ctl.parameters():
        #         print(p.grad)
        # Apply gradient clipping
        #torch.nn.utils.clip_grad_norm_(ctl.parameters(), 1)
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
                loss_valid = loss_fn.forward(x_log_valid, u_log_valid, valid_data[:, :, 4:7])
            msg += ' ---||--- validation loss: %.2f' % (loss_valid.item())
            # compare with the best valid loss
            if loss_valid.item() < best_valid_loss:
                best_valid_loss = loss_valid.item()
                best_params = copy.deepcopy(ctl.state_dict())
                # ctl.get_parameters_as_vector()  # record state dict if best on valid
                msg += ' (best so far)'
        duration = time.time() - t
        msg += ' ---||--- time: %.0f s' % duration
        print(msg)
        # plot trajectory
        random_sample = 12
        plot_data = torch.zeros(1, t_ext, valid_data.shape[-1])
        plot_data[:, 0, 0:7] = valid_data[random_sample, 0, :]
        plot_trajectories(x_log_valid[random_sample, :, :], T=t_ext, radius_robot=loss_fn.radius_robot, circles=True,
                          obstacle_radius=plot_data[:, 0, 6:7], obstacle_centers=plot_data[:, 0, 4:6])
        t = time.time()

# set to best seen during training
if args.return_best:
    ctl.load_state_dict(best_params)
    # ctl.set_parameters_as_vector(best_params)

# evaluate on the train data
print('[INFO] evaluating the trained controller on %i training rollouts.' % train_data.shape[0])
with torch.no_grad():
    x_log, _, u_log = sys.rollout(
        controller=ctl, data=train_data, train=False,
    )  # use the entire train data, not a batch
    # evaluate losses
    loss = loss_fn.forward(x_log, u_log, train_data[:, :, 4:7])
    print('Train loss: %.4f' % loss)

# evaluate on the test data
print('[INFO] evaluating the trained controller on %i test rollouts.' % test_data.shape[0])
with torch.no_grad():
    # simulate over horizon steps
    x_log, _, u_log = sys.rollout(
        controller=ctl, data=test_data, train=False,
    )
    # loss
    test_loss = loss_fn.forward(x_log, u_log, test_data[:, :, 4:7]).item()
    print("Test loss: %.4f" % test_loss)

# # plot closed-loop trajectories using the trained controller
# print('Plotting closed-loop trajectories using the trained controller...')
# x_log, _, u_log = sys.rollout(ctl, plot_data)
# plot_trajectories(
#     x_log[0, :, :], T=t_ext, radius_robot=loss_fn.radius_robot, circles=True,
#     obstacle_centers=loss_fn.obstacle_centers,
#     obstacle_radius=loss_fn.obstacle_radius,
#     #     save=True, filename="pb_robot"
# )
# plot_traj_vs_time(t_ext, x_log[0, :, :], u_log[0, :, :])


plot_data = torch.zeros(1, t_ext, train_data.shape[-1])
plot_data[:, 0, 0:7] = torch.tensor([2, 1, 0, 0, 1, 0.5, .9])
x_log, _, u_log = sys.rollout(ctl, plot_data)
plot_trajectories(x_log[0, :, :], T=t_ext, obstacle_radius=plot_data[:, 0, 6:7], obstacle_centers=plot_data[:, 0, 4:6])



plot_traj_vs_time(t_ext, x_log[0, :, :], u_log[0, :, :])
# ------------ Dataset for validation with wild initial conditions  ------------
dataset_wild = RobotsDataset(random_seed=args.random_seed, horizon=args.horizon, x0=torch.tensor([.3, 1.2, 0, 0]),
                             std_ini=.3)
wild_data = dataset_wild._generate_data(300)

# evaluate on the wild test data
print('[INFO] evaluating the trained controller on %i test rollouts.' % test_data.shape[0])
with torch.no_grad():
    # simulate over horizon steps
    x_log, _, u_log = sys.rollout(
        controller=ctl, data=wild_data, train=False,
    )
    # loss
    test_loss = loss_fn.forward(x_log, u_log).item()
    print("Test loss: %.4f" % test_loss)

# plot closed-loop trajectories using the trained controller on the wild
print('Plotting closed-loop trajectories using the trained controller...')
plot_data = wild_data[3, :, :].unsqueeze(0)
x_log, _, u_log = sys.rollout(ctl, plot_data)
plot_trajectories(
    x_log[0, :, :], T=t_ext, radius_robot=loss_fn.radius_robot, circles=True,
    obstacle_centers=loss_fn.obstacle_centers,
    obstacle_radius=loss_fn.obstacle_radius,
    #     save=True, filename="pb_robot"
)
plot_traj_vs_time(t_ext, x_log[0, :, :], u_log[0, :, :])

plot_data

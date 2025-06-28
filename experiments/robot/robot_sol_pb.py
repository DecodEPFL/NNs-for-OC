import torch
import time
import copy
from torch.utils.data import DataLoader
from experiments.robot.arg_parser import argument_parser, print_args
from plants.robots import RobotsSystem, RobotsDataset
from plants.robots.robots_dataset import RobotsDatasetMulti, RobotsDatasetMultiCircle, RobotsDatasetMultiCircle_v2
from plot_functions import plot_trajectories, plot_traj_vs_time
from controllers.PB_controller import PerfBoostController
from loss_functions import RobotsLoss
import os
from matplotlib import pyplot as plt
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

PATH = "my_model_weights.pth"

#torch.set_num_threads(10)

# Build model
config = DWNConfig(d_model=cfg.d_model, d_state=cfg.d_state, n_layers=cfg.n_layers, ff=cfg.ff, rmin=cfg.r_min,
                   rmax=cfg.r_max, max_phase=cfg.max_phase, gamma=cfg.gamma, trainable=cfg.trainable, gain=cfg.gain)

# ----- Overwriting arguments -----
args = argument_parser()
args.epochs = 900
# args.lr = 1e-3
args.num_rollouts = 500
args.log_epoch = args.epochs // 10 if args.epochs // 10 > 0 else 1
args.nn_type = "MI"
args.non_linearity = "coupling_layers"
args.batch_size = 60
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
dataset = RobotsDatasetMultiCircle_v2(random_seed=2, horizon=args.horizon, std_ini=args.std_init_plant)
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
optimizer = torch.optim.Adam(ctl.parameters(), lr=1e-3)

# ------------ 6. Training ------------
# ------------ 5. Setup for Training ------------
print('------------ Setting up training ------------')
# Initialize lists to store loss history for plotting
train_loss_history = []
valid_loss_history = []
epoch_log_points = []  # To store the epoch numbers for the x-axis

# Start the timer and initialize best loss
best_valid_loss = float('inf')
best_params = None

# ------------ 6. Training ------------
print('------------ Begin training ------------')
t_start_training = time.time()

# --- OUTER EPOCH LOOP ---
for epoch in range(args.epochs + 1):

    # --- A. TRAINING PHASE ---
    ctl.train()  # Set the model to training mode
    running_train_loss = 0.0  # Accumulator for the epoch's training loss

    # --- INNER BATCH LOOP (for training) ---
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
        torch.nn.utils.clip_grad_norm_(ctl.parameters(), max_norm=2.0)
        optimizer.step()

        # Add the loss of this batch to the accumulator
        running_train_loss += loss.item()

    # Calculate the average training loss for the entire epoch
    avg_epoch_train_loss = running_train_loss / len(train_dataloader)

    # --- B. VALIDATION AND LOGGING PHASE ---
    # This block is now OUTSIDE the batch loop and executes ONCE per epoch.
    if epoch % args.log_epoch == 0:
        t_log_start = time.time()

        # Append the average training loss for plotting
        train_loss_history.append(avg_epoch_train_loss)
        epoch_log_points.append(epoch)

        msg = f'Epoch: {epoch:4d} --- AVG train loss: {avg_epoch_train_loss:.2f}'

        # Validation logic
        if args.return_best:
            ctl.eval()  # Set the model to evaluation mode
            with torch.no_grad():
                # NOTE: It's good practice to average validation loss over all validation batches
                # For simplicity here, we use your single validation data tensor `valid_data`
                x_log_valid, _, u_log_valid = sys.rollout(
                    controller=ctl, data=valid_data, train=False,
                )
                loss_valid = loss_fn.forward(x_log_valid, u_log_valid, valid_data[:, :, 4:7])

            current_valid_loss = loss_valid.item()
            valid_loss_history.append(current_valid_loss)
            msg += f' ---||--- validation loss: {current_valid_loss:.2f}'

            # Check for best model
            if current_valid_loss < best_valid_loss:
                best_valid_loss = current_valid_loss
                best_params = copy.deepcopy(ctl.state_dict())
                msg += ' (best so far)'

        duration = time.time() - t_log_start
        msg += f' ---||--- log time: {duration:.0f}s'
        print(msg)  # This now prints only once per logging epoch!

        # Plotting logic (also now only runs once per logging epoch)
        # Note: 'x_log_valid' must be available from the validation step above
        if args.return_best:
            random_sample = 12
            if random_sample < valid_data.shape[0]:
                plot_data = torch.zeros(1, t_ext, valid_data.shape[-1])
                plot_data[:, 0, 0:7] = valid_data[random_sample, 0, :]
                plot_trajectories(
                    x_log_valid[random_sample, :, :],
                    T=t_ext,
                    radius_robot=loss_fn.radius_robot,
                    circles=True,
                    obstacle_radius=plot_data[:, 0, 6:7],
                    obstacle_centers=plot_data[:, 0, 4:6]
                )

# ------------ 7. Post-Training ------------
print('------------ End of training ------------')
total_duration_mins = (time.time() - t_start_training) / 60
print(f"Total training time: {total_duration_mins:.1f} minutes")

# --- PLOTTING THE LOSS CURVE ---
print("Generating and saving the loss curve plot...")
plt.figure(figsize=(12, 6))
plt.plot(epoch_log_points, train_loss_history, label='Avg. Training Loss', color='blue', marker='o')
if valid_loss_history:
    plt.plot(epoch_log_points, valid_loss_history, label='Validation Loss', color='orange', marker='x')
plt.title('Training and Validation Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('loss_curve.png')
plt.show()

# Load the best model if applicable
if args.return_best and best_params is not None:
    print(f"Loading best model from epoch with validation loss: {best_valid_loss:.2f}")
    ctl.load_state_dict(best_params)

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
plot_data[:, 0, 0:7] = torch.tensor([1.5, 1.5, 0, 0, 1, 0.5, .3])
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

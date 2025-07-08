import torch
import time
import copy
from plants.robots import RobotsSystem, RobotsDataset
from plants.robots.robots_dataset import RobotsDatasetMultiCircle_v2, generate_remedial_data, RobotsDatasetRemedial
from plot_functions import plot_trajectories, plot_traj_vs_time, plot_radius_sweep, plot_facet_grid, \
    plot_loss_landscape, plot_value_landscape
from controllers.PB_controller import PerfBoostController
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
    "d_model": 10,  #15
    "d_state": 14,  #24
    "n_layers": 1,
    "ff": "LMLP",  # GLU | MLP | LMLP
    "max_phase": math.pi / 50,
    "r_min": 0.7,
    "r_max": 0.98,
    "gamma": False,
    "trainable": True,
    "gain": 2.4
}
cfg = Namespace(**cfg)

PATH = "MI2_weights.pth"

#torch.set_num_threads(10)

# Build model
config = DWNConfig(d_model=cfg.d_model, d_state=cfg.d_state, n_layers=cfg.n_layers, ff=cfg.ff, rmin=cfg.r_min,
                   rmax=cfg.r_max, max_phase=cfg.max_phase, gamma=cfg.gamma, trainable=cfg.trainable, gain=cfg.gain)

# ----- Overwriting arguments -----
args = argument_parser()
args.epochs = 10
# args.lr = 1e-3
args.num_rollouts = 1200
args.log_epoch = args.epochs // 10 if args.epochs // 10 > 0 else 1
args.nn_type = "MI"
args.non_linearity = "coupling_layers"
args.batch_size = 80
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
torch.manual_seed(15)

# ------------ 1. Dataset ------------
# --- Generate the NEW, dynamic remedial dataset ---
# We are no longer fixing the obstacle, but defining the geometry of the "hard" cases.

obstacle_center_fixed = torch.tensor([1.0, 0.5])
min_radius_to_train = 0.05
max_radius_to_train = .3

# 1. Instantiate the dataset class (no change here)
dataset = RobotsDatasetRemedial(random_seed=12, horizon=args.horizon)

# 3. Call the NEW method to generate your data
train_data, test_data = dataset.get_data_for_fixed_center(
    num_train_samples=args.num_rollouts,
    num_test_samples=1200,

    # Pass the fixed center and radius range
    fixed_center=obstacle_center_fixed,
    min_radius=min_radius_to_train,
    max_radius=max_radius_to_train,

    # Pass the parameters that define the remedial strategy
    remedial_data_ratio=0.1,
    critical_angle_spread=math.pi / 2,
    critical_radial_extension=1,
    square_bounds=(-2, 2)
)

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

# ------------ 4. Loss ------------
Q = torch.eye(4) * 100
loss_fn = RobotsLoss_v2(
    Q=Q, alpha_u=args.alpha_u, alpha_corridor=.5, alpha_q_scaling=6
)

ctl.load_state_dict(torch.load(PATH, weights_only=True))
ctl.train()

x_log, _, u_log = sys.rollout(ctl, plot_data)
plot_trajectories(x_log[0, :, :], T=t_ext, obstacle_radius=plot_data[:, 0, 6:7], obstacle_centers=plot_data[:, 0, 4:6])
plot_traj_vs_time(t_ext, x_log[0, :, :], u_log[0, :, :])
total_n_params = sum(p.numel() for p in ctl.parameters() if p.requires_grad)
logger.info("[INFO] Number of parameters: %i" % total_n_params)

# ------------ 5. Optimizer ------------
valid_data = test_data
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

# ==============================================================================
# FINAL VISUALIZATION OF TRAINED CONTROLLER
# ==============================================================================
print("\n[INFO] Generating final performance visualizations...")

# --- Plot 1: Radius Sweep ---
start_pos_for_sweep = torch.tensor([2, 1])
obstacle_center_for_sweep = torch.tensor([1, 0.5])
radii_to_test_sweep = [0.2, 0.6, 1]
plot_radius_sweep(ctl, sys, start_pos_for_sweep, obstacle_center_for_sweep, radii_to_test_sweep, args.horizon)

# --- Plot 2: Facet Grid ---
# You need your symmetrical point generator for this
# Assuming RobotsDatasetMultiCircle_v2 has access to it.
from plants.robots.robots_dataset import generate_four_way_symmetrical_points

# Note: You may need to adjust this import path!

center_for_grid = torch.tensor([1, 0.5])
radii_for_grid = [0.2, 0.6, 1]
# Generate a set of symmetrical points to test from
_, _, symmetrical_points = generate_four_way_symmetrical_points(center=center_for_grid)
# Use just two points for a cleaner grid
start_points_for_grid = symmetrical_points[0:2]

plot_facet_grid(ctl, sys, start_points_for_grid, radii_for_grid, center_for_grid, args.horizon)

# --- Plot 3: Loss Landscape Heatmap ---
# Define a single, interesting scenario to analyze in detail.
start_point_for_landscape = torch.tensor([2.0, 1])
center_for_landscape = torch.tensor([1, 0.5])
radius_for_landscape = .16

# --- Plot 4: NEW Value Landscape Heatmap ---
# This plot shows the performance of the TRAINED CONTROLLER.
# It uses the same scenario parameters.
# Define a list of interesting starting points to visualize
points_to_plot = [
    [1, 0.5 + radius_for_landscape + 0.2],  # A standard start point
    [1.2, 0.5 + radius_for_landscape],  # Its symmetrical counterpart to check for bias
    [1.5, 0.5 + radius_for_landscape],  # A point VERY close to the top edge of the obstacle
    [.76, 0.5 + radius_for_landscape + 0.2]  # A point just inside the obstacle to see the "escape"
]

plot_value_landscape(
    loss_fn=loss_fn,
    ctl=ctl,
    sys=sys,
    center=center_for_landscape,
    radius=radius_for_landscape,
    resolution=200,
    horizon=400,
    bounds=(-.2, 2),
    batch_size=20000,
    overlay_trajectories_from=points_to_plot  # Pass the list here
)

# The loss_fn object is already defined and holds all our parameters.
# The ctl and sys objects are also trained and ready.
plot_loss_landscape(
    loss_fn=loss_fn,
    ctl=ctl,
    sys=sys,
    start_point=start_point_for_landscape,
    center=center_for_landscape,
    radius=radius_for_landscape,
    horizon=400,
    vmax_percentile=92.7
)

# This is a plot for a specific initial position and obstacle
plot_data = torch.zeros(1, t_ext, train_data.shape[-1])
plot_data[:, 0, 0:7] = torch.tensor([1.2, 1, 0, 0, 1, 0.5, .2])
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

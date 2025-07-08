#!/usr/bin/env python
import torch
import time
import copy
import os
import sys
import logging
from datetime import datetime
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt

# Add the project root to the Python path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

from experiments.robot_multi_obstacle.arg_parser import argument_parser, print_args
from experiments.robot_multi_obstacle.multi_obstacle_dataset import MultiObstacleDataset
from experiments.robot_multi_obstacle.robots_sys import RobotsSystem
from experiments.robot_multi_obstacle.loss_functions import RobotsLoss_v2
from controllers.PB_controller import PerfBoostController
from controllers.architectures import DWNConfig
from assistive_functions import WrapLogger
from experiments.robot_multi_obstacle.plot_functions import plot_trajectories

def main():
    args = argument_parser()

    # ----- SET UP LOGGER -----
    now = datetime.now().strftime("%m_%d_%H_%M_%S")
    save_path = os.path.join(BASE_DIR, 'experiments', 'robot_multi_obstacle', 'saved_results')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_folder = os.path.join(save_path, 'perf_boost_' + args.nn_type + '_' + now)
    os.makedirs(save_folder)
    logging.basicConfig(filename=os.path.join(save_folder, 'log'), format='%(asctime)s %(message)s', filemode='w')
    logger = logging.getLogger('perf_boost_' + args.nn_type + '_')
    logger.setLevel(logging.DEBUG)
    logger = WrapLogger(logger)

    msg = print_args(args)
    logger.info(msg)
    torch.manual_seed(args.random_seed)

    # ----- DATASET -----
    dataset = MultiObstacleDataset(num_samples=args.num_rollouts, horizon=args.horizon, num_obstacles=args.num_obstacles, random_seed=args.random_seed)
    train_data, test_data = dataset.get_data(num_train_samples=int(args.num_rollouts*0.8), num_test_samples=int(args.num_rollouts*0.2))
    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

    # ----- SYSTEM -----
    sys = RobotsSystem(xbar=dataset.xbar,
                       x_init=None,
                       u_init=None,
                       linear_plant=args.linearize_plant,
                       k=args.spring_const
                       )

    # ----- CONTROLLER -----
    d_model = 10
    d_state = 14
    n_layers = 1
    max_phase = 3.14 / 50
    r_min = 0.7
    r_max = 0.98

    config = DWNConfig(d_model=d_model, d_state=d_state, n_layers=n_layers, ff=args.non_linearity, rmin=r_min,
                       rmax=r_max, max_phase=max_phase, gamma=False, trainable=True, gain=2.4)

    dim_in2 = 4 + 3 * args.num_obstacles
    ctl = PerfBoostController(noiseless_forward=sys.noiseless_forward,
                              input_init=sys.x_init,
                              output_init=sys.u_init,
                              nn_type=args.nn_type,
                              non_linearity=args.non_linearity,
                              dim_internal=args.dim_internal,
                              dim_nl=args.dim_nl,
                              config=config,
                              dim_in2=dim_in2,
                              initialization_std=args.cont_init_std,
                              )

    # ----- LOSS -----
    Q = torch.eye(4) * 100
    loss_fn = RobotsLoss_v2(Q=Q, alpha_u=args.alpha_u)

    # ----- OPTIMIZER -----
    optimizer = torch.optim.Adam(ctl.parameters(), lr=args.lr)

    # ----- TRAINING -----
    print('------------ Begin training ------------')
    t_start_training = time.time()
    best_valid_loss = float('inf')
    best_params = None

    for epoch in range(args.epochs + 1):
        ctl.train()
        running_train_loss = 0.0

        for train_data_batch in train_dataloader:
            optimizer.zero_grad()

            x_log, _, u_log = sys.rollout(
                controller=ctl, data=train_data_batch, train=True, num_obstacles=args.num_obstacles
            )

            loss = loss_fn.forward(x_log, u_log, train_data_batch, num_obstacles=args.num_obstacles)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(ctl.parameters(), max_norm=2.0)
            optimizer.step()

            running_train_loss += loss.item()

        avg_epoch_train_loss = running_train_loss / len(train_dataloader)

        if epoch % args.log_epoch == 0:
            ctl.eval()
            with torch.no_grad():
                x_log_valid, _, u_log_valid = sys.rollout(
                    controller=ctl, data=test_data, train=False, num_obstacles=args.num_obstacles
                )
                loss_valid = loss_fn.forward(x_log_valid, u_log_valid, test_data, num_obstacles=args.num_obstacles)

            current_valid_loss = loss_valid.item()
            msg = f'Epoch: {epoch:4d} --- AVG train loss: {avg_epoch_train_loss:.4f} ---||--- validation loss: {current_valid_loss:.4f}'

            if current_valid_loss < best_valid_loss:
                best_valid_loss = current_valid_loss
                best_params = copy.deepcopy(ctl.state_dict())
                msg += ' (best so far)'

            print(msg)

    total_duration_mins = (time.time() - t_start_training) / 60
    print(f"Total training time: {total_duration_mins:.1f} minutes")

    if args.return_best and best_params is not None:
        print(f"Loading best model from epoch with validation loss: {best_valid_loss:.4f}")
        ctl.load_state_dict(best_params)

    # ----- EVALUATION -----
    print('[INFO] evaluating the trained controller on test data.')
    with torch.no_grad():
        x_log, _, u_log = sys.rollout(
            controller=ctl, data=test_data, train=False, num_obstacles=args.num_obstacles
        )
        loss = loss_fn.forward(x_log, u_log, test_data, num_obstacles=args.num_obstacles)
        print('Test loss: %.4f' % loss)

        # Visualize a random trajectory from the test set
        random_sample_idx = torch.randint(0, len(test_data), (1,)).item()
        sample_x_log = x_log[random_sample_idx]
        sample_data = test_data[random_sample_idx]
        obstacles_info = sample_data[0, 4:].view(args.num_obstacles, 3)

        plot_trajectories(
            sample_x_log,
            T=args.horizon,
            obstacles_data=obstacles_info,
            num_obstacles=args.num_obstacles,
            save=True,
            filename=os.path.join(save_folder, 'final_trajectory')
        )

if __name__ == '__main__':
    main()

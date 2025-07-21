import torch
import time
import copy
import os
import logging
import math
from datetime import datetime
from torch.utils.data import DataLoader, TensorDataset
from argparse import ArgumentParser
from tqdm import tqdm
from matplotlib import pyplot as plt

# --- MODIFIED ---
# Make sure these imports point to your files containing the final versions of these components.
from robots_sys import RobotsSystemMultiObstacle
from multi_obstacle_dataset import RobotsDatasetMultiObstacle, generate_slalom_scenario
from loss_functions import RobotsLossMultiObstacle as RobotsLossMultiObstacle  # Using the improved loss
from plot_functions import plot_multi_obstacle_value_landscape as plot_multi_obstacle_performance  # Using the best plot

from controllers.PB_controller import PerfBoostController
from assistive_functions import WrapLogger
from controllers.m_operators.ssm import SSMConfig

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def setup_experiment(args):
    """Initializes logging, directories, and seeds for reproducibility."""
    now = datetime.now().strftime("%m_%d_%H_%M_%S")
    save_folder = os.path.join(BASE_DIR, 'experiments', 'multi_obstacle', 'saved_results',
                               f'slalom_training_{args.nn_type}_{now}')
    os.makedirs(save_folder, exist_ok=True)
    log_file = os.path.join(save_folder, 'log.txt')
    logging.basicConfig(filename=log_file, format='%(asctime)s %(message)s', filemode='w')
    logger = logging.getLogger(f'perf_boost_{args.nn_type}_')
    logger.setLevel(logging.DEBUG)
    logger = WrapLogger(logger)
    logger.info("----- Experiment Configuration -----")
    for arg, value in sorted(vars(args).items()):
        logger.info(f"{arg}: {value}")
    logger.info("------------------------------------")
    torch.manual_seed(args.seed)
    return logger, save_folder


def load_data(args):
    """Loads and prepares the multi-obstacle slalom datasets."""
    dataset = RobotsDatasetMultiObstacle(random_seed=args.seed, horizon=args.horizon)
    train_data, test_data = dataset.get_data(
        num_train_samples=args.num_rollouts,
        num_test_samples=500
    )
    train_dataloader = DataLoader(TensorDataset(train_data), batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(TensorDataset(test_data), batch_size=args.batch_size, shuffle=False)
    xbar = torch.zeros(4)
    return train_data, test_data, train_dataloader, test_dataloader, xbar


def build_models_and_optimizer(args, xbar):
    """Builds the system, controller, loss function, and optimizer."""
    sys = RobotsSystemMultiObstacle(
        xbar=xbar, x_init=None, u_init=None,
        linear_plant=args.linearize_plant, k=args.spring_const,
        num_obstacles=args.num_obstacles
    )

    # --- MODIFIED ---
    # Programmatically calculate the dimension of the controller's context vector `w_controller`
    # This avoids "magic numbers" and makes the code robust to changes in feature engineering.
    # From _rollout_step_sorted_threat: robot_vel(2) + vec_to_goal(2) + sorted_dists(num_obs) + sorted_vecs(num_obs*2)
    dim_w_controller = 2 + 2 + args.num_obstacles + (args.num_obstacles * 2)

    ctl = PerfBoostController(
        noiseless_forward=sys.noiseless_forward, input_init=sys.x_init, output_init=sys.u_init,
        nn_type=args.nn_type, non_linearity=args.non_linearity,
        dim_internal=args.dim_internal, dim_nl=args.dim_nl,
        config=args.config,
        dim_in2=dim_w_controller,  # <-- Use the calculated dimension
        initialization_std=args.cont_init_std,
    )

    # Use the improved loss function for better stability and path efficiency
    loss_fn = RobotsLossMultiObstacle(
        Q=torch.eye(4) * 100, alpha_u=args.alpha_u,
        num_obstacles=args.num_obstacles
    )
    optimizer = torch.optim.Adam(ctl.parameters(), lr=args.lr)
    return sys, ctl, loss_fn, optimizer


def evaluate(ctl, sys, loss_fn, dataloader, device='cpu'):
    """Evaluates the model on a given dataset."""
    ctl.eval()
    total_loss = 0.0
    with torch.no_grad():
        for i, (data_batch,) in enumerate(dataloader):
            data_batch = data_batch.to(device)
            x_log, u_log = sys.rollout(controller=ctl, data=data_batch, train=False)
            loss = loss_fn.forward(x_log, u_log, data_batch)
            total_loss += loss.item()
    return total_loss / len(dataloader)


def train(args, ctl, sys, loss_fn, optimizer, train_dataloader, valid_dataloader, logger, save_folder):
    """The core training loop."""
    logger.info('------------ Begin training ------------')
    t_start_training = time.time()
    history = {'train_loss': [], 'valid_loss': [], 'epochs': []}
    best_valid_loss = float('inf')
    best_params = None

    for epoch in range(args.epochs):
        ctl.train()
        running_train_loss = 0.0
        epoch_iterator = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{args.epochs}", leave=False)
        for i, (train_batch,) in enumerate(epoch_iterator):
            optimizer.zero_grad()
            x_log, u_log = sys.rollout(controller=ctl, data=train_batch, train=True)
            loss = loss_fn.forward(xs_log=x_log, us_log=u_log, initial_data_batch=train_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ctl.parameters(), max_norm=2.0)
            optimizer.step()
            running_train_loss += loss.item()
            epoch_iterator.set_postfix(loss=f'{running_train_loss / (i + 1):.4f}')

        avg_epoch_train_loss = running_train_loss / len(train_dataloader)

        if epoch % args.log_epoch == 0:
            history['train_loss'].append(avg_epoch_train_loss)
            history['epochs'].append(epoch)
            log_msg = f"Epoch: {epoch:4d} --- Avg Train Loss: {avg_epoch_train_loss:.4f}"

            current_valid_loss = evaluate(ctl, sys, loss_fn, valid_dataloader)
            history['valid_loss'].append(current_valid_loss)
            is_best = current_valid_loss < best_valid_loss
            if is_best:
                best_valid_loss = current_valid_loss
                best_params = copy.deepcopy(ctl.state_dict())
                torch.save(best_params, os.path.join(save_folder, 'best_model.pth'))

            log_msg += f' | Validation Loss: {current_valid_loss:.4f}{" (new best)" if is_best else ""}'
            logger.info(log_msg)
            tqdm.write(log_msg)

    logger.info(f"Total training time: {(time.time() - t_start_training) / 60:.1f} minutes")
    if args.return_best and best_params:
        logger.info(f"Loading best model with validation loss: {best_valid_loss:.4f}")
        ctl.load_state_dict(best_params)
    return ctl, history


def plot_results(save_folder, history):
    """Plots the training and validation loss curves."""
    plt.figure(figsize=(10, 5))
    plt.plot(history['epochs'], history['train_loss'], label='Avg. Training Loss', marker='o')
    if history['valid_loss']:
        plt.plot(history['epochs'], history['valid_loss'], label='Validation Loss', marker='x')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch');
    plt.ylabel('Loss')
    plt.legend();
    plt.grid(True)
    plt.savefig(os.path.join(save_folder, 'loss_curve.png'))
    plt.close()  # Close the figure to free up memory


# --- NEW ---
def run_final_visualizations(ctl, sys, loss_fn, args, save_folder):
    """Generates final plots to visualize performance on a specific, deterministic slalom task."""
    print("\n[INFO] Generating final performance visualization for a specific test case...")

    # --- DEFINE YOUR PRECISE TEST CASE ---
    test_scenario = generate_slalom_scenario(
        num_obstacles=args.num_obstacles,
        stagger_distance=1.3,
        fixed_start_point=torch.tensor([-4.0, 3.5]),
        fixed_radii=[1.2, 1, 1.1]
    )

    # Use the ultimate plotting function
    plot_multi_obstacle_performance(
        loss_fn=loss_fn,
        ctl=ctl,
        sys=sys,
        scenario_for_plot=test_scenario,
        horizon=args.horizon,
        save=True,
        filename=os.path.join(save_folder, 'final_performance_slalom.png')
    )


def main():
    """Main function to run the multi-obstacle slalom training experiment."""
    parser = ArgumentParser(description="Multi-Obstacle Slalom Robot Training Experiment")
    parser.add_argument('--num_obstacles', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=60)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num_rollouts', type=int, default=500)
    parser.add_argument('--log_epoch', type=int, default=None)
    parser.add_argument('--return_best', action='store_true', default=True)
    parser.add_argument('--seed', type=int, default=2)
    parser.add_argument('--nn_type', type=str, default="MI")
    parser.add_argument('--non_linearity', type=str, default="LMLP")
    parser.add_argument('--dim_internal', type=int, default=64)
    parser.add_argument('--dim_nl', type=int, default=64)
    parser.add_argument('--cont_init_std', type=float, default=0.01)
    parser.add_argument('--horizon', type=int, default=180)
    parser.add_argument('--std_init_plant', type=float, default=0.1)
    parser.add_argument('--linearize_plant', action='store_true', default=False)
    parser.add_argument('--spring_const', type=float, default=0.1)
    parser.add_argument('--alpha_u', type=float, default=50.0)
    args = parser.parse_args()

    args = parser.parse_args()  # This line should be adapted to include all your args

    if args.log_epoch is None:
        args.log_epoch = args.epochs // 10 if args.epochs // 10 > 0 else 1

    ssm_cfg = {"d_model": 10, "d_state": 14, "n_layers": 1, "ff": "LMLP", "max_phase": math.pi / 50,
               "rmin": 0.7, "rmax": 0.98, "gamma": False, "trainable": True, "gain": 2.4}
    args.config = SSMConfig(**ssm_cfg)

    logger, save_folder = setup_experiment(args)
    train_data, test_data, train_dataloader, test_dataloader, xbar = load_data(args)
    sys, ctl, loss_fn, optimizer = build_models_and_optimizer(args, xbar)
    logger.info(f"[INFO] Controller Parameters: {sum(p.numel() for p in ctl.parameters() if p.requires_grad)}")

    ctl, history = train(args, ctl, sys, loss_fn, optimizer, train_dataloader, test_dataloader, logger, save_folder)

    plot_results(save_folder, history)

    # Final evaluation on the test set with the best model
    final_test_loss = evaluate(ctl, sys, loss_fn, test_dataloader)
    logger.info(f"Final Test Loss: {final_test_loss:.4f}")

    # Run the final, high-quality visualization
    run_final_visualizations(ctl, sys, loss_fn, args, save_folder)


if __name__ == "__main__":
    main()
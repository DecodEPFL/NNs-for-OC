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
# Make sure the imports point to the files containing the NEW slalom generator
# and the dataset class that uses it.
from robots_sys import RobotsSystemMultiObstacle
from multi_obstacle_dataset import RobotsDatasetMultiObstacle, generate_slalom_scenario
from loss_functions import RobotsLossMultiObstacle
from plot_functions import plot_multi_obstacle_performance

from controllers.PB_controller import PerfBoostController
from assistive_functions import WrapLogger
from controllers.m_operators.ssm import SSMConfig

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def setup_experiment(args):
    """Initializes logging, directories, and seeds for reproducibility."""
    now = datetime.now().strftime("%m_%d_%H_%M_%S")
    # Give a more descriptive folder name for the new task
    save_folder = os.path.join(BASE_DIR, 'experiments', 'multi_obstacle', 'saved_results',
                               f'slalom_training_{args.nn_type}_{now}')
    os.makedirs(save_folder, exist_ok=True)

    # ... (rest of the function is identical)
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
    """Loads and prepares the new multi-obstacle slalom datasets."""
    # This function assumes your `RobotsDatasetMultiObstacle` class has been
    # updated to use the `_generate_slalom_scenarios` method as we discussed.
    dataset = RobotsDatasetMultiObstacle(random_seed=args.seed, horizon=args.horizon)

    # --- MODIFIED ---
    # The `get_data` call is now simpler. The `challenge_ratio` is no longer
    # needed as every sample from the slalom generator is a challenge.
    train_data, test_data = dataset.get_data(
        num_train_samples=args.num_rollouts,
        num_test_samples=500
    )

    train_dataloader = DataLoader(TensorDataset(train_data), batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(TensorDataset(test_data), batch_size=args.batch_size, shuffle=False)
    xbar = torch.zeros(4)
    return train_data, test_data, train_dataloader, test_dataloader, xbar


def build_models_and_optimizer(args, xbar):
    """Builds the system, controller, etc. This function is already general enough and needs no changes."""
    sys = RobotsSystemMultiObstacle(
        xbar=xbar, x_init=None, u_init=None,
        linear_plant=args.linearize_plant, k=args.spring_const,
        num_obstacles=args.num_obstacles
    )
    ctl = PerfBoostController(
        noiseless_forward=sys.noiseless_forward, input_init=sys.x_init, output_init=sys.u_init,
        nn_type=args.nn_type, non_linearity=args.non_linearity,
        dim_internal=args.dim_internal, dim_nl=args.dim_nl,
        config=args.config, dim_in2=7,
        initialization_std=args.cont_init_std,
    )
    loss_fn = RobotsLossMultiObstacle(
        Q=torch.eye(4) * 100, alpha_u=args.alpha_u,
        num_obstacles=args.num_obstacles
    )
    optimizer = torch.optim.Adam(ctl.parameters(), lr=args.lr)
    return sys, ctl, loss_fn, optimizer


def evaluate(ctl, sys, loss_fn, dataloader, device='cpu'):
    """This function is general and needs no changes."""
    ctl.eval()
    total_loss = 0.0
    with torch.no_grad():
        for i, (data_batch,) in enumerate(dataloader):
            data_batch = data_batch.to(device)
            x_log, u_log = sys.rollout(controller=ctl, data=data_batch, train=False)
            loss = loss_fn.forward(x_log, u_log, data_batch)
            total_loss += loss.item()
    return total_loss / len(dataloader)


def train(args, ctl, sys, loss_fn, optimizer, train_dataloader, valid_dataloader, logger):
    """This core training loop is general and needs no changes."""
    # ... (The entire train function is identical to your provided version)
    logger.info('------------ Begin training ------------')
    t_start_training = time.time()
    history = {'train_loss': [], 'valid_loss': [], 'epochs': []}
    best_valid_loss = float('inf')
    best_params = None
    epoch_iterator = tqdm(range(args.epochs), desc="Training Progress", dynamic_ncols=True)
    postfix_dict = {}
    for epoch in epoch_iterator:
        ctl.train()
        running_train_loss = 0.0
        for i, (train_batch,) in enumerate(train_dataloader):
            optimizer.zero_grad()
            x_log, u_log = sys.rollout(controller=ctl, data=train_batch, train=True)
            loss = loss_fn.forward(xs_log=x_log, us_log=u_log, initial_data_batch=train_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ctl.parameters(), max_norm=2.0)
            optimizer.step()
            running_train_loss += loss.item()
            postfix_dict['Running Loss'] = f'{running_train_loss / (i + 1):.4f}'
            epoch_iterator.set_postfix(postfix_dict, refresh=False)
        epoch_iterator.refresh()
        avg_epoch_train_loss = running_train_loss / len(train_dataloader)
        if 'Running Loss' in postfix_dict: del postfix_dict['Running Loss']
        postfix_dict['Avg Train Loss'] = f'{avg_epoch_train_loss:.4f}'
        if epoch % args.log_epoch == 0:
            history['train_loss'].append(avg_epoch_train_loss)
            history['epochs'].append(epoch)
            log_msg = f"Epoch: {epoch:4d} --- Avg Train Loss: {avg_epoch_train_loss:.4f}"
            if args.return_best:
                current_valid_loss = evaluate(ctl, sys, loss_fn, valid_dataloader)
                history['valid_loss'].append(current_valid_loss)
                is_best = current_valid_loss < best_valid_loss
                if is_best:
                    best_valid_loss = current_valid_loss
                    best_params = copy.deepcopy(ctl.state_dict())
                val_loss_str = f'{current_valid_loss:.4f}{" (new best)" if is_best else ""}'
                postfix_dict['Validation Loss'] = val_loss_str
                postfix_dict['Best Val Loss'] = f'{best_valid_loss:.4f}'
                log_msg += f' | Validation Loss: {current_valid_loss:.4f}{" (best)" if is_best else ""}'
            logger.info(log_msg)
        epoch_iterator.set_description(f"Epoch {epoch + 1}/{args.epochs}")
        epoch_iterator.set_postfix(postfix_dict, refresh=True)
    epoch_iterator.close()
    logger.info(f"Total training time: {(time.time() - t_start_training) / 60:.1f} minutes")
    if args.return_best and best_params:
        logger.info(f"Loading best model with validation loss: {best_valid_loss:.4f}")
        ctl.load_state_dict(best_params)
    return ctl, history


def plot_results(save_folder, history):
    """This function is general and needs no changes."""
    plt.figure(figsize=(10, 5))
    plt.plot(history['epochs'], history['train_loss'], label='Avg. Training Loss', marker='o')
    if history['valid_loss']:
        plt.plot(history['epochs'], history['valid_loss'], label='Validation Loss', marker='x')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_folder, 'loss_curve.png'))
    plt.show()


def run_final_visualizations(ctl, sys, loss_fn, args):
    """Generates final plots to visualize performance on a specific, deterministic slalom task."""
    print("\n[INFO] Generating final performance visualization for a specific test case...")

    # --- DEFINE YOUR PRECISE TEST CASE ---
    # The exact start point you want to test from.
    start_pos_for_plot = torch.tensor([-4.0, 4.0])

    # The exact radii for the obstacles in the course.
    radii_for_plot = [1.1, 0.8, 1.5]

    # Generate the fully deterministic test scenario
    test_scenario = generate_slalom_scenario(
        num_obstacles=args.num_obstacles,
        stagger_distance=1.3,
        fixed_start_point=start_pos_for_plot,  # <-- Pass the fixed start point
        fixed_radii=radii_for_plot  # <-- Pass the fixed radii
    )

    # The plotting function remains unchanged, as it just consumes the scenario dictionary.
    plot_multi_obstacle_performance(
        loss_fn=loss_fn,
        ctl=ctl,
        sys=sys,
        scenario=test_scenario,
        horizon=args.horizon,
        save=True,
        filename='final_performance_deterministic_slalom.png'
    )


def main():
    """Main function to run the multi-obstacle slalom training experiment."""
    # This function is general and needs no changes beyond adding args if needed.
    parser = ArgumentParser(description="Multi-Obstacle Slalom Robot Training Experiment")
    parser.add_argument('--num_obstacles', type=int, default=3)
    # ... (all your other arguments are the same) ...
    parser.add_argument('--epochs', type=int, default=400)
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

    if args.log_epoch is None:
        args.log_epoch = args.epochs // 10 if args.epochs // 10 > 0 else 1

    ssm_cfg = {"d_model": 10, "d_state": 14, "n_layers": 1, "ff": "LMLP", "max_phase": math.pi / 50,
               "rmin": 0.7, "rmax": 0.98, "gamma": False, "trainable": True, "gain": 2.4}
    args.config = SSMConfig(**ssm_cfg)

    logger, save_folder = setup_experiment(args)
    train_data, test_data, train_dataloader, test_dataloader, xbar = load_data(args)
    sys, ctl, loss_fn, optimizer = build_models_and_optimizer(args, xbar)
    logger.info(f"[INFO] Controller Parameters: {sum(p.numel() for p in ctl.parameters() if p.requires_grad)}")

    ctl, history = train(args, ctl, sys, loss_fn, optimizer, train_dataloader, test_dataloader, logger)

    plot_results(save_folder, history)
    final_train_loss = evaluate(ctl, sys, loss_fn, train_dataloader)
    logger.info(f'Final Train Loss: {final_train_loss:.4f}')
    final_test_loss = evaluate(ctl, sys, loss_fn, test_dataloader)
    logger.info(f"Final Test Loss: {final_test_loss:.4f}")

    run_final_visualizations(ctl, sys, loss_fn, args)


if __name__ == "__main__":
    main()
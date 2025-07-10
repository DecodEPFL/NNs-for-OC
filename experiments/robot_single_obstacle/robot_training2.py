import torch
import time
import copy
import os
import logging
import math
from datetime import datetime
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from tqdm import tqdm
from matplotlib import pyplot as plt

# Local imports
from experiments.robot_single_obstacle.robots_sys import RobotsSystem
from experiments.robot_single_obstacle.datasets import RobotsDatasetMultiCircle_v2, generate_four_way_symmetrical_points
from plot_functions import plot_trajectories, plot_traj_vs_time, plot_radius_sweep, plot_facet_grid, \
    plot_loss_landscape, plot_value_landscape
from controllers.PB_controller import PerfBoostController
from loss_functions import RobotsLoss_v2
from assistive_functions import WrapLogger
from controllers.m_operators.ssm import SSMConfig

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def setup_experiment(args):
    """Initializes logging, directories, and seeds for reproducibility."""
    now = datetime.now().strftime("%m_%d_%H_%M_%S")
    save_folder = os.path.join(BASE_DIR, 'experiments', 'minimal_example', 'saved_results',
                               f'perf_boost_{args.nn_type}_{now}')
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
    """Loads and prepares the datasets and dataloaders."""
    dataset = RobotsDatasetMultiCircle_v2(random_seed=args.seed, horizon=args.horizon, std_ini=args.std_init_plant)
    train_data, test_data = dataset.get_data(num_train_samples=args.num_rollouts, num_test_samples=500)
    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    return train_data, test_data, train_dataloader, dataset.xbar


def build_models_and_optimizer(args, xbar):
    """Builds the system, controller, loss function, and optimizer."""
    sys = RobotsSystem(xbar=xbar,
                       x_init=None,
                       u_init=None,
                       linear_plant=args.linearize_plant,
                       k=args.spring_const
                       )

    ctl = PerfBoostController(
        noiseless_forward=sys.noiseless_forward,
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
    loss_fn = RobotsLoss_v2(Q=torch.eye(4) * 100, alpha_u=args.alpha_u)
    optimizer = torch.optim.Adam(ctl.parameters(), lr=args.lr)
    return sys, ctl, loss_fn, optimizer


def evaluate(ctl, sys, loss_fn, data, device='cpu'):
    """Evaluates the controller on a given dataset."""
    ctl.eval()
    with torch.no_grad():
        data = data.to(device)
        x_log, _, u_log = sys.rollout(controller=ctl, data=data, train=False)
        loss = loss_fn.forward(x_log, u_log, data[:, :, 4:7])
    return loss.item()


def train(args, ctl, sys, loss_fn, optimizer, train_dataloader, valid_data, logger):
        """Main training loop with a single, real-time updating tqdm progress bar."""
        logger.info('------------ Begin training ------------')
        t_start_training = time.time()

        history = {'train_loss': [], 'valid_loss': [], 'epochs': []}
        best_valid_loss = float('inf')
        best_params = None

        epoch_iterator = tqdm(range(args.epochs + 1), desc="Training Progress", dynamic_ncols=True)
        postfix_dict = {}

        for epoch in epoch_iterator:
            ctl.train()
            running_train_loss = 0.0

            for i, train_batch in enumerate(train_dataloader):
                optimizer.zero_grad()
                x_log, _, u_log = sys.rollout(controller=ctl, data=train_batch, train=True)
                loss = loss_fn.forward(x_log, u_log, circle=train_batch[:, :, 4:7].detach().clone())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(ctl.parameters(), max_norm=2.0)
                optimizer.step()

                running_train_loss += loss.item()

                # Update postfix with running loss for the current batch
                postfix_dict['Running Loss'] = f'{running_train_loss / (i + 1):.2f}'
                epoch_iterator.set_postfix(postfix_dict, refresh=False)

            epoch_iterator.refresh()

            avg_epoch_train_loss = running_train_loss / len(train_dataloader)
            if 'Running Loss' in postfix_dict:
                del postfix_dict['Running Loss']
            postfix_dict['Avg Train Loss'] = f'{avg_epoch_train_loss:.2f}'

            if epoch % args.log_epoch == 0:
                history['train_loss'].append(avg_epoch_train_loss)
                history['epochs'].append(epoch)
                log_msg = f"Epoch: {epoch:4d} --- Avg Train Loss: {avg_epoch_train_loss:.2f}"

                if args.return_best:
                    current_valid_loss = evaluate(ctl, sys, loss_fn, valid_data)
                    history['valid_loss'].append(current_valid_loss)

                    is_best = current_valid_loss < best_valid_loss
                    if is_best:
                        best_valid_loss = current_valid_loss
                        best_params = copy.deepcopy(ctl.state_dict())

                    val_loss_str = f'{current_valid_loss:.2f}'
                    if is_best:
                        val_loss_str += ' (new best)'

                    postfix_dict['Validation Loss'] = val_loss_str
                    postfix_dict['Best Val Loss'] = f'{best_valid_loss:.2f}'

                    log_msg += f' | Validation Loss: {current_valid_loss:.2f}'
                    if is_best:
                        log_msg += ' (best)'

                logger.info(log_msg)

            epoch_iterator.set_description(f"Epoch {epoch}/{args.epochs}")
            epoch_iterator.set_postfix(postfix_dict, refresh=True)

        epoch_iterator.close()
        logger.info(f"Total training time: {(time.time() - t_start_training) / 60:.1f} minutes")

        if args.return_best and best_params:
            logger.info(f"Loading best model with validation loss: {best_valid_loss:.2f}")
            ctl.load_state_dict(best_params)

        return ctl, history

def plot_results(save_folder, history):
    """Plots and saves the training and validation loss curves."""
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


def run_final_visualizations(ctl, sys, loss_fn, args, train_data):
    """Generates and displays all final plots."""
    print("\n[INFO] Generating final performance visualizations...")
    center = torch.tensor([1, 0.5])

    plot_value_landscape(
        loss_fn=loss_fn, ctl=ctl, sys=sys, center=center, radius=1.0, resolution=140, horizon=400,
        bounds=(-2, 3), batch_size=12000,
        overlay_trajectories_from=[[1.2, 1.7], [1.5, 1.5], [0.76, 1.7]]
    )
    plot_radius_sweep(ctl, sys, torch.tensor([2, 1]), center, [0.2, 0.6, 1], args.horizon)
    _, _, symmetrical_points = generate_four_way_symmetrical_points(center=center)
    plot_facet_grid(ctl, sys, symmetrical_points[0:2], [0.2, 0.6, 1], center, args.horizon)
    plot_loss_landscape(
        loss_fn=loss_fn, ctl=ctl, sys=sys, start_point=torch.tensor([2.0, 1]), center=center,
        radius=1.0, horizon=400, vmax_percentile=92.7
    )


def main():
        """Main function to run the experiment."""
        parser = ArgumentParser(description="Robot Training Experiment")
        # Training args
        parser.add_argument('--epochs', type=int, default=100)
        parser.add_argument('--batch_size', type=int, default=60)
        parser.add_argument('--lr', type=float, default=1e-3)
        parser.add_argument('--num_rollouts', type=int, default=500)
        parser.add_argument('--log_epoch', type=int, default=None) # Changed default to None
        parser.add_argument('--return_best', action='store_true', default=True)
        parser.add_argument('--seed', type=int, default=2)
        # Model args
        parser.add_argument('--nn_type', type=str, default="MI")
        parser.add_argument('--non_linearity', type=str, default="LMLP")
        parser.add_argument('--dim_internal', type=int, default=64)
        parser.add_argument('--dim_nl', type=int, default=64)
        parser.add_argument('--cont_init_std', type=float, default=0.01)
        # System & Loss args
        parser.add_argument('--horizon', type=int, default=180)
        parser.add_argument('--std_init_plant', type=float, default=0.1)
        parser.add_argument('--linearize_plant', action='store_true', default=False)
        parser.add_argument('--spring_const', type=float, default=0.1)
        parser.add_argument('--alpha_u', type=float, default=50.0)
        args = parser.parse_args()

        # Dynamically set log_epoch if not provided
        if args.log_epoch is None:
            args.log_epoch = args.epochs // 10 if args.epochs // 10 > 0 else 1

        # SSM Config
        ssm_cfg = {"d_model": 10, "d_state": 14, "n_layers": 1, "ff": "LMLP", "max_phase": math.pi / 50,
                   "rmin": 0.7, "rmax": 0.98, "gamma": False, "trainable": True, "gain": 2.4}
        args.config = SSMConfig(**ssm_cfg)

        logger, save_folder = setup_experiment(args)
        train_data, test_data, train_dataloader, xbar = load_data(args)
        sys, ctl, loss_fn, optimizer = build_models_and_optimizer(args, xbar)
        logger.info(f"[INFO] Parameters: {sum(p.numel() for p in ctl.parameters() if p.requires_grad)}")

        ctl, history = train(args, ctl, sys, loss_fn, optimizer, train_dataloader, train_data, logger)

        plot_results(save_folder, history)
        final_train_loss = evaluate(ctl, sys, loss_fn, train_data)
        logger.info(f'Final Train Loss: {final_train_loss:.4f}')
        final_test_loss = evaluate(ctl, sys, loss_fn, test_data)
        logger.info(f"Final Test Loss: {final_test_loss:.4f}")

        run_final_visualizations(ctl, sys, loss_fn, args, train_data)


if __name__ == "__main__":
    main()
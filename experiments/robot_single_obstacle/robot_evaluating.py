import torch
import os
import math
from argparse import ArgumentParser

# Local imports from your project structure
from experiments.robot_single_obstacle.robots_sys import RobotsSystem
from experiments.robot_single_obstacle.datasets import RobotsDatasetMultiCircle_v2, generate_four_way_symmetrical_points
from plot_functions import plot_radius_sweep, plot_facet_grid, plot_loss_landscape, plot_value_landscape
from controllers.PB_controller import PerfBoostController
from loss_functions import RobotsLoss_v2
from controllers.m_operators.ssm import SSMConfig

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def build_models(args, xbar):
    """Builds the system and controller models."""
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
    return sys, ctl, loss_fn


def run_final_visualizations(ctl, sys, loss_fn, args):
    """Generates and displays all final plots for the loaded model."""
    print("\n[INFO] Generating final performance visualizations for the loaded model...")
    center = torch.tensor([1, 0.5])

    # Run the series of plots
    plot_value_landscape(
        loss_fn=loss_fn, ctl=ctl, sys=sys, center=center, radius=0.7, resolution=140, horizon=args.horizon,
        bounds=(-1.2, 2.7), batch_size=12000,
        overlay_trajectories_from=[[1.2, 1.7], [1.5, 1.5], [0.76, 1.7]]
    )
    plot_radius_sweep(ctl, sys, torch.tensor([2, 1]), center, [0.2, 0.6, 1], args.horizon)
    _, _, symmetrical_points = generate_four_way_symmetrical_points(center=center)
    plot_facet_grid(ctl, sys, symmetrical_points[0:2], [0.2, 0.6, 1], center, args.horizon)
    plot_loss_landscape(
        loss_fn=loss_fn, ctl=ctl, sys=sys, start_point=torch.tensor([2.0, 1]), center=center,
        radius=1.0, horizon=args.horizon, vmax_percentile=92.7
    )
    print("\n[INFO] Visualization complete.")


def main():
    """Main function to load a model and run visualizations."""
    parser = ArgumentParser(description="Robot Model Evaluation and Visualization")

    # Determine the default path for the weights file, assuming it's in a 'trained_models'
    # folder in the same directory as the script.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_weights_path = os.path.join(script_dir, 'trained_models', 'MI2_weights.pth')

    # Add an argument for the weights file path with the new default
    parser.add_argument('--weights_path', type=str, default=default_weights_path,
                        help="Path to the saved model weights (.pth file)")
    # Model args (must match the architecture of the saved model)
    parser.add_argument('--nn_type', type=str, default="MI")
    parser.add_argument('--non_linearity', type=str, default="LMLP")
    parser.add_argument('--dim_internal', type=int, default=64)
    parser.add_argument('--dim_nl', type=int, default=64)
    parser.add_argument('--cont_init_std', type=float, default=0.01)
    # System & Loss args
    parser.add_argument('--horizon', type=int, default=600)
    parser.add_argument('--std_init_plant', type=float, default=0.1)
    parser.add_argument('--linearize_plant', action='store_true', default=False)
    parser.add_argument('--spring_const', type=float, default=0.1)
    parser.add_argument('--alpha_u', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=2)
    args = parser.parse_args()

    # SSM Config (must match the trained model's config)
    ssm_cfg = {"d_model": 10, "d_state": 14, "n_layers": 1, "ff": "LMLP", "max_phase": math.pi / 50,
               "rmin": 0.7, "rmax": 0.98, "gamma": False, "trainable": True, "gain": 2.4}
    args.config = SSMConfig(**ssm_cfg)

    torch.manual_seed(args.seed)

    # 1. Load dataset to get system parameters like xbar
    dataset = RobotsDatasetMultiCircle_v2(random_seed=args.seed, horizon=args.horizon, std_ini=args.std_init_plant)
    xbar = dataset.xbar

    # 2. Build the model structure
    print("[INFO] Building model structure...")
    sys, ctl, loss_fn = build_models(args, xbar)
    print(f"[INFO] Total Parameters: {sum(p.numel() for p in ctl.parameters())}")

    # 3. Load the saved weights
    if not os.path.exists(args.weights_path):
        print(f"[ERROR] Weights file not found at: {args.weights_path}")
        return

    print(f"[INFO] Loading weights from: {args.weights_path}")
    ctl.load_state_dict(torch.load(args.weights_path))
    ctl.eval()  # Set the model to evaluation mode

    # 4. Run visualizations
    run_final_visualizations(ctl, sys, loss_fn, args)
if __name__ == "__main__":
    main()
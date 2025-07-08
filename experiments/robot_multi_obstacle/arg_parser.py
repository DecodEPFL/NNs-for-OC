import argparse
import math


# argument parser
def argument_parser():
    parser = argparse.ArgumentParser(description="Robots multi-obstacle experiment.")

    # experiment
    parser.add_argument('--random-seed', type=int, default=5, help='Random seed. Default is 5.')
    parser.add_argument('--num-obstacles', type=int, default=3, help='Number of obstacles. Default is 3.')

    # dataset
    parser.add_argument('--horizon', type=int, default=180, help='Time horizon for the computation. Default is 180.')
    parser.add_argument('--num-rollouts', type=int, default=1200,
                        help='Number of rollouts in the training data. Default is 1200.')
    parser.add_argument('--std-init-plant', type=float, default=0.2,
                        help='std of the plant initial conditions. Default is 0.2.')

    # plant
    parser.add_argument('--spring-const', type=float, default=1.0, help='Spring constant. Default is 1.0.')
    parser.add_argument('--linearize-plant', type=bool, default=False, help='Linearize plant or not. Default is False.')

    # controller
    parser.add_argument('--nn-type', type=str, default='MI',
                        help='Type of the NN for operator Emme in controller. Options: REN, SSM, MI. Default is MI')
    parser.add_argument('--non-linearity', type=str, default='LMLP',
                        help='Type of non_linearity in SSMs. Options: MLP, coupling_layers, hamiltonian, tanh, LMLP. '
                             'Default LMLP.')
    parser.add_argument('--cont-init-std', type=float, default=0.1,
                        help='Initialization std for controller params. Default is 0.1.')
    parser.add_argument('--dim-internal', type=int, default=8,
                        help='Dimension of the internal state of the controller. '
                             'Adjusts the size of the linear part of REN. Default is 8.')
    parser.add_argument('--dim-nl', type=int, default=8, help='size of the non-linear part of REN. Default is 8.')

    # loss
    parser.add_argument('--alpha-u', type=float, default=0.2,
                        help='Weight of the loss due to control input "u". Default is 0.2.')

    # optimizer
    parser.add_argument('--batch-size', type=int, default=80,
                        help='Number of forward trajectories of the closed-loop system at each step. Default is 80.')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Total number of epochs for training. Default is 10.')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate. Default is 1e-3.')
    parser.add_argument('--log-epoch', type=int, default=1,
                        help='Frequency of logging in epochs. Default is 1.')
    parser.add_argument('--return-best', type=bool, default=True,
                        help='Return the best model on the validation data among all logged iterations. '
                             'The train data can be used instead of validation data. The Default is True.')

    args = parser.parse_args()
    return args

def print_args(args):
    msg = '\n[INFO] Dataset: num_obstacles: %i' % args.num_obstacles + ' -- num_rollouts: %i' % args.num_rollouts
    msg += ' -- std_ini: %.2f' % args.std_init_plant + ' -- time horizon: %i' % args.horizon

    msg += '\n[INFO] Plant: spring constant: %.2f' % args.spring_const
    msg += ' -- use linearized plant: ' + str(args.linearize_plant)

    msg += '\n[INFO] Controller using %s: non_linearity: %s' % (args.nn_type, args.non_linearity)

    msg += '\n[INFO] Loss:  alpha_u: %.6f' % args.alpha_u

    msg += '\n[INFO] Optimizer: lr: %.2e' % args.lr
    msg += ' -- epochs: %i,' % args.epochs
    msg += ' -- batch_size: %i,' % args.batch_size
    msg += ' -- return best model for validation data among logged epochs: ' + str(args.return_best)

    return msg


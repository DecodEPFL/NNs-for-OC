import torch
from torch.utils.data import DataLoader

from arg_parser import argument_parser, print_args
from plants.tanks import TankSystem, TankDataset
from controllers.PB_controller import PerfBoostController
from loss_functions import TankLoss


# ----- Overwriting arguments -----
args = argument_parser()
args.horizon = 200
args.epochs = 30  # 5000
args.lr = 5e-2  # SSM
# args.lr = 5e-3  # REN
args.num_rollouts = 8
args.log_epoch = 1
args.nn_type = "SSM"
args.dim_internal = 2
args.dim_nl = 2
args.non_linearity = "coupling_layers"
args.batch_size = 2
args.Q = 1e3
args.R = 2.5e3

args.std_init_plant = 0.005

args.xbar = 0.2

dataset = TankDataset(random_seed=args.random_seed, horizon=args.horizon, std_ini=args.std_init_plant)
# divide to train and test
train_data, test_data = dataset.get_data(num_train_samples=args.num_rollouts, num_test_samples=500)
# batch the data
train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

# ------------ 2. Plant (with prestabilizing controller) ------------
xbar = torch.Tensor([args.xbar])
plant_input_init = None     # all zero
plant_state_init = torch.Tensor([args.std_init_plant])
sys = TankSystem(xbar=xbar,
                 x_init=plant_state_init,
                 u_init=plant_input_init,
                 )

# ------------ 3. Controller ------------
ctl = PerfBoostController(noiseless_forward=sys.noiseless_forward,
                          input_init=sys.x_init,
                          output_init=sys.u_init,
                          nn_type=args.nn_type,
                          non_linearity=args.non_linearity,
                          dim_internal=args.dim_internal,
                          dim_nl=args.dim_nl,
                          initialization_std=args.cont_init_std,
                          )

# ------------ 4. Loss ------------
loss_fn = TankLoss(Q=args.Q, R=args.R, xbar=xbar, u_bar=sys.a * torch.sqrt(sys.xbar))

# ------------ 5. Optimizer ------------
valid_data = train_data      # use the entire train data for validation
assert not (valid_data is None and args.return_best)
optimizer = torch.optim.Adam(ctl.parameters(), lr=args.lr)

# ------------ 6. Training ------------
print('------------ Begin training ------------')
# Add your code here
print("Add your training code here...")

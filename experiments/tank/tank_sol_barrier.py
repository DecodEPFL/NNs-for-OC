import torch
import time
import copy
from torch.utils.data import DataLoader

from arg_parser import argument_parser
from plants.tanks import TankSystem, TankDataset
from controllers.PB_controller import PerfBoostController
from plot_functions import plot_traj_vs_time
from loss_functions import TankLoss

# ----- Overwriting arguments -----
args = argument_parser()
args.horizon = 200
args.epochs = 50  # 5000
args.lr = 5e-2  # SSM
# args.lr = 5e-3  # REN
args.num_rollouts = 5
args.log_epoch = 4  # args.epochs//10 if args.epochs//10 > 0 else 1
args.nn_type = "SSM"
args.dim_internal = 2
args.dim_nl = 2
args.non_linearity = "coupling_layers"
args.batch_size = 2
args.Q = 1e3
args.R = 2.5e3
args.alpha_smooth = 1e3
args.alpha_u_pos = 1e4*0
args.alpha_barrier = 1e4
t_ext = args.horizon
args.std_init_plant = 0.005

args.plant_state_init = 0.2
args.xbar = 0.01
args.xmax = 0.71

dataset = TankDataset(random_seed=args.random_seed, horizon=args.horizon, std_ini=args.std_init_plant)
# divide to train and test
train_data, test_data = dataset.get_data(num_train_samples=args.num_rollouts, num_test_samples=500)
# data for plots
plot_data = torch.zeros(1, t_ext, train_data.shape[-1])
# plot_data = train_data[0:1,:,:]
# batch the data
train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

# ------------ 2. Plant (with prestabilizing controller) ------------
xbar = torch.Tensor([args.xbar])
plant_state_init = torch.Tensor([args.plant_state_init])
sys = TankSystem(xbar=xbar,
                 x_init=plant_state_init,
                 )
u_bar = sys.a * torch.sqrt(sys.xbar)

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

# Count numer of parameters
total_n_params = sum(p.numel() for p in ctl.parameters() if p.requires_grad)
print("[INFO] Number of parameters: %i" % total_n_params)

# plot closed-loop trajectories before training the controller
x_log, _, u_log = sys.rollout(ctl, plot_data)
plot_traj_vs_time(args.horizon, x_log[0, :args.horizon, :], u_log[0, :args.horizon, :])

# ------------ 4. Loss ------------
loss_fn = TankLoss(Q=args.Q, R=args.R, alpha_smooth=args.alpha_smooth, alpha_u_pos=args.alpha_u_pos,
                   alpha_barrier=args.alpha_barrier, xbar=xbar, u_bar=u_bar, x_max=args.xmax, sys=sys)
loss = loss_fn.forward(x_log[0, :args.horizon, :].unsqueeze(0), u_log[0, :args.horizon, :].unsqueeze(0))
print("Loss before training: %.4f" % loss)

# ------------ 5. Optimizer ------------
valid_data = train_data      # use the entire train data for validation
assert not (valid_data is None and args.return_best)
optimizer = torch.optim.Adam(ctl.parameters(), lr=args.lr)

# ------------ 6. Training ------------
print('------------ Begin training ------------')
best_valid_loss = 1e6
best_params = ctl.state_dict()
loss = 1e6
t = time.time()
for epoch in range(1+args.epochs):
    # iterate over all data batches
    for train_data_batch in train_dataloader:
        optimizer.zero_grad()
        # simulate over horizon steps
        x_log, _, u_log = sys.rollout(
            controller=ctl, data=train_data_batch, train=True,
        )
        # loss of this rollout
        loss = loss_fn.forward(x_log, u_log)
        # take a step
        loss.backward()
        optimizer.step()

    # print info
    if epoch % args.log_epoch == 0:
        msg = 'Epoch: %i --- train loss: %.4f' % (epoch, loss)

        if args.return_best:
            # rollout the current controller on the valid data
            with torch.no_grad():
                x_log_valid, _, u_log_valid = sys.rollout(
                    controller=ctl, data=valid_data, train=False,
                )
                # loss of the valid data
                loss_valid = loss_fn.forward(x_log_valid, u_log_valid)
            msg += ' ---||--- validation loss: %.4f' % (loss_valid.item())
            # compare with the best valid loss
            if loss_valid.item() < best_valid_loss:
                best_valid_loss = loss_valid.item()
                best_params = copy.deepcopy(ctl.state_dict())
                msg += ' (best so far)'
        duration = time.time() - t
        msg += ' ---||--- time: %.0f s' % duration
        print(msg)
        # plot trajectory
        x_log, x_nonfilter_log, u_log = sys.rollout(ctl, plot_data)
        plot_traj_vs_time(t_ext, x_log[0, :, :], u_log[0, :, :], x_bar=xbar, x_nonfilter_log=x_nonfilter_log[0,:,:])
        t = time.time()

# set to best seen during training
if args.return_best:
    ctl.load_state_dict(best_params)
    # ctl.set_parameters_as_vector(best_params)

# Plot trajectories after training
x_log, _, u_log = sys.rollout(ctl, plot_data)
plot_traj_vs_time(t_ext, x_log[0, :, :], u_log[0, :, :],  x_bar=xbar)
# plot_traj_vs_time(t_ext, x_log[0, :, :], u_log[0, :, :], filename='tank_barrier', save=True,  x_bar=xbar)

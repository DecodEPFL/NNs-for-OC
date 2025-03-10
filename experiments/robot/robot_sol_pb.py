import torch
import time
import copy
from torch.utils.data import DataLoader

from experiments.robot.arg_parser import argument_parser, print_args
from plants.robots import RobotsSystem, RobotsDataset
from plot_functions import plot_trajectories, plot_traj_vs_time
from controllers.PB_controller import PerfBoostController
from loss_functions import RobotsLoss


# ----- Overwriting arguments -----
args = argument_parser()
args.epochs = 200
# args.lr = 1e-3
args.num_rollouts = 8
args.log_epoch = 4
args.nn_type = "SSM"
args.dim_internal = 4
args.dim_nl = 4
args.non_linearity = "coupling_layers"
args.batch_size = 2

# ------------ 1. Dataset ------------
dataset = RobotsDataset(random_seed=args.random_seed, horizon=args.horizon, std_ini=args.std_init_plant)
# divide to train and test
train_data, test_data = dataset.get_data(num_train_samples=args.num_rollouts, num_test_samples=500)
# data for plots
t_ext = args.horizon
plot_data = torch.zeros(1, t_ext, train_data.shape[-1])
plot_data[:, 0, :] = (dataset.x0.detach() - dataset.xbar)
# batch the data
train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

# ------------ 2. Plant ------------
plant_input_init = None     # all zero
plant_state_init = None     # same as xbar
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
                          initialization_std=args.cont_init_std,
                          )
# plot closed-loop trajectories before training the controller
x_log, _, u_log = sys.rollout(ctl, plot_data)
plot_trajectories(x_log[0, :, :], T=t_ext)
plot_traj_vs_time(t_ext, x_log[0, :, :], u_log[0, :, :])

# ------------ 4. Loss ------------
Q = torch.eye(4) *100
loss_fn = RobotsLoss(
    Q=Q, alpha_u=args.alpha_u,
    alpha_obst=args.alpha_obst,
)

# ------------ 5. Optimizer ------------
valid_data = train_data      # use the entire train data for validation
assert not (valid_data is None and args.return_best)
optimizer = torch.optim.Adam(ctl.parameters(), lr=args.lr)

# ------------ 6. Training ------------
print('------------ Begin training ------------')
best_valid_loss = 1e6
best_params = ctl.state_dict()  # ctl.get_parameters_as_vector()
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
        msg = 'Epoch: %i --- train loss: %.2f' % (epoch, loss)

        if args.return_best:
            # rollout the current controller on the valid data
            with torch.no_grad():
                x_log_valid, _, u_log_valid = sys.rollout(
                    controller=ctl, data=valid_data, train=False,
                )
                # loss of the valid data
                loss_valid = loss_fn.forward(x_log_valid, u_log_valid)
            msg += ' ---||--- validation loss: %.2f' % (loss_valid.item())
            # compare with the best valid loss
            if loss_valid.item() < best_valid_loss:
                best_valid_loss = loss_valid.item()
                best_params = copy.deepcopy(ctl.state_dict())
                # ctl.get_parameters_as_vector()  # record state dict if best on valid
                msg += ' (best so far)'
        duration = time.time() - t
        msg += ' ---||--- time: %.0f s' % duration
        print(msg)
        # plot trajectory
        plot_trajectories(x_log_valid[0, :, :], T=t_ext, radius_robot=loss_fn.radius_robot, circles=True,
                          obstacle_centers=loss_fn.obstacle_centers, obstacle_radius=loss_fn.obstacle_radius)
        t = time.time()

# set to best seen during training
if args.return_best:
    ctl.load_state_dict(best_params)
    # ctl.set_parameters_as_vector(best_params)

# evaluate on the train data
print('[INFO] evaluating the trained controller on %i training rollouts.' % train_data.shape[0])
with torch.no_grad():
    x_log, _, u_log = sys.rollout(
        controller=ctl, data=train_data, train=False,
    )   # use the entire train data, not a batch
    # evaluate losses
    loss = loss_fn.forward(x_log, u_log)
    print('Train loss: %.4f' % loss)

# evaluate on the test data
print('[INFO] evaluating the trained controller on %i test rollouts.' % test_data.shape[0])
with torch.no_grad():
    # simulate over horizon steps
    x_log, _, u_log = sys.rollout(
        controller=ctl, data=test_data, train=False,
    )
    # loss
    test_loss = loss_fn.forward(x_log, u_log).item()
    print("Test loss: %.4f" % test_loss)


# plot closed-loop trajectories using the trained controller
print('Plotting closed-loop trajectories using the trained controller...')
x_log, _, u_log = sys.rollout(ctl, plot_data)
plot_trajectories(
    x_log[0, :, :], T=t_ext, radius_robot=loss_fn.radius_robot, circles=True,
    obstacle_centers=loss_fn.obstacle_centers,
    obstacle_radius=loss_fn.obstacle_radius,
#     save=True, filename="pb_robot"
)
plot_traj_vs_time(t_ext, x_log[0, :, :], u_log[0, :, :])

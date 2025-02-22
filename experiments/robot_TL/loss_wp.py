import torch
import torch.nn.functional as F


def _loss_TL_always_implies_next_always_not(p, one_value=True):
    t_end = p.shape[0]
    minus_p_next = -p[1:]
    mask = torch.triu(torch.ones([t_end-1, t_end-1]))
    aux_negativity = torch.max(minus_p_next).detach()
    minus_p_next = minus_p_next - aux_negativity
    min_on_minus_p_next = torch.min(minus_p_next.unsqueeze(0).repeat(t_end-1, 1) * mask, dim=1).values
    min_on_minus_p_next = min_on_minus_p_next + aux_negativity
    for_min_on_interval = torch.maximum(-p, torch.cat((min_on_minus_p_next, -p[-1:])))
    if one_value:
        out = torch.min(for_min_on_interval, dim=0).values
        return out
    return


def _f_tl_goal(x, u, sys):
    t_end = x.shape[0]
    pos = x.reshape(-1, 4)[:, 0:2].reshape(-1, 2)
    posbar = sys.xbar.reshape(-1, 4)[:, 0:2].reshape(2)
    target_distance = torch.norm(pos - posbar.unsqueeze(0).repeat(t_end, 1), dim=1)
    mask = torch.triu(torch.ones([t_end, t_end]))
    min_on_goal = 0.05 + torch.min((- target_distance).repeat(t_end, 1) * mask, dim=1).values
    maxmin_on_goal = torch.max(min_on_goal, dim=0).values
    return maxmin_on_goal.unsqueeze(0)


def _f_tl_input(u, sys, u_max):
    t_end = u.shape[0]
    min_on_u = 0.1*torch.min(-torch.relu(u - u_max))
    return min_on_u.unsqueeze(0)


def _f_tl_obstacle(x, u, sys, obstacle_pos, obstacle_radius):
    n_obstacles = obstacle_pos.shape[0]
    t_end = x.shape[0]
    stack_for_min = torch.tensor([])
    pos = x.reshape(-1, 4)[:, 0:2].reshape(-1, 2)
    for i in range(n_obstacles):
        radius = obstacle_radius[i,0]
        o = obstacle_pos[i,:]
        o_distance = torch.norm((pos - o.unsqueeze(0).repeat(t_end, 1)).reshape(-1, 2), dim=1)
        min_on_obst = torch.min(o_distance - radius, dim=0).values.unsqueeze(0)
        stack_for_min = torch.cat([stack_for_min, min_on_obst])
    min_on_obst = torch.min(stack_for_min, dim=0).values
    return min_on_obst.unsqueeze(0)


def _f_loss_states(x, sys, Q=None):
    t_end = x.shape[0]
    if Q is None:
        Q = torch.eye(sys.n)
    dx = x
    xQx = (F.linear(dx, Q) * dx).sum(dim=1)
    return xQx.sum().unsqueeze(0)

def _f_loss_u(u, sys, R=None):
    if R is None:
        R = torch.eye(sys.m)
    uRu = (F.linear(u, R) * u).sum(dim=1)
    return uRu.sum().unsqueeze(0)


def loss_TL_waypoints(x):
    t_end = x.shape[0]
    dist_error = 0.05
    stack_for_min = torch.tensor([])
    center_red = torch.tensor([0, 0.])
    dist_x1_red = torch.norm(x[:,0:2] - center_red.unsqueeze(0).repeat(t_end, 1), dim=1)
    psi_goal_r1 = dist_error - dist_x1_red

    # Loss AINAN robot
    loss_ainan_r1 = _loss_TL_always_implies_next_always_not(psi_goal_r1).unsqueeze(0)
    stack_for_min = torch.cat([stack_for_min, loss_ainan_r1])
    print("\t Robot ---||--- Loss_r: %.4f" % loss_ainan_r1)

    return -torch.min(stack_for_min)


def f_loss_tl(x, u, sys, dict_tl):
    text_to_print = ""
    stack_for_min = torch.tensor([])
    if dict_tl["goal"]:
        maxmin_on_goal = _f_tl_goal(x, u, sys)
        text_to_print = text_to_print + ("\t Goal %.4f ---" % maxmin_on_goal)
        stack_for_min = torch.cat([stack_for_min, maxmin_on_goal])
    if dict_tl["input"]:
        min_on_u = _f_tl_input(u, sys, dict_tl["max_u"])
        text_to_print = text_to_print + ("\t Input %.4f ---" % min_on_u)
        stack_for_min = torch.cat([stack_for_min, min_on_u])
    if dict_tl["obstacle"]:
        min_on_obst = _f_tl_obstacle(x, u, sys, dict_tl["obstacle_pos"],
                                     1.2*(dict_tl["robot_radius"]+dict_tl["obstacle_radius"]))
        text_to_print = text_to_print + ("\t Obstacle %.4f ---" % min_on_obst)
        stack_for_min = torch.cat([stack_for_min, min_on_obst])
    if not stack_for_min.shape[0] == 0:
        print(text_to_print)
    else:
        stack_for_min = torch.zeros(1)
    return -torch.min(stack_for_min)


def f_loss_sum(xx, uu, sys, dict_sum_loss):
    text_to_print = ""
    stack_for_sum = torch.tensor([])
    alphas = torch.tensor([])
    t_end = xx.shape[0]
    if dict_sum_loss["state"]:
        loss_x = _f_loss_states(xx, sys, dict_sum_loss["Q"])
        stack_for_sum = torch.cat([stack_for_sum, loss_x])
        alphas = torch.cat([alphas, dict_sum_loss["alpha_x"]])
        text_to_print = text_to_print + ("\t Loss_x %.4f ---" % (alphas[-1]*loss_x))
    if dict_sum_loss["input"]:
        loss_u = _f_loss_u(uu, sys)
        stack_for_sum = torch.cat([stack_for_sum, loss_u])
        alphas = torch.cat([alphas, dict_sum_loss["alpha_u"]])
        text_to_print = text_to_print + ("\t Loss_u %.4f ---" % (alphas[-1] * loss_u))
    if not stack_for_sum.shape[0] == 0:
        print(text_to_print)
    else:
        stack_for_sum = torch.zeros(1)
        alphas = torch.zeros(1)
    return (alphas * stack_for_sum).sum()

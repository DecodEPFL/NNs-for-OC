import torch
import numpy as np
import matplotlib.pyplot as plt


plt.rcParams['text.usetex'] = True

def plot_trajectories(
    x, save=True, filename='', T=100, obst=False,
    dots=False, circles=False, radius_robot=1, f=5,
    obstacle_centers=None, obstacle_radius=None
):
    # fig = plt.figure(f)
    fig, ax = plt.subplots(figsize=(f,f))
    # plot obstacles

    colors = ['tab:blue', 'tab:orange']
    ax.plot(
        x[:T+1,0].detach(), x[:T+1,1].detach(),
        color=colors[0], linewidth=1
    )
    ax.plot(
        x[T:,0].detach(), x[T:,1].detach(),
        color='k', linewidth=0.1, linestyle='dotted', dashes=(3, 15)
        )
    ax.plot(
        x[0,0].detach(), x[0,1].detach(),
        color=colors[0], marker='8'
    )
    if dots:
        for j in range(T):
            ax.plot(
                x[j, 0].detach(), x[j, 1].detach(),
                color=colors[0], marker='o'
            )
    if circles:
        r = radius_robot
        circle = plt.Circle((x[T-1, 0].detach(), x[T-1, 1].detach()),
                            r, color=colors[0], alpha=0.5, zorder=10
                           )
        ax.add_patch(circle)
    if obstacle_centers is not None:
        r = obstacle_radius[0,0]
        circle = plt.Circle((obstacle_centers[0,0],obstacle_centers[0,1]),
                            r, color='k', alpha=0.1, zorder=10
                            )
        ax.add_patch(circle)
    if save:
        plt.savefig(filename + '.pdf', format='pdf')
    plt.show()


def plot_traj_vs_time(t_end, x, u=None, save=True, filename=''):
    t = torch.linspace(0,t_end-1, t_end)
    if u is not None:
        p = 3
    else:
        p = 2
    plt.figure(figsize=(4*p, 4))
    plt.subplot(1, p, 1)
    plt.plot(t, x[:,0].detach())
    plt.plot(t, x[:,1].detach())
    plt.xlabel(r'$t$')
    plt.title(r'$p(t)$ - position')
    plt.subplot(1, p, 2)
    plt.plot(t, x[:,2].detach())
    plt.plot(t, x[:,3].detach())
    plt.xlabel(r'$t$')
    plt.title(r'$q(t)$ - velocity')
    if p == 3:
        plt.subplot(1, 3, 3)
        plt.plot(t, u[:, 0].detach())
        plt.plot(t, u[:, 1].detach())
        plt.xlabel(r'$t$')
        plt.title(r'$u(t)$')
    if save:
        plt.savefig(filename+'.pdf', format='pdf')
    else:
        plt.show()

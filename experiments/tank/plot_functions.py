import torch
import numpy as np
import matplotlib.pyplot as plt


# plt.rcParams['text.usetex'] = True

def plot_traj_vs_time(t_end, x, u=None, save=False, filename='', u_bar=None, x_bar=None, x_nonfilter_log=None):
    if x.dim() == 2:
        x = x.unsqueeze(0)
    if u is not None and u.dim() == 2:
        u = u.unsqueeze(0)
    assert x.dim() == 3
    if u is not None:
        assert u.dim() == 3
        p = 2
    else:
        p = 1
    if u_bar is not None:
        u_bar = float(u_bar)
    t = torch.linspace(0,t_end-1, t_end)
    plt.figure(figsize=(4*p, 4))
    plt.subplot(1, p, 1)
    for i in range(x.shape[0]):
        plt.plot(t, x[i,:,0].detach())
    if x_bar is not None:
        plt.plot(t, (x_bar * torch.ones_like(t)), 'k--', linewidth=0.1)
        plt.plot(t, torch.zeros_like(t), 'k--', linewidth=0.1)
    if x_nonfilter_log is not None:
        plt.plot(t, x_nonfilter_log[:, 0].detach())
    plt.xlabel(r'$t$')
    plt.title(r'$x(t)$')
    if p == 2:
        plt.subplot(1, 2, 2)
        for i in range(u.shape[0]):
            plt.plot(t, u[i, :, 0].detach())
        if u_bar is not None:
            plt.plot(t, (-u_bar * torch.ones_like(t)), 'k--', linewidth=0.1)
        plt.xlabel(r'$t$')
        plt.title(r'$u(t)$')
    if save:
        plt.savefig(''+filename+'.pdf', format='pdf')
    else:
        plt.show()
        plt.close()

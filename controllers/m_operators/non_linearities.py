import torch
import torch.nn as nn
import torch.nn.functional as F


class HamiltonianSIE(nn.Module):
    # Hamiltonian neural network
    # General ODE: \dot{y} = J(y,t) K(t) \tanh( K^T(t) y(t) + b(t) )
    # Constraints:
    #   J(y,t) = J_1 = [ 0 I ; -I 0 ]
    # Discretization method: Semi-Implicit Euler
    def __init__(self, n_layers, nf=4, t_end=0.5, random=True, bias=True):
        super().__init__()

        self.n_layers = n_layers
        self.h = t_end / self.n_layers
        self.act = nn.Tanh()
        if not nf % 2 == 0:
            raise ValueError('Number of features need to be and even number -- Currently it is %i' % nf)
        self.nf = nf
        self.half_state_dim = nf//2
        if random:
            k1 = 0.1 * torch.randn(self.half_state_dim, self.half_state_dim).repeat(self.n_layers, 1, 1)
            k2 = 0.1 * torch.randn(self.half_state_dim, self.half_state_dim).repeat(self.n_layers, 1, 1)
            b1 = 0.1 * torch.randn(1, self.half_state_dim).repeat(self.n_layers, 1, 1)
            b2 = 0.1 * torch.randn(1, self.half_state_dim).repeat(self.n_layers, 1, 1)
        else:
            k1 = torch.eye(self.half_state_dim).repeat(self.n_layers, 1, 1)
            k2 = torch.eye(self.half_state_dim).repeat(self.n_layers, 1, 1)
            b1 = torch.zeros(self.n_layers, 1, self.half_state_dim)
            b2 = torch.zeros(self.n_layers, 1, self.half_state_dim)

        self.k1 = nn.Parameter(k1)
        self.k2 = nn.Parameter(k2)
        if bias:
            self.b1 = nn.Parameter(b1)
            self.b2 = nn.Parameter(b2)
        else:
            self.b1 = torch.zeros(self.n_layers, 1, self.half_state_dim)
            self.b2 = torch.zeros(self.n_layers, 1, self.half_state_dim)

    def forward(self, x0, ini=0, end=None):
        # the size of x0 is (sampleNumber, 1, nf)
        if end is None:
            end = self.n_layers
        # x = x0.clone()
        p, q = torch.split(x0.clone(), [self.half_state_dim, self.half_state_dim], dim=2)
        for j in range(ini, end):
            p = p - self.h * F.linear(self.act(F.linear(q, self.k2[j].transpose(0,1)) + self.b1[j]), self.k2[j])
            q = q + self.h * F.linear(self.act(F.linear(p, self.k1[j].transpose(0,1)) + self.b2[j]), self.k1[j])
        x = torch.cat([p, q], dim=2)
        return x


class FCNN(nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden, act=nn.Tanh):
        super(FCNN, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(dim_in, dim_hidden, bias=False), act(),
            # nn.Linear(hidden_dim, hidden_dim), act(),
            nn.Linear(dim_hidden, dim_out, bias=False)
        )

    def forward(self, x):
        return self.network(x)


class CouplingLayer(nn.Module):
    """  An implementation of a coupling layer from RealNVP (https://arxiv.org/abs/1605.08803). """
    def __init__(self, dim_inputs, dim_hidden):
        super(CouplingLayer, self).__init__()

        self.dim_inputs = dim_inputs
        self.mask = torch.arange(0, dim_inputs) % 2  # alternating inputs

        self.scale_net = FCNN(dim_in=dim_inputs, dim_out=dim_inputs, dim_hidden=dim_hidden)
        self.translate_net = FCNN(dim_in=dim_inputs, dim_out=dim_inputs, dim_hidden=dim_hidden)

        nn.init.normal_(self.translate_net.network[0].weight.data, std=0.1)
        nn.init.normal_(self.translate_net.network[2].weight.data, std=0.1)

        nn.init.normal_(self.scale_net.network[0].weight.data, std=0.1)
        nn.init.normal_(self.scale_net.network[2].weight.data, std=0.1)

    def forward(self, inputs, mode='direct'):
        mask = self.mask
        masked_inputs = inputs * mask

        log_s = self.scale_net(masked_inputs) * (1 - mask)
        t = self.translate_net(masked_inputs) * (1 - mask)

        if mode == 'direct':
            s = torch.exp(log_s)
            return inputs * s + t
        else:
            s = torch.exp(-log_s)
            return (inputs - t) * s

""" Lipschitz-bounded MLPs base layers from Manchester paper """

def cayley(W):
    if len(W.shape) == 2:
        return cayley(W[None])[0]
    _, cout, cin = W.shape
    if cin > cout:
        return cayley(W.transpose(1, 2)).transpose(1, 2)
    U, V = W[:, :cin], W[:, cin:]
    I = torch.eye(cin, dtype=W.dtype, device=W.device)[None, :, :]
    A = U - U.conj().transpose(1, 2) + V.conj().transpose(1, 2) @ V
    iIpA = torch.inverse(I + A)
    return torch.cat((iIpA @ (I - A), -2 * V @ iIpA), axis=1)

class FirstChannel(nn.Module):
    def __init__(self, cout, scale=1.0):
        super().__init__()
        self.cout = cout
        self.scale = scale

    def forward(self, x):
        xdim = len(x.shape)
        if xdim == 4:
            return self.scale * x[:, :self.cout, :, :]
        elif xdim == 2:
            return self.scale * x[:, :self.cout]
        elif xdim == 3:
            return self.scale * x[:, :, :]



class SandwichLin(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, scale=1.0, AB=False):
        super().__init__(in_features + out_features, out_features, bias)
        self.alpha = nn.Parameter(torch.ones(1, dtype=torch.float32, requires_grad=True))
        self.alpha.data = self.weight.norm()
        self.scale = scale
        self.AB = AB
        self.Q = None

    def forward(self, x):
        fout, _ = self.weight.shape
        if self.training or self.Q is None:
            self.Q = cayley(self.alpha * self.weight / self.weight.norm())
        Q = self.Q if self.training else self.Q.detach()
        x = F.linear(self.scale * x, Q[:, fout:])  # B @ x
        if self.AB:
            x = 2 * F.linear(x, Q[:, :fout].T)  # 2 A.T @ B @ x
        if self.bias is not None:
            x += self.bias
        return x


class SandwichFc(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, scale=1.0):
        super().__init__(in_features + out_features, out_features, bias)
        self.alpha = nn.Parameter(torch.ones(1, dtype=torch.float32, requires_grad=True))
        self.alpha.data = self.weight.norm()
        self.scale = scale
        self.psi = nn.Parameter(torch.zeros(out_features, dtype=torch.float32, requires_grad=True))
        self.Q = None

    def forward(self, x):
        fout, _ = self.weight.shape
        if self.training or self.Q is None:
            self.Q = cayley(self.alpha * self.weight / self.weight.norm())
        Q = self.Q if self.training else self.Q.detach()
        x = F.linear(self.scale * x, Q[:, fout:])  # B*h
        if self.psi is not None:
            x = x * torch.exp(-self.psi) * (2 ** 0.5)  # sqrt(2) \Psi^{-1} B * h
        if self.bias is not None:
            x += self.bias
        x = F.relu(x) * torch.exp(self.psi)  # \Psi z
        x = 2 ** 0.5 * F.linear(x, Q[:, :fout].T)  # sqrt(2) A^top \Psi z
        return x
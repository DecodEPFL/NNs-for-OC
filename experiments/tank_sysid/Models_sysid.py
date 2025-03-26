import math
import torch
import torch.nn as nn
from dataclasses import dataclass
from scan_utils import associative_scan, binary_operator_diag
import torch.nn.functional as F
from collections import OrderedDict


# Data class to set up the SSM

@dataclass
class DWNConfig:
    d_model: int = 10  # input/output size of the LRU (u and y)
    d_state: int = 64  # state size of the LRU (n)
    n_layers: int = 6  # number of SSMs blocks in cascade for deep structures
    dropout: float = 0.0  # set it different from 0 if you want to introduce dropout regularization
    bias: bool = False  # bias of MLP layers
    rmin: float = 0.0  # min. magnitude of the eigenvalues at initialization in the complex parametrization
    rmax: float = 1.0  # max. magnitude of the eigenvalues at initialization in the complex parametrization
    max_phase: float = 2 * math.pi  # maximum phase of the eigenvalues at initialization in the complex parametrization
    ff: str = "MLP"  # non-linear block used in the scaffolding


# LRU block -------------------------------------------

class LRU(nn.Module):
    """ Linear Recurrent Unit. The LRU is simulated using Parallel Scan (fast!) when
     "scan" is set to True (default) in the forward pass, otherwise recursively (slow)."""

    def __init__(
            self, in_features: int, out_features: int, state_features: int, rmin=0.9, rmax=1.0, max_phase=6.283
    ):
        super().__init__()
        self.out_features = out_features
        self.D = nn.Parameter(
            torch.randn([out_features, in_features]) / math.sqrt(in_features)
        )
        u1 = torch.rand(state_features)
        u2 = torch.rand(state_features)
        self.nu_log = nn.Parameter(
            torch.log(-0.5 * torch.log(u1 * (rmax + rmin) * (rmax - rmin) + rmin ** 2))
        )
        self.theta_log = nn.Parameter(torch.log(max_phase * u2))
        lambda_abs = torch.exp(-torch.exp(self.nu_log))
        self.gamma_log = nn.Parameter(
            torch.log(
                torch.sqrt(torch.ones_like(lambda_abs) - torch.square(lambda_abs))
            )
        )
        B_re = torch.randn([state_features, in_features]) / math.sqrt(2 * in_features)
        B_im = torch.randn([state_features, in_features]) / math.sqrt(2 * in_features)
        self.B = nn.Parameter(torch.complex(B_re, B_im))  # N, U
        C_re = torch.randn([out_features, state_features]) / math.sqrt(state_features)
        C_im = torch.randn([out_features, state_features]) / math.sqrt(state_features)
        self.C = nn.Parameter(torch.complex(C_re, C_im))  # H, N

        self.in_features = in_features
        self.out_features = out_features
        self.state_features = state_features

    def ss_params(self):
        lambda_abs = torch.exp(-torch.exp(self.nu_log))
        lambda_phase = torch.exp(self.theta_log)

        lambda_re = lambda_abs * torch.cos(lambda_phase)
        lambda_im = lambda_abs * torch.sin(lambda_phase)
        lambdas = torch.complex(lambda_re, lambda_im)
        gammas = torch.exp(self.gamma_log).unsqueeze(-1).to(self.B.device)
        B = gammas * self.B
        return lambdas, B, self.C, self.D

    def ss_real_matrices(self, to_numpy=True):

        lambdas, B, self.C, self.D = self.ss_params()

        lambdas_full = torch.zeros(2 * self.state_features, device=lambdas.device, dtype=lambdas.dtype)
        lambdas_full[::2] = lambdas
        lambdas_full[1::2] = lambdas.conj()

        # First convert to complex conjugate system....
        A_full = torch.diag(lambdas_full)
        B_full = torch.zeros((2 * self.state_features, self.in_features), device=lambdas.device, dtype=lambdas.dtype)
        B_full[::2] = B
        B_full[1::2] = B.conj()
        C_full = torch.zeros((self.out_features, 2 * self.state_features), device=lambdas.device, dtype=lambdas.dtype)
        C_full[:, ::2] = 0.5 * self.C  # we take the real part of the complex conjugate system as output...
        C_full[:, 1::2] = 0.5 * self.C.conj()
        D_full = self.D

        # Then apply transformation to real domain
        T_block = torch.tensor([[1, 1], [1j, -1j]], device=lambdas.device, dtype=lambdas.dtype)
        T_block_inv = torch.linalg.inv(T_block)
        T_full = torch.block_diag(*([T_block] * self.state_features))
        T_full_inv = torch.block_diag(*([T_block_inv] * self.state_features))

        A_real = (T_full @ A_full @ T_full_inv).real
        B_real = (T_full @ B_full).real
        C_real = (C_full @ T_full_inv).real
        D_real = D_full

        ss_real_params = [A_real, B_real, C_real, D_real]
        if to_numpy:
            ss_real_params = [ss_real_param.detach().numpy() for ss_real_param in ss_real_params]

        return (*ss_real_params,)

    def forward_loop(self, input, state=None):

        # Input size: (B, L, H)
        lambdas, B, C, D = self.ss_params()
        output = torch.empty(
            [i for i in input.shape[:-1]] + [self.out_features], device=self.B.device
        )

        states = []
        for u_step in input.split(1, dim=1):  # 1 is the time dimension

            u_step = u_step.squeeze(1)
            state = lambdas * state + u_step.to(B.dtype) @ B.T
            states.append(state)

        states = torch.stack(states, 1)
        output = (states @ C.mT).real + input @ D.T

        return output, states

    @torch.compiler.disable
    def forward_scan(self, input, state=None):

        # Only handles input of size (B, L, H) Batched parallel scan, borrows heavily from
        # https://colab.research.google.com/drive/1RgIv_3WAOW53CS0BnT7_782VKTYis9WG?usp=sharing which in turn borrows
        # from https://github.com/i404788/s5-pytorch
        lambdas, B, C, D = self.ss_params()

        # lambdas is shape (N,) but needs to be repeated to shape (L, N),
        # since input_sequence has shape (B, L, H).
        lambda_elements = lambdas.tile(input.shape[1], 1)
        # Calculate B@u for each step u of each input sequence in the batch.
        # Bu_elements will have shape (B, L, N)
        Bu_elements = input.to(B.dtype) @ B.T
        if state is not None:
            Bu_elements[:, 0, :] = Bu_elements[:, 0, :] + lambdas * state
            # Vmap the associative scan since Bu_elements is a batch of B sequences.
        # Recall that Lambda_elements has been repeated L times to (L, N),
        # while Bu_seq has shape (B, L, N)
        inner_state_fn = lambda Bu_seq: associative_scan(binary_operator_diag, (lambda_elements, Bu_seq))[1]
        # inner_states will be of shape (B, L, N)
        inner_states = torch.vmap(inner_state_fn)(Bu_elements)

        # y = (inner_states @ self.C.T).real + input_sequences * self.D
        y = (inner_states @ C.T).real + input @ D.T
        return y, inner_states

    def forward(self, input, state=None, mode="scan"):

        if state is None:
            state = torch.view_as_complex(
                torch.zeros((self.state_features, 2), device=input.device)
            )  # default initial state, size N

        match mode:
            case "scan":
                y, st = self.forward_scan(input, state)
            case "loop":
                y, st = self.forward_loop(input, state)
        return y


# Static non-linearities ------------------------------------

class MLPC(nn.Module):
    """ Standard Transformer MLP """

    def __init__(self, config: DWNConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.d_model, 14, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(14, config.d_model, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, activation=nn.ReLU, dropout=0.0):
        """
        Initialize an MLP.

        Parameters:
        - input_dim (int): Number of input features.
        - hidden_dims (list of int): List containing the number of neurons for each hidden layer.
        - output_dim (int): Number of output features.
        - activation (torch.nn.Module): Activation function class to be used (default: nn.ReLU).
        - dropout (float): Dropout probability (default: 0.0, meaning no dropout).
        """
        super(MLP, self).__init__()
        layers = []
        last_dim = input_dim
        # Add hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(activation())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            last_dim = hidden_dim

        # Add output layer
        layers.append(nn.Linear(last_dim, output_dim))

        # Combine all layers into a Sequential module
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass of the network.

        Parameters:
        - x (torch.Tensor): Input tensor.

        Returns:
        - torch.Tensor: Output of the MLP.
        """
        return self.network(x)


# SSM model -----------------------------------------------------

class SSMLayer(nn.Module):
    """ SSM block: LRU --> MLP + skip connection """

    def __init__(self, config: DWNConfig):
        super().__init__()
        self.ln = nn.LayerNorm(config.d_model, bias=config.bias)

        self.lru = LRU(config.d_model, config.d_model, config.d_state,
                       rmin=config.rmin, rmax=config.rmax, max_phase=config.max_phase)

        self.ff = MLPC(config)

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, state=None, mode: str = "scan"):
        z = x
        z = self.ln(z)  # prenorm

        z = self.lru(z, state=state, mode=mode)

        z = self.ff(z)  # MLP, GLU or LMLP
        z = self.dropout(z)

        # Residual connection
        x = z + x

        return x


class DeepSSM(nn.Module):
    """ Deep SSMs block: encoder --> cascade of n SSM layers --> decoder  """

    def __init__(self, n_u: int, n_y: int, config: DWNConfig):
        super().__init__()
        self.encoder = nn.Linear(n_u, config.d_model, bias=False)
        self.decoder = nn.Linear(config.d_model, n_y, bias=False)
        self.blocks = nn.ModuleList([SSMLayer(config) for _ in range(config.n_layers)])

    def forward(self, u, state=None, mode="scan"):
        x = self.encoder(u)
        for layer, block in enumerate(self.blocks):
            state_block = state[layer] if state is not None else None
            x = block(x, state=state_block, mode=mode)
        x = self.decoder(x)

        return x


# RENs -----------------------------------------------------------------------------


class ContractiveREN(nn.Module):
    """
    Acyclic contractive recurrent equilibrium network, following the paper:
    "Recurrent equilibrium networks: Flexible dynamic models with guaranteed
    stability and robustness, Revay M et al. ."

    The mathematical model of RENs relies on an implicit layer embedded in a recurrent layer.
    The model is described as,

                    [  E . x_t+1 ]  =  [ F    B_1  B_2   ]   [  x_t ]   +   [  b_x ]
                    [  Λ . v_t   ]  =  [ C_1  D_11  D_12 ]   [  w_t ]   +   [  b_w ]
                    [  y_t       ]  =  [ C_2  D_21  D_22 ]   [  u_t ]   +   [  b_u ]

    where E is an invertible matrix and Λ is a positive-definite diagonal matrix. The model parameters
    are then {E, Λ , F, B_i, C_i, D_ij, b} which form a convex set according to the paper.

    NOTE: REN has input "u", output "y", and internal state "x". When used in closed-loop,
          the REN input "u" would be the noise reconstruction ("w") and the REN output ("y")
          would be the input to the plant. The internal state of the REN ("x") should not be mistaken
          with the internal state of the plant.
    """

    def __init__(
            self, dim_in: int, dim_out: int, dim_internal: int,
            dim_nl: int, internal_state_init=None, initialization_std: float = 0.5,
            pos_def_tol: float = 0.001, contraction_rate_lb: float = 1.0
    ):
        """
        Args:
            dim_in (int): Input (u) dimension.
            dim_out (int): Output (y) dimension.
            dim_internal (int): Internal state (x) dimension. This state evolves with contraction properties.
            dim_nl (int): Dimension of the input ("v") and ouput ("w") of the nonlinear static block.
            initialization_std (float, optional): Weight initialization. Set to 0.1 by default.
            internal_state_init (torch.Tensor or None, optional): Initial condition for the internal state. Defaults to 0 when set to None.
            epsilon (float, optional): Positive and negligible scalar to force positive definite matrices.
            contraction_rate_lb (float, optional): Lower bound on the contraction rate. Defaults to 1.
        """
        super().__init__()

        # set dimensions
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_internal = dim_internal
        self.dim_nl = dim_nl

        # set functionalities
        self.contraction_rate_lb = contraction_rate_lb

        # auxiliary elements
        self.epsilon = pos_def_tol

        # initialize internal state
        if internal_state_init is None:
            self.x = torch.zeros(1, 1, self.dim_internal)
        else:
            assert isinstance(internal_state_init, torch.Tensor)
            self.x = internal_state_init.reshape(1, 1, self.dim_internal)
        self.register_buffer('init_x', self.x.detach().clone())

        # define matrices shapes
        # auxiliary matrices
        self.X_shape = (2 * self.dim_internal + self.dim_nl, 2 * self.dim_internal + self.dim_nl)
        self.Y_shape = (self.dim_internal, self.dim_internal)
        # nn state dynamics
        self.B2_shape = (self.dim_internal, self.dim_in)
        # nn output
        self.C2_shape = (self.dim_out, self.dim_internal)
        self.D21_shape = (self.dim_out, self.dim_nl)
        self.D22_shape = (self.dim_out, self.dim_in)
        # v signal
        self.D12_shape = (self.dim_nl, self.dim_in)

        # define trainable params
        self.training_param_names = ['X', 'Y', 'B2', 'C2', 'D21', 'D22', 'D12']
        self._init_trainable_params(initialization_std)

        # mask
        self.register_buffer('eye_mask_H', torch.eye(2 * self.dim_internal + self.dim_nl))
        self.register_buffer('eye_mask_w', torch.eye(self.dim_nl))

    def _update_model_param(self):
        """
        Update non-trainable matrices according to the REN formulation to preserve contraction.
        """
        # dependent params
        H = torch.matmul(self.X.T, self.X) + self.epsilon * self.eye_mask_H
        h1, h2, h3 = torch.split(H, [self.dim_internal, self.dim_nl, self.dim_internal], dim=0)
        H11, H12, H13 = torch.split(h1, [self.dim_internal, self.dim_nl, self.dim_internal], dim=1)
        H21, H22, _ = torch.split(h2, [self.dim_internal, self.dim_nl, self.dim_internal], dim=1)
        H31, H32, H33 = torch.split(h3, [self.dim_internal, self.dim_nl, self.dim_internal], dim=1)
        P = H33

        # nn state dynamics
        self.F = H31
        self.B1 = H32

        # nn output
        self.E = 0.5 * (H11 + self.contraction_rate_lb * P + self.Y - self.Y.T)
        self.E_inv = self.E.inverse()

        # v signal for strictly acyclic REN
        self.Lambda = 0.5 * torch.diag(H22)
        self.D11 = -torch.tril(H22, diagonal=-1)
        self.C1 = -H21

    def forward_onstep(self, u_in):
        """
        Forward pass of REN.

        Args:
            u_in (torch.Tensor): Input with the size of (batch_size, 1, self.dim_in).

        Return:
            y_out (torch.Tensor): Output with (batch_size, 1, self.dim_out).
        """

        batch_size = u_in.shape[0]

        w = torch.zeros(batch_size, 1, self.dim_nl, device=u_in.device)

        # update each row of w using Eq. (8) with a lower triangular D11
        for i in range(self.dim_nl):
            #  v is element i of v with dim (batch_size, 1)
            v = F.linear(self.x, self.C1[i, :]) + F.linear(w, self.D11[i, :]) + F.linear(u_in, self.D12[i, :])
            w = w + (self.eye_mask_w[i, :] * torch.tanh(v / self.Lambda[i])).reshape(batch_size, 1, self.dim_nl)

        # compute next state using Eq. 18
        self.x = F.linear(F.linear(self.x, self.F) + F.linear(w, self.B1) + F.linear(u_in, self.B2), self.E_inv)

        # compute output
        y_out = F.linear(self.x, self.C2) + F.linear(w, self.D21) + F.linear(u_in, self.D22)
        return y_out

    def reset(self):
        self.x = self.init_x  # reset the REN state to the initial value

    # init trainable params
    def _init_trainable_params(self, initialization_std):
        for training_param_name in self.training_param_names:  # name of one of the training params, e.g., X
            # read the defined shapes of the selected training param, e.g., X_shape
            shape = getattr(self, training_param_name + '_shape')
            # define the selected param (e.g., self.X) as nn.Parameter
            setattr(self, training_param_name, nn.Parameter((torch.randn(*shape) * initialization_std)))

    # setters and getters
    def get_parameter_shapes(self):
        param_dict = OrderedDict(
            (name, getattr(self, name).shape) for name in self.training_param_names
        )
        return param_dict

    def get_named_parameters(self):
        param_dict = OrderedDict(
            (name, getattr(self, name)) for name in self.training_param_names
        )
        return param_dict

    def forward(self, u_in, mode=None):
        self.reset()
        self._update_model_param()

        """
        Runs the forward pass of REN for a whole input sequence of length horizon.

        Args:
            x0_sys: initial condition of the real plant
            u_in (torch.Tensor): Input with the size of (batch_size, horizon, self.input_dim).

        Return:
            y_out (torch.Tensor): Output with (batch_size, horizon, self.output_dim).
        """
        horizon = u_in.shape[1]
        batch_size = u_in.shape[0]

        # Storage for trajectories
        y_traj = []

        for t in range(horizon):
            y = self.forward_onstep(u_in[:, t:t + 1, :])
            y_traj.append(y)  # Store output
            # note that the last input is not used

        y_out = torch.cat(y_traj, dim=1)  # Shape: (batch_size, horizon, output_dim)

        return y_out


# RNN Model ------------------------------------------

class SimpleRNN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers):
        super(SimpleRNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers, batch_first=True, nonlinearity="relu")
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, h=None, mode=None):
        # Initialize hidden state if not provided
        if h is None:
            h = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=x.device)
        out, h = self.rnn(x, h)
        out = self.fc(out)  # Apply linear layer to all time steps
        return out

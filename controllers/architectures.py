import math
from dataclasses import dataclass
import torch
import torch.nn as nn
from .LRU import LRU, LRU_Robust
from .L_bounded_MLPs import FirstChannel, SandwichFc, SandwichLin
from collections import OrderedDict

""" Data class to set up the model (values here are used just to initialize all fields) """


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
    scale: float = 1  # Lipschitz constant of the Lipschitz bounded MLP (LMLP)
    dim_amp: int = 4  # controls the hidden layer's dimension of the MLP
    gamma: bool = True  # set this to true if you want to use the l2 gain parametrization for the SSM. If set to false,
    # the complex diagonal parametrization of the LRU will be used instead.
    gain: float = 8  # set the overall l2 gain in case you want to keep it fixed and not trainable
    trainable: bool = True  # set this to true if you want a trainable l2 gain.

    # Parallel scan must be selected in the forward call. It will be disabled when gamma is set to True.

    """ Scaffolding Layers """


class MLP(nn.Module):
    """ Standard Transformer MLP """

    def __init__(self, config: DWNConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.d_model, config.dim_amp * config.d_model, bias=False)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(config.dim_amp * config.d_model, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class LMLP(nn.Module):
    """ Implements a Lipschitz.-bounded MLP with sandwich layers. The square root
    # of the Lipschitz bound is given by scale """

    def __init__(self, config: DWNConfig):
        super().__init__()
        layers = [FirstChannel(config.d_model, scale=config.scale),
                  SandwichFc(config.d_model, config.dim_amp * config.d_model, bias=False, scale=config.scale),
                  SandwichFc(config.dim_amp * config.d_model, config.dim_amp * config.d_model, bias=False,
                             scale=config.scale),
                  SandwichFc(config.dim_amp * config.d_model, config.dim_amp * config.d_model, bias=False,
                             scale=config.scale),
                  SandwichLin(config.dim_amp * config.d_model, config.d_model, bias=False, scale=config.scale),
                  nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()]
        self.model = nn.Sequential(*layers)

    def forward(self, input):
        x = self.model(input)
        return x


class GLU(nn.Module):
    """ The static non-linearity used in the S4 paper """

    def __init__(self, config: DWNConfig):
        super().__init__()
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()
        self.output_linear = nn.Sequential(
            nn.Linear(config.d_model, 2 * config.d_model),
            # nn.Conv1d(config.d_model, 2 * config.d_model, kernel_size=1),
            nn.GLU(dim=-1),
        )

    def forward(self, x):
        x = self.dropout(self.activation(x))
        x = self.output_linear(x)
        return x

    """ SSMs blocks """


class SSL(nn.Module):
    """ State Space Layer: LRU --> MLP + skip connection """

    def __init__(self, config: DWNConfig):
        super().__init__()
        self.ln = nn.LayerNorm(config.d_model, bias=config.bias)

        if config.gamma:
            self.lru = LRU_Robust(config.d_model, config.trainable)

        else:
            self.lru = LRU(config.d_model, config.d_model, config.d_state,
                           rmin=config.rmin, rmax=config.rmax, max_phase=config.max_phase)
        if config.ff == "GLU":
            self.ff = GLU(config)
        elif config.ff == "MLP":
            self.ff = MLP(config)
        elif config.ff == "LMLP":
            self.ff = LMLP(config)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, gamma=None, state=None, mode: str = "loop"):

        z = x
        #  z = self.ln(z)  # prenorm

        z, st = self.lru(z, state=state, mode=mode)

        z = self.ff(z)  # MLP, GLU or LMLP
        z = self.dropout(z)

        # Residual connection
        x = z + x

        return x, st


class DeepSSM(nn.Module):
    """ Deep SSMs block: encoder --> cascade of n SSM blocks --> decoder  """

    def __init__(self, n_u: int, n_y: int, config: DWNConfig):
        super().__init__()

        self.config = config

        self.encoder = nn.Linear(n_u, config.d_model, bias=False)
        self.decoder = nn.Linear(config.d_model, n_y, bias=False)

        if not config.trainable:  # parameters needed for when the l2 gain is fixed and prescribed
            self.register_buffer('gamma_t', torch.tensor(config.gain))

            self.encoder = nn.Parameter(torch.randn(config.d_model, n_u))
            self.decoder = nn.Parameter(torch.randn(n_y, config.d_model))

        self.blocks = nn.ModuleList([SSL(config) for _ in range(config.n_layers)])

    def forward_fixed_gamma(self, u, state=None, mode="loop", gammaT=None):

        gamma_t = torch.abs(self.gamma_t) if gammaT is None else gammaT
        gammaLRU = [block.lru.gamma for layer, block in enumerate(self.blocks)]
        decoder = (gamma_t * self.decoder / (torch.norm(self.decoder, 2) * torch.norm(self.encoder, 2)) /
                   (torch.prod(torch.abs(torch.tensor(gammaLRU))) + 1))
        x = u @ self.encoder.T
        for layer, block in enumerate(self.blocks):
            state_block = state[layer] if state is not None else None
            x, st = block(x, state=state_block, mode=mode)
        x = x @ decoder.T

        return x, st

    def forward_trainable_gamma(self, u, state=None, mode="loop"):

        x = self.encoder(u)
        for layer, block in enumerate(self.blocks):
            state_block = state[layer] if state is not None else None
            x, st = block(x, state=state_block, mode=mode)
        x = self.decoder(x)

        return x, st

    def forward(self, u, state=None, mode="loop", gamma=None):

        if not self.config.trainable:
            x, st = self.forward_fixed_gamma(u=u, state=state, mode=mode, gammaT=gamma)
        else:
            x, st = self.forward_trainable_gamma(u=u, state=state, mode=mode)

        return x

    def reset(self):
        for layer, block in enumerate(self.blocks):
            block.lru.reset()  # default initial state, size N

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


""" Work in progresso on multi - input """


class MLPtoSquareMatrix(nn.Module):
    def __init__(self, w_dim: int, x_dim: int, y_dim: int,
                 sensitive_feature_index: int,
                 hidden_dim: int = 64, depth: int = 4):
        """
        A general-purpose MLP that is highly sensitive to a specific feature.

        Args:
            w_dim (int): Dimension of the 'w' input vector.
            x_dim (int): Dimension of the 'x' input vector.
            y_dim (int): The side dimension of the output square matrix (y_dim x y_dim).
            sensitive_feature_index (int): The index of the feature in 'x' that the
                                           network should be highly sensitive to.
            hidden_dim (int): The number of neurons in the hidden layers.
            depth (int): The number of layers in the main MLP.
        """
        super().__init__()

        # Validate the sensitive feature index
        if not (0 <= sensitive_feature_index < x_dim):
            raise ValueError(f"sensitive_feature_index must be between 0 and {x_dim - 1}")

        self.sensitive_feature_index = sensitive_feature_index

        # The "static" inputs consist of 'w' and all features of 'x' EXCEPT the sensitive one.
        num_static_features_x = x_dim - 1
        main_input_dim = w_dim + num_static_features_x

        # 1. Main MLP for static inputs
        layers = [nn.Linear(main_input_dim, hidden_dim), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        self.main_mlp_base = nn.Sequential(*layers)

        # The final layer that maps modulated features to the output
        self.final_layer = nn.Linear(hidden_dim, y_dim * y_dim)

        # 2. Gating MLP (the "Controller")
        # Takes only the single sensitive feature as input
        self.gating_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 2 * hidden_dim)  # Outputs gain and bias
        )

        self.y_dim = y_dim
        self.hidden_dim = hidden_dim

        # Pre-calculate the indices for static features for efficiency
        all_indices = list(range(x_dim))
        # This removes the sensitive feature's index from the list
        self.static_indices = [i for i in all_indices if i != self.sensitive_feature_index]

    def forward(self, w: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        w: (B, 1, w_dim)
        x: (B, 1, x_dim)
        returns: (B, y_dim, y_dim)
        """
        assert w.dim() == 3 and x.dim() == 3, "Inputs must be (B, 1, N)"
        B = w.shape[0]
        w_flat = w.squeeze(1)  # (B, w_dim)
        x_flat = x.squeeze(1)  # (B, x_dim)

        # Separate the sensitive feature from the static features in x
        sensitive_input = x_flat[:, self.sensitive_feature_index].unsqueeze(1)  # (B, 1)
        static_x_inputs = x_flat[:, self.static_indices]  # (B, x_dim - 1)

        # Combine all static inputs
        combined_static_inputs = torch.cat([w_flat, static_x_inputs], dim=1)

        # --- Main forward pass ---

        # 1. Pass static inputs through the base of the main MLP
        features = self.main_mlp_base(combined_static_inputs)  # (B, hidden_dim)

        # 2. Pass the sensitive feature through its dedicated gating MLP
        modulation = self.gating_mlp(sensitive_input)  # (B, 2 * hidden_dim)

        # Split into gain and bias
        gain = modulation[:, :self.hidden_dim]
        bias = modulation[:, self.hidden_dim:]

        # 3. Apply the modulation (FiLM step)
        modulated_features = (features * gain) + bias

        # 4. Pass through the final layer
        out = self.final_layer(modulated_features)
        out = out.view(B, self.y_dim, self.y_dim)

        return out


class GeneralSensitiveMLP_Gating_LN(nn.Module):
    def __init__(self, w_dim: int, x_dim: int, y_dim: int,
                 sensitive_feature_index: int,
                 hidden_dim: int = 64, depth: int = 4):
        super().__init__()

        # --- Same as before ---
        if not (0 <= sensitive_feature_index < x_dim):
            raise ValueError(f"sensitive_feature_index must be between 0 and {x_dim - 1}")
        self.sensitive_feature_index = sensitive_feature_index

        main_input_dim = w_dim + (x_dim - 1)

        # --- MODIFICATION: We will apply LayerNorm and ReLU manually ---
        # We need to break up the main_mlp_base to insert LayerNorm
        self.main_mlp_layers = nn.ModuleList()
        # Input layer
        self.main_mlp_layers.append(nn.Linear(main_input_dim, hidden_dim))

        # Hidden layers
        for _ in range(depth - 1):
            # Each "block" is a Linear layer followed by LayerNorm and ReLU
            self.main_mlp_layers.append(nn.Linear(hidden_dim, hidden_dim))

        # The final layer that maps modulated features to the output
        self.final_layer = nn.Linear(hidden_dim, y_dim * y_dim)

        # Gating MLP (Controller) - No change here
        self.gating_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim // 2),
            nn.ReLU(),  # ReLU is fine here; the network is small
            nn.Linear(hidden_dim // 2, 2 * hidden_dim)
        )

        # *** THE KEY ADDITION: A LayerNorm layer ***
        # It will normalize the `hidden_dim` features.
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # --- Same as before ---
        self.y_dim = y_dim
        self.hidden_dim = hidden_dim
        all_indices = list(range(x_dim))
        self.static_indices = [i for i in all_indices if i != self.sensitive_feature_index]

    def forward(self, w: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        B = w.shape[0]
        w_flat = w.squeeze(1)
        x_flat = x.squeeze(1)

        sensitive_input = x_flat[:, self.sensitive_feature_index].unsqueeze(1)
        static_x_inputs = x_flat[:, self.static_indices]
        combined_static_inputs = torch.cat([w_flat, static_x_inputs], dim=1)

        # --- MODIFIED FORWARD PASS ---

        # 1. Pass static inputs through the first layer
        features = self.main_mlp_layers[0](combined_static_inputs)
        # For simplicity, we apply modulation after the first layer.
        # This is a common and effective pattern.

        # 2. Get gain and bias from the sensitive input
        modulation = self.gating_mlp(sensitive_input)
        gain = modulation[:, :self.hidden_dim]
        bias = modulation[:, self.hidden_dim:]

        # 3. Apply the modulation (FiLM step)
        modulated_features = (features * gain) + bias

        # 4. *** NORMALIZE and ACTIVATE ***
        # This is the crucial step that prevents explosions
        normed_features = self.layer_norm(modulated_features)
        activated_features = nn.functional.relu(normed_features)  # Use the original ReLU!

        # Pass through the rest of the main MLP
        # (This example modulates one layer; you could add more modulation blocks)
        hidden_out = activated_features
        for layer in self.main_mlp_layers[1:]:
            hidden_out = nn.functional.relu(layer(hidden_out))

        # 5. Pass through the final layer
        out = self.final_layer(hidden_out)
        out = out.view(B, self.y_dim, self.y_dim)

        return out

class Multi(nn.Module):
    """ Multi input operator  """

    def __init__(self, n_u: int, n_x: int, n_y: int, config: DWNConfig):
        super().__init__()

        self.config = config
        self.m1 = DeepSSM(n_u, n_y, config)
        self.m2 = GeneralSensitiveMLP_Gating_LN(n_u, n_x, n_y, sensitive_feature_index=6)

    def forward(self, w, x):
        output = torch.bmm(self.m2(w, x), self.m1(w, state=None, mode="loop", gamma=None).squeeze(1).unsqueeze(2))
        output = output.transpose(-1, -2)
        return output

    def reset(self):
        for layer, block in enumerate(self.m1.blocks):
            block.lru.reset()  # default initial state, size N

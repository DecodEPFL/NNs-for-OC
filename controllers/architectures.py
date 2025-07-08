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
        # Pre-compute hidden dimension for efficiency
        self.hidden_dim = config.dim_amp * config.d_model

        self.c_fc = nn.Linear(config.d_model, self.hidden_dim, bias=False)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(self.hidden_dim, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return self.dropout(x)


class LMLP(nn.Module):
    """ Implements a Lipschitz.-bounded MLP with sandwich layers. The square root
    # of the Lipschitz bound is given by scale """

    def __init__(self, config: DWNConfig):
        super().__init__()
        # Pre-compute hidden dimension for efficiency
        hidden_dim = config.dim_amp * config.d_model

        # More efficient layer construction using list comprehension
        layers = [
            FirstChannel(config.d_model, scale=config.scale),
            SandwichFc(config.d_model, hidden_dim, bias=False, scale=config.scale),
            SandwichFc(hidden_dim, hidden_dim, bias=False, scale=config.scale),
            SandwichFc(hidden_dim, hidden_dim, bias=False, scale=config.scale),
            SandwichLin(hidden_dim, config.d_model, bias=False, scale=config.scale)
        ]

        # Only add dropout if needed
        if config.dropout > 0:
            layers.append(nn.Dropout(config.dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, input):
        return self.model(input)


class GLU(nn.Module):
    """ The static non-linearity used in the S4 paper """

    def __init__(self, config: DWNConfig):
        super().__init__()
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()

        # More efficient sequential construction
        self.output_linear = nn.Sequential(
            nn.Linear(config.d_model, 2 * config.d_model),
            nn.GLU(dim=-1),
        )

    def forward(self, x):
        x = self.dropout(self.activation(x))
        return self.output_linear(x)

    """ SSMs blocks """


class SSL(nn.Module):
    """ State Space Layer: LRU --> MLP + skip connection """

    def __init__(self, config: DWNConfig):
        super().__init__()
        self.ln = nn.LayerNorm(config.d_model, bias=config.bias)

        # More efficient LRU initialization
        if config.gamma:
            self.lru = LRU_Robust(config.d_model, config.trainable)
        else:
            self.lru = LRU(config.d_model, config.d_model, config.d_state,
                           rmin=config.rmin, rmax=config.rmax, max_phase=config.max_phase)

        # Dictionary for layer selection
        ff_layers = {
            "GLU": lambda: GLU(config),
            "MLP": lambda: MLP(config),
            "LMLP": lambda: LMLP(config)
        }

        if config.ff not in ff_layers:
            raise ValueError(f"Unknown feedforward type: {config.ff}")

        self.ff = ff_layers[config.ff]()
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, gamma=None, state=None, mode: str = "loop"):
        z = x
        # z = self.ln(z)  # prenorm

        z, st = self.lru(z, state=state, mode=mode)
        z = self.ff(z)  # MLP, GLU or LMLP
        z = self.dropout(z)

        # Residual connection
        return z + x, st


class DeepSSM(nn.Module):
    """ Deep SSMs block: encoder --> cascade of n SSM blocks --> decoder  """

    def __init__(self, n_u: int, n_y: int, config: DWNConfig):
        super().__init__()

        self.config = config

        # Simplified initialization - only handle trainable gamma for LRU_Robust
        if config.gamma:
            # Using LRU_Robust - need to handle trainable vs fixed gamma
            if config.trainable:
                self.encoder = nn.Linear(n_u, config.d_model, bias=False)
                self.decoder = nn.Linear(config.d_model, n_y, bias=False)
            else:
                # Fixed gamma case - use Parameter tensors
                self.register_buffer('gamma_t', torch.tensor(config.gain))
                self.encoder = nn.Parameter(torch.randn(config.d_model, n_u))
                self.decoder = nn.Parameter(torch.randn(n_y, config.d_model))
        else:
            # Using regular LRU - always use Linear layers (no gamma considerations)
            self.encoder = nn.Linear(n_u, config.d_model, bias=False)
            self.decoder = nn.Linear(config.d_model, n_y, bias=False)

        self.blocks = nn.ModuleList([SSL(config) for _ in range(config.n_layers)])

    def forward_sequential_efficient(self, u, state=None, mode="loop"):
        """
        Efficient sequential processing: single loop over time, passing through all layers at each timestep.
        This replaces the inefficient approach of each layer processing the entire sequence separately.
        """
        batch_size, seq_len, input_dim = u.shape

        # Initialize states for all layers if not provided
        if state is None:
            layer_states = [None] * len(self.blocks)
        else:
            layer_states = state if isinstance(state, list) else [state] * len(self.blocks)

        # Pre-allocate output tensor with correct dimensions
        if isinstance(self.decoder, nn.Linear):
            output_dim = self.decoder.out_features
        else:
            output_dim = self.decoder.shape[0]

        # Pre-allocate outputs for better memory efficiency
        outputs = torch.empty(batch_size, seq_len, output_dim, device=u.device, dtype=u.dtype)

        # Process encoder once for entire sequence - more efficient
        if isinstance(self.encoder, nn.Linear):
            x = self.encoder(u)  # (B, L, d_model)
        else:
            x = u @ self.encoder.T

        # Cache frequently used values outside the loop
        ff_blocks = [block.ff for block in self.blocks]
        dropout_blocks = [block.dropout for block in self.blocks]
        lru_blocks = [block.lru for block in self.blocks]

        # Single optimized loop over time
        for t in range(seq_len):
            x_t = x[:, t, :]  # Current timestep: (B, d_model)

            # Optimized layer processing
            for layer_idx in range(len(self.blocks)):
                lru = lru_blocks[layer_idx]

                if hasattr(lru, 'forward_step'):
                    # Use efficient single-timestep processing
                    x_t, layer_states[layer_idx] = lru.forward_step(x_t, layer_states[layer_idx])
                    # Apply feedforward and dropout
                    z = ff_blocks[layer_idx](x_t)
                    z = dropout_blocks[layer_idx](z)
                    x_t = z + x_t  # Residual connection
                else:
                    # Fallback to regular forward
                    x_t_expanded = x_t.unsqueeze(1)
                    x_t_out, st = self.blocks[layer_idx](x_t_expanded, state=layer_states[layer_idx], mode=mode)
                    x_t = x_t_out.squeeze(1)
                    layer_states[layer_idx] = st

            # Decoder for this timestep
            if isinstance(self.decoder, nn.Linear):
                outputs[:, t, :] = self.decoder(x_t)
            else:
                outputs[:, t, :] = x_t @ self.decoder.T

        return outputs, layer_states

    def forward_lru_robust(self, u, state=None, mode="loop", gamma=None):
        """Handle LRU_Robust case with trainable/fixed gamma logic"""
        if self.config.trainable:
            # Trainable gamma case
            x = self.encoder(u)

            st = None
            for layer, block in enumerate(self.blocks):
                state_block = state[layer] if state is not None else None
                x, st = block(x, state=state_block, mode=mode)

            x = self.decoder(x)
            return x, st
        else:
            # Fixed gamma case
            gamma_t = torch.abs(self.gamma_t) if gamma is None else gamma

            st = None

            # More efficient gamma collection using list comprehension
            gammaLRU = [torch.abs(block.lru.gamma) for block in self.blocks]
            gammaLRU_tensor = torch.stack(gammaLRU)

            # More efficient decoder computation
            encoder_norm = torch.norm(self.encoder, 2)
            decoder_norm = torch.norm(self.decoder, 2)
            gamma_prod = torch.prod(gammaLRU_tensor) + 1

            decoder_scaled = (gamma_t * self.decoder) / (encoder_norm * decoder_norm * gamma_prod)

            # Use matrix multiplication directly with Parameter tensors
            x = u @ self.encoder.T

            for layer, block in enumerate(self.blocks):
                state_block = state[layer] if state is not None else None
                x, st = block(x, state=state_block, mode=mode)

            x = x @ decoder_scaled.T
            return x, st

    def forward_regular_lru(self, u, state=None, mode="loop"):
        """Handle regular LRU case - much simpler, no gamma considerations"""
        if mode == "loop_efficient":
            return self.forward_sequential_efficient(u, state, mode)

        # Standard processing for regular LRU
        x = self.encoder(u)

        st = None
        for layer, block in enumerate(self.blocks):
            state_block = state[layer] if state is not None else None
            x, st = block(x, state=state_block, mode=mode)

        x = self.decoder(x)
        return x, st

    def forward(self, u, state=None, mode="loop", gamma=None):
        # Default to efficient mode for loop processing
        if mode == "loop":
            mode = "loop_efficient"

        if self.config.gamma:
            # Using LRU_Robust - handle trainable/fixed gamma logic
            return self.forward_lru_robust(u, state, mode, gamma)
        else:
            # Using regular LRU - simple case, no gamma considerations
            return self.forward_regular_lru(u, state, mode)

    def reset(self):
        # More efficient reset using direct iteration
        for block in self.blocks:
            block.lru.reset()

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
        self.y_dim = y_dim
        self.hidden_dim = hidden_dim

        # Pre-compute dimensions for efficiency
        num_static_features_x = x_dim - 1
        main_input_dim = w_dim + num_static_features_x

        # 1. More efficient main MLP construction
        layers = [nn.Linear(main_input_dim, hidden_dim), nn.Tanh()]
        layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()] for _ in range(depth - 1))
        self.main_mlp_base = nn.Sequential(*layers)

        # The final layer that maps modulated features to the output
        self.final_layer = nn.Linear(hidden_dim, y_dim * y_dim)

        # 2. More efficient gating MLP construction
        self.gating_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 2 * hidden_dim)  # Outputs gain and bias
        )

        # Pre-calculate the indices for static features for efficiency
        self.static_indices = [i for i in range(x_dim) if i != self.sensitive_feature_index]

    def forward(self, w: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        w: (B, 1, w_dim)
        x: (B, 1, x_dim)
        returns: (B, y_dim, y_dim)
        """
        assert w.dim() == 3 and x.dim() == 3, "Inputs must be (B, 1, N)"
        B = w.shape[0]

        # More efficient tensor operations
        w_flat = w.squeeze(1)  # (B, w_dim)
        x_flat = x.squeeze(1)  # (B, x_dim)

        # Separate the sensitive feature from the static features in x
        sensitive_input = x_flat[:, self.sensitive_feature_index:self.sensitive_feature_index+1]  # (B, 1)
        static_x_inputs = x_flat[:, self.static_indices]  # (B, x_dim - 1)

        # Combine all static inputs
        combined_static_inputs = torch.cat([w_flat, static_x_inputs], dim=1)

        # --- Main forward pass ---

        # 1. Pass static inputs through the base of the main MLP
        features = self.main_mlp_base(combined_static_inputs)  # (B, hidden_dim)

        # 2. Pass the sensitive feature through its dedicated gating MLP
        modulation = self.gating_mlp(sensitive_input)  # (B, 2 * hidden_dim)

        # Split into gain and bias more efficiently
        gain, bias = modulation.chunk(2, dim=1)

        # 3. Apply the modulation (FiLM step)
        modulated_features = features * gain + bias

        # 4. Pass through the final layer and reshape
        out = self.final_layer(modulated_features)
        return out.view(B, self.y_dim, self.y_dim)


class GeneralSensitiveMLP_Gating_LN(nn.Module):
    def __init__(self, w_dim: int, x_dim: int, y_dim: int,
                 sensitive_feature_index: int,
                 hidden_dim: int = 64, depth: int = 4):
        super().__init__()

        # Validate inputs
        if not (0 <= sensitive_feature_index < x_dim):
            raise ValueError(f"sensitive_feature_index must be between 0 and {x_dim - 1}")

        self.sensitive_feature_index = sensitive_feature_index
        self.y_dim = y_dim
        self.hidden_dim = hidden_dim

        # Pre-compute dimensions
        main_input_dim = w_dim + (x_dim - 1)

        # More efficient layer construction
        self.main_mlp_layers = nn.ModuleList([
            nn.Linear(main_input_dim, hidden_dim),
            *[nn.Linear(hidden_dim, hidden_dim) for _ in range(depth - 1)]
        ])

        # The final layer that maps modulated features to the output
        self.final_layer = nn.Linear(hidden_dim, y_dim * y_dim)

        # Gating MLP (Controller) - more efficient construction
        self.gating_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2 * hidden_dim)
        )

        # LayerNorm layer
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # Pre-calculate static indices for efficiency
        self.static_indices = [i for i in range(x_dim) if i != self.sensitive_feature_index]

    def forward(self, w: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        B = w.shape[0]
        w_flat = w.squeeze(1)
        x_flat = x.squeeze(1)

        # More efficient indexing
        sensitive_input = x_flat[:, self.sensitive_feature_index:self.sensitive_feature_index+1]
        static_x_inputs = x_flat[:, self.static_indices]
        combined_static_inputs = torch.cat([w_flat, static_x_inputs], dim=1)

        # --- MODIFIED FORWARD PASS ---

        # 1. Pass static inputs through the first layer
        features = self.main_mlp_layers[0](combined_static_inputs)

        # 2. Get gain and bias from the sensitive input
        modulation = self.gating_mlp(sensitive_input)
        gain, bias = modulation.chunk(2, dim=1)

        # 3. Apply the modulation (FiLM step)
        modulated_features = features * gain + bias

        # 4. Normalize and activate
        normed_features = self.layer_norm(modulated_features)
        activated_features = torch.relu(normed_features)

        # Pass through the rest of the main MLP
        hidden_out = activated_features
        for layer in self.main_mlp_layers[1:]:
            hidden_out = torch.relu(layer(hidden_out))

        # 5. Pass through the final layer and reshape
        out = self.final_layer(hidden_out)
        return out.view(B, self.y_dim, self.y_dim)


class Multi(nn.Module):
    """ Multi input operator  """

    def __init__(self, n_u: int, n_x: int, n_y: int, config: DWNConfig):
        super().__init__()

        self.config = config
        self.m1 = DeepSSM(n_u, n_y, config)
        self.m2 = GeneralSensitiveMLP_Gating_LN(n_u, n_x, n_y, sensitive_feature_index=6)

    def forward(self, w, x):
        # More efficient computation by avoiding unnecessary operations
        m1_output, _ = self.m1(w, state=None, mode="loop", gamma=None)  # Unpack tuple to get just the output
        m2_output = self.m2(w, x)

        # More efficient batch matrix multiplication
        m1_reshaped = m1_output.squeeze(1).unsqueeze(2)
        output = torch.bmm(m2_output, m1_reshaped)
        return output.transpose(-1, -2)

    def reset(self):
        # More efficient reset
        for block in self.m1.blocks:
            block.lru.reset()

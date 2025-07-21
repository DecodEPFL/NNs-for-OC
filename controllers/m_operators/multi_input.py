import torch
import torch.nn as nn
from controllers.m_operators.ssm import DeepSSM, SSMConfig


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

        # --- FORWARD PASS ---

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
    """ Multi input operator M(w,x) = M1(w) @ M2(w,x)
     w: l_p sequence
     x: l_infinity sequence  """

    def __init__(self, n_w: int, n_x: int, n_y: int, config: SSMConfig):
        super().__init__()

        self.config = config
        self.m1 = DeepSSM(n_w, n_y, config)
        self.m2 = GeneralSensitiveMLP_Gating_LN(n_w, n_x, n_y, sensitive_feature_index=4) #6

    def forward(self, w, x):
        # Factorization: M(w,x) = M1(w) @ M2(w,x)
        m1_output, _ = self.m1(w, state=None, mode="loop", gamma=None)  # Unpack tuple to get just the output
        m2_output = self.m2(w, x)

        # Batch matrix multiplication
        m1_reshaped = m1_output.squeeze(1).unsqueeze(2)
        output = torch.bmm(m2_output, m1_reshaped)
        return output.transpose(-1, -2)

    def reset(self):
        # Reset initial states of dynamical operators
        for block in self.m1.blocks:
            block.lru.reset()

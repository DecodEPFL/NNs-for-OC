import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, Union, Callable

from .architectures import DWNConfig, DeepSSM, Multi
from .contractive_ren import ContractiveREN

# More flexible device management
def get_optimal_device():
    """Get the optimal device for computation"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("cpu")
    else:
        return torch.device("cpu")

device = get_optimal_device()


class PerfBoostController(nn.Module):
    """
    Performance boosting controller, following the paper:
        "Learning to Boost the Performance of Stable Nonlinear Systems".
    Implements a state-feedback controller with stability guarantees.
    NOTE: When used in closed-loop, the controller input is the measured state of the plant
          and the controller output is the input to the plant.
    This controller has a memory for the last input ("self.last_input") and the last output ("self.last_output").
    """

    def __init__(self,
                 noiseless_forward: Callable,
                 input_init: torch.Tensor,
                 output_init: torch.Tensor,
                 nn_type: str = "REN",
                 dim_internal: int = 8,
                 config: Optional[DWNConfig] = None,
                 dim_in2: int = 1,
                 dim_nl: int = 8,
                 # SSM properties
                 non_linearity: Optional[str] = None,
                 dim_middle: int = 6,
                 # acyclic REN properties
                 initialization_std: float = 0.5,
                 pos_def_tol: float = 0.001,
                 contraction_rate_lb: float = 1.0,
                 ren_internal_state_init: Optional[torch.Tensor] = None,
                 # misc
                 output_amplification: float = 20,
                 target_device: Optional[torch.device] = None,
                 ):
        """
         Args:
            noiseless_forward:            System dynamics without process noise. It can be TV.
            input_init (torch.Tensor):    Initial input to the controller.
            output_init (torch.Tensor):   Initial output from the controller before anything is calculated.
            nn_type (str):                Which NN model to use for the Emme operator (Options: 'REN', 'SSM', 'MI')
            non_linearity (str):          Non-linearity used in SSMs for scaffolding.
            target_device:                Target device for computation. If None, uses optimal device.
            ##### the following are the same as AcyclicREN args:
            dim_internal (int):           Internal state (x) dimension.
            dim_nl (int):                 Dimension of the input ("v") and output ("w") of the NL static block of REN.
            initialization_std (float):   [Optional] Weight initialization. Set to 0.1 by default.
            pos_def_tol (float):          [Optional] Positive and negligible scalar to force positive definite matrices.
            contraction_rate_lb (float):  [Optional] Lower bound on the contraction rate. Default to 1.
            ren_internal_state_init (torch.Tensor): [Optional] Initial state of the REN. Default to 0 when None.
        """
        super().__init__()

        # Set device
        self.device = target_device or device

        # Validate nn_type early
        valid_nn_types = {"REN", "SSM", "MI"}
        if nn_type not in valid_nn_types:
            raise ValueError(f"nn_type must be one of {valid_nn_types}, got {nn_type}")

        # Pre-process and validate initial conditions
        self.input_init = input_init.clone().detach().reshape(1, -1).to(self.device)
        self.output_init = output_init.clone().detach().reshape(1, -1).to(self.device)

        # Pre-compute dimensions for efficiency
        self.dim_in = self.input_init.shape[-1]
        self.dim_out = self.output_init.shape[-1]
        self.dim_in2 = dim_in2

        # Use default config if none provided, and customize based on parameters
        self.config = config or DWNConfig()

        # Update config with passed parameters for SSM models
        if nn_type in ["SSM", "MI"] and non_linearity is not None:
            self.config.ff = non_linearity

        self.nn_type = nn_type

        # Store all parameters for model creation
        self._model_params = {
            'dim_internal': dim_internal,
            'dim_nl': dim_nl,
            'initialization_std': initialization_std,
            'ren_internal_state_init': ren_internal_state_init,
            'pos_def_tol': pos_def_tol,
            'contraction_rate_lb': contraction_rate_lb,
            'dim_middle': dim_middle,
            'output_amplification': output_amplification,
        }

        # Create model using factory pattern
        self.emme = self._create_model()

        # Store noiseless forward function
        self.noiseless_forward = noiseless_forward

        # Pre-allocate internal variables for better memory management
        self.t: int = 0
        self.last_input: Optional[torch.Tensor] = None
        self.last_output: Optional[torch.Tensor] = None

        # Cache for parameter operations
        self._param_shapes_cache: Optional[Dict] = None
        self._num_params_cache: Optional[int] = None

        # Initialize internal variables
        self.reset()

    def _create_model(self) -> nn.Module:
        """Factory method for creating the appropriate model with proper parameter handling"""

        if self.nn_type == "REN":
            return ContractiveREN(
                dim_in=self.dim_in,
                dim_out=self.dim_out,
                dim_internal=self._model_params['dim_internal'],
                dim_nl=self._model_params['dim_nl'],
                initialization_std=self._model_params['initialization_std'],
                internal_state_init=self._model_params['ren_internal_state_init'],
                pos_def_tol=self._model_params['pos_def_tol'],
                contraction_rate_lb=self._model_params['contraction_rate_lb']
            ).to(self.device)

        elif self.nn_type == "SSM":
            # For SSM, we could potentially use dim_middle, output_amplification, etc.
            # but DeepSSM currently only takes config, so we pass the updated config
            return DeepSSM(self.dim_in, self.dim_out, self.config).to(self.device)

        elif self.nn_type == "MI":
            # For Multi-input, similar to SSM
            return Multi(self.dim_in, self.dim_in2, self.dim_out, self.config).to(self.device)

        else:
            raise ValueError(f"Unknown nn_type: {self.nn_type}")  # Should never reach here due to early validation

    def reset(self) -> None:
        """Reset controller to initial state with optimized tensor operations"""
        self.t = 0
        # More efficient cloning without detach (already detached in init)
        self.last_input = self.input_init.clone()
        self.last_output = self.output_init.clone()

        # Reset model state
        if hasattr(self.emme, 'reset'):
            self.emme.reset()

    def forward(self, t: int, input_t: torch.Tensor, input_t2: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Optimized forward pass of the controller.

        Args:
            t: Current time step
            input_t (torch.Tensor): Input with the size of (batch_size, 1, self.dim_in).
            input_t2: Optional second input for MI models
            # NOTE: when used in closed-loop, "input_t" is the measured states.

        Return:
            y_out (torch.Tensor): Output with (batch_size, 1, self.dim_out).
        """
        # Ensure input is on correct device
        input_t = input_t.to(self.device)
        if input_t2 is not None:
            input_t2 = input_t2.to(self.device)

        # Apply noiseless forward to get noiseless input
        u_noiseless = self.noiseless_forward(
            t=self.t,
            x=self.last_input,
            u=self.last_output
        )

        # Ensure noiseless output is on correct device
        if isinstance(u_noiseless, torch.Tensor):
            u_noiseless = u_noiseless.to(self.device)

        # Reconstruct the noise efficiently
        w_ = input_t - u_noiseless

        # Apply model based on type - optimized branching
        if self.nn_type == "MI":
            output = self.emme(w_, input_t2)
        else:
            output = self.emme(w_)

        # Update internal states efficiently
        self.last_input = input_t
        self.last_output = output
        self.t += 1

        return output

    @property
    def num_params(self) -> int:
        """Cached property for number of parameters"""
        if self._num_params_cache is None:
            self._num_params_cache = sum(p.numel() for p in self.emme.parameters())
        return self._num_params_cache

    def get_parameter_shapes(self) -> Dict:
        """Cached parameter shapes for better performance"""
        if self._param_shapes_cache is None:
            if self.nn_type == 'SSM':
                # Implement for SSM if needed
                if hasattr(self.emme, 'get_parameter_shapes'):
                    self._param_shapes_cache = self.emme.get_parameter_shapes()
                else:
                    self._param_shapes_cache = {
                        name: param.shape for name, param in self.emme.named_parameters()
                    }
            else:
                self._param_shapes_cache = self.emme.get_parameter_shapes()
        return self._param_shapes_cache

    def get_named_parameters(self) -> Dict:
        """Optimized parameter getter"""
        if self.nn_type == 'SSM':
            if hasattr(self.emme, 'get_named_parameters'):
                return self.emme.get_named_parameters()
            else:
                return dict(self.emme.named_parameters())
        return self.emme.get_named_parameters()

    def get_parameters_as_vector(self) -> torch.Tensor:
        """Optimized parameter vectorization using PyTorch instead of numpy"""
        return torch.cat([p.detach().flatten() for p in self.emme.parameters()])

    def set_parameter(self, name: str, value: Union[torch.Tensor, np.ndarray]) -> None:
        """Optimized parameter setting with better type handling"""
        if self.nn_type == 'SSM':
            print("Warning: This function might not work optimally for SSMs")

        current_val = getattr(self.emme, name)

        # More efficient tensor conversion
        if isinstance(value, np.ndarray):
            value = torch.from_numpy(value)
        elif not isinstance(value, torch.Tensor):
            value = torch.tensor(value)

        # Ensure correct device and dtype
        value = value.to(device=self.device, dtype=current_val.dtype)
        value = torch.nn.Parameter(value.reshape(current_val.shape))

        setattr(self.emme, name, value)

        if self.nn_type == 'REN' and hasattr(self.emme, '_update_model_param'):
            self.emme._update_model_param()

        # Clear cache since parameters changed
        self._param_shapes_cache = None
        self._num_params_cache = None

    def set_parameters(self, param_dict: Dict[str, Union[torch.Tensor, np.ndarray]]) -> None:
        """Optimized batch parameter setting"""
        for name, value in param_dict.items():
            self.set_parameter(name, value)

    def set_parameters_as_vector(self, value: torch.Tensor) -> None:
        """Optimized vectorized parameter setting"""
        # Ensure value is a tensor on the correct device
        if isinstance(value, np.ndarray):
            value = torch.from_numpy(value).to(self.device)
        else:
            value = value.to(self.device)

        # Flatten vector if not batched
        if value.numel() == self.num_params:
            value = value.flatten()

        if self.nn_type == 'SSM':
            print("Warning: This function might not work optimally for SSMs")

        idx = 0
        param_shapes = self.get_parameter_shapes()

        with torch.no_grad():  # More efficient context management
            for name, shape in param_shapes.items():
                # More efficient dimension calculation
                dim = torch.tensor(shape).prod().item()
                idx_next = idx + dim

                # More efficient value extraction
                if value.ndim == 1:
                    value_tmp = value[idx:idx_next]
                elif value.ndim == 2:
                    value_tmp = value[:, idx:idx_next]
                elif value.ndim == 3:
                    value_tmp = value[:, :, idx:idx_next]
                else:
                    raise ValueError(f"Unsupported tensor dimension: {value.ndim}")

                # Set parameter more efficiently
                self.set_parameter(name, value_tmp.reshape(shape))
                idx = idx_next

        assert idx == value.shape[-1], f"Parameter vector size mismatch: expected {idx}, got {value.shape[-1]}"

    def to(self, device: torch.device) -> 'PerfBoostController':
        """Enhanced device movement with internal state handling"""
        super().to(device)
        self.device = device

        # Move internal states to new device
        if self.last_input is not None:
            self.last_input = self.last_input.to(device)
        if self.last_output is not None:
            self.last_output = self.last_output.to(device)

        self.input_init = self.input_init.to(device)
        self.output_init = self.output_init.to(device)

        return self

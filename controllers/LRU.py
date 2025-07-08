import math
import torch
import torch.nn as nn
from .scan_utils import associative_scan, binary_operator_diag
import torch.jit as jit


class LRU(nn.Module):
    """ Linear Recurrent Unit. The LRU is simulated using Parallel Scan (fast!) when
     "scan" is set to True (default) in the forward pass, otherwise recursively (slow)."""

    def __init__(
            self, in_features: int, out_features: int, state_features: int, internal_state_init=None, rmin=0.9,
            rmax=1.0, max_phase=6.283
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.state_features = state_features

        # Pre-compute constants for efficiency
        self._sqrt_in_features = math.sqrt(in_features)
        self._sqrt_2_in_features = math.sqrt(2 * in_features)
        self._sqrt_state_features = math.sqrt(state_features)
        self._rmin_rmax_diff = rmax - rmin
        self._rmin_rmax_sum = rmax + rmin
        self._rmin_squared = rmin ** 2

        self.D = nn.Parameter(
            torch.randn([out_features, in_features]) / self._sqrt_in_features
        )

        u1 = torch.rand(state_features)
        u2 = torch.rand(state_features)
        self.nu_log = nn.Parameter(
            torch.log(-0.5 * torch.log(u1 * self._rmin_rmax_sum * self._rmin_rmax_diff + self._rmin_squared))
        )
        self.theta_log = nn.Parameter(torch.log(max_phase * u2))

        lambda_abs = torch.exp(-torch.exp(self.nu_log))
        self.gamma_log = nn.Parameter(
            torch.log(torch.sqrt(1.0 - lambda_abs.square()))  # More efficient than torch.ones_like and torch.square
        )

        # More efficient initialization using single complex tensor creation
        B_complex = torch.complex(
            torch.randn([state_features, in_features]) / self._sqrt_2_in_features,
            torch.randn([state_features, in_features]) / self._sqrt_2_in_features
        )
        self.B = nn.Parameter(B_complex)  # N, U

        C_complex = torch.complex(
            torch.randn([out_features, state_features]) / self._sqrt_state_features,
            torch.randn([out_features, state_features]) / self._sqrt_state_features
        )
        self.C = nn.Parameter(C_complex)  # H, N

        # initialize internal state
        self.state = None

        # Pre-compute transformation matrices for ss_real_matrices method
        self._T_block = None
        self._T_block_inv = None

    def ss_params(self):
        lambda_abs = torch.exp(-torch.exp(self.nu_log))
        lambda_phase = torch.exp(self.theta_log)

        # More efficient complex number creation
        lambdas = lambda_abs * torch.exp(1j * lambda_phase)
        gammas = torch.exp(self.gamma_log).unsqueeze(-1)
        B = gammas * self.B
        return lambdas, B, self.C, self.D

    def ss_real_matrices(self, to_numpy=True):
        lambdas, B, C, D = self.ss_params()

        # Pre-allocate with correct dtype and device
        device, dtype = lambdas.device, lambdas.dtype
        state_features_2 = 2 * self.state_features

        # More efficient tensor creation using stack instead of manual indexing
        lambdas_conjugate = torch.stack([lambdas, lambdas.conj()], dim=1).flatten()
        A_full = torch.diag(lambdas_conjugate)

        # More efficient B_full creation
        B_conjugate = torch.stack([B, B.conj()], dim=1).view(state_features_2, self.in_features)

        # More efficient C_full creation
        C_half = 0.5 * C
        C_conjugate = torch.stack([C_half, C_half.conj()], dim=2).view(self.out_features, state_features_2)

        # Cache transformation matrices to avoid recomputation
        if self._T_block is None or self._T_block.device != device:
            self._T_block = torch.tensor([[1, 1], [1j, -1j]], device=device, dtype=dtype)
            self._T_block_inv = torch.linalg.inv(self._T_block)

        T_full = torch.block_diag(*([self._T_block] * self.state_features))
        T_full_inv = torch.block_diag(*([self._T_block_inv] * self.state_features))

        # More efficient matrix operations using @ operator consistently
        A_real = (T_full @ A_full @ T_full_inv).real
        B_real = (T_full @ B_conjugate).real
        C_real = (C_conjugate @ T_full_inv).real
        D_real = D

        ss_real_params = [A_real, B_real, C_real, D_real]
        if to_numpy:
            ss_real_params = [param.detach().cpu().numpy() for param in ss_real_params]

        return tuple(ss_real_params)

    def forward_loop(self, input, state=None):
        batch_size = input.shape[0]

        # More efficient state management
        if self.state is None or self.state.shape[0] != batch_size:
            self.state = torch.zeros(batch_size, self.state_features,
                                   device=input.device, dtype=torch.complex64)

        lambdas, B, C, D = self.ss_params()

        # More efficient state computation using pre-converted input
        input_B_dtype = input.to(B.dtype)
        B_T = B.mT  # Cache transpose

        # Optimized loop with pre-allocated tensor for states
        seq_len = input.shape[1]
        states = torch.empty(batch_size, seq_len, self.state_features,
                           device=input.device, dtype=torch.complex64)

        # Vectorized state updates - much more efficient
        for t, u_step in enumerate(input_B_dtype.unbind(dim=1)):
            self.state = lambdas * self.state + u_step @ B_T
            states[:, t] = self.state

        # More efficient output computation using all states
        output = (states @ C.mT).real + input @ D.T

        return output, states

    @torch.compiler.disable
    def forward_scan(self, input, state=None):
        lambdas, B, C, D = self.ss_params()

        # More efficient lambda tiling
        lambda_elements = lambdas.unsqueeze(0).expand(input.shape[1], -1)

        # Pre-compute input transformation
        Bu_elements = input.to(B.dtype) @ B.mT

        if state is not None:
            Bu_elements[:, 0, :] += lambdas * state

        # More efficient vmap usage with cleaner lambda
        def scan_fn(Bu_seq):
            return associative_scan(binary_operator_diag, (lambda_elements, Bu_seq))[1]

        inner_states = torch.vmap(scan_fn)(Bu_elements)

        # More efficient state expansion and concatenation
        if state is not None:
            state_expanded = state.unsqueeze(1).expand(-1, 1, -1)
            inner_states = torch.cat([state_expanded, inner_states], dim=1)[:, :-1, :]
        else:
            # Handle case where state is None to avoid uninitialized variable warning
            zero_state = torch.zeros(inner_states.shape[0], 1, inner_states.shape[2],
                                   device=inner_states.device, dtype=inner_states.dtype)
            inner_states = torch.cat([zero_state, inner_states], dim=1)[:, :-1, :]

        # More efficient output computation
        y = (inner_states @ C.mT).real + input @ D.T
        return y, inner_states

    def forward(self, input, gamma=None, state=None, mode="loop"):
        if state is None:
            state = torch.zeros(self.state_features, dtype=torch.complex64, device=input.device)

        if mode == "scan":
            return self.forward_scan(input, state)
        elif mode in ["loop", "loop_efficient"]:
            return self.forward_loop(input, state)
        else:
            raise ValueError(f"Unknown mode: {mode}. Expected 'scan', 'loop', or 'loop_efficient'.")

    def forward_step(self, input_step, state=None):
        """
        Process a single timestep efficiently for sequential processing.

        Args:
            input_step: (B, H) - single timestep input
            state: optional state from previous timestep

        Returns:
            output_step: (B, out_features) - single timestep output
            new_state: updated state for next timestep
        """
        batch_size = input_step.shape[0]

        # Initialize or validate state
        if self.state is None or self.state.shape[0] != batch_size:
            self.state = torch.zeros(batch_size, self.state_features,
                                   device=input_step.device, dtype=torch.complex64)

        # If external state provided, use it
        if state is not None:
            self.state = state

        lambdas, B, C, D = self.ss_params()

        # Update state for single timestep - more efficient
        input_B_dtype = input_step.to(B.dtype)
        self.state = lambdas * self.state + input_B_dtype @ B.mT

        # Compute output for single timestep
        output_step = (self.state @ C.mT).real + input_step @ D.T

        return output_step, self.state.clone()

    def reset(self):
        self.state = None  # reset the SSM state to the initial value


# WORK IN PROGRESS

class LRU_Robust(jit.ScriptModule):
    """ Implements a Linear Recurrent Unit (LRU) with trainable or prescribed l2 gain gamma.
    No parallel scan implementation available at the moment. """

    def __init__(self, state_features: int, trainable: bool):
        super().__init__()
        self.state_features = state_features
        self.register_buffer('state', torch.zeros(state_features))
        self.register_buffer('ID', torch.eye(state_features))

        self.alpha = nn.Parameter(torch.tensor(4.1))  # controls the initialization of the matrix A:
        # the larger the alpha at initialization, the closer the eigenvalues of A will be
        # to the boundary of the unitary circle at initialization. This helps the SSM to obtain long memory properties.

        self.gamma = nn.Parameter(torch.tensor(33.0))  # l2 gain - more efficient initialization
        self.epsilon = nn.Parameter(torch.tensor(-99.9))  # Regularization

        self.Skew = nn.Parameter(0.01 * torch.randn(state_features, state_features))

        # Define each block of X as a parameter
        self.X11 = nn.Parameter(torch.eye(state_features))
        self.X22 = nn.Parameter(0.01 * torch.eye(state_features))
        self.X21 = nn.Parameter(torch.eye(state_features))

        self.C = nn.Parameter(torch.eye(state_features))
        self.D = nn.Parameter(torch.eye(state_features))

        # self.X11 = nn.Parameter(torch.randn(state_features, state_features))
        # self.X22 = nn.Parameter(torch.randn(state_features, state_features))
        # self.X21 = nn.Parameter(torch.randn(state_features, state_features))
        #
        # self.C = nn.Parameter(torch.randn(state_features, state_features))
        # self.D = nn.Parameter(torch.randn(state_features, state_features))

    @jit.script_method
    def set_param(self):  # Parameter update for l2 gain (free param)

        gamma = self.gamma
        # Auxiliary Parameters
        X11 = self.X11
        X22 = self.X22

        Sk = self.Skew - self.Skew.T
        Q = (self.ID - Sk) @ torch.linalg.inv(self.ID + Sk)
        Z = self.X21 @ self.X21.T + X22 @ X22.T + self.D.T @ self.D + torch.exp(self.epsilon) * self.ID
        beta = gamma ** 2 * torch.sigmoid(self.alpha) / torch.norm(Z, 2)
        H11 = X11 @ X11.T + self.C.T @ self.C + beta * torch.exp(self.epsilon) * self.ID
        H12 = torch.sqrt(beta) * (X11 @ self.X21.T + self.C.T @ self.D)
        V = Z * beta - gamma ** 2 * self.ID
        R = H12 @ torch.linalg.inv(V.T) @ H12.T
        CR = torch.linalg.cholesky(-R)
        CRH = torch.linalg.cholesky(-R + H11)

        # Parameters

        A = torch.linalg.inv(CRH).T @ Q @ CR.T
        B = A @ torch.linalg.inv(H12.T) @ V.T
        C = self.C
        D = torch.sqrt(beta) * self.D

        # # Assemble H*H.T in block form
        # HHt = torch.cat([
        #     torch.cat([HHt_11, HHt_12], dim=1),
        #     torch.cat([HHt_21, HHt_22n], dim=1)
        # ], dim=0)
        #P = -Atilde @ R @ Atilde.T
        #la= torch.abs(torch.linalg.eigvals(A))
        # lp = torch.linalg.eigvals(self.P)
        # row1 = torch.cat([-A.T@P@ A+P, -A.T@P@B], dim=1)
        # row2 = torch.cat([-(A.T@P@B).T, -B.T@P@B+(gamma**2*self.ID)], dim=1)
        # M = torch.cat([row1, row2], dim=0)
        # eigs = torch.linalg.eigvals(M)

        #eigs
        return A, B, C, D

    #state: Optional[torch.Tensor]
    @jit.script_method
    def forward(self, input, state=None, gamma=None, mode: str = "scan"):
        # Input size: (B, L, H)
        state = torch.zeros(self.state_features, device=self.C.device)
        A, B, C, D = self.set_param()
        if input.dim() == 1:
            input = input.unsqueeze(0).unsqueeze(0)
        # output = torch.empty(
        #     [i for i in input.shape[:-1]] + [self.state_features], device=self.C.device
        # )

        states = []
        for u_step in input.split(1, dim=1):  # 1 is the time dimension

            u_step = u_step.squeeze(1)
            state = state @ A.T + u_step @ B.T
            states.append(state)

        states = torch.stack(states, 1)
        output = states @ C.mT + input @ D.T
        return output, states

    @jit.script_method
    def forward_step(self, input_step, state=None):
        """
        Process a single timestep efficiently for sequential processing.

        Args:
            input_step: (B, H) - single timestep input
            state: optional state from previous timestep

        Returns:
            output_step: (B, out_features) - single timestep output
            new_state: updated state for next timestep
        """
        if state is None:
            state = torch.zeros(self.state_features, device=input_step.device)

        A, B, C, D = self.set_param()

        # Handle batch dimension if present
        if input_step.dim() == 1:
            input_step = input_step.unsqueeze(0)

        # Update state for single timestep
        new_state = state @ A.T + input_step @ B.T

        # Compute output for single timestep
        output_step = new_state @ C.mT + input_step @ D.T

        return output_step, new_state

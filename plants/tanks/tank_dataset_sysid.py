import matplotlib.pyplot as plt
import torch
import os

def generate_trajectories_dataset(horizon=200, num_train=400, num_val=200, std_noise=0.003, save_dir=None):
    """
    Generate training and validation datasets of trajectories using a TankSystem.

    Parameters:
        horizon (int): Number of time steps per trajectory.
        num_train (int): Number of training trajectories.
        num_val (int): Number of validation trajectories.
        std_noise (float): Standard deviation of the additive white Gaussian noise.
        save_dir (str or None): If provided, the trajectories will be saved in this directory.

    Returns:
        train_trajectories (dict): Dictionary containing training inputs 'u' and states 'x'.
        val_trajectories (dict): Dictionary containing validation inputs 'u' and states 'x'.
    """

    # Define the provided TankSystem class
    class TankSystem(torch.nn.Module):
        def __init__(self, x_init=None, a: float = 0.6):
            super().__init__()
            self.x_init = torch.tensor(1.1) if x_init is None else x_init.reshape(1, -1)
            self.state_dim = 1
            self.in_dim = 1
            self.h = 0.1
            self.a = a
            self.b = 0.5

        def dynamics(self, x, u):
            f = 1 / (self.b + x) * (-self.a * torch.sqrt(x) + self.a * torch.sqrt(u))
            return f

        def noiseless_forward(self, x: torch.Tensor, u: torch.Tensor):
            f = self.dynamics(x, u)
            x_ = x + self.h * f
            return x_

        def forward(self, x, u, w):
            x_plus = self.noiseless_forward(x, u) + w.view(-1, 1, self.state_dim)
            return torch.relu(x_plus), x_plus

        def simulate(self, u, w):
            horizon = u.shape[1]
            y_traj = []
            x = self.x_init  # Broadcasts to batch size
            for t in range(horizon):
                x, _ = self.forward(x, u[:, t:t + 1, :], w[:, t:t + 1, :])
                y_traj.append(x)
            y_out = torch.cat(y_traj, dim=1)
            return y_out

    # Create a tank system instance
    tank_system = TankSystem()

    # Function to generate piecewise constant inputs
    def generate_piecewise_constant_inputs(num_traj, horizon, num_segments=5, min_val=0.0, max_val=2.0):
        seg_len = horizon // num_segments
        u_segments = torch.rand(num_traj, num_segments, 1) * (max_val - min_val) + min_val
        u = torch.repeat_interleave(u_segments, seg_len, dim=1)
        return u

    # Function to generate sinusoidal inputs with random phase
    def generate_sinusoidal_inputs(num_traj, horizon, omega=2 * torch.pi / 40, amplitude=1.0):
        t = torch.arange(horizon).float()
        phase = torch.rand(num_traj, 1, 1) * 2 * torch.pi
        u = amplitude * (1 + torch.sin(omega * t[None, :, None] + phase))
        return u

    # Function to generate exotic inputs for validation trajectories
    def generate_exotic_inputs(num_traj, horizon):
        t = torch.linspace(0, 1, horizon)
        inputs = torch.zeros(num_traj, horizon, 1)
        for i in range(num_traj):
            choice = torch.randint(0, 3, (1,)).item()
            if choice == 0:
                # Chirp signal: frequency increases linearly
                freq_start = 1.0
                freq_end = 10.0
                freqs = freq_start + (freq_end - freq_start) * t
                phase = torch.rand(1) * 2 * torch.pi
                signal = torch.sin(2 * torch.pi * freqs * t + phase)
            elif choice == 1:
                # Sum of two sinusoids with different frequencies and phases
                phase1 = torch.rand(1) * 2 * torch.pi
                phase2 = torch.rand(1) * 2 * torch.pi
                signal = 0.5 * torch.sin(2 * torch.pi * 3 * t + phase1) + \
                         0.5 * torch.sin(2 * torch.pi * 7 * t + phase2)
            else:
                # Modulated square wave: square wave modulated by a slow sine
                square_wave = torch.sign(torch.sin(2 * torch.pi * 5 * t))
                modulator = 0.5 * torch.sin(2 * torch.pi * 0.5 * t) + 0.5
                signal = square_wave * modulator

            # Scale signal to be in the range [0, 2]
            signal_min = signal.min()
            signal_max = signal.max()
            signal = (signal - signal_min) / (signal_max - signal_min + 1e-6) * 2.0
            inputs[i, :, 0] = signal
        return inputs

    # Generate training inputs: half piecewise, half sinusoidal
    u_train_piecewise = generate_piecewise_constant_inputs(num_train // 2, horizon)
    u_train_sinusoidal = generate_sinusoidal_inputs(num_train // 2, horizon)
    u_train = torch.cat([u_train_piecewise, u_train_sinusoidal], dim=0)

    # Generate validation inputs using the exotic input generator
    u_val = generate_exotic_inputs(num_val, horizon)

    # Generate additive white Gaussian noise
    w_train = torch.normal(0, std_noise, size=(num_train, horizon, 1))
    w_val = torch.normal(0, std_noise, size=(num_val, horizon, 1))

    # Simulate trajectories
    x_train = tank_system.simulate(u_train, w_train)
    x_val = tank_system.simulate(u_val, w_val)

    train_trajectories = {'u': u_train, 'x': x_train}
    val_trajectories = {'u': u_val, 'x': x_val}

    # Save trajectories if a save directory is provided
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        torch.save(train_trajectories, os.path.join(save_dir, 'train_trajectories.pt'))
        torch.save(val_trajectories, os.path.join(save_dir, 'val_trajectories.pt'))
        print(f"Trajectories saved in directory: {save_dir}")
    else:
        torch.save(train_trajectories, 'train_trajectories.pt')
        torch.save(val_trajectories, 'val_trajectories.pt')
        print("Trajectories saved as 'train_trajectories.pt' and 'val_trajectories.pt'")

    # Optional: Plot a few sample trajectories from the training data
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    sample_indices = [0, 1, num_train // 2, num_train // 2 + 1]
    time = torch.arange(horizon)

    for i, idx in enumerate(sample_indices):
        ax = axes[i // 2, i % 2]
        ax.plot(time, x_train[idx, :, 0], label='State $x$')
        ax.plot(time, u_train[idx, :, 0], label='Input $u$', linestyle='--')
        ax.set_title(f'Trajectory {idx + 1}')
        ax.set_xlabel('Time step k')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.show()

    return train_trajectories, val_trajectories
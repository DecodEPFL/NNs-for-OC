import torch
from plants.custom_dataset import CustomDataset


def generate_points_outside_circle(vertices, center, radius, max_attempts=10_000, N=1):
    """
    Generates N random points inside a square (defined by 4 vertices)
    but excluding points inside a given circle.

    Args:
        N (int): Number of desired points.
        vertices (Tensor): Shape (4, 2), 4 vertices of the square in order.
        center (Tensor): Shape (2,), center of the circle.
        radius (float): Radius of the exclusion circle.
        max_attempts (int): Safety limit to avoid infinite loops.

    Returns:
        Tensor: Shape (N, 2), the sampled points.
    """
    # Convert square to axis-aligned bounding box
    x_min, _ = torch.min(vertices[:, 0], dim=0)
    x_max, _ = torch.max(vertices[:, 0], dim=0)
    y_min, _ = torch.min(vertices[:, 1], dim=0)
    y_max, _ = torch.max(vertices[:, 1], dim=0)

    accepted = []
    attempts = 0

    while len(accepted) < N and attempts < max_attempts:
        # Generate in bulk
        batch_size = (N - len(accepted)) * 2
        samples = torch.empty(batch_size, 2).uniform_(0, 1)
        samples[:, 0] = samples[:, 0] * (x_max - x_min) + x_min
        samples[:, 1] = samples[:, 1] * (y_max - y_min) + y_min

        # Distance to circle center
        dist_sq = torch.sum((samples - center) ** 2, dim=1)
        mask = dist_sq >= radius ** 2

        # Append valid points
        valid = samples[mask]
        accepted.extend(valid.tolist())

        attempts += 1

    if len(accepted) < N:
        raise RuntimeError(f"Only generated {len(accepted)} points after {max_attempts} attempts.")

    return torch.tensor(accepted[:N])


class RobotsDataset(CustomDataset):
    def __init__(self, random_seed, horizon, x0=torch.tensor([2., .5, 0, 0]), std_ini=0.2, n_agents=1):
        # experiment and file names
        exp_name = 'robot'
        file_name = 'data_T' + str(horizon) + '_stdini' + str(std_ini) + '_agents' + str(n_agents) + '_RS' + str(
            random_seed) + '.pkl'

        super().__init__(random_seed=random_seed, horizon=horizon, exp_name=exp_name, file_name=file_name)

        self.std_ini = std_ini
        self.n_agents = n_agents

        # initial state TODO: set as arg
        self.x0 = x0
        self.xbar = torch.zeros(4)

    # ---- data generation ----
    def _generate_data(self, num_samples):
        state_dim = 4 * self.n_agents
        data = torch.zeros(num_samples, self.horizon, state_dim)
        for rollout_num in range(num_samples):
            data[rollout_num, 0, :] = \
                (self.x0 - self.xbar) + self.std_ini * torch.randn(self.x0.shape)

        assert data.shape[0] == num_samples
        return data


class RobotsDatasetMulti(CustomDataset):
    def __init__(self, random_seed, horizon, x0=torch.tensor([2., .5, 0, 0]), std_ini=0.2, n_agents=1):
        # experiment and file names
        exp_name = 'robot'
        file_name = 'data_T' + str(horizon) + '_stdini' + str(std_ini) + '_agents' + str(n_agents) + '_RS' + str(
            random_seed) + '.pkl'

        super().__init__(random_seed=random_seed, horizon=horizon, exp_name=exp_name, file_name=file_name)

        self.std_ini = std_ini
        self.n_agents = n_agents

        # initial state TODO: set as arg
        self.x0 = x0
        self.xbar = torch.zeros(4)

    # ---- data generation ----
    def _generate_data(self, num_samples):
        a = torch.tensor([-1.0, -1.0])
        b = torch.tensor([3.0, -1.0])
        c = torch.tensor([-1.0, 3.0])
        d = torch.tensor([3.0, 3.0])
        state_dim = 4 * self.n_agents
        data = torch.zeros(num_samples, self.horizon, state_dim)
        for rollout_num in range(num_samples):
            p0 = generate_points_outside_circle(vertices=torch.stack([a, b, c, d]), radius=0.5,
                                                center=torch.tensor([[1., 0.5]]))
            pr = torch.tensor([p0[0, 0], p0[0, 1], 0, 0])
            data[rollout_num, 0, :] = \
                (pr - self.xbar)

        assert data.shape[0] == num_samples
        return data

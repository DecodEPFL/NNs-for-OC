import torch
from plants.custom_dataset import CustomDataset

def generate_fixed_center_circle_and_point(
    center=torch.tensor([1.0, 0.5]),
    square_bounds=(-5, 5),
    min_radius=0.5,
    max_radius=2.5,
    max_attempts=1000
):
    """
    Generate a circle with fixed center and random radius such that:
    - The circle lies within a square
    - The circle does not contain or touch the origin
    - A point is sampled inside the square and outside the circle
    - The line from the origin to the point intersects the circle (if possible)

    Returns:
        center: Tensor of shape (2,)
        radius: float
        point: Tensor of shape (2,)
    """
    min_val, max_val = square_bounds
    cx, cy = center.tolist()

    for _ in range(max_attempts):
        # Sample radius and compute its square
        r = torch.empty(1).uniform_(min_radius, max_radius).item()
        r2 = r * r

        # Check that the circle stays inside the square
        if not (min_val + r <= cx <= max_val - r and min_val + r <= cy <= max_val - r):
            continue

        # Check that it does not contain or touch the origin
        dist2_to_origin = cx * cx + cy * cy
        if dist2_to_origin <= r2:
            continue

        # Try to find a point outside the circle such that the line origin→point intersects the circle
        for _ in range(max_attempts):
            pt = torch.empty(2).uniform_(min_val, max_val)
            dx = pt[0].item() - cx
            dy = pt[1].item() - cy
            if dx * dx + dy * dy <= r2:
                continue  # point is inside or on the circle

            # Project circle center onto the line from origin to point
            denom = torch.dot(pt, pt).item()
            if denom == 0:
                continue  # point is at origin

            t = (cx * pt[0].item() + cy * pt[1].item()) / denom
            if not (0.0 < t < 1.0):
                continue  # projection is not on the segment

            # Compute closest point on line, check distance to circle center
            closest_x = t * pt[0].item()
            closest_y = t * pt[1].item()
            ddx = cx - closest_x
            ddy = cy - closest_y
            dist2_to_line = ddx * ddx + ddy * ddy

            if dist2_to_line <= r2:
                return center, r, pt

    raise RuntimeError("Failed to generate a valid circle + intersecting point.")

def generate_random_circles_and_points(
        square_bounds=(-5, 5),
        min_radius=0.5,
        max_radius=2.5,
        max_attempts=1000
):
    """
    Generate one random circle inside a square and a point inside the square but outside the circle,
    such that the line segment from (0,0) to that point intersects the circle. Also the circle must
    not contain the origin.

    Returns:
        center: Tensor of shape (2,)
        radius: float
        point: Tensor of shape (2,)
    """
    min_val, max_val = square_bounds
    origin = torch.zeros(2)

    for _ in range(max_attempts):
        # 1) Sample a radius
        r = torch.empty(1).uniform_(min_radius, max_radius).item()

        # 2) Sample a center so the circle stays fully inside the square
        cx = torch.empty(1).uniform_(min_val + r, max_val - r).item()
        cy = torch.empty(1).uniform_(min_val + r, max_val - r).item()
        center = torch.tensor([cx, cy])

        # 3) Reject if circle would contain the origin
        if torch.norm(center) < r:
            continue

        # 4) Now sample points until we find one that:
        #    a) lies outside the circle, and
        #    b) the segment origin→point intersects the circle
        for _ in range(max_attempts):
            pt = torch.empty(2).uniform_(min_val, max_val)
            # must be outside circle
            if torch.norm(pt - center) < r:
                continue

            # projection parameter of center onto line through origin→pt
            # t = dot(center, pt) / dot(pt, pt)
            denom = torch.dot(pt, pt)
            if denom == 0:
                continue
            t = torch.dot(center, pt) / denom

            # compute distance from center to the line
            closest = t * pt
            dist_to_line = torch.norm(center - closest)

            # intersects iff 0 < t < 1 and dist_to_line <= r
            if 0.0 < t < 1.0 and dist_to_line <= r:
                return center, r, pt

    raise RuntimeError("Failed to generate a valid circle+point after many attempts.")

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


class RobotsDatasetMultiCircle(CustomDataset):
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
        circles = []
        for rollout_num in range(num_samples):
            p0 = generate_fixed_center_circle_and_point()
            pr = torch.tensor([p0[2][0], p0[2][1], 0, 0])
            data[rollout_num, 0, :] = \
                (pr - self.xbar)
            circles.append(torch.cat((p0[0], torch.tensor(p0[1]).unsqueeze(0))))

        assert data.shape[0] == num_samples

        # Process: concatenate each pair to a single 1D tensor of length 3
        stacked = torch.stack(circles)  # shape: (N, 3)
        finalC = stacked.unsqueeze(1)  # shape: (N, 1, 3)
        finalC = finalC.repeat(1, self.horizon, 1)
        Final_data = torch.cat((data, finalC), dim=2)

        return Final_data

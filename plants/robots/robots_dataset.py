import torch
from plants.custom_dataset import CustomDataset
from torch.utils.data import Dataset
import random
import math


# Assume your other generator functions (v2 and 4-way) are available
# from your_data_generation_file import generate_four_way_symmetrical_points

def generate_remedial_data(
        num_samples: int,
        fixed_center: torch.Tensor,
        min_radius: float,
        max_radius: float,
        remedial_data_ratio: float = 0.5,
        critical_angle_spread: float = math.pi / 2,
        critical_radial_extension: float = 1.5,
        square_bounds=(-3, 3)
):
    """
    Generates a mixed dataset for fine-tuning on a FIXED obstacle CENTER
    but with VARYING radii.

    - "Remedial" samples have a random radius and a start point in the dynamic
      critical zone relative to that circle.
    - "General" samples have a random radius and use the 4-way symmetrical method.
    """
    dataset = []
    num_remedial_samples = int(num_samples * remedial_data_ratio)
    num_general_samples = num_samples - num_remedial_samples

    print(f"Generating remedial dataset for fixed center {fixed_center.tolist()}...")
    print(f" - {num_remedial_samples} will be targeted remedial samples.")
    print(f" - {num_general_samples} will be general symmetrical samples.")

    # --- Part 1: Generate bespoke "hard" samples ---
    for _ in range(num_remedial_samples):
        # 1. Generate a random radius for the fixed center.
        radius = torch.empty(1).uniform_(min_radius, max_radius).item()

        # 2. Define the critical region relative to THIS specific circle.
        angle_to_center = torch.atan2(fixed_center[1], fixed_center[0])
        dist_to_center = torch.norm(fixed_center)

        min_rho = dist_to_center + radius
        max_rho = min_rho + critical_radial_extension


        min_phi = angle_to_center - (critical_angle_spread / 2)
        max_phi = angle_to_center + (critical_angle_spread / 2)
        # a fixed "upward" direction.
        target_angle_center = math.pi / 2  # 90 degrees = straight up

        # The angle spread determines the width of the "above" sector.
        # min_phi = target_angle_center - (critical_angle_spread / 2)
        # max_phi = target_angle_center + (critical_angle_spread / 2)

        # 3. Sample a point directly from this region.
        rho = torch.empty(1).uniform_(min_rho, max_rho)
        phi = torch.empty(1).uniform_(min_phi, max_phi)
        point = torch.tensor([rho * torch.cos(phi), rho * torch.sin(phi)])

        dataset.append({
            'center': fixed_center,
            'radius': radius,
            'start_point': point
        })

    # --- Part 2: Generate general symmetrical data for the rest ---
    general_generated = 0
    while general_generated < num_general_samples:
        # Use our new tweaked generator function
        center, radius, points_list = generate_four_way_symmetrical_points(
            center=fixed_center,
            min_radius=min_radius,
            max_radius=max_radius,
            square_bounds=square_bounds
        )
        for point in points_list:
            if general_generated >= num_general_samples:
                break
            dataset.append({'center': center, 'radius': radius, 'start_point': point})
            general_generated += 1

    random.shuffle(dataset)
    return dataset[:num_samples]


def generate_four_way_symmetrical_points(
        **kwargs
):
    """
    Generates a single fixed circle and a set of up to four symmetrical starting points.
    """
    max_attempts = kwargs.get('max_attempts', 100)
    square_bounds = kwargs.get('square_bounds', (-5, 5))
    min_val, max_val = square_bounds

    for _ in range(max_attempts):
        center, radius, p1 = generate_fixed_center_circle_and_point_v2(**kwargs)
        points_to_check = [
            p1, p1 * torch.tensor([-1.0, 1.0]),
                p1 * torch.tensor([1.0, -1.0]), p1 * torch.tensor([-1.0, -1.0])
        ]
        valid_points = []
        for pt in points_to_check:
            in_bounds = (min_val <= pt[0] <= max_val and min_val <= pt[1] <= max_val)
            outside_circle = torch.norm(pt - center) > radius
            if in_bounds and outside_circle:
                is_duplicate = any(torch.allclose(pt, vp) for vp in valid_points)
                if not is_duplicate:
                    valid_points.append(pt)
        if len(valid_points) >= 2:
            return center, radius, valid_points

    raise RuntimeError("Failed to generate a valid symmetrical point set.")


def generate_symmetrical_points_for_fixed_circle(
        **kwargs
):
    """
    Generates a single fixed circle and two symmetrical starting points for the robot.

    This function is designed to de-bias the training data. For a single obstacle,
    it provides two starting points, one on each side (flipped across the y-axis).
    This forces the neural controller to learn how to navigate around the *same*
    obstacle from different starting conditions, preventing it from learning a
    biased, one-sided maneuver.

    It validates that the flipped point is also a valid starting position (i.e.,
    not inside the circle).

    Args:
        **kwargs: Keyword arguments to be passed to the underlying generator
                  `generate_fixed_center_circle_and_point_v2`. This includes
                  `center`, `square_bounds`, `min_radius`, etc.

    Returns:
        A tuple of: (center, radius, point1, point2)
        - center (Tensor): The (2,) coordinate tensor of the circle's center.
        - radius (float): The radius of the circle.
        - point1 (Tensor): The primary generated starting point.
        - point2 (Tensor): The symmetrical (flipped) starting point.
    """
    max_attempts = kwargs.get('max_attempts', 100)
    square_bounds = kwargs.get('square_bounds', (-5, 5))

    for _ in range(max_attempts):
        # Step 1: Generate a valid circle and our primary starting point
        center, radius, point1 = generate_fixed_center_circle_and_point_v2(**kwargs)

        # Step 2: Create the symmetrical point by flipping the x-coordinate
        point2 = point1 * torch.tensor([-1.0, 1.0])

        # Step 3: **Crucial Validation** - Ensure the flipped point is also valid
        # Check 1: Is the flipped point still within the square bounds?
        in_bounds = (square_bounds[0] <= point2[0] <= square_bounds[1] and
                     square_bounds[0] <= point2[1] <= square_bounds[1])

        # Check 2: Is the flipped point outside the obstacle?
        outside_circle = torch.norm(point2 - center) > radius

        if in_bounds and outside_circle:
            # Both points are valid, we are done!
            return center, radius, point1, point2

    # If the loop finishes, we failed to find a valid pair. This can happen if
    # the circle is not centered on the y-axis and flipping the point makes it invalid.
    raise RuntimeError("Failed to generate a valid symmetrical point pair. "
                       "Try moving the circle center closer to the y-axis.")


def generate_fixed_center_circle_and_point_v2(
        center=torch.tensor([1.0, 0.5]),
        square_bounds=(-5, 5),
        min_radius=0.2,
        max_radius=2.5,
        opposite_side_prob=0.6,  # <-- NEW: Probability of generating a point on the "opposite side"
        max_attempts=100
):
    """
    Generates a valid circle and a starting point for the robot with more control.

    A circle is generated with a fixed center and random radius such that:
    - The circle lies fully within the specified square bounds.
    - The circle does not contain or touch the origin.

    A point is then generated with two distinct strategies:
    1.  With probability `opposite_side_prob`: The point is generated in a cone-shaped
        region on the far side of the circle, as viewed from the origin. This ensures
        the circle is between the origin and the point.
    2.  With probability `1 - opposite_side_prob`: The point is generated anywhere else
        in the square, as long as it's outside the circle.

    This approach is more robust and provides a richer dataset for training.

    Returns:
        center: Tensor of shape (2,)
        radius: float
        point: Tensor of shape (2,)
    """
    min_val, max_val = square_bounds
    cx, cy = center.tolist()

    # --- Step 1: Generate a valid circle ---
    # We use a loop that is almost guaranteed to succeed quickly.
    for _ in range(max_attempts):
        r = torch.empty(1).uniform_(min_radius, max_radius).item()

        # Check if the circle is fully inside the square bounds
        if not (min_val + r <= cx <= max_val - r and min_val + r <= cy <= max_val - r):
            continue

        # Check if the circle contains the origin
        if torch.norm(center) <= r:
            continue

        # Found a valid circle, break the loop
        break
    else:
        # This will rarely happen with reasonable parameters
        raise RuntimeError("Failed to generate a valid circle within the given constraints.")

    # --- Step 2: Generate a point based on the desired strategy ---
    point = None

    # Decide which strategy to use
    if torch.rand(1).item() < opposite_side_prob:
        # --- STRATEGY A: Generate on the "opposite side" (Constructive Method) ---
        # This is the majority case. We construct the point directly.

        # 1. Define the "opposite side" cone using polar coordinates.
        angle_to_center = torch.atan2(center[1], center[0])
        dist_to_center = torch.norm(center)

        # Define the angular spread of the cone (e.g., +/- 45 degrees)
        angle_spread = torch.pi / 4

        # 2. Sample a random angle and radius within this cone.
        phi = torch.empty(1).uniform_(angle_to_center - angle_spread, angle_to_center + angle_spread)

        # The point must be beyond the far edge of the circle.
        min_rho = dist_to_center + r
        # The max distance is roughly the corner of the square.
        max_rho = max(abs(min_val), abs(max_val))
        rho = torch.empty(1).uniform_(min_rho, max_rho)

        # 3. Convert back to Cartesian coordinates.
        px = rho * torch.cos(phi)
        py = rho * torch.sin(phi)
        point = torch.tensor([px, py])

        # Ensure the point is within the square bounds as a final check
        point.clamp_(min=min_val, max=max_val)

    else:
        # --- STRATEGY B: Generate anywhere else (Rejection Sampling) ---
        # This is the minority case. Simple rejection sampling is efficient here.
        for _ in range(max_attempts):
            pt_candidate = torch.empty(2).uniform_(min_val, max_val)

            # Check if the point is outside the circle
            if torch.norm(pt_candidate - center) > r:
                point = pt_candidate
                break

        if point is None:
            raise RuntimeError("Failed to generate a valid 'other side' point.")

    return center, r, point


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
            p0 = generate_fixed_center_circle_and_point_v2()
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


class RobotsDatasetMultiCircle_v2(CustomDataset):
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

        # Generate the data upon initialization
        # In a real use case, you might wrap this in a check to load from file first
        self.data = self._generate_data(num_samples=10000)  # Example: generate 10k samples

    # ---- data generation ----
    def _generate_data(self, num_samples):
        """
        Generates a balanced dataset using the 4-way symmetrical point strategy.
        """
        state_dim = 4 * self.n_agents

        # We will collect data in python lists first, as the number of points
        # per scenario is variable. This is more flexible than pre-allocating.
        initial_states_list = []
        circles_list = []

        print(f"Generating approximately {num_samples} balanced training samples...")

        # Loop until we have enough samples
        while len(initial_states_list) < num_samples:
            # Generate one scenario: one circle and a set of symmetrical points
            center, radius, points_list = generate_four_way_symmetrical_points()

            # For each valid symmetrical point, create one training sample
            for point in points_list:
                # Construct the initial state vector [px, py, 0, 0]
                pr = torch.tensor([point[0], point[1], 0, 0])
                initial_state = pr - self.xbar
                initial_states_list.append(initial_state)

                # The circle info is the same for all points in this set
                circle_info = torch.cat((center, torch.tensor(radius).unsqueeze(0)))
                circles_list.append(circle_info)

                # Stop if we have generated enough samples
                if len(initial_states_list) >= num_samples:
                    break

        # --- Post-processing: Convert lists to final tensors ---

        # 1. Truncate to the exact number of samples requested
        initial_states_list = initial_states_list[:num_samples]
        circles_list = circles_list[:num_samples]

        # 2. Create the final robot state tensor
        data = torch.zeros(num_samples, self.horizon, state_dim)
        # Place the list of initial states into the first time step
        data[:, 0, :] = torch.stack(initial_states_list)

        # 3. Create the final circle information tensor
        stacked_circles = torch.stack(circles_list)  # shape: (N, 3)
        # Repeat circle info across the time horizon
        finalC = stacked_circles.unsqueeze(1).repeat(1, self.horizon, 1)  # shape: (N, T, 3)

        # 4. Concatenate robot states and circle info
        Final_data = torch.cat((data, finalC), dim=2)

        print(f"Successfully generated {Final_data.shape[0]} samples.")
        assert Final_data.shape[0] == num_samples

        return Final_data


class RobotsDatasetRemedial(Dataset):
    """
    A versatile Dataset class for generating various types of fine-tuning data.
    """

    def __init__(self, random_seed, horizon, xbar=torch.zeros(4)):
        self.random_seed = random_seed
        torch.manual_seed(random_seed)
        random.seed(random_seed)

        self.horizon = horizon
        self.xbar = xbar
        self.train_data = None
        self.test_data = None

    def _format_data_to_tensor(self, data_list: list):
        # This helper function is perfect as is. No changes needed.
        num_samples = len(data_list)
        if num_samples == 0: return torch.empty(0)
        state_dim = 4
        initial_states = torch.zeros(num_samples, state_dim)
        circles_info = torch.zeros(num_samples, 3)
        for i, sample in enumerate(data_list):
            point = sample['start_point']
            initial_states[i] = torch.tensor([point[0], point[1], 0, 0]) - self.xbar
            circles_info[i] = torch.cat((sample['center'], torch.tensor(sample['radius']).unsqueeze(0)))
        data = torch.zeros(num_samples, self.horizon, state_dim)
        data[:, 0, :] = initial_states
        finalC = circles_info.unsqueeze(1).repeat(1, self.horizon, 1)
        final_data = torch.cat((data, finalC), dim=2)
        return final_data

    # ==============================================================================
    # --- THIS IS THE NEW, CORRECTED METHOD FOR YOUR CURRENT GOAL ---
    # ==============================================================================
    def get_data_for_fixed_center(
            self,
            num_train_samples: int,
            num_test_samples: int,
            # Parameters defining the specific fine-tuning task
            fixed_center: torch.Tensor,
            min_radius: float,
            max_radius: float,
            # Parameters for the remedial data generation strategy
            remedial_data_ratio: float = 0.5,
            critical_angle_spread: float = math.pi / 2,
            critical_radial_extension: float = 1.5,
            square_bounds=(-5, 5)
    ):
        """
        Generates and splits data for a FIXED obstacle CENTER but VARYING radii.
        This is the correct method for your current fine-tuning goal.
        """
        print("--- Generating Training Data (Fixed Center, Varying Radii) ---")
        train_list = generate_remedial_data(
            num_samples=num_train_samples,
            fixed_center=fixed_center,
            min_radius=min_radius,
            max_radius=max_radius,
            remedial_data_ratio=remedial_data_ratio,
            critical_angle_spread=critical_angle_spread,
            critical_radial_extension=critical_radial_extension,
            square_bounds=square_bounds
        )
        self.train_data = self._format_data_to_tensor(train_list)
        print(f"Generated {self.train_data.shape[0]} training samples.\n")

        print("--- Generating Testing Data (Fixed Center, Varying Radii) ---")
        test_list = generate_remedial_data(
            num_samples=num_test_samples,
            fixed_center=fixed_center,
            min_radius=min_radius,
            max_radius=max_radius,
            remedial_data_ratio=remedial_data_ratio,
            critical_angle_spread=critical_angle_spread,
            critical_radial_extension=critical_radial_extension,
            square_bounds=square_bounds
        )
        self.test_data = self._format_data_to_tensor(test_list)
        print(f"Generated {self.test_data.shape[0]} testing samples.\n")

        return self.train_data, self.test_data

    # __len__ and __getitem__ can remain as they are, they are standard Dataset methods.
    def __len__(self):
        return self.train_data.shape[0] if self.train_data is not None else 0

    def __getitem__(self, idx):
        return self.train_data[idx]
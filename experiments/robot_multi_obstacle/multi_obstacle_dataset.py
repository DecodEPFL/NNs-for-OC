import torch
import math
import random
from torch.utils.data import Dataset
import numpy as np


def generate_narrow_passage_scenario(
        num_obstacles: int = 3,
        # --- Passage Geometry Parameters ---
        passage_axis_angle: float = math.pi / 4,  # 45 degrees, angle of the passage from origin
        passage_dist_from_origin: float = 2.0,  # How far the passage center is from the origin
        passage_width: float = 0.8,  # The gap between the flanking obstacles
        # --- Obstacle Radii Parameters ---
        main_radius_range: tuple = (0.5, 0.7),
        flank_radius_range: tuple = (0.3, 0.5),
        # --- Robot Start Point Parameters ---
        start_point_dist_from_passage: float = 1.5,
        start_point_angle_spread: float = math.pi / 4  # 45 degree cone
):
    """
    Constructs a scenario with multiple obstacles forming a narrow passage.

    This function is deterministic in its construction to guarantee a valid and
    challenging scenario every time.

    Returns:
        A dictionary containing:
        - 'centers': Tensor of shape (num_obstacles, 2)
        - 'radii': Tensor of shape (num_obstacles,)
        - 'start_point': Tensor of shape (2,) for the robot
    """
    assert num_obstacles == 3, "This specific constructor is designed for 3 obstacles."

    # --- Step 1: Define the passage axis and center point ---
    axis_vec = torch.tensor([math.cos(passage_axis_angle), math.sin(passage_axis_angle)])
    passage_center_pos = axis_vec * passage_dist_from_origin

    # --- Step 2: Generate random radii for the obstacles ---
    r_main = torch.empty(1).uniform_(*main_radius_range).item()
    r_flank1 = torch.empty(1).uniform_(*flank_radius_range).item()
    r_flank2 = torch.empty(1).uniform_(*flank_radius_range).item()
    radii = torch.tensor([r_main, r_flank1, r_flank2])

    # --- Step 3: Place the obstacles to form the passage ---
    # The main, central obstacle
    center_main = passage_center_pos

    # The flanking obstacles are placed perpendicular to the passage axis
    perp_vec = torch.tensor([-axis_vec[1], axis_vec[0]])  # Rotated by 90 degrees

    # Position the flanking obstacles
    # They are offset from the main obstacle's center by half the passage width + their own radius
    offset1 = perp_vec * (passage_width / 2 + r_flank1)
    center_flank1 = center_main + offset1

    offset2 = -perp_vec * (passage_width / 2 + r_flank2)
    center_flank2 = center_main + offset2

    centers = torch.stack([center_main, center_flank1, center_flank2])

    # --- Step 4: Validate non-intersection (optional but good practice) ---
    # We constructed them to not overlap, but a check can be added here if needed.
    # For instance, check if d(center_flank1, center_main) > r_flank1 + r_main.
    # Our construction already guarantees this.

    # --- Step 5: Place the challenging start point behind the passage ---
    # Define the starting region in polar coordinates relative to the origin
    # The angle is the passage axis, the distance is beyond the passage
    min_rho = passage_dist_from_origin + r_main + 0.2  # Start a bit behind the main obs
    max_rho = min_rho + start_point_dist_from_passage

    min_phi = passage_axis_angle - (start_point_angle_spread / 2)
    max_phi = passage_axis_angle + (start_point_angle_spread / 2)

    # Sample and convert to Cartesian
    rho_start = torch.empty(1).uniform_(min_rho, max_rho)
    phi_start = torch.empty(1).uniform_(min_phi, max_phi)
    start_point = torch.tensor([rho_start * torch.cos(phi_start), rho_start * torch.sin(phi_start)])

    return {
        'centers': centers,
        'radii': radii,
        'start_point': start_point
    }


import torch
import math
import random


def generate_slalom_scenario(
        num_obstacles: int = 3,
        # --- NEW: Optional argument for a fixed start point ---
        fixed_start_point: torch.Tensor = None,
        fixed_radii: list = None,
        # --- Default Parameters for Random Generation ---
        start_dist_range: tuple = (5.0, 7.0),
        obstacle_radius_range: tuple = (0.4, 0.6),
        stagger_distance: float = 1.0,
        max_attempts: int = 100
):
    """
    Constructs a slalom course with full optional control over the start point
    and obstacle radii for deterministic testing.

    If `fixed_start_point` or `fixed_radii` are None, those elements will be
    randomized as before.
    """
    # --- NEW: Validation for fixed arguments ---
    if fixed_radii is not None:
        assert len(fixed_radii) == num_obstacles, \
            f"The 'fixed_radii' list must have {num_obstacles} elements."
    if fixed_start_point is not None:
        assert fixed_start_point.shape == (2,), \
            "The 'fixed_start_point' must be a tensor of shape (2,)."

    # The outer loop is mainly for random generation; for fixed scenarios, it will run once.
    for _ in range(max_attempts):

        # --- MODIFIED: Use fixed start point if provided, else randomize ---
        if fixed_start_point is not None:
            start_point = fixed_start_point
        else:
            rho_start = torch.empty(1).uniform_(*start_dist_range)
            phi_start = torch.empty(1).uniform_(0, 2 * math.pi)
            start_point = torch.tensor([rho_start * math.cos(phi_start), rho_start * math.sin(phi_start)])

        # The rest of the course is constructed relative to the start_point
        path_vector = -start_point
        path_length = torch.norm(path_vector)
        path_unit_vec = path_vector / path_length
        perp_unit_vec = torch.tensor([-path_unit_vec[1], path_unit_vec[0]])

        # ... (Obstacle placement logic is the same) ...
        centers = []
        radii = []
        for i in range(num_obstacles):
            frac = (i + 1) / (num_obstacles + 1)
            base_pos = start_point + path_vector * frac
            stagger_direction = (-1) ** i
            stagger_offset = stagger_direction * stagger_distance * perp_unit_vec
            center = base_pos + stagger_offset

            if fixed_radii is not None:
                radius = fixed_radii[i]
            else:
                radius = torch.empty(1).uniform_(*obstacle_radius_range).item()

            centers.append(center)
            radii.append(radius)

        centers_tensor = torch.stack(centers)
        radii_tensor = torch.tensor(radii)

        # ... (Validation and return logic is the same) ...
        is_valid = True
        for i in range(num_obstacles):
            for j in range(i + 1, num_obstacles):
                dist_sq = torch.sum((centers_tensor[i] - centers_tensor[j]) ** 2)
                radius_sum_sq = (radii_tensor[i] + radii_tensor[j]) ** 2
                if dist_sq < radius_sum_sq:
                    is_valid = False
                    break
            if not is_valid:
                break

        if is_valid:
            return {
                'centers': centers_tensor,
                'radii': radii_tensor,
                'start_point': start_point
            }

    raise RuntimeError("Failed to generate a valid non-overlapping slalom scenario.")

class RobotsDatasetMultiObstacle(Dataset):
    def __init__(self, random_seed, horizon, xbar=torch.zeros(4)):
        self.random_seed = random_seed
        torch.manual_seed(random_seed)
        random.seed(random_seed)

        self.horizon = horizon
        self.xbar = xbar
        self.train_data = None
        self.test_data = None

    def _format_data_to_tensor(self, data_list: list):
        """Converts a list of scenario dicts to the final tensor format."""
        num_samples = len(data_list)
        if num_samples == 0: return torch.empty(0)

        num_obstacles = data_list[0]['centers'].shape[0]
        state_dim = 4
        # Final tensor shape: (S, T, state_dim + num_obs * (pos_dim + radius_dim))
        # (S, T, 4 + 3 * 3) = (S, T, 13)
        final_dim = state_dim + num_obstacles * 3

        final_tensor = torch.zeros(num_samples, self.horizon, final_dim)

        for i, sample in enumerate(data_list):
            # Set initial robot state
            point = sample['start_point']
            initial_state = torch.tensor([point[0], point[1], 0, 0]) - self.xbar
            final_tensor[i, 0, :state_dim] = initial_state

            # Prepare and set obstacle data
            centers = sample['centers']  # (3, 2)
            radii = sample['radii'].unsqueeze(1)  # (3, 1)
            obstacle_info = torch.cat([centers, radii], dim=1).flatten()  # (9,)

            # Repeat obstacle info across the horizon
            final_tensor[i, :, state_dim:] = obstacle_info.repeat(self.horizon, 1)

        return final_tensor

    def get_data(self, num_train_samples, num_test_samples):
        """Generates and splits the multi-obstacle slalom dataset."""
        print("--- Generating Multi-Obstacle SLALOM Training Data ---")
        train_list = self._generate_slalom_scenarios(num_train_samples)
        self.train_data = self._format_data_to_tensor(train_list)
        print(f"Generated {len(train_list)} training samples.\n")

        print("--- Generating Multi-Obstacle SLALOM Testing Data ---")
        test_list = self._generate_slalom_scenarios(num_test_samples)
        self.test_data = self._format_data_to_tensor(test_list)
        print(f"Generated {len(test_list)} testing samples.\n")

        return self.train_data, self.test_data

    def _generate_slalom_scenarios(self, num_samples):
        """
        Internal helper to generate a list of slalom scenario dictionaries.
        This directly replaces your old _generate_scenarios method.
        """
        scenarios = []
        for _ in range(num_samples):
            # Each sample is now a challenging slalom course
            scenario = generate_slalom_scenario(
                num_obstacles=3,
                stagger_distance=1.2,  # A good starting value to tune
                obstacle_radius_range=(0.4, 0.7)
            )
            scenarios.append(scenario)

        # Shuffling is good practice
        random.shuffle(scenarios)
        return scenarios

import torch
from torch.utils.data import Dataset
import random
import math

def generate_multi_obstacle_scenario(num_obstacles=3, square_bounds=(-4, 4), min_radius=0.3, max_radius=0.8, max_attempts=100):
    """
    Generates a scenario with multiple non-overlapping obstacles and a valid start point.
    """
    min_val, max_val = square_bounds
    obstacles = []

    # Generate obstacles
    for _ in range(num_obstacles):
        for _ in range(max_attempts):
            radius = torch.empty(1).uniform_(min_radius, max_radius).item()
            center_x = torch.empty(1).uniform_(min_val + radius, max_val - radius).item()
            center_y = torch.empty(1).uniform_(min_val + radius, max_val - radius).item()
            center = torch.tensor([center_x, center_y])

            # Check for overlap with existing obstacles
            is_overlapping = False
            for obs_center, obs_radius in obstacles:
                if torch.norm(center - obs_center) < radius + obs_radius + 0.2: # Add a small buffer
                    is_overlapping = True
                    break

            if not is_overlapping:
                obstacles.append((center, radius))
                break
        else:
            raise RuntimeError("Failed to place a non-overlapping obstacle.")

    # Generate a valid starting point for the robot_single_obstacle
    for _ in range(max_attempts):
        start_point = torch.empty(2).uniform_(min_val, max_val)
        is_inside_any_obstacle = False
        for center, radius in obstacles:
            if torch.norm(start_point - center) < radius + 0.1: # Add a small buffer
                is_inside_any_obstacle = True
                break
        if not is_inside_any_obstacle:
            break
    else:
        raise RuntimeError("Failed to find a valid starting point for the robot_single_obstacle.")

    return start_point, obstacles

class MultiObstacleDataset(Dataset):
    def __init__(self, num_samples, horizon, num_obstacles=3, random_seed=42):
        self.num_samples = num_samples
        self.horizon = horizon
        self.num_obstacles = num_obstacles
        self.random_seed = random_seed
        torch.manual_seed(random_seed)
        random.seed(random_seed)

        self.data = self._generate_data()
        self.xbar = torch.zeros(4) # Nominal equilibrium point

    def _generate_data(self):
        data_list = []
        for _ in range(self.num_samples):
            start_point, obstacles = generate_multi_obstacle_scenario(self.num_obstacles)

            # Robot initial state
            robot_state = torch.tensor([start_point[0], start_point[1], 0, 0])

            # Obstacles info
            obstacles_tensor = torch.zeros(self.num_obstacles, 3)
            for i, (center, radius) in enumerate(obstacles):
                obstacles_tensor[i, 0:2] = center
                obstacles_tensor[i, 2] = radius

            # Combine into a single sample tensor
            # Format: [robot_x, robot_y, robot_vx, robot_vy, obs1_cx, obs1_cy, obs1_r, obs2_cx, ...]
            sample_t0 = torch.cat([robot_state, obstacles_tensor.flatten()])
            data_list.append(sample_t0)

        data_tensor = torch.stack(data_list).unsqueeze(1) # Add time dimension
        return data_tensor.repeat(1, self.horizon, 1)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx]

    def get_data(self, num_train_samples, num_test_samples):
        # Simple split for demonstration
        if num_train_samples + num_test_samples > self.num_samples:
            raise ValueError("Not enough samples generated for the requested split.")
        train_data = self.data[:num_train_samples]
        test_data = self.data[num_train_samples:num_train_samples + num_test_samples]
        return train_data, test_data


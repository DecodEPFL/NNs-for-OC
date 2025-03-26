import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import math
from argparse import Namespace
from tqdm import tqdm
from Models_sysid import DeepSSM, DWNConfig, ContractiveREN, SimpleRNN

# Set seed for reproducibility
seed = 9
torch.manual_seed(seed)

dtype = torch.float
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(device)

# Load Data
# Load the training trajectories file
train_data = torch.load('train_trajectories.pt')

# Extract inputs and states
u_train = train_data['u']  # Shape: (400, 200, 1)
y_train = train_data['x']  # Shape: (400, 200, 1)

# Load the validation trajectories file
val_data = torch.load('val_trajectories.pt')

# Extract inputs and states
u_val = val_data['u']  # Shape: (200, 200, 1)
y_val = val_data['x']  # Shape: (200, 200, 1)

"""
SSM set up ------------------------------------------
"""
# Define model configuration (SSM)
cfg = Namespace(
    n_u=1, n_y=1, d_model=11, d_state=20, n_layers=3,
    ff="MLP", max_phase=math.pi / 50, r_min=0.7, r_max=0.98
)

# Initialize model (SSM)
config = DWNConfig(
    d_model=cfg.d_model, d_state=cfg.d_state, n_layers=cfg.n_layers,
    ff=cfg.ff, rmin=cfg.r_min, rmax=cfg.r_max, max_phase=cfg.max_phase
)

#model = DeepSSM(cfg.n_u, cfg.n_y, config).to(device)

"""
REN set up ------------------------------------------
"""

model = ContractiveREN(1, 1, 8, 8)

#model = SimpleRNN(1, 1, 10, 8)

# Configure optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Track total parameters
print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")


# Training function
def train_step(model, optimizer, u, y):
    model.train()
    optimizer.zero_grad()
    y_pred = model(u, mode="scan")
    loss = F.mse_loss(y_pred, y)
    loss.backward()
    optimizer.step()
    return loss.item()


# Validation function
def validate(model, u, y):
    model.eval()
    with torch.no_grad():
        y_pred = model(u, mode="scan")
        loss = F.mse_loss(y_pred, y)
    return loss.item(), y_pred


# Training loop
num_epochs = 1500
train_losses, val_losses = [], []

tqdm_bar = tqdm(range(num_epochs), desc="Training")
for epoch in tqdm_bar:
    train_loss = train_step(model, optimizer, u_train[:, :, :], y_train[:, :, :])
    val_loss, _ = validate(model, u_val[:, :, :], y_val[:, :, :])

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    tqdm_bar.set_postfix(train_loss=train_loss, val_loss=val_loss)

# Plot training and validation loss
plt.figure()
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.yscale('log')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss (log scale)')
plt.legend()
plt.title('Training & Validation Loss')
plt.show()


# Function to plot multiple trajectories
def plot_trajectories(y_true, y_pred, u, indices, title_prefix, num_plots=4):
    """
    Plot multiple trajectories comparing true and predicted scalar states.

    Args:
        y_true: True states tensor, shape (num_traj, horizon, 1)
        y_pred: Predicted states tensor, shape (num_traj, horizon, 1)
        u: Input tensor, shape (num_traj, horizon, 1)
        indices: List of trajectory indices to plot
        title_prefix: String prefix for plot titles ('Training' or 'Validation')
        num_plots: Number of subplots (default 4)
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))  # 2x2 grid for 4 plots
    axes = axes.flatten()  # Flatten to easily index

    for i, idx in enumerate(indices[:num_plots]):  # Limit to num_plots
        ax = axes[i]
        time = torch.arange(y_true.shape[1])  # Horizon steps
        ax.plot(time.cpu(), y_true[idx, :, 0].cpu(), label='True State $x$', linestyle='dashed')
        ax.plot(time.cpu(), y_pred[idx, :, 0].cpu(), label='Predicted State $x$')
        ax.plot(time.cpu(), u[idx, :, 0].cpu(), label='Input $u$', linestyle='dotted')
        ax.set_title(f'{title_prefix} Trajectory {idx}')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    # Save the figure as a PDF
    plt.savefig("fit.pdf", format="pdf", bbox_inches="tight")
    plt.show()


# Get predictions (replace with your actual model and validate function)
_, y_train_pred = validate(model, u_train[:, :, :], y_train[:, :, :])
_, y_val_pred = validate(model, u_val[:, :, :], y_val[:, :, :])

# Define indices for plotting
train_indices = [0, 1, 205, 205]  # Example: two piecewise, two sinusoidal
val_indices = [0, 1, 105, 105]  # Example: two piecewise, two sinusoidal

# Plot training trajectories
plot_trajectories(y_train, y_train_pred, u_train, train_indices, "Training")

# Plot validation trajectories
plot_trajectories(y_val, y_val_pred, u_val, val_indices, "Validation")

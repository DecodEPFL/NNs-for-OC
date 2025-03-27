# Neural Networks for Optimal Control

This repository contains a Python exercises for the 
"Neural Networks for Optimal Control" course.

## Project Setup

In order to run this project it is recommended to use a virtual environment
and required to install some dependencies. 
The setup process for both actions is automated using `setup.py`.

### Installation

Follow these steps to set up the project:

#### 1. Clone the repository
```bash
git clone https://github.com/DecodEPFL/NNs-for-OC
cd NNs_for_OC
```

#### 2. Run the setup script
Execute the following command to create a virtual environment and install dependencies:
```bash
python setup.py
```

#### 3. Activate the virtual environment
After installation, activate the environment (if working on the console):
- **On macOS/Linux:**
  ```bash
  source venv/bin/activate
  ```
- **On Windows (CMD or PowerShell):**
  ```powershell
  venv\Scripts\activate
  ```
Otherwise, activate the environment through the UI 
of your preferred development environment (PyCharm, VS Code, etc).

### Dependencies
The project requires the following dependencies, which are automatically installed from `requirements.txt`:
- `torch`
- `numpy`
- `matplotlib`
- `jax`
- `pip`
- `tqdm`


### Notes
- Ensure you have Python installed.
- If you encounter permission issues, try running commands with `python3` instead of `python`.

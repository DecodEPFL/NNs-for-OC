import torch
from plants.custom_dataset import CustomDataset


class TankDataset(CustomDataset):
    def __init__(self, random_seed, horizon, std_ini=0.2):
        # experiment and file names
        exp_name = 'tank'
        file_name = 'data_T' + str(horizon) + '_stdini' + str(std_ini) + '_RS' + str(random_seed) + '.pkl'

        super().__init__(random_seed=random_seed, horizon=horizon, exp_name=exp_name, file_name=file_name)

        self.std_ini = std_ini

    # ---- data generation ----
    def _generate_data(self, num_samples):
        state_dim = 1
        data = self.std_ini * torch.randn(num_samples, self.horizon, state_dim)
        assert data.shape[0] == num_samples
        return data

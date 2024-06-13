import torch
import numpy as np
from torchvision import transforms as T
from torchvision.transforms import functional as F

class ActiveDatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset: torch.utils.data.Dataset):
        super().__init__()
        self.dataset = dataset
        assert 'idx' in dataset[0].keys(), 'Dataset needs to return idx'
        self.original_ds_size = len(self.dataset)

    def set_budget(self, budget):
        assert budget <= len(self.dataset)
        self.budget = budget
        self.probs = np.ones(len(self.dataset)) / len(self.dataset)
        self.counts = np.zeros(len(self.dataset))
        self.populate_samples()

    def populate_samples(self):
        self.idxs = np.random.choice(
            np.arange(self.original_ds_size),
            self.budget,
            p=self.probs,
            replace=False
        )
        self.counts[self.idxs] += 1
        print('samples_updated\n\n')

    def update_probs(self, probs):
        self.probs = probs

    def __getitem__(self, idx):
        return self.dataset[self.idxs[idx]]

    def __len__(self):
        return self.budget

    def __getattr__(self, attr):
        return getattr(self.dataset, attr)

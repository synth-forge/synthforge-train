import torch
from lightning.pytorch.utilities.types import (
    TRAIN_DATALOADERS,
    EVAL_DATALOADERS,
)
from ...data.active import ActiveDatasetWrapper
from ...data.synthforge import SynthForgeDataset
from .base_datamodule import BaseDataModule


class ActiveLoaderDataModule(BaseDataModule):
    COLORMAP = SynthForgeDataset.COLORMAP

    def __init__(
        self,
        train_dataset: ActiveDatasetWrapper,
        corpus_dataset: torch.utils.data.Dataset,
        batch_size: int,
        val_batch_size: int,
        budget: int,
        val_dataset: torch.utils.data.Dataset,
        train_dataloader_args=None,
        val_dataloader_args=None,
    ):
        super().__init__()
        self.train_dataset = train_dataset
        self.corpus_dataset = corpus_dataset
        self.val_dataset = val_dataset
        self.budget = budget
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.train_dataloader_kwargs = train_dataloader_args or {}
        self.val_dataloader_kwargs = val_dataloader_args or {}
        self.train_dataset.set_budget(self.budget)

    def set_probs(self, probs):
        self.train_dataset.update_probs(probs)
    
    def update_samples(self):
        self.train_dataset.populate_samples()

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return torch.utils.data.DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            **self.train_dataloader_kwargs,
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        corpus_dl = torch.utils.data.DataLoader(
            dataset=self.corpus_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            **self.val_dataloader_kwargs,
        )
        validation_dl = torch.utils.data.DataLoader(
            dataset=self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            **self.val_dataloader_kwargs,
        )
        return [corpus_dl, validation_dl]

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return torch.utils.data.DataLoader(
            dataset=self.val_dataset,
            batch_size=self.val_batch_size,
            **self.val_dataloader_kwargs,
        )

    @property
    def num_classes(self):
        return SynthForgeDataset.NUM_CLASSES

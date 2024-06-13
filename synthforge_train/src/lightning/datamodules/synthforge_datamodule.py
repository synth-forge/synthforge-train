import torch
from lightning.pytorch.utilities.types import (
    TRAIN_DATALOADERS,
    EVAL_DATALOADERS,
)
from ...data.synthforge import SynthForgeDataset
from .base_datamodule import BaseDataModule


class SynthForgeDataModule(BaseDataModule):
    COLORMAP = SynthForgeDataset.COLORMAP

    def __init__(
        self,
        train_dataset_path,
        val_dataset_path,
        batch_size,
        val_batch_size,
        crop_size=None,
        disable_crop_op=False,
        train_dataloder_args=None,
        val_dataloader_args=None,
    ):
        super().__init__()
        self.train_dataset_path = train_dataset_path
        self.val_dataset_path = val_dataset_path
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.train_dataloader_kwargs = train_dataloder_args or {}
        self.val_dataloader_kwargs = val_dataloader_args or {}
        self.disable_crop_op = disable_crop_op
        self.crop_size = crop_size

    def setup(self, stage: str) -> None:
        super().setup(stage)
        self.train_dataset = SynthForgeDataset(
            self.train_dataset_path,
            disable_crop_op=self.disable_crop_op,
            crop_size=self.crop_size,
        )
        self.val_dataset = SynthForgeDataset(
            self.val_dataset_path,
            is_train=False,
            disable_crop_op=self.disable_crop_op,
            crop_size=self.crop_size,
        )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return torch.utils.data.DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            **self.train_dataloader_kwargs,
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return torch.utils.data.DataLoader(
            dataset=self.val_dataset,
            batch_size=self.val_batch_size,
            **self.val_dataloader_kwargs,
        )

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return torch.utils.data.DataLoader(
            dataset=self.val_dataset,
            batch_size=self.val_batch_size,
            **self.val_dataloader_kwargs,
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return torch.utils.data.DataLoader(
            dataset=self.val_dataset,
            batch_size=self.val_batch_size,
            **self.val_dataloader_kwargs,
        )

    def predit_dataloader(self) -> EVAL_DATALOADERS:
        return self.val_dataloader()

    @property
    def num_classes(self):
        return SynthForgeDataset.NUM_CLASSES

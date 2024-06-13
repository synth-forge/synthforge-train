import torch
from lightning.pytorch.utilities.types import (
    TRAIN_DATALOADERS,
    EVAL_DATALOADERS,
)
from ...data.ms_synth import MSSynthDataset
from .base_datamodule import BaseDataModule


class MSSynthDataModule(BaseDataModule):
    COLORMAP = MSSynthDataset.COLORMAP

    def __init__(
        self,
        dataset_path,
        batch_size,
        val_batch_size,
        crop_size=None,
        disable_crop_op=True,
        train_dataloder_args=None,
        val_dataloader_args=None,
    ):
        super().__init__()
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        default_train_dl_args = {
            'num_workers': 32
        }
        self.train_dataloader_kwargs = train_dataloder_args or default_train_dl_args 

        default_val_dl_args = {
            'num_workers': 32
        }
        self.val_dataloader_kwargs = val_dataloader_args or default_val_dl_args 

        self.disable_crop_op = disable_crop_op
        self.crop_size = crop_size

    def setup(self, stage: str) -> None:
        super().setup(stage)
        self.train_dataset = MSSynthDataset(
            self.dataset_path,
            disable_crop_op=self.disable_crop_op,
            crop_size=self.crop_size,
        )
        self.val_dataset = MSSynthDataset(
            self.dataset_path,
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

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return torch.utils.data.DataLoader(
            dataset=self.val_dataset,
            batch_size=self.val_batch_size,
            **self.val_dataloader_kwargs,
        )

    @property
    def num_keypoints(self):
        return 68

    @property
    def num_classes(self):
        return MSSynthDataset.NUM_CLASSES

import torch
from lightning.pytorch.utilities.types import (
    TRAIN_DATALOADERS,
    EVAL_DATALOADERS,
)
from ...data.lapa_d import LaPaDepthDataset
from .base_datamodule import BaseDataModule


class LaPaDepthDataModule(BaseDataModule):
    COLORMAP = LaPaDepthDataset.COLORMAP

    def __init__(
        self,
        dataset_path,
        batch_size,
        val_batch_size,
        force_tightcrop=False,
        train_dataloder_args=None,
        val_dataloader_args=None,
    ):
        super().__init__()
        self.root = dataset_path
        self.batch_size = batch_size
        self.force_tightcrop = force_tightcrop
        self.val_batch_size = val_batch_size
        default_train_dl_args = {"num_workers": 16}
        self.train_dataloader_kwargs = (
            train_dataloder_args or default_train_dl_args
        )

        default_val_dl_args = {"num_workers": 4}
        self.val_dataloader_kwargs = val_dataloader_args or default_val_dl_args

    def setup(self, stage: str) -> None:
        super().setup(stage)
        self.train_dataset = LaPaDepthDataset(self.root, split="train")
        self.val_dataset = LaPaDepthDataset(
            self.root, split="val", force_tightcrop=self.force_tightcrop
        )
        self.test_dataset = LaPaDepthDataset(
            self.root, split="test", force_tightcrop=self.force_tightcrop
        )
        self.dls = {}

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        if self.dls.get("train") is None:
            self.dls["train"] = torch.utils.data.DataLoader(
                dataset=self.train_dataset,
                batch_size=self.batch_size,
                **self.train_dataloader_kwargs,
            )
        return self.dls["train"]

    def val_dataloader(self) -> EVAL_DATALOADERS:
        if self.dls.get("val") is None:
            self.dls["val"] = torch.utils.data.DataLoader(
                dataset=self.val_dataset,
                batch_size=self.val_batch_size,
                **self.val_dataloader_kwargs,
            )
        return self.dls["val"]

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return torch.utils.data.DataLoader(
            dataset=self.test_dataset,
            batch_size=self.val_batch_size,
            **self.val_dataloader_kwargs,
        )

    @property
    def num_keypoints(self):
        return 106

    @property
    def num_classes(self):
        return LaPaDepthDataset.NUM_CLASSES

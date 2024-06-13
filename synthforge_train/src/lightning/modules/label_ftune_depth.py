import torch
import torch.nn as nn


from typing import Any, Optional
from torchvision.utils import *
from torchvision import transforms as T
from .base_module import BaseModule
from lightning.pytorch.utilities.types import STEP_OUTPUT
from ...models import UNet
from .multimodal import MultiModalLightningModule
from ...data.synthforge import SynthForgeDataset


def NME(target, pred, iod):
    loss = (target - pred).norm(dim=-1)  # [B, K]
    loss = loss.mean(dim=-1) / iod  # [B]
    return loss.mean()


def freeze(model):
    for k, v in model.named_parameters():
        v.requires_grad = False
        # print(f"Freezing {k}")


def z2o(x):
    return (x - x.min()) / (x.max() - x.min())


class DepthLabelFinetuneLightningModule(BaseModule):
    def __init__(
        self,
        multimodal_ckpt=None,
        multimodal_model=None,
        include_features=False,
        include_segmentation=False,
        include_keypoints=False,
        learning_rate=1e-4,
        enable_ssl_loss=False,
        regress_heatmaps: bool = False,
        log_gradients: bool = False,
        accumulate_grad_batches: Optional[int] = None,
        num_keypoints: Optional[int] = None,
        datamodule: Optional[Any] = None,
    ):
        super().__init__(learning_rate=learning_rate)
        if multimodal_model is None:
            self.mm_model = MultiModalLightningModule.load_from_checkpoint(
                multimodal_ckpt,
                regress_heatmaps=regress_heatmaps,
                learning_rate=learning_rate,
                num_classes_seg=SynthForgeDataset.NUM_CLASSES,
                num_keypoints=num_keypoints,
                datamodule=None,
            )
        else:
            self.mm_model = multimodal_model
        self.loss_fn = nn.L1Loss()
        self.datamodule = datamodule
        self.include_features = include_features
        self.include_segmentation = include_segmentation
        self.include_keypoints = include_keypoints
        self.is_setup = False
        self.plotter = self.mm_model.plotter
        self.mm_model.eval()
        self.automatic_optimization = False
        self.accumulate_grad_batches = accumulate_grad_batches
        self.log_gradients = log_gradients
        self.enable_ssl_loss = enable_ssl_loss
        self.setup("")

    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            self.trainer.max_epochs,
            eta_min=1e-6,
            last_epoch=-1,
            verbose=False,
        )
        return optimizer

    def setup(self, stage):
        if self.is_setup:
            return
        x = self.mm_model(torch.rand(1, 3, 256, 256))
        in_channels = x["depth"].size(1)
        if self.include_features:
            in_channels += x["fusion_maps"].size(1)
        if self.include_segmentation:
            in_channels += x["seg"].size(1)

        self.ftune_model = UNet(in_channels, 1)
        freeze(self.mm_model.backbone)
        freeze(self.mm_model.seg_model)
        freeze(self.mm_model.dep_model)
        freeze(self.mm_model.kps_model)
        self.is_setup = True

    def forward(self, imgs):
        with torch.no_grad():
            fusion_maps = self.mm_model.backbone(imgs)
            depth = self.mm_model.dep_model(fusion_maps.clone())
            outs = {
                "depth": depth,
                "fusion_maps": fusion_maps,
            }
        x = outs["depth"]

        if self.include_features:
            x = torch.cat([x, fusion_maps], dim=1)
        if self.include_segmentation:
            seg_maps = self.mm_model.seg_model(fusion_maps.clone())
            outs["seg"] = seg_maps
            outs["seg_label"] = torch.argmax(seg_maps, dim=1)
            x = torch.cat([x, seg_maps], dim=1)
        if self.include_keypoints:
            raise NotImplementedError
        outs["la_depth"] = self.ftune_model(x)
        return outs

    def get_ssl_loss(self, batch, outs, return_labels=False):
        raise NotImplementedError

    def get_losses_and_metrics(self, batch, return_pred=False, outs=None):
        if outs is None:
            outs = self(batch["img"])

        ssl_loss = 0.0
        if self.enable_ssl_loss:
            ssl_loss = self.get_ssl_loss(batch, outs)

        loss = self.loss_fn(outs["la_depth"], batch["depth"])

        metrics = {"loss": loss}
        if self.enable_ssl_loss:
            metrics["ssl_loss"] = ssl_loss
        if return_pred:
            return metrics, outs
        return metrics

    def training_step(self, batch, batch_idx, dataloder_idx=0) -> STEP_OUTPUT:
        batch_size = batch["img"].size(0)
        metrics = self.get_losses_and_metrics(batch)
        self.manual_backward(metrics["loss"])
        is_last_epoch = self.last_iter_in_epoch(batch_idx)
        if (
            self.current_iter(batch_idx) % self.accumulate_grad_batches == 0
            or is_last_epoch
        ):
            opt = self.optimizers()
            opt.step()
            opt.zero_grad()

        self.log_dict(
            {
                "lr": self.current_lr,
                "train/loss": metrics["loss"],
            },
            prog_bar=True,
            add_dataloader_idx=False,
            on_step=True,
            batch_size=batch_size,
        )
        self.log_gradients_in_model(self.current_iter(batch_idx))
        return {"train/loss": metrics["loss"]}

    def on_train_epoch_end(self) -> None:
        self.scheduler.step()

    @torch.no_grad()
    def validation_step(self, batch, batch_idx, dataloder_idx=0) -> STEP_OUTPUT:
        batch_size = batch["img"].size(0)
        metrics, outs = self.get_losses_and_metrics(batch, return_pred=True)
        self.log_dict(
            {"val/loss": metrics["loss"]},
            prog_bar=True,
            add_dataloader_idx=False,
            on_epoch=True,
        )
        if batch_idx == 0:
            plot = torch.tensor(
                self.plotter.get_plot(
                    batch["img"],
                    tgt_depth_map=batch["depth"],
                    pred_depth_map=outs["depth"],
                    pred_depth_map_la=outs["la_depth"],
                )
            )
            self.logger.experiment.add_image(
                f"val/out",
                plot.permute(2, 0, 1),
                self.current_epoch,
            )
            self.plotter.close_plot()
        return {"val/loss": metrics["loss"]}

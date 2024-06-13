import torch
import torch.nn as nn


from typing import Any, Dict, Optional
from importlib import import_module
from torchvision.utils import *
from torchvision import transforms as T
from .base_module import BaseModule
from lightning.pytorch.utilities.types import STEP_OUTPUT
from ...models import UNet
from ...utils.viz import Visualizer
from ...data.utils import seg_classes_to_colors
from ...metrics.metrics import F1Score
from .multimodal import MultiModalLightningModule
from ...data.synthforge import SynthForgeDataset


def NME(target, pred, iod):
    loss = (target - pred).norm(dim=-1)  # [B, K]
    loss = loss.mean(dim=-1) / iod  # [B]
    return loss.mean()


def freeze(model):
    for k, v in model.named_parameters():
        v.requires_grad = False
        print(f"Freezing {k}")


def z2o(x):
    return (x - x.min()) / (x.max() - x.min())


class SegmentationLabelFinetuneLightningModule(BaseModule):
    def __init__(
        self,
        multimodal_ckpt=None,
        multimodal_model=None,
        multimodal_lightning_module=None,
        include_features=False,
        include_depth=False,
        include_keypoints=False,
        learning_rate=1e-4,
        enable_ssl_loss=False,
        regress_heatmaps: bool = False,
        log_gradients: bool = False,
        accumulate_grad_batches: Optional[int] = None,
        loss_fn: Optional[nn.Module] = None,
        num_keypoints: Optional[int] = None,
        datamodule: Optional[Any] = None,
        num_classes_seg=None,
        multimodal_no_depth=False,
    ):
        super().__init__(learning_rate=learning_rate)
        if multimodal_model is None and multimodal_ckpt is not None:
            if multimodal_lightning_module is None:
                multimodal_lightning_module = MultiModalLightningModule
            else:
                mod = import_module("face_data_aug.src.lightning.modules")
                multimodal_lightning_module = getattr(
                    mod, multimodal_lightning_module
                )
            self.mm_model = multimodal_lightning_module.load_from_checkpoint(
                multimodal_ckpt,
                regress_heatmaps=regress_heatmaps,
                learning_rate=learning_rate,
                num_classes_seg=num_classes_seg or SynthForgeDataset.NUM_CLASSES,
                num_keypoints=num_keypoints,
                datamodule=None,
                no_depth=multimodal_no_depth,
            )
        else:
            self.mm_model = multimodal_model
        self.loss_fn = loss_fn or nn.CrossEntropyLoss()
        self.datamodule = datamodule
        self.include_features = include_features
        self.include_depth = include_depth
        self.include_keypoints = include_keypoints
        self.is_setup = False
        if self.mm_model is None:
            self.plotter = Visualizer(n_cols=2, n_rows=4)
        else:
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
        x = self.mm_model(torch.rand(1, 3, 256, 256).to(self.device))
        in_channels = x["seg"].size(1)
        if self.include_features:
            in_channels += x["fusion_maps"].size(1)
        if self.include_depth:
            in_channels += x["depth"].size(1)

        self.ftune_model = UNet(in_channels, self.datamodule.num_classes)
        freeze(self.mm_model.backbone)
        freeze(self.mm_model.seg_model)
        if not self.mm_model.no_depth:
            freeze(self.mm_model.dep_model)
        freeze(self.mm_model.kps_model)
        self.f1 = F1Score(
            "seg_label",
            "pred_label",
            label_names=list(self.datamodule.COLORMAP.keys()) + ["background"],
        )
        self.f1.init_evaluation()
        self.is_setup = True

    def forward(self, imgs, return_bb_outs=False):
        with torch.no_grad():
            self.mm_model.eval()
            if return_bb_outs:
                outs = self.mm_model(imgs)
                fusion_maps = outs["fusion_maps"]
            else:
                fusion_maps = self.mm_model.backbone_forward(imgs)
                seg_maps = self.mm_model.seg_model(fusion_maps.clone())
                outs = {
                    "seg": seg_maps,
                    "seg_label": torch.argmax(seg_maps, dim=1),
                    "fusion_maps": fusion_maps,
                }

        x = outs["seg"]

        if self.include_features:
            x = torch.cat([x, fusion_maps], dim=1)
        if self.include_depth:
            if outs.get("depth", None) is None:
                outs["depth"] = self.mm_model.dep_model(fusion_maps.clone())
            x = torch.cat([x, outs["depth"]], dim=1)
        if self.include_keypoints:
            raise NotImplementedError
        outs["la_seg_scores"] = self.ftune_model(x)
        return outs

    def get_ssl_loss(self, batch, outs, return_labels=False):
        raise NotImplementedError

    def get_losses_and_metrics(self, batch, return_pred=False, outs=None):
        if outs is None:
            outs = self(batch["img"])
        ssl_loss = 0.0
        if self.enable_ssl_loss:
            ssl_loss = self.get_ssl_loss(batch, outs)

        loss = self.loss_fn(outs["la_seg_scores"], batch["seg_label"])
        self.f1.evaluate(
            {
                "seg_label": batch["seg_label"].cpu().numpy(),
                "pred_label": outs["la_seg_scores"].argmax(dim=1).cpu().numpy(),
            }
        )
        metrics = {"loss": loss}
        if self.enable_ssl_loss:
            metrics["ssl_loss"] = ssl_loss
        if return_pred:
            return metrics, outs
        return metrics

    def on_train_epoch_start(self) -> None:
        self.f1.init_evaluation()

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
        ff = self.f1.finalize_evaluation()
        self.log_dict(
            {f"train/f1_{k}": v for k, v in ff.items()},
            add_dataloader_idx=False,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            rank_zero_only=True,
        )
        self.scheduler.step()
        return {f"train/f1_{k}": v for k, v in ff.items()}

    def on_validation_epoch_start(self) -> None:
        self.f1.init_evaluation()

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
            plot_colors_lapa = [
                torch.tensor(x, device=self.device)
                for x in self.datamodule.COLORMAP.values()
            ]
            plot_colors = [
                torch.tensor(x, device=self.device)
                for x in SynthForgeDataset.COLORMAP.values()
            ]
            tgt_seg_map = seg_classes_to_colors(
                batch["seg_label"].unsqueeze(1), plot_colors_lapa
            )
            pred_seg_map = seg_classes_to_colors(
                outs["seg_label"].unsqueeze(1), plot_colors
            )
            pred_seg_map_la = seg_classes_to_colors(
                outs["la_seg_scores"].argmax(dim=-3, keepdims=True),
                plot_colors_lapa,
            )
            plot = torch.tensor(
                self.plotter.get_plot(
                    batch["img"],
                    tgt_seg_map=tgt_seg_map,
                    pred_seg_map=pred_seg_map,
                    pred_seg_map_la=pred_seg_map_la,
                )
            )
            self.logger.experiment.add_image(
                f"val/out",
                plot.permute(2, 0, 1),
                self.current_epoch,
            )
            self.plotter.close_plot()
        return {"val/loss": metrics["loss"]}

    def on_validation_epoch_end(self) -> None:
        f1 = self.f1.finalize_evaluation()
        if self.local_rank == 0:
            for k, v in f1.items():
                self.trainer.logger.experiment.add_scalar(
                    f"val/f1_{k}", v, self.current_epoch
                )
        return {f"val/f1_{k}": v for k, v in f1.items()}

    def on_test_epoch_end(self) -> None:
        return super().on_validation_epoch_end()

    def on_test_end(self) -> None:
        f1 = self.f1.finalize_evaluation()
        for k, v in f1.items():
            self.trainer.logger.experiment.add_scalar(
                f"val/f1_{k}", v, self.current_epoch
            )
            print(f"F1_{k.upper()}: {v*100:5.3f}")
        return f1

    def test_step(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        return self.validation_step(*args, **kwargs)

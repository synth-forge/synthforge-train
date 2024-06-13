import torch
import torch.nn as nn
from munch import Munch

from typing import Any, Optional
from torchvision.utils import *
from torchvision import transforms as T
from .base_module import BaseModule
from lightning.pytorch.utilities.types import STEP_OUTPUT
from ...models import StackedHGNetV2
from ...utils.viz import Visualizer
from ...data.d300w import D300WDataset
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


class LandmarksFinetuneLightningModule(BaseModule):
    def __init__(
        self,
        multimodal_ckpt=None,
        multimodal_model=None,
        include_features=False,
        include_depth=False,
        include_segmentation=False,
        learning_rate=1e-4,
        regress_heatmaps: bool = False,
        log_gradients: bool = False,
        accumulate_grad_batches: Optional[int] = None,
        loss_fn: Optional[nn.Module] = None,
        enable_ssl_loss: bool = False,
        num_keypoints: Optional[int] = None,
        datamodule: Optional[Any] = None,
        num_classes_seg: Optional[Any] = None,
        val_on_la: bool = True,
        multimodal_no_depth= False,
        strict_loading: Optional[bool] = True,
    ):
        super().__init__(learning_rate=learning_rate)
        if multimodal_model is None:
            self.mm_model = MultiModalLightningModule.load_from_checkpoint(
                multimodal_ckpt,
                regress_heatmaps=regress_heatmaps,
                learning_rate=learning_rate,
                num_classes_seg=num_classes_seg or SynthForgeDataset.NUM_CLASSES,
                num_keypoints=num_keypoints,
                datamodule=None,
                no_depth= multimodal_no_depth,
                strict=strict_loading
            )
        else:
            self.mm_model = multimodal_model
        self.loss_fn = loss_fn or nn.CrossEntropyLoss()
        self.datamodule = datamodule
        self.include_features = include_features
        self.include_depth = include_depth
        self.include_segmentation = include_segmentation
        self.is_setup = False
        self.plotter = Visualizer(n_cols=4, n_rows=8)
        self.mm_model.eval()
        self.automatic_optimization = False
        self.accumulate_grad_batches = accumulate_grad_batches
        self.log_gradients = log_gradients
        self.enable_ssl_loss = enable_ssl_loss
        self.val_on_la = val_on_la
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

        pre_in_channels = 0
        if self.include_features:
            pre_in_channels += x["fusion_maps"].size(1)
        if self.include_depth:
            pre_in_channels += x["depth"].size(1)
        if self.include_segmentation:
            pre_in_channels += x["seg"].size(1)

        common_conf = {"width": 256, "height": 256}
        kps_config = Munch(
            {"out_height": 64, "out_width": 64, "use_AAM": True, **common_conf}
        )
        self.ftune_model = StackedHGNetV2(
            config=kps_config,
            classes_num=[
                self.datamodule.num_keypoints,
                9,
                self.datamodule.num_keypoints,
            ],
            edge_info=D300WDataset.EDGE_INFO,
            nstack=2,
            add_coord=False,
            x_in_channel=pre_in_channels,
            in_channel=68,
            skip_pre=(pre_in_channels == 0),
        )
        freeze(self.mm_model.backbone)
        freeze(self.mm_model.seg_model)
        if not self.mm_model.no_depth:
            freeze(self.mm_model.dep_model)
        freeze(self.mm_model.kps_model)
        self.is_setup = True
        self.kps_decoder = self.mm_model.kps_decoder

    def forward(self, imgs):
        with torch.no_grad():
            fusion_maps = self.mm_model.backbone(imgs)
            _, heatmaps = self.mm_model.kps_model(fusion_maps)
            heatmaps = heatmaps[-1]
            kps = self.kps_decoder.get_coords_from_heatmap(heatmaps)

            outs = {
                "fusion_maps": fusion_maps,
                "heatmaps": heatmaps,
                "kps": kps * 0.5 + 0.5,
                "n_kps": kps,
                "px_kps": ((kps * 0.5 + 0.5) * imgs.size(-1)),
            }
        inp = []

        if self.include_features:
            inp.append(fusion_maps)
        if self.include_depth:
            outs["depth"] = self.mm_model.dep_model(fusion_maps.clone())
            inp.append(outs["depth"])
        if self.include_segmentation:
            seg_maps = self.mm_model.seg_model(fusion_maps.clone())
            outs["seg"] = seg_maps
            outs["seg_label"] = torch.argmax(seg_maps, dim=1)
            inp.append(seg_maps)

        if not (
            self.include_depth
            or self.include_features
            or self.include_segmentation
        ):
            _, la_heatmaps = self.ftune_model(heatmaps)
        else:
            _, la_heatmaps, feature_maps = self.ftune_model(
                torch.cat(inp, dim=1), kp_res=heatmaps, return_pre_out=True
            )
            outs["feature_maps"] = feature_maps
        n_kps = self.kps_decoder.get_coords_from_heatmap(la_heatmaps[-1])
        outs.update(
            {
                "la_kps": n_kps * 0.5 + 0.5,
                "la_n_kps": n_kps,
                "la_px_kps": ((n_kps * 0.5 + 0.5) * imgs.size(-1)),
                "la_heatmaps": la_heatmaps[-1],
            }
        )
        return outs

    def get_losses_and_metrics(self, batch, return_pred=False, outs=None):
        if self.val_on_la:
            px_key = "la_px_kps"
            n_key = "la_n_kps"
        else:
            px_key = "px_kps"
            n_key = "n_kps"
        if outs is None:
            outs = self(batch["img"])

        ssl_loss = 0.0
        if self.enable_ssl_loss:
            ssl_loss = self.get_ssl_loss(batch, outs)

        n_tgts = (batch["kps"] / batch["img"].size(-1)) * 2 - 1
        kp_loss = (n_tgts - outs[n_key]).norm(dim=-1)
        kp_loss = kp_loss.sum(dim=-1).mean()
        nme = NME(batch["kps"], outs[px_key], batch["iod"])
        metrics = {
            "nme": nme,
            # "kp_loss": kp_loss,
            "loss": kp_loss + ssl_loss,
        }
        if self.enable_ssl_loss:
            metrics["ssl_loss"] = ssl_loss
        if return_pred:
            return metrics, outs
        return metrics

    def get_ssl_loss(self, batch, og_outs, return_labels=False):
        raise NotImplementedError

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
                **{f"train/{k}": v for k, v, in metrics.items()},
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
    def validation_step(
        self, batch, batch_idx, dataloader_idx=0, dataloader_name=None
    ) -> STEP_OUTPUT:
        batch_size = batch["img"].size(0)
        metrics, outs = self.get_losses_and_metrics(batch, return_pred=True)
        dl_name = "" if dataloader_name is None else f"_{dataloader_name}"
        self.log_dict(
            {f"val/{k}{dl_name}": v for k, v in metrics.items()},
            prog_bar=True,
            add_dataloader_idx=False,
            on_epoch=True,
            batch_size=batch_size,
        )
        if batch_idx == 0:
            plot = torch.tensor(
                self.plotter.get_plot(
                    batch["img"],
                    px_tgts=batch["kps"],
                    px_pred=outs["px_kps"],
                    px_pred_la=outs["la_px_kps"],
                )
            )
            self.logger.experiment.add_image(
                f"val/out{dl_name}",
                plot.permute(2, 0, 1),
                self.current_epoch,
            )
            self.plotter.close_plot()
        return {f"val/{k}{dl_name}": v for k, v in metrics.items()}

    def test_step(self, batch, batch_idx, dataloader_idx=0) -> STEP_OUTPUT:
        dm = self.trainer.datamodule
        kwargs = {}
        if isinstance(self.trainer.test_dataloaders, (list, dict)):
            dl_idx = kwargs["dataloader_idx"] = dataloader_idx
            kwargs["dataloader_name"] = dm.dataloader_names[dl_idx]
        return self.validation_step(batch, batch_idx, **kwargs)

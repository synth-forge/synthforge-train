import os
import sys
from munch import Munch
from typing import Any, Optional
from lightning.pytorch.utilities.types import STEP_OUTPUT
import torch
import torch.nn as nn

from torchvision.utils import *
from torchvision import transforms as T
from .base_module import BaseModule
from ...data.d300w import D300WDataset
from ...models import StackedHGNetV2, UNet, get_decoder
from ...metrics.metrics import F1Score
from ...metrics.dice_loss import DiceLoss
from ...data.utils import seg_classes_to_colors
from ...data.synthforge import SynthForgeDataset
from ...utils.viz import Visualizer
from kornia import geometry as G


def NME(target, pred, iod):
    loss = (target - pred).norm(dim=-1)  # [B, K]
    loss = loss.mean(dim=-1) / iod  # [B]
    return loss.mean()


def init_weights_xavier(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


def z2o(x):
    return (x - x.min()) / (x.max() - x.min())


class MultiModalLightningModule(BaseModule):
    def __init__(
        self,
        regress_heatmaps: bool = False,
        learning_rate: float = 1e-3,
        log_gradients: bool = False,
        accumulate_grad_batches: Optional[int] = None,
        loss_fn: Optional[nn.Module] = None,
        num_classes_seg: Optional[int] = None,
        num_keypoints: Optional[int] = None,
        datamodule: Optional[Any] = None,
        enable_ssl_loss: bool = False,
        ssl_type: Optional[str] = None,
        no_depth: Optional[bool] = False,
    ):
        super().__init__(learning_rate)
        self.log_gradients = log_gradients
        self.plotter = Visualizer(n_cols=2, n_rows=4)
        self.regress_heatmaps = regress_heatmaps
        self.accumulate_grad_batches = accumulate_grad_batches or 1
        self.loss_fn = loss_fn or nn.CrossEntropyLoss()
        self.num_classes_seg = num_classes_seg
        self.num_keypoints = num_keypoints
        self.datamodule = datamodule
        self.enable_ssl_loss = enable_ssl_loss
        self.ssl_type = ssl_type
        self.no_depth = no_depth
        if enable_ssl_loss:
            valid_ssl_types = ["full", "features", "outputs"]
            assert (
                self.ssl_type is not None and self.ssl_type in valid_ssl_types
            ), "specify ssl type: [full, features, outputs]"
        self.setup("")

    def configure_optimizers(self) -> Any:
        params = [
            {"params": self.backbone.parameters(), "lr": self.lr * 0.1},
            {"params": self.seg_model.parameters()},
            {"params": self.kps_model.parameters()},
        ]
        if not self.no_depth:
            params.append({"params": self.dep_model.parameters()})

        optimizer = torch.optim.AdamW(
            params,
            lr=self.lr,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            self.trainer.max_epochs,
            eta_min=1e-5,
            last_epoch=-1,
            verbose=False,
        )
        return optimizer

    def setup(self, stage: str) -> None:
        common_feat_dim = 64
        common_conf = {"width": 256, "height": 256}

        self.backbone = self.encoder = UNet(3, common_feat_dim)

        num_kps = 68
        if self.num_classes_seg is None:
            num_classes = self.datamodule.num_classes
        else:
            num_classes = self.num_classes_seg
        self.seg_model = UNet(common_feat_dim, num_classes)
        print(num_classes, self.seg_model.outc.conv.weight.shape)
        if not self.no_depth:
            self.dep_model = UNet(common_feat_dim, 1)

        kps_config = Munch(
            {"out_height": 64, "out_width": 64, "use_AAM": True, **common_conf}
        )

        self.kps_model = StackedHGNetV2(
            config=kps_config,
            classes_num=[num_kps, 9, num_kps],
            edge_info=D300WDataset.EDGE_INFO,
            nstack=2,
            add_coord=False,
            x_in_channel=common_feat_dim,
            in_channel=common_feat_dim,
            skip_pre=False,
        )

        self.kps_decoder = get_decoder()
        self.decoders = {
            "kps": self.kps_model,
            "seg": self.seg_model,
        }
        if not self.no_depth:
            self.decoders["dep"] = (self.dep_model,)
        try:
            colormap = self.datamodule.COLORMAP
        except:
            colormap = SynthForgeDataset.COLORMAP
        self.f1 = F1Score(
            "seg_label",
            "pred_label",
            label_names=list(colormap.keys()) + ["background"],
        )
        self.f1.init_evaluation()
        self.automatic_optimization = False

    def backbone_forward(self, imgs):
        return self.backbone(imgs)

    def forward(self, imgs, batch=None):
        fusion_maps = self.backbone_forward(imgs)
        _, kps_maps = self.kps_model(fusion_maps.clone())
        kps = self.kps_decoder.get_coords_from_heatmap(kps_maps[-1])
        # kps is in range [-1, 1]
        fusion_mapsx = fusion_maps.clone()
        seg_maps = self.seg_model(fusion_mapsx)
        dep_maps = None
        if not self.no_depth:
            dep_maps = self.dep_model(fusion_mapsx)

        return {
            "kps": kps * 0.5 + 0.5,
            "n_kps": kps,
            "px_kps": ((kps * 0.5 + 0.5) * imgs.size(-1)),
            "heatmaps": kps_maps[-1],
            "seg": seg_maps,
            "seg_scores": seg_maps,
            "seg_label": torch.argmax(seg_maps, dim=1),
            "depth": dep_maps,
            "fusion_maps": fusion_maps,
        }

    def get_ssl_loss(self, batch, og_outs, return_labels=False, reduce=True):
        batch_size, _, h, w = batch["img"].shape
        d = self.device
        angle = torch.rad2deg(torch.rand(batch_size, device=d) * 2 - 1)
        transl = (torch.rand(batch_size, 2, device=d) * 2 - 1) * (h // 3)
        img_center = torch.tensor([127.5], device=d).repeat(batch_size, 2)
        A = G.Affine(angle=angle, translation=transl, center=img_center)
        aug_outs = self(A(batch["img"]), batch=batch)
        seg_loss, depth_loss, feats_loss, heatmaps_loss = 0, 0, 0, 0
        if not self.no_depth:
            valid_mask = A(batch["depth_mask"])
            num_valid = valid_mask.nonzero().size(0)
        labels = {}
        if self.ssl_type in ["full", "outputs"]:
            seg_label = torch.argmax(A(og_outs["seg"].detach()), dim=1)
            seg_loss = self.loss_fn(aug_outs["seg"], seg_label)
            if not self.no_depth:
                labels["depth"] = A(og_outs["depth"].detach())
                depth_loss = (aug_outs["depth"] - labels["depth"]).abs()
                depth_loss = (depth_loss * valid_mask).sum() / num_valid
        if self.ssl_type in ["full", "features"]:
            labels["features"] = A(og_outs["fusion_maps"].detach())
            feats_loss = (aug_outs["fusion_maps"] - labels["features"]).abs()
            feats_loss = feats_loss.sum()
        if self.ssl_type in ["full", "keypoints"]:
            valid_mask = A(batch["heatmaps"][:, :1])
            num_valid = valid_mask.nonzero().size(0)
            labels["heatmaps"] = A(og_outs["heatmaps"].detach())
            heatmaps_loss = (aug_outs["heatmaps"] - labels["features"]).abs()
            heatmaps_loss = (heatmaps_loss * valid_mask).sum() / num_valid

        loss = seg_loss + depth_loss + feats_loss + heatmaps_loss
        if return_labels:
            loss, labels
        return loss

    def get_losses_and_metrics(self, batch, return_pred=False):
        outs = self(batch['img'], batch=batch)

        ssl_loss = 0.0
        if self.enable_ssl_loss:
            ssl_loss = self.get_ssl_loss(batch, outs)

        n_tgts = (batch["kps"] / batch["img"].size(-1)) * 2 - 1
        if self.regress_heatmaps:
            assert outs["heatmaps"] is not None
            deviation = (outs["heatmaps"] - outs["heatmaps"]).abs()
            topk = deviation.flatten(start_dim=-2, end_dim=-1).topk(128).values
            kp_loss = topk.sum(dim=(-1, -2)).mean()
        else:
            kp_loss = (n_tgts - outs["n_kps"]).norm(dim=-1)

        kp_loss = kp_loss.sum(dim=-1).mean()
        if isinstance(self.loss_fn, DiceLoss) and self.current_epoch < 10:
            seg_loss = nn.CrossEntropyLoss()(outs["seg"], batch["seg_label"])
        else:
            seg_loss = self.loss_fn(outs["seg"], batch["seg_label"])

        depth_loss = 0
        if not self.no_depth:
            depth_loss = (outs["depth"] - batch["depth"]).abs()
            depth_loss *= batch["depth_mask"]
            depth_loss = depth_loss.sum()
            depth_loss /= torch.nonzero(batch["depth_mask"]).size(0)

        nme = NME(batch["kps"], outs["px_kps"], batch["iod"])
        self.f1.evaluate(
            {
                "seg_label": batch["seg_label"].cpu().numpy(),
                "pred_label": outs["seg_label"].cpu().numpy(),
            }
        )
        metrics = {
            "nme": nme,
            "kps_loss": kp_loss,
            "seg_loss": seg_loss,
            "depth_loss": depth_loss,
            "loss": kp_loss + seg_loss + depth_loss + ssl_loss,
        }
        if self.enable_ssl_loss:
            metrics["ssl_loss"] = ssl_loss

        if return_pred:
            return metrics, outs
        return metrics

    def on_train_epoch_start(self) -> None:
        self.f1.init_evaluation()

    def training_step(self, batch, batch_idx, dataloder_idx=0) -> STEP_OUTPUT:
        imgs = batch["img"]
        batch_size = imgs.size(0)
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
                **{f"train/{key}": v for key, v in metrics.items()},
            },
            prog_bar=True,
            add_dataloader_idx=False,
            on_step=True,
            batch_size=batch_size,
            sync_dist=True,
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
    def validation_step(self, batch, batch_idx, dataloder_idx=0, dl_tag=None) -> STEP_OUTPUT:
        dl_tag = f'{dl_tag}/' if dl_tag is not None else ''
        imgs, kps = batch["img"], batch["kps"]
        batch_size = imgs.size(0)
        metrics, outs = self.get_losses_and_metrics(batch, return_pred=True)
        px_pred = outs["px_kps"]

        self.log_dict(
            {f"{dl_tag}val/{key}": v for key, v in metrics.items()},
            prog_bar=False,
            add_dataloader_idx=False,
            on_epoch=True,
            batch_size=batch_size,
            sync_dist=True,
        )
        if batch_idx == 0 and self.local_rank == 0:
            plot_colors = [
                torch.tensor(x, device=self.device)
                for x in self.datamodule.COLORMAP.values()
            ]
            tgt_seg_map = seg_classes_to_colors(
                batch["seg_label"].unsqueeze(1), plot_colors
            )
            pred_seg_map = seg_classes_to_colors(
                outs["seg_label"].unsqueeze(1), plot_colors
            )
            plot = torch.tensor(
                self.plotter.get_plot(
                    batch["img"],
                    px_tgts=kps,
                    px_pred=px_pred,
                    tgt_seg_map=tgt_seg_map,
                    pred_seg_map=pred_seg_map,
                    tgt_depth_map=batch.get("depth", None),
                    pred_depth_map=outs.get("depth", None),
                )
            )
            self.logger.experiment.add_image(
                f"{dl_tag}val/out",
                plot.permute(2, 0, 1),
                self.current_epoch,
            )
            self.plotter.close_plot()

        return {
            f"{dl_tag}val/loss": metrics["loss"], 
            f"{dl_tag}val/nme": metrics["nme"]
        }

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

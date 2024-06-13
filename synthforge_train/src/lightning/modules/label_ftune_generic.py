import torch
from typing import Any, Optional


from munch import Munch
from torchvision.utils import *
from lightning.pytorch.utilities.types import STEP_OUTPUT
from kornia import geometry as G
from kornia.geometry.transform.imgwarp import get_affine_matrix2d


from .base_module import BaseModule
from .multimodal import MultiModalLightningModule
from .label_ftune_landmarks import LandmarksFinetuneLightningModule
from .label_ftune_depth import DepthLabelFinetuneLightningModule
from .label_ftune_segmentation import SegmentationLabelFinetuneLightningModule
from ...models.unet_la import OutConv
from ...models import StackedHGNetV2
from ...data.synthforge import SynthForgeDataset
from ...data.lapa_d import LaPaDepthDataset
from ...data.d300w import D300WDataset
from ...data.utils import seg_classes_to_colors
from ...utils.viz import Visualizer
from ...metrics.metrics import F1Score


def NME(target, pred, iod):
    loss = (target - pred).norm(dim=-1)  # [B, K]
    loss = loss.mean(dim=-1) / iod  # [B]
    return loss.mean()


def freeze(model):
    for k, v in model.named_parameters():
        v.requires_grad = False
        # print(f"Freezing {k}")


def unfreeze(model):
    for k, v in model.named_parameters():
        v.requires_grad = True
        # print(f"Freezing {k}")


def z2o(x):
    return (x - x.min()) / (x.max() - x.min())


class GenericLabelFinetuneLightningModule(BaseModule):
    def __init__(
        self,
        multimodal_ckpt=None,
        include_segmentation=False,
        include_keypoints=False,
        include_depth=False,
        learning_rate=1e-4,
        enable_ssl_loss=False,
        regress_heatmaps: bool = False,
        log_gradients: bool = False,
        accumulate_grad_batches: Optional[int] = None,
        num_keypoints: Optional[int] = None,
        datamodule: Optional[Any] = None,
        optimize_last_layer_only: bool = False,
        num_classes_seg= None,
        strict_loading: Optional[bool] = True,
    ):
        super().__init__(learning_rate=learning_rate)
        self.mm_model = MultiModalLightningModule.load_from_checkpoint(
            multimodal_ckpt,
            regress_heatmaps=regress_heatmaps,
            learning_rate=learning_rate,
            num_classes_seg=num_classes_seg or SynthForgeDataset.NUM_CLASSES,
            num_keypoints=num_keypoints,
            datamodule=None,
            no_depth=not include_depth,
            strict=strict_loading
        )
        self.datamodule = datamodule
        self.include_depth = include_depth
        self.include_segmentation = include_segmentation
        self.include_keypoints = include_keypoints
        self.optimize_last_layer_only = optimize_last_layer_only

        if self.include_depth:
            if optimize_last_layer_only:
                self.mm_model.dep_model.set_last_layer(OutConv(64, 1))
            else:
                self.ft_dep_model = DepthLabelFinetuneLightningModule(
                    multimodal_model=self.mm_model,
                    include_features=True,
                    include_segmentation=False,
                    include_keypoints=False,
                    datamodule=datamodule,
                )
        if self.include_keypoints:
            if optimize_last_layer_only:
                kps_config = Munch(
                    {
                        "out_height": 64,
                        "out_width": 64,
                        "use_AAM": True,
                        "width": 256,
                        "height": 256,
                    }
                )
                num_kps = self.datamodule.num_keypoints
                ft_kps_model = StackedHGNetV2(
                    config=kps_config,
                    classes_num=[num_kps, 9, num_kps],
                    edge_info=D300WDataset.EDGE_INFO,
                    nstack=2,
                    add_coord=False,
                    x_in_channel=64,
                    in_channel=64,
                    skip_pre=False,
                )
                ft_kps_model.pre = self.mm_model.kps_model.pre
                self.mm_model.kps_model = ft_kps_model
                # self.mm_model.kps_model.set_last_layers(
                # *ft_kps_model.last_layers
                # )
                del ft_kps_model
            else:
                ft_kps_model = LandmarksFinetuneLightningModule(
                    multimodal_model=self.mm_model,
                    include_features=True,
                    include_segmentation=False,
                    include_depth=False,
                    datamodule=datamodule,
                )
                self.ft_kps_model = ft_kps_model
        if self.include_segmentation:
            if optimize_last_layer_only:
                self.mm_model.seg_model.set_last_layer(
                    OutConv(64, datamodule.num_classes)
                )
            else:
                self.ft_seg_model = SegmentationLabelFinetuneLightningModule(
                    multimodal_model=self.mm_model,
                    include_features=True,
                    include_keypoints=False,
                    include_depth=False,
                    datamodule=datamodule,
                )
        self.plotter = Visualizer(n_cols=4, n_rows=8)

        self.is_setup = False
        self.mm_model.eval()
        self.automatic_optimization = False
        self.accumulate_grad_batches = accumulate_grad_batches
        self.log_gradients = log_gradients
        self.enable_ssl_loss = enable_ssl_loss
        self.setup("")

    def configure_optimizers(self) -> Any:
        params = []

        if self.include_depth:
            if self.optimize_last_layer_only:
                params += list(self.mm_model.dep_model.last_layer.parameters())
            else:
                params += list(self.ft_dep_model.ftune_model.parameters())
        if self.include_keypoints:
            if self.optimize_last_layer_only:
                last_layers = self.mm_model.kps_model.last_layers
                params += list(last_layers[0].parameters())
                params += list(last_layers[1].parameters())
                params += list(last_layers[2].parameters())
            else:
                params += list(self.ft_kps_model.ftune_model.parameters())
        if self.include_segmentation:
            if self.optimize_last_layer_only:
                params += list(self.mm_model.seg_model.last_layer.parameters())
            else:
                params += list(self.ft_seg_model.ftune_model.parameters())

        optimizer = torch.optim.AdamW(params, lr=self.lr)
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
        freeze(self.mm_model.backbone)
        freeze(self.mm_model.seg_model)
        if not self.mm_model.no_depth:
            freeze(self.mm_model.dep_model)
        freeze(self.mm_model.kps_model)
        if self.optimize_last_layer_only:
            self.mm_model.f1 = self.f1 = F1Score(
                "seg_label",
                "pred_label",
                label_names=list(self.datamodule.COLORMAP.keys())
                + ["background"],
            )
            if self.include_depth:
                unfreeze(self.mm_model.dep_model.last_layer)
            if self.include_keypoints:
                unfreeze(self.mm_model.kps_model.last_layers[0])
                unfreeze(self.mm_model.kps_model.last_layers[1])
                unfreeze(self.mm_model.kps_model.last_layers[2])
            if self.include_segmentation:
                unfreeze(self.mm_model.seg_model.last_layer)
        self.is_setup = True

    def forward(self, imgs):
        with torch.no_grad():
            fusion_maps = self.mm_model.backbone(imgs)
            outs = {"feature_maps": fusion_maps}
        if self.include_depth:
            depth = self.mm_model.dep_model(fusion_maps.clone())
            if self.optimize_last_layer_only:
                outs["la_depth"] = depth
            else:
                outs["depth"] = depth.detach()
                outs["la_depth"] = self.ft_dep_model.ftune_model(
                    torch.cat([outs["depth"], fusion_maps.clone()], dim=1)
                )
        if self.include_keypoints:
            _, heatmaps = self.mm_model.kps_model(fusion_maps.clone())
            heatmaps = heatmaps[-1]
            kps = self.mm_model.kps_decoder.get_coords_from_heatmap(heatmaps)
            if self.optimize_last_layer_only:
                outs.update(
                    {
                        "la_kps": kps * 0.5 + 0.5,
                        "la_n_kps": kps,
                        "la_px_kps": ((kps * 0.5 + 0.5) * imgs.size(-1)),
                        "la_heatmaps": heatmaps[-1],
                    }
                )
            else:
                outs.update(
                    {
                        "kps": kps * 0.5 + 0.5,
                        "n_kps": kps,
                        "px_kps": ((kps * 0.5 + 0.5) * imgs.size(-1)),
                    }
                )
                _, la_heatmaps, _ = self.ft_kps_model.ftune_model(
                    fusion_maps.clone(),
                    kp_res=heatmaps.detach(),
                    return_pre_out=True,
                )
                n_kps = self.ft_kps_model.kps_decoder.get_coords_from_heatmap(
                    la_heatmaps[-1]
                )
                outs.update(
                    {
                        "la_kps": n_kps * 0.5 + 0.5,
                        "la_n_kps": n_kps,
                        "la_px_kps": ((n_kps * 0.5 + 0.5) * imgs.size(-1)),
                        "la_heatmaps": la_heatmaps[-1],
                    }
                )
        if self.include_segmentation:
            seg = self.mm_model.seg_model(fusion_maps.clone())
            if self.optimize_last_layer_only:
                outs["la_seg_scores"] = seg
            else:
                outs["seg"] = seg.detach()
                outs["la_seg_scores"] = self.ft_seg_model.ftune_model(
                    torch.cat([outs["seg"], fusion_maps.clone()], dim=1)
                )
                outs["seg_label"] = torch.argmax(outs["seg"], dim=1)
        return outs

    def get_ssl_loss(self, batch, outs, return_labels=False):
        batch_size, _, h, w = batch["img"].shape
        d = self.device
        angle = torch.rad2deg(torch.rand(batch_size, device=d) * 2 - 1)
        transl = (torch.rand(batch_size, 2, device=d) * 2 - 1) * (h // 3)
        img_center = torch.tensor([127.5], device=d).repeat(batch_size, 2)
        A = G.Affine(angle=angle, translation=transl, center=img_center)
        affine_mat = get_affine_matrix2d(
            A.translation, A.center, A.scale_factor, -A.angle
        )

        outs = self(A(batch["img"]))
        # import pdb; pdb.set_trace()
        if self.include_segmentation:
            seg_loss = self.ft_seg_model.loss_fn(
                outs["la_seg_scores"], A(batch["seg_label"])
            )
        if self.include_depth:
            dep_loss = self.ft_dep_model.loss_fn(
                outs["la_depth"], A(batch["depth"])
            )
        if batch.get('kps') is not None and self.include_keypoints:
            kps_aug = affine_mat[:, :2, :3].unsqueeze(1) @ torch.cat(
                [batch["kps"], torch.ones_like(batch["kps"][..., :1])], dim=-1
            ).unsqueeze(-1)
            kps_aug = kps_aug.squeeze()
            n_aug_kps = (kps_aug / batch["img"].size(-1)) * 2 - 1
            kps_loss = (n_aug_kps - outs["la_n_kps"]).norm(dim=-1)
            kps_loss = kps_loss.sum(dim=-1).mean()
        return {"kps": kps_loss, "seg": seg_loss, "dep": dep_loss}

    def get_losses_and_metrics(self, batch, return_pred=False):
        if self.optimize_last_layer_only:
            metrics = self.mm_model.get_losses_and_metrics(
                batch, return_pred=return_pred
            )
            if return_pred:
                metrics, outs = metrics
                outs = {f"la_{k}": v for k, v in outs.items()}
                metrics = metrics, outs
            return metrics
        outs = self(batch["img"])

        ssl_loss = 0.0
        if self.enable_ssl_loss:
            ssl_loss = self.get_ssl_loss(batch, outs)

        metrics = {}
        weights = {}
        if self.include_depth:
            weights['dep'] = 0.5
            metrics["dep"] = self.ft_dep_model.get_losses_and_metrics(
                batch=batch, outs=outs
            )
        if self.include_segmentation:
            weights['seg'] = 1.0
            metrics["seg"] = self.ft_seg_model.get_losses_and_metrics(
                batch=batch, outs=outs
            )
        if self.include_keypoints:
            weights['kps'] = 0.2
            metrics["kps"] = self.ft_kps_model.get_losses_and_metrics(
                batch=batch, outs=outs
            )

        loss = sum(v["loss"] * weights[k] for k, v in metrics.items())

        metrics = {
            f"{k}_{kk}": vv for k, v in metrics.items() for kk, vv in v.items()
        }
        metrics["loss"] = loss

        if self.enable_ssl_loss:
            metrics["ssl_loss"] = ssl_loss
        if return_pred:
            return metrics, outs
        return metrics

    def on_train_epoch_start(self) -> None:
        if self.include_segmentation:
            if self.optimize_last_layer_only:
                return self.mm_model.on_train_epoch_start()
            return self.ft_seg_model.on_train_epoch_start()

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
                **{f"train/{k}": v for k, v in metrics.items()},
            },
            prog_bar=True,
            add_dataloader_idx=False,
            on_step=True,
            batch_size=batch_size,
            sync_dist=True
        )
        self.log_gradients_in_model(self.current_iter(batch_idx), ignore_frozen=True)
        return {"train/loss": metrics["loss"]}

    def on_train_epoch_end(self) -> None:
        if self.include_segmentation:
            if self.optimize_last_layer_only:
                ff = self.f1.finalize_evaluation()
            else:
                ff = self.ft_seg_model.f1.finalize_evaluation()
            self.log_dict(
                {f"train/f1_{k}": v for k, v in ff.items()},
                add_dataloader_idx=False,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                rank_zero_only=True,
            )
        self.scheduler.step()

    def on_validation_epoch_start(self) -> None:
        if self.include_segmentation:
            if self.optimize_last_layer_only:
                return self.mm_model.on_validation_epoch_start()
            return self.ft_seg_model.on_validation_epoch_start()

    @torch.no_grad()
    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        batch_size = batch["img"].size(0)
        metrics, outs = self.get_losses_and_metrics(batch, return_pred=True)
        
        self.log_dict(
            {f"val/{k}": v for k, v in metrics.items()},
            prog_bar=True,
            add_dataloader_idx=False,
            on_epoch=True,
            sync_dist=True
        )
        if batch_idx == 0:
            tgt_seg_map = None
            pred_seg_map = None
            pred_seg_map_la = None
            if self.include_segmentation:
                plot_colors_lapa = [
                    torch.tensor(x, device=self.device)
                    for x in LaPaDepthDataset.COLORMAP.values()
                ]
                plot_colors = [
                    torch.tensor(x, device=self.device)
                    for x in SynthForgeDataset.COLORMAP.values()
                ]
                tgt_seg_map = seg_classes_to_colors(
                    batch["seg_label"].unsqueeze(1), plot_colors_lapa
                )
                if self.optimize_last_layer_only:
                    pred_seg_map = None
                else:
                    pred_seg_map = seg_classes_to_colors(
                        outs["seg_label"].unsqueeze(1), plot_colors
                    )
                pred_seg_map_la = seg_classes_to_colors(
                    outs["la_seg_scores"].argmax(dim=-3, keepdims=True),
                    plot_colors_lapa,
                )
            px_tgts = batch["kps"] if self.include_keypoints else None
            tgt_depth_map = batch["depth"] if self.include_depth else None
            plot = torch.tensor(
                self.plotter.get_plot(
                    batch["img"],
                    px_tgts=px_tgts,
                    px_pred=outs.get("px_kps", None),
                    px_pred_la=outs["la_px_kps"],
                    tgt_depth_map=tgt_depth_map,
                    pred_depth_map=outs.get("depth", None),
                    pred_depth_map_la=outs.get("la_depth", None),
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
        if self.include_segmentation:
            if self.optimize_last_layer_only:
                return self.mm_model.on_validation_epoch_end()
            return self.ft_seg_model.on_validation_epoch_end()

    def on_test_epoch_start(self):
        return self.on_validation_epoch_start()

    def on_test_epoch_end(self):
        return self.on_validation_epoch_end()

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def on_test_end(self):
        if self.include_segmentation:
            if self.optimize_last_layer_only:
                return self.mm_model.on_test_end()
            return self.ft_seg_model.on_test_end()

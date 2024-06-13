import os
from munch import Munch
from typing import Any, Optional
from lightning.pytorch.utilities.types import STEP_OUTPUT
import torch
import torch.nn as nn
import numpy as np

from torchvision.utils import *
from torchvision import transforms as T
from ...metrics.dice_loss import DiceLoss
from kornia import geometry as G
from .multimodal import MultiModalLightningModule, NME

class ActiveMultiModalLightningModule(MultiModalLightningModule):
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
        enable_grad_balancing: bool = False,
        enable_ssl_loss: bool = False,
        ssl_type: Optional[str] = None,
        no_depth: Optional[bool] = False,
        use_roi_tanh_warping: Optional[bool] = False,
        update_samples_every_n_epochs: Optional[int] = 10,
        tau: Optional[float] = 0.2,
    ):
        loss_fn = loss_fn or nn.CrossEntropyLoss(reduction='none')
        super().__init__(
            regress_heatmaps=regress_heatmaps,
            learning_rate=learning_rate,
            log_gradients=log_gradients,
            accumulate_grad_batches=accumulate_grad_batches,
            loss_fn=loss_fn,
            num_classes_seg=num_classes_seg,
            num_keypoints=num_keypoints,
            datamodule=datamodule,
            enable_grad_balancing=enable_grad_balancing,
            enable_ssl_loss=enable_ssl_loss,
            ssl_type=ssl_type,
            no_depth=no_depth,
            use_roi_tanh_warping=use_roi_tanh_warping,
        )
        self.update_samples_every_n_epochs = update_samples_every_n_epochs
        self.tau = tau

    def __register_stats(self):
        assert self.trainer.datamodule is not None, \
            'Datamodule is not defined at this stage!'
        try:
            self.counts
        except:
            active_wrapper = self.trainer.datamodule.train_dataset
            counts = torch.tensor(active_wrapper.counts)
            cum_loss = torch.zeros_like(counts)
            self.register_buffer('cum_loss', cum_loss.to(self.device).clone())
            self.register_buffer('counts', counts.to(self.device).clone())

    def load_state_dict(self, *args, **kwargs):
        self.__register_stats()
        super().load_state_dict(*args, **kwargs)
        self.start_epoch = self.current_epoch

    def on_fit_start(self):
        self.__register_stats()
        assert self.update_samples_every_n_epochs % self.trainer.check_val_every_n_epoch == 0, \
        'update_samples_every_n_epochs should be a multiple of '\
        'trainer.check_val_every_n_epoch'
        os.makedirs(f'{self.log_path}/stats')
        
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
        labels = {}
        if self.ssl_type in ["full", "outputs"]:
            seg_label = torch.argmax(A(og_outs["seg"].detach()), dim=1)
            seg_loss = self.loss_fn(aug_outs["seg"], seg_label)
            seg_loss = seg_loss.mean(dim=(-1, -2))
            if not self.no_depth:
                labels["depth"] = A(og_outs["depth"].detach())
                depth_loss = (aug_outs["depth"] - labels["depth"]).abs()
                depth_loss = (depth_loss * valid_mask).sum(dim=(1, 2, 3)) / valid_mask.sum(dim=(1, 2, 3))
        if self.ssl_type in ["full", "features"]:
            labels["features"] = A(og_outs["fusion_maps"].detach())
            feats_loss = (aug_outs["fusion_maps"] - labels["features"]).abs()
            feats_loss = feats_loss.mean(dim=(2, 3)).sum(dim=-1)
        if self.ssl_type in ["full", "keypoints"]:
            valid_mask = A(batch["heatmaps"])
            labels["heatmaps"] = A(og_outs["heatmaps"].detach())
            heatmaps_loss = (aug_outs["heatmaps"] - labels["heatmaps"]).abs()
            heatmaps_loss = (heatmaps_loss * valid_mask).sum(dim=(1, 2, 3))

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
            kp_loss = topk.sum(dim=(-1, -2))
        else:
            kp_loss = (n_tgts - outs["n_kps"]).norm(dim=-1)
        kp_loss = kp_loss.sum(dim=-1)
        
        if isinstance(self.loss_fn, DiceLoss) and self.current_epoch < 10:
            seg_loss = nn.CrossEntropyLoss(reduction='none')(outs["seg"], batch["seg_label"])
        else:
            seg_loss = self.loss_fn(outs["seg"], batch["seg_label"]).mean(dim=(1, 2))

        depth_loss = 0
        if not self.no_depth:
            depth_loss = (outs["depth"] - batch["depth"]).abs()
            depth_loss *= batch["depth_mask"]
            depth_loss = depth_loss.sum(dim=(1, 2, 3)) / batch["depth_mask"].sum(dim=(1, 2, 3))
            

        nme = NME(batch["kps"], outs["px_kps"], batch["iod"])
        if self.trainer.state.stage == 'train' or not self.corpus_mode:
            self.f1.evaluate(
                {
                    "seg_label": batch["seg_label"].cpu().numpy(),
                    "pred_label": outs["seg_label"].cpu().numpy(),
                }
            )
        loss = kp_loss + seg_loss + depth_loss + ssl_loss
        if self.trainer.state.stage == 'validate' and self.corpus_mode:
            self.cum_loss[batch['idx'].cpu()] += loss.detach()

        metrics = {
            "nme": nme.mean(),
            "kps_loss": kp_loss.mean(),
            "seg_loss": seg_loss.mean(),
            "depth_loss": depth_loss.mean(),
            "loss": loss.mean(),
        }
        if self.enable_ssl_loss:
            metrics["ssl_loss"] = ssl_loss.mean()

        if return_pred:
            return metrics, outs
        return metrics

    @torch.no_grad()
    def validation_step(self, batch, batch_idx, dataloader_idx=0) -> STEP_OUTPUT:
        if dataloader_idx == 0:
            self.corpus_mode = True
            
            return super().validation_step(batch, batch_idx, dataloader_idx, dl_tag='corpus')
        elif dataloader_idx == 1:
            self.corpus_mode = False
            return super().validation_step(batch, batch_idx, dataloader_idx, dl_tag='val')
        else:
            raise NotImplementedError

    def on_validation_epoch_end(self) -> None:
        if self.current_epoch < 1 or self.current_epoch < self.start_epoch:
            return
        self.counts = torch.tensor(
            self.trainer.datamodule.train_dataset.counts
        ).to(self.counts.device)

        # import pdb; pdb.set_trace()
        if ((self.current_epoch + 1) % self.update_samples_every_n_epochs) == 0:
            # print("IS THIS CALLED?  "*10)
            probs = torch.nn.functional.softmax(
                self.cum_loss * self.tau,
                dim=-1
            )
            self.trainer.datamodule.set_probs(
                probs.detach().cpu().numpy()
            )
            self.trainer.datamodule.update_samples()
            with open(f'{self.log_path}/stats/counts.csv', 'ab+') as f:
                counts = self.counts.cpu().long().numpy().tolist()
                data = [self.current_epoch] + counts
                np.savetxt(f, [data], delimiter=',', fmt='%d')
                f.write(b'\n')
            with open(f'{self.log_path}/stats/probs.csv', 'ab+') as f:
                data = [self.current_epoch] + probs.cpu().numpy().tolist()
                np.savetxt(f, [data], delimiter=',', newline='\n')
                f.write(b'\n')
        return super().on_validation_epoch_end()

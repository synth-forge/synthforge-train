import os
import json
import torch
import numpy as np
from torchvision import transforms as T
from torchvision.transforms import functional as F
from .utils import crop_to_keypoints, Erosion2d, Dilation2d
from .augmentation import Augmentation
from ..models.encoder_default import get_encoder
from .synthforge import SynthForgeDataset
import kornia as K


def load_json(json_file):
    with open(json_file, "r") as f:
        x = json.load(f)
    return x


class SynthForgeDecaDataset(SynthForgeDataset):
    def __init__(
        self,
        dataset_path,
        is_train=True,
        disable_crop_op=False,
        include_depth=True,
        crop_size=None,
    ):
        self.dataset_path = dataset_path
        self.annotations_dir = os.path.join(dataset_path, "annotations")
        self.images_dir = os.path.join(dataset_path, "images")
        self.depth_dir = os.path.join(dataset_path, "depth")
        self.seg_dir = os.path.join(dataset_path, "seg")
        self.images = os.listdir(self.images_dir)
        self.include_depth = include_depth
        if crop_size is not None:
            self.crop_size = crop_size
        else:
            self.crop_size = 256

        # Copied from STAR.lib.dataset.alignmentDataset.AlignmentDataset
        self.encoder = get_encoder(256, 256, encoder_type="default")
        crop_op = True
        self.disable_crop_op = disable_crop_op
        self.is_train = is_train
        target_face_scale = 1.0 if crop_op else 1.25
        self.augmentation = Augmentation(
            is_train=self.is_train,
            aug_prob=1.0,
            image_size=self.crop_size,
            crop_op=crop_op,
            std_lmk_5pts=None,
            target_face_scale=target_face_scale,
            flip_rate=0.5,
            flip_mapping=self.FLIP_MAPPING,
            random_shift_sigma=0.05,
            random_rot_sigma=np.pi / 180 * 18,
            random_scale_sigma=0.1,
            random_gray_rate=0.2,
            random_occ_rate=0.4,
            random_blur_rate=0.3,
            random_gamma_rate=0.2,
            random_nose_fusion_rate=0.2,
        )
        self.erode = Erosion2d(1, 1, 3, soft_max=False)
        self.dilate = Dilation2d(1, 1, 3, soft_max=False)

    def __getitem__(self, idx):
        # rets = self._annots[idx]
        img = self.get_img(idx)
        annot = self.get_annotation(idx)
        seg = self.get_segmentation(idx)
        depth = self.get_depth(idx)
        og_img = img.copy()
        rets = {"img": img, **{k: np.array(v) for k, v in annot.items()}}
        scale = 2.4
        if self.is_train:
            scale = np.clip(2.6 + np.random.randn() * 0.75, 2.2, 4)
        # rets["2d_kps"] *= self.crop_size
        center_w, center_h = rets["2d_kps"].mean(axis=0)
        (
            rets["img"],
            rets["kps"],
            matrix,
            is_flipped,
            rets["seg"],
            rets["depth"],
        ) = self.augmentation.process(
            img,
            # self._imgs[idx].numpy(),
            rets["2d_kps"],
            scale=scale,
            center_h=center_h,
            center_w=center_w,
            seg=seg,
            depth=depth.astype("f") / 255.0,
            # seg=self._segs[idx].numpy(),
        )
        rets["img"] = T.ToTensor()(rets["img"]) * 2 - 1
        rets["seg"] = T.ToTensor()(rets["seg"])
        rets["depth"] = torch.tensor(rets["depth"])
        kps = torch.tensor(rets["kps"])
        if not self.disable_crop_op:
            rets["img"], rets["kps"] = crop_to_keypoints(
                rets["img"], kps.clone(), self.crop_size, padding=10
            )
            rets["seg"], _ = crop_to_keypoints(
                rets["seg"], kps.clone(), self.crop_size, padding=10
            )
            rets["depth"], _ = crop_to_keypoints(
                rets["depth"], kps.clone(), self.crop_size, padding=10
            )
            # print(rets['depth'].shape)
        else:
            rets["kps"] = torch.tensor(rets["kps"])
        rets["iod"] = (rets["kps"][..., 36, :] - rets["kps"][..., 45, :]).norm()
        rets["img_path"] = f"{self.images_dir}/{self.images[idx]}"
        rets["og_img"] = T.ToTensor()(og_img)
        rets["heatmaps"] = self.encoder.generate_heatmap(rets["kps"])
        rets["seg_masks"] = self.get_seg_map(rets["seg"])

        if is_flipped:
            for a, b in self.FLIP_MAPPING_SEG:
                rets["seg_masks"][[a, b]] = rets["seg_masks"][[b, a]]

        rets["seg_label"] = torch.argmax(rets["seg_masks"], dim=0)
        rets["depth"], rets["depth_mask"] = rets["depth"].unsqueeze(dim=1)
        rets["matrix"] = matrix
        rets["idx"] = idx
        return rets

    def get_seg_map(self, seg):
        seg_LAB = K.color.rgb_to_lab(seg)
        masks = []
        for k, v in SynthForgeDataset.COLORMAP.items():
            if k == "background":
                continue
            v_LAB = K.color.rgb_to_lab(torch.tensor(v).reshape(3, 1, 1))
            delta_E = (seg_LAB - v_LAB).norm(dim=0)
            mask = torch.zeros_like(seg[0])
            mask[torch.where(delta_E < 2.3)] = 1.0
            mask = self.dilate(mask.reshape(1, 1, *mask.shape)).squeeze()
            masks.append(mask)
        bg_seg = sum(masks).clamp(0, 1).reshape(1, 1, *masks[0].shape)
        return torch.stack([*masks, 1 - bg_seg.squeeze()])

    def __len__(self):
        if self.h5_mode:
            return len(self._h5["imgs"])
        return len(self.images)

    @property
    def num_classes(self):
        return 12

import os
import cv2
import torch
import numpy as np
from colour import Color
from torchvision import transforms as T
from torchvision.transforms import functional as F

from .augmentation import Augmentation
from .utils import crop_to_keypoints


class LaPaDepthDataset(torch.utils.data.Dataset):
    """LaPa face parsing dataset

    Args:
        root (str): The directory that contains subdirs 'image', 'labels'
    """

    COLORMAP = {
        "hair": Color("#440f85").rgb,
        "face_lr_rr": Color("#f5dccb").rgb,
        "lb": Color("#440f85").rgb,
        "rb": Color("#9dab3b").rgb,
        "le": Color("#e4f18f").rgb,
        "re": Color("#c602ac").rgb,
        "nose": Color("#d38f8f").rgb,
        "ul": Color("#00ff3c").rgb,
        "im": Color("#656b38").rgb,
        "ll": Color("#fcff00").rgb,
    }
    LABEL_NAMES = [
        "hair",
        "face_lr_rr",
        "lb",
        "rb",
        "le",
        "re",
        "nose",
        "ul",
        "im",
        "ll",
        "background",
    ]
    NUM_CLASSES = len(LABEL_NAMES)
    FLIP_MAPPING_SEG = [[2, 3], [4, 5]]

    def __init__(
        self,
        root,
        split,
        force_augmentation=False,
        force_tightcrop=False,
        skip_aug=False,
    ):
        assert os.path.isdir(root)
        self.root = root
        self.split = split
        assert split in [
            "train",
            "val",
            "test",
        ], "choose from: train, val, test"
        self.dep_root = (
            "/pfs01/performance-tier/rd_algo/algo_bin/abrawat/datasets/LaPa-D"
        )
        self.is_train = split == "train"
        self.force_tightcrop = force_tightcrop
        target_face_scale = 1
        crop_op = True
        self.info = []
        self.skip_aug = skip_aug
        self.augmentation = Augmentation(
            is_train=self.is_train or force_augmentation,
            aug_prob=1.0,
            image_size=256,
            crop_op=crop_op,
            std_lmk_5pts=None,
            target_face_scale=target_face_scale,
            flip_rate=0.0,
            flip_mapping=None,
            random_shift_sigma=0.05,
            random_rot_sigma=np.pi / 180 * 18,
            random_scale_sigma=0.1,
            random_gray_rate=0.2,
            random_occ_rate=0.4,
            random_blur_rate=0.3,
            random_gamma_rate=0.2,
            random_nose_fusion_rate=0.2,
        )

        with open(os.path.join(self.dep_root, f"{split}_list.txt")) as f:
            self.files = map(lambda x: x.strip(), f.readlines())

        for name in self.files:
            image_path = os.path.join(self.root, split, "images", f"{name}.jpg")
            label_path = os.path.join(self.root, split, "labels", f"{name}.png")
            landmark_path = os.path.join(
                self.root, split, "landmarks", f"{name}.txt"
            )
            depth_path = os.path.join(self.dep_root, "depth", f"{name}.png")
            assert os.path.exists(image_path), f"{image_path} not found"
            assert os.path.exists(label_path), f"{label_path} not found"
            assert os.path.exists(landmark_path), f"{landmark_path} not found"
            landmarks = [
                float(v) for v in open(landmark_path, "r").read().split()
            ]
            assert landmarks[0] == 106 and len(landmarks) == 106 * 2 + 1
            landmarks = np.reshape(
                np.array(landmarks[1:], np.float32), [106, 2]
            )
            sample_name = f"{split}.{name}"
            self.info.append(
                {
                    "image_path": image_path,
                    "label_path": label_path,
                    "landmarks": landmarks,
                    "sample_name": sample_name,
                    "depth_path": depth_path,
                }
            )

    def __getitem__(self, index):
        info = self.info[index]
        img = cv2.imread(info["image_path"])[:, :, ::-1]
        img = img.astype("f").astype("uint8")
        label = cv2.imread(info["label_path"], cv2.IMREAD_GRAYSCALE)
        depth = cv2.imread(info["depth_path"], cv2.IMREAD_GRAYSCALE)
        og_label = label.copy()
        og_label = torch.tensor(og_label)

        x, y = torch.where(og_label > 0)
        imh, imw = og_label.shape[-2:]
        min_y, min_x = float(x.min()), float(y.min())
        max_y, max_x = float(x.max()), float(y.max())
        h, w = max_x - min_x, max_y - min_y

        kps = info["landmarks"]
        og_kps = kps.copy()
        center_h, center_w = 0.5 * (min_x + max_x), 0.5 * (min_y + max_y)
        scale = max(img.shape) / 256
        # print(f"center_h [{center_h}] | center_w [{center_w}]")
        is_flipped = False
        if self.is_train and not self.skip_aug:
            (
                img,
                kps,
                matrix,
                is_flipped,
                label,
                depth,
            ) = self.augmentation.process(
                img,
                kps,  # if self.use_fan else rets["2d_kps"],
                scale=scale,
                center_h=center_h,
                center_w=center_w,
                seg=label,
                depth=depth,
                # seg=self._segs[idx].numpy(),
            )
            img = T.ToTensor()(img)
            label = torch.tensor(label)
            depth = torch.tensor(depth.astype("f"))
            depth, depth_mask = depth.squeeze(dim=0)
            depth = depth / 255.0
            kps = torch.tensor(kps)
        else:
            img = T.ToTensor()(img)
            label = torch.tensor(label)
            depth = torch.tensor(depth).float() / 255.0
            if self.force_tightcrop:
                bounds = torch.tensor(kps).clone()
            else:
                bounds = torch.tensor([[min_x, min_y], [max_x, max_y]])
            img, _, tf = crop_to_keypoints(
                img, bounds.clone(), 256, padding=80, get_tf=True
            )
            label, _ = crop_to_keypoints(
                label,
                bounds.clone(),
                256,
                padding=80,
                interpolation=F.InterpolationMode.NEAREST,
            )
            depth, _ = crop_to_keypoints(
                depth,
                bounds,
                256,
                padding=80,
                interpolation=F.InterpolationMode.BILINEAR,
            )
            label = label.squeeze(dim=0)
            depth = depth.squeeze(dim=0)
            depth_mask = torch.ones_like(depth)
            kps = tf(torch.tensor(kps))
        img = img * 2 - 1

        # label = torch.tensor(og_label)
        # img = T.ToTensor()(og_img) * 2 - 1
        # kps = torch.tensor(og_kps)
        _kps = kps.clone()

        label = label.long()
        bg = label == 0
        hair = label == 10
        label[bg] = 10
        label[hair] = 0
        if is_flipped:
            for a, b in self.FLIP_MAPPING_SEG:
                mx_a = label == a
                mx_b = label == b
                label[mx_a] = b
                label[mx_b] = a
        rets = {
            "img": img,
            "seg_label": label,
            "kps": kps,
            "depth": depth.unsqueeze(0),
            "depth_mask": depth_mask.unsqueeze(0),
            "idx": index,
        }
        rets["iod"] = (rets["kps"][..., 66, :] - rets["kps"][..., 79, :]).norm()
        return rets

    def __len__(self):
        return len(self.info)

    def sample_name(self, index):
        return self.info[index]["sample_name"]

    @property
    def label_names(self):
        return self.LABEL_NAMES

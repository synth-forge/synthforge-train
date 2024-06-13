import os
import tqdm
import json
import torch
import numpy as np
from collections import OrderedDict
from PIL import Image
from torchvision import transforms as T
from torchvision.transforms import functional as F
from scipy.io import loadmat
from .utils import crop_to_keypoints, Erosion2d, Dilation2d
from .augmentation import Augmentation
from ..models.encoder_default import get_encoder
from colour import Color
import kornia as K


def load_json(json_file):
    with open(json_file, "r") as f:
        x = json.load(f)
    return x


class SynthForgeDataset(torch.utils.data.Dataset):
    EDGE_INFO = (
        (
            False,
            (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16),
        ),  # FaceContour
        (False, (17, 18, 19, 20, 21)),  # RightEyebrow
        (False, (22, 23, 24, 25, 26)),  # LeftEyebrow
        (False, (27, 28, 29, 30)),  # NoseLine
        (False, (31, 32, 33, 34, 35)),  # Nose
        (True, (36, 37, 38, 39, 40, 41)),  # RightEye
        (True, (42, 43, 44, 45, 46, 47)),  # LeftEye
        (True, (48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59)),  # OuterLip
        (True, (60, 61, 62, 63, 64, 65, 66, 67)),  # InnerLip
    )
    FLIP_MAPPING = (
        [0, 16],
        [1, 15],
        [2, 14],
        [3, 13],
        [4, 12],
        [5, 11],
        [6, 10],
        [7, 9],
        [17, 26],
        [18, 25],
        [19, 24],
        [20, 23],
        [21, 22],
        [31, 35],
        [32, 34],
        [36, 45],
        [37, 44],
        [38, 43],
        [39, 42],
        [40, 47],
        [41, 46],
        [48, 54],
        [49, 53],
        [50, 52],
        [61, 63],
        [60, 64],
        [67, 65],
        [58, 56],
        [59, 55],
    )
    COLORMAP = OrderedDict(
        {
            "face": Color("#f5dccb").rgb,
            "forehead": Color("#440f85").rgb,
            "eye_region": Color("#9dab3b").rgb,
            "neck": Color("#e4f18f").rgb,
            "right_eye_region": Color("#757ae4").rgb,
            "left_eye_region": Color("#e5a5e7").rgb,
            "right_eyeball": Color("#0000ff").rgb,
            "left_eyeball": Color("#f600ff").rgb,
            "left_eyebrow": Color("#8fa9d3").rgb,
            "right_eyebrow": Color("#d38f8f").rgb,
            "eye_region+forehead": Color("#656b38").rgb,
            "left_ear": Color("#00ff3c").rgb,
            "right_ear": Color("#fcff00").rgb,
            "lower_lip": Color("#008880").rgb,
            "upper_lip": Color("#d91717").rgb,
            "inner_mouth": Color("#002f99").rgb,
            "nose": Color("#c602ac").rgb,
            "scalp": Color("#683516").rgb,
            "neck_back": Color("#50443d").rgb,
        }
    )

    SEG_COLORMAP = OrderedDict(
        {
            "face": Color("#f5dccb").rgb,
            "forehead": Color("#f5dccb").rgb,
            "eye_region": Color("#f5dccb").rgb,
            "neck": Color("#000000").rgb,
            "right_eye_region": Color("#f5dccb").rgb,
            "left_eye_region": Color("#f5dccb").rgb,
            "right_eyeball": Color("#0000ff").rgb,
            "left_eyeball": Color("#f600ff").rgb,
            "left_eyebrow": Color("#8fa9d3").rgb,
            "right_eyebrow": Color("#d38f8f").rgb,
            "eye_region+forehead": Color("#f5dccb").rgb,
            "left_ear": Color("#00ff3c").rgb,
            "right_ear": Color("#fcff00").rgb,
            "lower_lip": Color("#008880").rgb,
            "upper_lip": Color("#d91717").rgb,
            "inner_mouth": Color("#002f99").rgb,
            "nose": Color("#c602ac").rgb,
            "scalp": Color("#f5dccb").rgb,
            "neck_back": Color("#000000").rgb,
        }
    )
    # NUM_CLASSES = 11 + 1
    NUM_CLASSES = len(COLORMAP) + 1
    FLIP_MAPPING_SEG = [[4, 5], [6, 7], [8, 9], [11, 12]]
    COLORMAP["background"] = Color("#454545").rgb
    SEG_COLORMAP["background"] = Color("#000000").rgb

    def __init__(
        self,
        dataset_path: str,
        is_train: bool = True,
        disable_crop_op: bool = False,
        include_depth: bool = True,
        crop_size: bool = None,
        debug_mode: bool = False,
    ):
        self.dataset_path = dataset_path
        self.debug_mode = debug_mode
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

    def _load_ds(self):
        pbar = tqdm.tqdm(
            range(len(self)), desc="Loading Images to RAM", total=len(self)
        )
        self._imgs = []
        self._segs = []
        self._deps = []
        self._annots = []
        for idx in pbar:
            img_id = self.images[idx].split(".")[0]
            img_path = f"{self.images_dir}/{self.images[idx]}"
            self._imgs.append(np.array(Image.open(img_path)))
            seg_path = f"{self.seg_dir}/{self.images[idx]}"
            self._segs.append(np.array(Image.open(seg_path)))
            annot = load_json(f"{self.annotations_dir}/{img_id}.json")
            self._annots.append({k: np.array(v) for k, v in annot.items()})
            depth_path = f"{self.depth_dir}/{self.images[idx]}"
            self._deps.append(np.array(Image.open(depth_path)))

    def _get_crop_bb_new(self, kpt2d, img):
        x0, y0 = kpt2d.min(dim=0).values
        x1, y1 = kpt2d.max(dim=0).values

        # Keep extra buffer
        x0 = x0 - 10 if x0 - 10 > 0 else 0
        y0 = y0 - 10 if y0 - 10 > 0 else 0
        x1 = x1 + 10 if x1 + 10 < img.shape[0] else img.shape[0]
        y1 = y1 + 10 if y1 + 10 < img.shape[1] else img.shape[1]
        return x0, x1, y0, y1

    def get_img(self, idx):
        img_path = f"{self.images_dir}/{self.images[idx]}"
        return np.array(Image.open(img_path))

    def get_annotation(self, idx):
        img_id = self.images[idx].split(".")[0]
        return load_json(f"{self.annotations_dir}/{img_id}.json")

    def get_segmentation(self, idx):
        seg_path = f"{self.seg_dir}/{self.images[idx]}"
        return np.array(Image.open(seg_path))

    def get_depth(self, idx):
        depth_path = f"{self.depth_dir}/{self.images[idx]}"
        return np.array(Image.open(depth_path))

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
        if self.debug_mode:
            return 100
        return len(self.images)

    @property
    def num_classes(self):
        return 12

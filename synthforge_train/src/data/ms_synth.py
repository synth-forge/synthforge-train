import torch
import numpy as np
from collections import OrderedDict
from PIL import Image
from torchvision import transforms as T
from torchvision.transforms import functional as F
from .utils import crop_to_keypoints, Erosion2d, Dilation2d
from .augmentation import Augmentation
from ..models.encoder_default import get_encoder
from colour import Color


class MSSynthDataset(torch.utils.data.Dataset):
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
            "FACEWEAR": Color("#f5dccb").rgb,
            "SKIN": Color("#440f85").rgb,
            "NOSE": Color("#9dab3b").rgb,
            "HAIR": Color("#008880").rgb,
            "RIGHT_EYE": Color("#e4f18f").rgb,
            "LEFT_EYE": Color("#757ae4").rgb,
            "RIGHT_BROW": Color("#e5a5e7").rgb,
            "LEFT_BROW": Color("#0000ff").rgb,
            "RIGHT_EAR": Color("#f600ff").rgb,
            "LEFT_EAR": Color("#8fa9d3").rgb,
            "MOUTH_INTERIOR": Color("#d38f8f").rgb,
            "TOP_LIP": Color("#656b38").rgb,
            "BOTTOM_LIP": Color("#00ff3c").rgb,
            "NECK": Color("#fcff00").rgb,
            "BEARD": Color("#d91717").rgb,
            "CLOTHING": Color("#002f99").rgb,
            "GLASSES": Color("#c602ac").rgb,
            "HEADWEAR": Color("#683516").rgb,
        }
    )
    # NUM_CLASSES = 11 + 1
    NUM_CLASSES = len(COLORMAP) + 1
    FLIP_MAPPING_SEG = [[4, 5], [6, 7], [8, 9]]

    def __init__(
        self,
        dataset_path,
        is_train=True,
        disable_crop_op=True,
        crop_size=None,
    ):
        self.dataset_path = dataset_path
        if crop_size is not None:
            self.crop_size = crop_size
        else:
            self.crop_size = 256

        crop_op = True
        self.disable_crop_op = disable_crop_op
        self.is_train = is_train
        target_face_scale = 1.25 if crop_op else 1.25
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

        self.encoder = get_encoder(256, 256, encoder_type="default")
        self.erode = Erosion2d(1, 1, 3, soft_max=False)
        self.dilate = Dilation2d(1, 1, 3, soft_max=False)

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
        img_path = f"{self.dataset_path}/{idx:0>6}.png"
        return np.array(Image.open(img_path))

    def get_annotation(self, idx):
        return np.genfromtxt(
            f"{self.dataset_path}/{idx:0>6}_ldmks.txt", delimiter=" "
        )[:68]

    def get_segmentation(self, idx):
        seg_path = f"{self.dataset_path}/{idx:0>6}_seg.png"
        seg = np.array(Image.open(seg_path))
        seg[seg == 255] = 0
        return seg

    def __getitem__(self, idx):
        # rets = self._annots[idx]
        img = self.get_img(idx)
        annot = self.get_annotation(idx)
        seg = self.get_segmentation(idx)
        og_img = img.copy()
        rets = {"img": img, "kps": annot}
        scale = 1.5
        if self.is_train:
            scale = np.clip(1.2 + np.random.randn() * 0.5, 1.0, 2.0)
        center_w, center_h = rets["kps"].mean(axis=0)
        (
            rets["img"],
            rets["kps"],
            matrix,
            is_flipped,
            rets["seg"],
        ) = self.augmentation.process(
            img,
            # self._imgs[idx].numpy(),
            rets["kps"],  # if self.use_fan else rets["2d_kps"],
            scale=scale,
            center_h=center_h,
            center_w=center_w,
            seg=seg,
        )
        rets["img"] = T.ToTensor()(rets["img"]) * 2 - 1
        rets["seg"] = torch.from_numpy(rets["seg"]).long()
        kps = torch.tensor(rets["kps"])
        if not self.disable_crop_op:
            rets["img"], rets["kps"] = crop_to_keypoints(
                rets["img"], kps.clone(), self.crop_size, padding=10
            )
            rets["seg"], _ = crop_to_keypoints(
                rets["seg"],
                kps.clone(),
                self.crop_size,
                padding=10,
                interpolation=F.InterpolationMode.NEAREST,
            )
        else:
            rets["kps"] = torch.tensor(rets["kps"])
        rets["iod"] = (rets["kps"][..., 36, :] - rets["kps"][..., 45, :]).norm()
        # rets["img_path"] = img_path
        rets["og_img"] = T.ToTensor()(og_img)
        rets["heatmaps"] = self.encoder.generate_heatmap(rets["kps"])
        rets["seg_masks"] = rets["seg"]

        bg = rets["seg"] == 0
        hair = rets["seg"] == 18
        rets["seg"][bg] = 18
        rets["seg"][hair] = 0

        if is_flipped:
            for a, b in self.FLIP_MAPPING_SEG:
                mx_a = rets["seg"] == a
                mx_b = rets["seg"] == b
                rets["seg"][mx_a] = b
                rets["seg"][mx_b] = a

        rets["seg_label"] = rets["seg"]
        rets["matrix"] = matrix
        rets["idx"] = idx
        return rets

    def __len__(self):
        if not self.is_train:
            return 64
        return 100000

    @property
    def num_classes(self):
        return len(self.COLORMAP) + 1

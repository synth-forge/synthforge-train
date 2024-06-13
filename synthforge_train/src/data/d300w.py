import os
import torch
import numpy as np
from torchvision import transforms as T
from STAR.lib.dataset import AlignmentDataset


def z2o(x):
    return (x - x.min()) / (x.max() - x.min())


class D300WDataset(torch.utils.data.Dataset):
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
    NME_LEFT_IDX = 36  # ocular
    NME_RIGHT_IDX = 45  # ocular

    colormap = {
        "face": (0.7, 0.3, 0.5),
        "forehead": (0.6, 0.2, 0.4),
        "eye_region": (0.8, 0.2, 0.2),
        "neck": (0.2, 0.8, 0.2),
        "right_eye_region": (0.2, 0.8, 0.8),
        "left_eye_region": (0.3, 0.5, 0.7),
        "left_eyeball": (0.2, 0.2, 0.8),
        "right_eyeball": (0.8, 0.2, 0.8),
        "lips": (0.4, 0.6, 0.2),
        "nose": (0.2, 0.4, 0.6),
    }

    def __init__(self, is_train, split=None, splits_path=None):
        self.img_dir = "/pfs/rdi/cei/synthetic_data/public_dataset/300W"
        annot_dir = (
            "/pfs/rdi/cei/synthetic_data/public_dataset/300W/data_annot/"
        )
        self.splits_path = splits_path or annot_dir
        mode = "train" if is_train else "test"
        self.dataset = AlignmentDataset(
            f"{annot_dir}/{mode}.tsv",
            image_dir=self.img_dir,
            transform=T.Compose([T.ToTensor()]),
            width=256,
            height=256,
            channels=3,
            means=(127.5, 127.5, 127.5),
            scale=1 / 127.5,
            classes_num=[68, 9, 68],
            crop_op=False,
            aug_prob=1.0 if is_train else 0.0,
            edge_info=self.EDGE_INFO,
            flip_mapping=self.FLIP_MAPPING,
            is_train=is_train,
            encoder_type="default",
        )
        assert split is None or is_train, "Can't split a test set"
        if split is not None:
            os.makedirs(splits_path, exist_ok=True)
            splits_path = f"{splits_path}/train_{split:0>3.2f}.pt"
            if not os.path.exists(splits_path):
                len_ds = len(self.dataset)
                len_sample_ds = int(np.ceil(len_ds))
                random_select = np.random.choice(
                    len(self.dataset), len_sample_ds
                )
                self.split_mapping = {i: j for i, j in enumerate(random_select)}
                torch.save(self.split_mapping, splits_path)
                print(f"Made Split: {splits_path}")
            else:
                self.split_mapping = torch.load(splits_path)
                print(f"Loaded Split: {splits_path}")
        else:
            self.split_mapping = {i: i for i in range(len(self.dataset))}

    def __getitem__(self, index):
        idx = self.split_mapping[index]
        data = self.dataset[idx]
        kps = (data["label"][0] + 1) * 127.5
        iod = kps[self.NME_LEFT_IDX] - kps[self.NME_RIGHT_IDX]
        iod = iod.norm()
        rets = {
            "img": data["data"][[2, 1, 0]],  # BGR to RGB
            "kps": kps,
            "heatmaps": data["label"][1],
            "iod": iod,
            "landmarks_5pts": data["landmarks_5pts"],
            "landmarks_tgtt": data["landmarks_tgtt"],
            "matrix": data["matrix"],
        }
        return rets

    def __len__(self):
        return len(self.dataset)

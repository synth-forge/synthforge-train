import torch
import numpy as np
from scipy.integrate import simps
import torch.distributed as dist

from typing import Mapping, Optional, List


def NME(target, pred, iod):
    nme = (target - pred).norm(dim=-1)  # [B, K]
    nme = nme.mean(dim=-1) / iod  # [B]
    return nme.mean()


class NME_tracker:
    def __init__(self):
        self._nmes = []

    def update_batch(self, nmes):
        self._nmes.append(nmes)

    def compute_and_update(self, target, pred, iod):
        nme = (target - pred).norm(dim=-1)  # [B, K]
        nme = nme.mean(dim=-1) / iod  # [B]
        self.update_batch(nme)

    @property
    def nmes(self):
        return torch.cat(self._nmes)

    @property
    def nme(self):
        return self.nmes.mean()


class FR_AUC:
    def __init__(self, thresh, nme):
        self.thresh = thresh
        self.nme = nme

    def __repr__(self):
        return "FR_AUC()"

    def test(self, nmes=None, thres=None, step=0.0001):
        if thres is None:
            thres = self.thresh
        nmes = nmes or self.nme.nmes.cpu().numpy()
        num_data = len(nmes)
        xs = np.arange(0, thres + step, step)
        ys = np.array([np.count_nonzero(nmes <= x) for x in xs]) / float(
            num_data
        )
        fr = 1.0 - ys[-1]
        auc = simps(ys, x=xs) / thres
        return [round(fr, 4), round(auc, 6)]


class F1Score:
    """Compute F1 score among label and pred_label.

    Args:
        label_tag (str): The tag for groundtruth label, which is a np.ndarray with dtype=int.
        pred_label_tag (str): The tag for predicted label, which is a np.ndarray with dtype=int
            and same shape with groundtruth label.
        label_names (List[str]): Names of the label values.
        num_labels (int): The number of valid label values.
    """

    def __init__(
        self,
        label_tag: str = "label",
        pred_label_tag: str = "pred_label",
        label_names: Optional[List[str]] = None,
        num_labels: Optional[int] = None,
        compute_fg_mean: bool = True,
        bg_label_name: str = "background",
    ) -> None:
        self.label_tag = label_tag
        self.pred_label_tag = pred_label_tag
        if label_names is None and num_labels is None:
            raise RuntimeError(
                "The label_names and the num_labels should never both be None."
            )
        if label_names is None:
            label_names = [f"label.{i}" for i in range(num_labels)]
        if num_labels is None:
            num_labels = len(label_names)
        self.label_names = label_names
        self.num_labels = num_labels
        self.compute_fg_mean = compute_fg_mean
        self.bg_label_name = bg_label_name

    def init_evaluation(self):
        self.hists_sum = np.zeros(
            [self.num_labels, self.num_labels], dtype=np.int64
        )
        self.count = 0
        self.num_pixels = 0

    def evaluate(self, data: Mapping[str, np.ndarray]):
        label = data[self.label_tag]
        pred_label = data[self.pred_label_tag]
        if label.shape != pred_label.shape:
            raise RuntimeError(
                f"The label shape {label.shape} mismatches the pred_label shape {pred_label.shape}"
            )

        hist = __class__._collect_hist(
            label, pred_label, self.num_labels, self.num_labels
        )
        self.hists_sum += hist
        self.count += label.shape[0]
        self.num_pixels += np.prod(label.shape)

    def finalize_evaluation(self) -> Mapping[str, float]:
        # gather all hists_sum
        hists_sum = torch.from_numpy(self.hists_sum).contiguous().cuda()
        if dist.is_initialized():
            dist.all_reduce(hists_sum)
        count_sum = torch.tensor(self.count, dtype=torch.int64, device="cuda")
        if dist.is_initialized():
            dist.all_reduce(count_sum)
        num_pixels = torch.tensor(
            self.num_pixels, dtype=torch.int64, device="cuda"
        )
        if dist.is_initialized():
            dist.all_reduce(num_pixels)

        assert hists_sum.sum() == num_pixels

        # compute F1 score
        A = hists_sum.sum(0).to(dtype=torch.float64)
        B = hists_sum.sum(1).to(dtype=torch.float64)
        intersected = hists_sum.diagonal().to(dtype=torch.float64)
        f1 = 2 * intersected / (A + B)

        f1s = {
            self.label_names[i]: f1[i].item() for i in range(self.num_labels)
        }
        if self.compute_fg_mean:
            f1s_fg = [
                f1[i].item()
                for i in range(self.num_labels)
                if self.label_names[i] != self.bg_label_name
            ]
            f1s["fg_mean"] = sum(f1s_fg) / len(f1s_fg)
        return f1s

    @staticmethod
    def _collect_hist(
        a: np.ndarray, b: np.ndarray, na: int, nb: int
    ) -> np.ndarray:
        """
        fast histogram calculation

        Args:
            a, b: Non negative label ids, a.shape == b.shape, a in [0, ... na-1], b in [0, ..., nb-1]

        Returns:
            hist (np.ndarray): The histogram matrix with shape [na, nb].
        """
        assert np.all((a >= 0) & (a < na) & (b >= 0) & (b < nb))
        hist = np.bincount(
            nb * a.reshape([-1]).astype(np.int64)
            + b.reshape([-1]).astype(np.int64),
            minlength=na * nb,
        ).reshape(na, nb)
        assert np.sum(hist) == a.size
        return hist

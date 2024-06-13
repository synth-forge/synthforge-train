import math
import torch
from torch import nn
from torchvision.transforms import functional as F


def crop_to_keypoints(
    image,
    kps,
    size,
    padding=50,
    get_tf=False,
    interpolation=F.InterpolationMode.BILINEAR,
):
    # Calculate the bounding box around keypoints
    min_x, min_y = torch.min(kps, dim=0).values
    max_x, max_y = torch.max(kps, dim=0).values

    # Calculate the square bounding box size around keypoints
    max_range = max(max_x - min_x, max_y - min_y) + padding
    center_x, center_y = (min_x + max_x) / 2, (min_y + max_y) / 2

    # Calculate cropping coordinates
    crop_left = int(center_x - max_range / 2)
    crop_top = int(center_y - max_range / 2)
    crop_right = int(center_x + max_range / 2)
    crop_bottom = int(center_y + max_range / 2)

    scale = size / max_range

    # Crop and resize the image
    if image.ndim == 2:
        image = image[None]
        interpolation = F.InterpolationMode.NEAREST
    image = F.resized_crop(
        image,
        crop_top,
        crop_left,
        crop_bottom - crop_top,
        crop_right - crop_left,
        (size, size),
        interpolation=interpolation,
        # antialias=True,
    )
    if get_tf:

        def tf(kps):
            kps[:, 0] -= crop_left
            kps[:, 1] -= crop_top
            kps *= scale
            return kps

        return image, tf(kps), tf

    kps[:, 0] -= crop_left
    kps[:, 1] -= crop_top
    kps *= scale
    return image, kps


class Morphology(nn.Module):
    """
    Base class for morpholigical operators
    For now, only supports stride=1, dilation=1, kernel_size H==W, and padding='same'.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=5,
        soft_max=True,
        beta=15,
        type=None,
    ):
        """
        in_channels: scalar
        out_channels: scalar, the number of the morphological neure.
        kernel_size: scalar, the spatial size of the morphological neure.
        soft_max: bool, using the soft max rather the torch.max(), ref: Dense Morphological Networks: An Universal Function Approximator (Mondal et al. (2019)).
        beta: scalar, used by soft_max.
        type: str, dilation2d or erosion2d.
        """
        super(Morphology, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.soft_max = soft_max
        self.beta = beta
        self.type = type

        self.weight = nn.Parameter(
            torch.zeros(out_channels, in_channels, kernel_size, kernel_size),
            requires_grad=False,
        )
        self.unfold = nn.Unfold(kernel_size, dilation=1, padding=0, stride=1)

    def forward(self, x):
        """
        x: tensor of shape (B,C,H,W)
        """
        # padding
        x = fixed_padding(x, self.kernel_size, dilation=1)

        # unfold
        x = self.unfold(x)  # (B, Cin*kH*kW, L), where L is the numbers of patches
        x = x.unsqueeze(1)  # (B, 1, Cin*kH*kW, L)
        L = x.size(-1)
        L_sqrt = int(math.sqrt(L))

        # erosion
        weight = self.weight.view(self.out_channels, -1)  # (Cout, Cin*kH*kW)
        weight = weight.unsqueeze(0).unsqueeze(-1)  # (1, Cout, Cin*kH*kW, 1)

        if self.type == "erosion2d":
            x = weight - x  # (B, Cout, Cin*kH*kW, L)
        elif self.type == "dilation2d":
            x = weight + x  # (B, Cout, Cin*kH*kW, L)
        else:
            raise ValueError

        if not self.soft_max:
            x, _ = torch.max(x, dim=2, keepdim=False)  # (B, Cout, L)
        else:
            x = (
                torch.logsumexp(x * self.beta, dim=2, keepdim=False) / self.beta
            )  # (B, Cout, L)

        if self.type == "erosion2d":
            x = -1 * x

        # instead of fold, we use view to avoid copy
        x = x.view(-1, self.out_channels, L_sqrt, L_sqrt)  # (B, Cout, L/2, L/2)

        return x


class Dilation2d(Morphology):
    def __init__(
        self, in_channels, out_channels, kernel_size=5, soft_max=True, beta=20
    ):
        super(Dilation2d, self).__init__(
            in_channels, out_channels, kernel_size, soft_max, beta, "dilation2d"
        )


class Erosion2d(Morphology):
    def __init__(
        self, in_channels, out_channels, kernel_size=5, soft_max=True, beta=20
    ):
        super(Erosion2d, self).__init__(
            in_channels, out_channels, kernel_size, soft_max, beta, "erosion2d"
        )


def fixed_padding(inputs, kernel_size, dilation):
    kernel_size_effective = kernel_size + (kernel_size - 1) * (dilation - 1)
    pad_total = kernel_size_effective - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    padded_inputs = F.pad(inputs, (pad_beg, pad_end, pad_beg, pad_end))
    return padded_inputs


def seg_classes_to_colors(seg_out, class_colors):
    return sum(
        [
            torch.where(seg_out == i, 1, 0) * c.reshape(3, 1, 1)
            for i, c in enumerate(class_colors)
        ]
    )

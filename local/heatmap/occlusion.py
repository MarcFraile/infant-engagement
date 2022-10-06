from typing import Tuple, Optional

from tqdm import trange
import numpy as np

import torch
from torch import Tensor
from torch.nn import Module
import torch.nn.functional as F

from .common import clean, exactly_one_non_null


# TODO: Center occlusion volume on removed tile, make sure tiles cover all available space (see other repos).
def occlusion_2d(input: Tensor, net: Module, class_idx: int, device = None, occluder_size = (32, 32), occluder_step = (16, 16)) -> Tensor:
    """
    Performs a class heatmap by iteratively occluding different 2D squares in the image or video.

    * Acts frame-by-frame on video (faster than occlusion_3d()).
    * Input must have shape (B)CTHW, with B=1 (if present), and C=3.
    * Output is
    """

    input = clean(input)
    (_C, T, H, W) = input.shape

    img_size      = torch.tensor((H, W))
    occluder_size = torch.tensor(occluder_size)
    occluder_step = torch.tensor(occluder_step)

    steps = (img_size - occluder_size) // occluder_step

    heatmap = torch.zeros(T, steps[0], steps[1])

    net.eval()
    with torch.no_grad():

        for t in trange(T):
            for j in range(steps[0]):
                for i in range(steps[1]):

                    offset = torch.tensor((j, i)) * occluder_step
                    end    = offset + occluder_size + 1
                    y0, y1 = offset[0], end[0]
                    x0, x1 = offset[1], end[1]

                    occluded = input[:, t, :, :].clone()
                    occluded[:, y0:y1, x0:x1] = 0

                    if device != None:
                        occluded = occluded.to(device)
                    occluded = occluded[None, :, None, :, :]

                    probs = net(occluded)
                    probs = probs.squeeze()

                    heat = 1.0 - probs[class_idx]

                    heatmap[t, j, i] = heat

    return heatmap


# TODO: Verify that this logic makes sense. I'm not sure that the sampling points are correctly placed along the volumne.
def occlusion_3d(
    input: Tensor,
    net: Module,
    device: Optional[torch.device] = None,
    occluder_radius: Tuple[int, int, int] = (2, 16, 16),
    occluder_stride: Tuple[int, int, int] = (1, 16, 16),
    class_idx: Optional[int] = None,
    target_value: Optional[float] = None,
) -> Tensor:
    """
    Performs a class heatmap by iteratively occluding different 3D volumes of the video.

    * Acts on volumes (slower than occlusion_2d()).
    * Occluder radius and stride shape: THW.
    * Input must have shape (B)CTHW, with B=1 (if present), and C=3.
    * Assumes network returns scores. Sigmoid applied in binary case, otherwise softmax.
    * For multi-class/multi-label, pass in `class_idx`.
    * For binary classification, pass in `target_value`.
    """

    assert exactly_one_non_null(class_idx, target_value), "Exactly one of `class_idx` (for multiclass) or `target_value` (for binary) must be non-null."
    is_binary = (class_idx == None)

    input = clean(input)
    (_C, T, H, W) = input.shape

    (rT, rH, rW) = occluder_radius
    (sT, sH, sW) = occluder_stride

    # Output sizes.
    oT = int(np.ceil(T / sT))
    oH = int(np.ceil(H / sH))
    oW = int(np.ceil(W / sW))

    heatmap = torch.zeros(oT, oH, oW)

    net.eval()
    with torch.no_grad():

        for out_t in trange(0, oT, desc="T"):
            # for out_h in tnrange(0, oH, desc="H"):
            #     for out_w in tnrange(0, oW, desc="W"):
            for out_h in range(0, oH):
                for out_w in range(0, oW):

                    occluded = input.clone()

                    t = out_t * sT
                    h = out_h * sH
                    w = out_w * sW

                    t0 = max(0, t - rT)
                    t1 = min(T, t + rT + 1)

                    h0 = max(0, h - rH)
                    h1 = min(H, h + rH + 1)

                    w0 = max(0, w - rW)
                    w1 = min(W, w + rW + 1)

                    occluded[:, t0:t1, h0:h1, w0:w1] = 0

                    occluded = occluded.unsqueeze(dim=0)
                    if device != None:
                        occluded = occluded.to(device)

                    scores : Tensor = net(occluded)

                    if is_binary:
                        assert target_value is not None # Make MyPy happy.
                        prob = scores.sigmoid()
                        if target_value < 0.5:
                            prob = 1 - prob
                    else:
                        probs = F.softmax(scores, dim=1)
                        prob  = probs[..., class_idx]

                    prob = prob.squeeze()

                    heat = 1.0 - prob
                    heatmap[out_t, out_h, out_w] = heat

    return heatmap

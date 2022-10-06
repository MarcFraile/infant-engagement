from typing import Optional

import torch
from torch import Tensor
from torch.nn import Module, functional as F

from .common import ActivationsGradients, exactly_one_non_null


# NOTE: Original paper: https://openaccess.thecvf.com/content_iccv_2017/html/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.html (Selvaraju 2017).
def gradcam(net: Module, target_layer: Module, input: Tensor, class_idx: Optional[int] = None, target_value: Optional[float] = None) -> Tensor:
    """
    Calculates basic Grad-CAM over a network.

    * Set `target_layer` to the last convolutional layer for maximum impact.
    * Estimates positive evidence for a given output signal.
    * Procedure:
        1. Weight per channel: spatial average of the channel's derivatives.
        2. Calculate the weighted average of channel activations (1 weight per channel).
        3. Take the positive part (ReLU) to remove negative evidence.
    * The results are *NOT* scaled spatially. You can use `heatmap.scale_to_match()` for that.
    * For multi-class/multi-label, pass in `class_idx`.
    * For binary classification, pass in `target_value`.
    * Expected input shape: (B, C, H, W) or (B, C, T, H, W) -- batch dimension mandatory!
    """

    assert exactly_one_non_null(class_idx, target_value), "Exactly one of `class_idx` (for multiclass) or `target_value` (for binary) must be non-null."
    is_binary = (class_idx == None)

    with ActivationsGradients(net, target_layer) as (get_activation, get_gradient):
        net.eval()
        output = net(input)

        if is_binary:
            assert target_value is not None # Make mypy happy
            # NOTE: See heatmap README.
            if target_value < 0.5:
                output = -output
        else:
            output = output[..., class_idx]

        net.zero_grad()
        output.backward()

        activation = get_activation()
        gradient   = get_gradient()

        # We do a weighted average over all channels, using the sum of each channel's gradients as its weight.
        # We then ReLU it to keep positive evidence only.
        weight  = gradient.mean(dim=(-2, -1)) # Per-channel weights: spatial average pooling of gradient.
        average = torch.sum(activation * weight[..., None, None], dim=1) # Do the weighted average.
        gradcam = F.relu(average) # ReLU to keep positive evidence only.

    return gradcam

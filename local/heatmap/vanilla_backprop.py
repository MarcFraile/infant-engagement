from typing import Optional

from torch import Tensor
from torch.nn import Module

from .common import clean, exactly_one_non_null


def vanilla_backprop(net: Module, input: Tensor, class_idx: Optional[int] = None, target_value: Optional[float] = None) -> Tensor:
    """
    Calculates Vanilla Backpropagation.

    * For multi-class/multi-label, pass in `class_idx`.
    * For binary classification, pass in `target_value`.
    """

    assert exactly_one_non_null(class_idx, target_value), "Exactly one of `class_idx` (for multiclass) or `target_value` (for binary) must be non-null."
    is_binary = (class_idx == None)

    input = input.detach().requires_grad_(True)
    net.eval()
    output = net(input)

    if is_binary:
        assert target_value is not None # Make MyPy happy.
        # NOTE: See heatmap README.
        if target_value < 0.5:
            output = -output
    else:
        output = output[..., class_idx]

    net.zero_grad()
    output.backward()
    gradient = clean(input.grad)

    return gradient

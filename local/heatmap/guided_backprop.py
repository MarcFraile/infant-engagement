from typing import List, Optional, Tuple

import torch
from torch import Tensor
from torch.nn import Module, ReLU
from torch.nn import functional as F
from torch.utils.hooks import RemovableHandle

from .common import clean, exactly_one_non_null


 # NOTE: Original paper: https://arxiv.org/abs/1412.6806 Springenberg 2014.
def guided_backprop(net: Module, input: Tensor, class_idx: Optional[int] = None, target_value: Optional[float] = None) -> Tensor:
    """
    Calculates Guided Backpropagation.

    * Modify ReLU (and ReLU-like activations) to filter gradients:
        * For activation < 0 (as usual backprop),
        * For gradient < 0.
    * For multi-class/multi-label, pass in `class_idx`.
    * For binary classification, pass in `target_value`.
    """

    assert exactly_one_non_null(class_idx, target_value), "Exactly one of `class_idx` (for multiclass) or `target_value` (for binary) must be non-null."

    is_binary = (class_idx == None)

    handles     : List[RemovableHandle]       = []
    activations : List[Tuple[Module, Tensor]] = []

    # DEBUG: This is not needed for the computations.
    def forward_hook(module: Module, inputs: Tuple[Tensor, ...], output: Tensor) -> None:
        assert type(output) == Tensor
        assert torch.all(output >= 0)
        activations.append((module, output))

    # NOTE: Called right after calculating `grad_in = module.backward(grad_out)`.
    #       `grad_out` is a tuple containing the gradients coming back from the next layer.
    #       `grad_in` is a tuple containing the gradients being passed to the previous layer.
    #       The hook can return `None` to keep `grad_in` unaltered, or return a modified gradient.
    def backward_hook(module: Module, grad_in: Tuple[Tensor, ...], grad_out: Tuple[Tensor, ...]) -> Tuple[Tensor, ...]:
        output_gradient = grad_out[0] # The gradient flowing from one layer up.
        input_gradient  = grad_in [0] # The normal gradient that would be backpropagated by ReLU: (input > 0) * output_gradient

        clipped_gradient = F.relu(input_gradient) # After filtering: clipped_gradient = (input > 0) * (grad_out > 0) * grad_out

        # DEBUG
        activation_module, activation = activations.pop()
        assert activation_module == module
        assert torch.all(clipped_gradient[activation <= 0] == 0)
        assert torch.all(clipped_gradient[output_gradient <= 0] == 0)

        return (clipped_gradient, )

    for module in net.modules():
        if type(module) == ReLU:
            forward_handle  = module.register_forward_hook (forward_hook )
            backward_handle = module.register_backward_hook(backward_hook)
            handles.append(forward_handle)
            handles.append(backward_handle)

    input = input.detach().requires_grad_(True)
    net.eval()
    output: Tensor = net(input)

    if is_binary:
        assert target_value is not None # Make MyPy happy.
        # NOTE: See heatmap README.
        if target_value < 0.5:
            output = -output
    else:
        output = output[..., class_idx]

    net.zero_grad()
    output.backward()
    guided_gradient = clean(input.grad)

    for h in handles:
        h.remove()

    return guided_gradient

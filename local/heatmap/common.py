from typing import Tuple, Callable, List

import torch
from torch import Tensor
from torch.nn import Module

from local import util


def clean(tensor: Tensor) -> Tensor:
    """Common tensor cleanup before operating (detach(), cpu(), ...)."""
    return tensor.detach().cpu().squeeze()


# NOTE: See https://arxiv.org/abs/1312.6034 (Simonyan 2014) for the choice of channel reduction.
# NOTE: Utku Ozbulak's implementation averages over channels, which disagrees with the original paper and possibly muddies the resulting map.
#       It also uses the 99-th percentile instead of the max, which I think is nonstandard (original paper does not have linked code), but judging by histograms seems very useful.
# TODO: Consider using percentiles instead of max and/or min.
def saliency(tensor: Tensor) -> Tensor:
    """
    Returns the normalized saliency of a multi-channel tensor.

    * Input:  Channel-first tensor, shape: C x (rest).
    * Output: Tensor in range [0, 1], shape: (rest).
    * Reduces channels by taking the maximum of the absolute values, following Simonyan 2014.
    """
    saliency = tensor.abs().max(dim=0).values
    return util.remap(saliency)


def normalize_per_frame(tensor: Tensor) -> Tensor:
    """
    Rescale every frame to 0-1 range.

    * Returns a copy.
    * Expected input size: T(...).
    """
    T = tensor.shape[0]
    out = torch.empty_like(tensor)
    for t in range(T):
        frame = tensor[t, ...]
        out[t, ...] = util.remap(frame)
    return out


def exactly_one_non_null(*args) -> bool:
    found = False
    for arg in args:
        if arg != None:
            if found == True:
                return False
            else:
                found = True
    return found


class ActivationsGradients:
    """
    Use in a with block to automatically hook and unhook activation and gradient capture:

    ```
        with ActivationGradients(net, target) as (get_activation, get_gradient):
            output = net(input)

            net.zero_grad()
            output.backward()

            activation = get_activation()
            gradient   = get_gradient()

            ...
    ```

    The activation and gradient will be available after the forward resp. backward pass.

    """

    def __init__(self, net: Module, target: Module):
        self.net    = net
        self.target = target

    def __enter__(self) -> Tuple[Callable[[], Tensor], Callable[[], Tensor]]:
        self.activations : List[Tensor] = []
        self.gradients   : List[Tensor] = []

        self.forward_hook  = self.target.register_forward_hook (self._save_activations)
        self.backward_hook = self.target.register_backward_hook(self._save_gradients  )

        return (self.get_activation, self.get_gradient)

    def __exit__(self, exc_type, exc_value, traceback):
        # TODO: Exceptions?
        self.forward_hook .remove()
        self.backward_hook.remove()

    def _save_activations(self, module: Module, input, output) -> None:
        activation: Tensor = output.cpu().detach().clone()
        self.activations.append(activation) # Callbacks happen front to back -> we keep order.

    def _save_gradients(self, module: Module, grad_input, grad_output) -> None:
        gradient: Tensor = grad_output[0].cpu().detach().clone()
        self.gradients.insert(0, gradient) # Callbacks happen back to front -> we reverse order.

    def get_activation(self) -> Tensor:
        return self.activations[-1]

    def get_gradient(self) -> Tensor:
        return self.gradients[-1]

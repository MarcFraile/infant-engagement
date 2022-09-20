import sys
from datetime import timedelta
from typing import Any, Collection, Optional, Tuple, TypeVar

import matplotlib.pyplot as plt
import numpy as np
import imageio

import torch
from torch import nn, cuda, Tensor
import torchvision


T = TypeVar("T")


def eprint(*args, **kwargs) -> None:
    """
    Print to stderr.

    Works the same way as print().
    """
    print(*args, file=sys.stderr, **kwargs)


def remap(x: T, source: Optional[Tuple[Any, Any]] = None, dest: Tuple[Any, Any] = (0, 1)) -> T:
    """
    Remap a value `x` in range `(source[0], source[1])` to the range `(dest[0], dest[1])`.

    * `source` and `dest` should be 2-tuples of a compatible type with x.
    * `source` defaults to `(x.min(), x.max())` (will fail if the type of `x` does not have these methods!)
    * `dest` defaults to `(0, 1)` (will fail if the type is not compatible with Python `int`s!)
    """
    if source is None:
        source = (x.min(), x.max())
    (sm, sM) = source
    (dm, dM) = dest
    return dm + (dM - dm) * (x - sm) / (sM - sm)


def clamp(x, low, high):
    """
    Clamps x between low and high.

    * WARN: Does not work with tensors! (use Tensor.clamp(min, max) instead)
    """
    return max(low, min(high, x))


def count_params(model: nn.Module) -> int:
    """Returns the total number of parameters in a model."""
    num_params = 0

    for params in model.parameters():
        count = 1
        for s in params.size():
            count *= s
        num_params += count

    return num_params


def average_duration(deltas: Collection[timedelta]) -> timedelta:
    counter = timedelta(0)
    for t in deltas:
        counter += t
    return counter / len(deltas)


def _imshow_internal(img: Tensor, mean=None, std=None) -> plt.Figure:
    """Convert from normalized PyTorch format to PyPlot format, and display."""
    img = img.squeeze().cpu() # Conversion to NumPy type will fail if img is in CUDA device.

    # Ensure we get a 3-tensor, stacking samples as grid if necessary.
    if len(img.shape) == 4:
        img = img.permute(1, 0, 2, 3)
        img = torchvision.utils.make_grid(img)

    if mean == None:
        mean = torch.tensor([1, 1, 1])
    elif type(mean) != Tensor:
        mean = torch.tensor(mean)

    if std == None:
        std = torch.tensor([1, 1, 1])
    elif type(std) != Tensor:
        std = torch.tensor(std)

    if list(mean.shape) != [3] or list(std.shape) != [3]:
        raise ValueError("Mean and std must be a 3-dimensional vector.")

    mean = mean[..., None, None]
    std  = std [..., None, None]

    # print(f"[imshow] img: {img.shape}; mean: {mean.shape}; std: {std.shape}.")

    img = (img * std + mean).int() # Un-normalize
    npimg = np.transpose(img.numpy(), (1, 2, 0)) # Un-tensorize
    npimg = np.clip(npimg, 0, 255) # Ensure correct range.

    fig = plt.figure()
    plt.imshow(npimg)
    plt.axis("off")

    return fig


def imshow(img: Tensor, mean=None, std=None) -> None:
    fig = _imshow_internal(img, mean, std)
    fig.show()
    plt.close(fig)


def imsave(img: Tensor, path, mean=None, std=None) -> None:
    fig = _imshow_internal(img, mean, std)
    fig.savefig(str(path), bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def save_vid(path, vid, fps, normalize=False):
    """
        Save the tensor 'vid' to the given 'path' (including termination).
        Takes care of some amount of input cleaning (squeeze/unsqueeze, detach, cpu, permute).
    """
    if(type(vid) == Tensor):
        vid = vid.cpu().detach().squeeze()
        if len(vid.shape) == 3: # BW
            vid = vid.unsqueeze(dim=3)
        if vid.shape[3] > 3: # C is not in last pos -> PyTorch dim order (CxTxHxW) needs to be mapped to NumPy order (TxHxWxC).
            vid = vid.permute(1, 2, 3, 0)
        vid = vid.numpy()

    M, m = vid.max(), vid.min()

    if normalize == True:
        vid = remap(vid, (m, M), (0, 255))
    elif m >= 0 and M <= 1:
        vid = remap(vid, (0, 1), (0, 255))
    elif m >= -1 and M <= 1:
        vid = remap(vid, (-1, 1), (0, 255))

    vid = vid.astype(np.uint8)
    vid = vid.clip(0, 255)

    # TODO: Test if we can substitute this for iio.imwrite(...) (using v3).
    path = str(path) # Just in case pathlib.Path() causes problems.
    writer = imageio.get_writer(path, fps=fps)
    for img in vid:
        writer.append_data(img)
    writer.close()


def to_torch(x, num_dims: int, dim: int) -> Tensor:
    """
    Convert `x` to a Tensor. Output object has given `num_dims`, and contains the 1D input in `dim`.

    * `x` expected to be 1D (arrays or tuples of numbers or 0D tensors, etc.)
    * `dim` works with any integer (wrapped around `num_dims`).
    """

    x = torch.tensor(x)
    dim = dim % num_dims

    for _ in range(dim):
        x = x.unsqueeze(dim=0)

    for _ in range(num_dims - dim - 1):
        x = x.unsqueeze(dim=-1)

    return x


def normalize(x: Tensor, mean, std, dim: int) -> Tensor:
    """
    normalize the entries in `x`, according to `mean` and `std`.

    * `mean` and `std` are expected to be 1D objects that can be converted to Tensor (arrays or tuples of numbers or 0D tensors, etc.)
    * `mean` and `std` are expected to have the same length as `x.shape[dim]`.
    * `dim` works with any integer (wrapped around `x.dim()`).
    """
    mean = to_torch(mean, x.dim(), dim)
    std  = to_torch(std , x.dim(), dim)

    return (x - mean) / std


def denormalize(y: Tensor, mean, std, dim: int) -> Tensor:
    """
    De-normalize the entries in `x`, according to `mean` and `std`.

    * `mean` and `std` are expected to be 1D objects that can be converted to Tensor (arrays or tuples of numbers or 0D tensors, etc.)
    * `mean` and `std` are expected to have the same length as `x.shape[dim]`.
    * `dim` works with any integer (wrapped around `x.dim()`).
    """

    mean = to_torch(mean, y.dim(), dim)
    std  = to_torch(std , y.dim(), dim)

    return y * std + mean


# TODO: Phase out in favor of PrettyCli-version in e2e/cli_helpers.py
def print_device_details(device: torch.device) -> None:
    raise Warning("Outdated method. Use e2e/cli_helpers.py")
    props = cuda.get_device_properties(device)
    print(f"name:    {props.name}")
    print(f"version: {props.major}.{props.minor}")
    print(f"memory:  {props.total_memory / (1024 * 1024 * 1024):4.02f} GB")
    print(f"cores:   {props.multi_processor_count}")


def next_letter(letter: str) -> str:
    return chr(ord(letter) + 1)

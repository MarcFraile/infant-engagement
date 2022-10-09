from typing import Tuple

import torch
from torch import fft, Tensor
from  torchvision.transforms import functional as F, InterpolationMode


EPSILON = torch.finfo(torch.float32).eps

INTERPOLATION_TYPES = {
    "nearest" : InterpolationMode.NEAREST,
    "linear"  : InterpolationMode.BILINEAR,
    "cubic"   : InterpolationMode.BICUBIC,
}


def fourier_downsample(img: Tensor, out_size: int) -> Tensor:
    """
    Reduces `img` spatial dimensions to `(out_size, out_size)` by brick-wall limiting.

    * `img` should have shape (..., H, W).
    * `img` is converted to 2D Fourier frequencies, only the first `out_size` are kept, and the final image is recovered.
    * This process can produce ringing artifacts!
    """

    offset = (out_size // 2)
    last   = (out_size + 1) // 2 + 1

    transform = fft.rfft2(img)
    rolled = torch.roll(transform, offset, -2)
    filtered = rolled[..., :out_size, :last]
    unrolled = torch.roll(filtered, -offset, -2)
    output = fft.irfft2(unrolled)

    assert output.shape[-2:] == (out_size, out_size)
    return output


def fourier_lowpass(img: Tensor, limit: float) -> Tensor:
    """
    Brick-wall limits `img`, while keeping its shape.

    * `img` should have shape (..., H, W).
    * `limit` should be a fraction in [0, 1] indicating how many Fourier frequencies to keep.
    * All frequencies above `limit` (regardless of sign) are set to 0.
    * All other frequencies are kept intact.
    * This process can produce ringing artifacts!
    """

    pass1 = torch.abs(2 * fft.rfftfreq(img.shape[-1])) <= limit
    pass2 = torch.abs(2 * fft.fftfreq (img.shape[-2])) <= limit
    kernel = torch.outer(pass2, pass1)

    transform = fft.rfft2(img)
    filtered = transform * kernel
    output = fft.irfft2(filtered, s=img.shape[-2:])

    return output


def gauss_lowpass(img: Tensor, sigma: float) -> Tensor:
    """
    Gaussian blur `img` by `sigma`.

    * `img` expected to take shape (..., H, W).
    * `sigma` expected to be > 0.
    """

    assert len(img.shape) >= 2
    assert sigma > 0

    kernel_size = max(1, int(5 * sigma))
    if kernel_size % 2 == 0: # TorchVision gets sad if this is even (not mentioned in the docs!).
        kernel_size += 1

    return F.gaussian_blur(img, kernel_size, sigma) # TODO: Verify this works plane by plane on (..., H, W) tensors.


def gauss_downsample(img: Tensor, stride: int, sigma: float) -> Tensor:
    """
    Gaussian blur `img` by `sigma` and then downsample by taking every `stride`-th cell.

    * `img` expected to take shape (..., H, W).
    * `sigma` expected to be >= 0.
    * Blurring will only be applied if `sigma` is above f32 epsilon.
    """

    assert len(img.shape) >= 2
    assert sigma >= 0

    if sigma > EPSILON:
        img = gauss_lowpass(img, sigma)

    img = img[..., ::stride, ::stride] # TODO: Should we center this for bigger strides? What would the correct formula be?

    return img


# TODO: Revisit what `antialias` does!
# TODO: This is based on local.heatmap.viz.scale_to_match(), but is expected to be more trustworthy than the original. Maybe we should substitute the old function for this one in all usages.
def scale_spatial_dimensions(img: Tensor, size: Tuple[int, int], mode: str = "nearest", antialias: bool = True) -> Tensor:
    """
    Rescale the spatial dimensions of `img` to match `size`, using standard PyTorch algorithms.

    * `img` expected to take shape (..., H, W).
    * `size` expected to be (H', W') -- the new height and width.
    * `mode` expected to be one of `['nearest', 'linear', 'cubic']`. Defaults to `'nearest'`.
    * `antialias` set to `True` by default.
    """

    assert len(img.shape) >= 2
    for s in size:
        assert s > 0
    assert mode in INTERPOLATION_TYPES, f"Expected mode to be one of {INTERPOLATION_TYPES.keys()}; found: '{mode}'."

    interpolation = INTERPOLATION_TYPES[mode]
    return F.resize(img, size, interpolation, antialias=antialias)

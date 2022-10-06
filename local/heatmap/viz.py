import cv2

import torch
from torch import Tensor
from torch.nn import functional as F

from local import util

from .common import clean


# TODO: This seems to be interpolating over time and possibly batch as well.
#       We should enforce that B and T dims match, and only (H, W) dims are stretched.
def scale_to_match(img: Tensor, overlay: Tensor, mode: str = "nearest") -> Tensor:
    """
    Scales `overlay` to match `img`.

    * Uses nearest neighbours by default. Change with `mode` (e.g., `mode="linear"`).
    * Expected `img` shape: (B?) x C x (T?) x H x W.
    * Expected `overlay` shape: (B?) x (C?) x (T'?) x H' x W'.
    * Expect B=1, if present.
    * Expect T present iif T' present.
    """

    img     = clean(img    )
    overlay = clean(overlay)

    size = img.shape[1:] # C(T?)HW => (T?)HW

    while (len(size) + 2) > overlay.dim():
        overlay = overlay.unsqueeze(dim=0) # (C'?)(T'?)H'W' => BC'(T'?)HW

    if mode == "linear":
        if overlay.dim() == 4:
            mode = "bilinear"
        elif overlay.dim() == 5:
            mode = "trilinear"

    output = F.interpolate(input=overlay, size=size, mode=mode)
    return output.squeeze()


def _colormap_single(grayscale: Tensor) -> Tensor:
    """
    Convert a single grayscale image (or frame) into a colormap.

    * Input Tensor should take values in [0, 1] and have shape (H, W).
    * Output tensor takes values in [0, 1] and has shape (H, W, C).
    """
    cv2_input  = (256 * grayscale).clamp(0, 255).type(torch.uint8).numpy()
    cv2_output = cv2.applyColorMap(cv2_input, cv2.COLORMAP_JET)
    cv2_output = cv2.cvtColor(cv2_output, cv2.COLOR_BGR2RGB)
    color      = torch.tensor(cv2_output, dtype=torch.float32)
    color      = color / 255.0
    return color


def colormap(grayscale: Tensor) -> Tensor:
    """
    Convert a grayscale image or video into a colormap.

    * Input tensor should takve values in [0, 1], and have shape (H, W) or (T, H, W).
    * Output tensor takes values in [0, 1], and has shape (C, T?, H, W).
    """
    if grayscale.dim() == 2: # HW (image)
        color = _colormap_single(grayscale)
    elif grayscale.dim() == 3: # THW (video)
        stack = []
        for t in range(grayscale.shape[0]):
            frame = _colormap_single(grayscale[t, ...])
            stack.append(frame)
        color = torch.stack(stack)
    else:
        raise ValueError(f"Unknown grayscale shape to colormap (should be (T?, H, W)): {grayscale.shape}")

    # NOTE: This sucks, but I couldn't figure a simple way to permute "last dim to first".
    #       In any case, it matches the shapes I indicate in the docstring.
    if color.dim() == 3:
        color = color.permute(2, 0, 1) # HWC => CHW
    elif color.dim() == 4:
        color = color.permute(3, 0, 1, 2) # THWC => CTHW
    else:
        raise ValueError(f"Unknown shape after color mapping (should be (T?, H, W, C): {color.shape}")

    return color


def multiply_overlay(img: Tensor, overlay: Tensor, mode: str = "nearest") -> Tensor:
    """
    Scales `overlay` to match `img` using nearest neighbors, and multiplies them together.

    * Uses nearest neighbours by default. Change with `mode` (e.g., `mode="linear"`).
    * Expected `img` shape: (B?, C, T?, H, W).
    * Expected `overlay` shape: (B?, T'?, H', W').
    * Expect B=1, if present.
    * Expect T present iif T' present.
    """

    img     = clean(img    )
    overlay = clean(overlay)

    overlay = scale_to_match(img, overlay, mode)

    return img * overlay


def colormap_overlay(img: Tensor, overlay: Tensor, mode: str = "nearest", use_alpha: bool = True) -> Tensor:
    """
    Scales `overlay` to match `img`, and combines them with a colormap.

    * Uses nearest neighbours by default. Change with `mode` (e.g., `mode="linear"`).
    * Expected `img` shape: (1?, C, T?, H, W).
    * Expected `overlay` shape: (1?, T'?, H', W').
    * Expect T present iif T' present.
    * `overlay` *IS NORMALIZED* to [0, 1].
    * `img` is *NOT NORMALIZED*, but expected to take values in [0, 1] range.
    * Output has shape (C, T?, H, W).
    """

    img     = clean(img    )
    overlay = clean(overlay)

    overlay = util.remap(overlay) # Ensure `overlay` in [0,1] range.
    color = colormap(overlay)
    color   = scale_to_match(img, color, mode=mode)

    if use_alpha:
        overlay = overlay.sqrt() # Make alpha more lenient (gets opaque sooner).
        overlay = scale_to_match(img, overlay, mode=mode)
        return (1 - overlay) * img + overlay * color
    else:
        return (img + color) / 2

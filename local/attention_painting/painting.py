import numpy as np

from local.util import clamp


def gaussian_kernel(sigma: float = 1.0) -> np.ndarray:
    """
    Creates a Gaussian kernel with the given `sigma` (in px) and side length `5 * sigma`.
    """
    length = max(1, int(5 * sigma))

    x            = np.linspace(-(length - 1) / 2.0, +(length - 1) / 2.0, length)
    kernel_slice = np.exp(-0.5 * np.square(x) / np.square(sigma))
    full_kernel  = np.outer(kernel_slice, kernel_slice)
    normalized   = full_kernel / np.sum(full_kernel)

    return normalized


def add_template(base: np.ndarray, template: np.ndarray, x: float, y: float, t: int) -> None:
    assert len(base    .shape) == 3 # THW
    assert len(template.shape) == 2 #  HW

    (T, H, W) = base.shape
    (h, w)    = template.shape

    # TODO: Should we floor() instead?
    # Low index in `base`, assuming no clipping.
    y_low = round(y - h / 2)
    x_low = round(x - w / 2)

    # Low index in `base`, with clipping taken into account.
    Y0 = clamp(y_low, 0, H - 1)
    X0 = clamp(x_low, 0, W - 1)

    # High index in `base`, with clipping taken into account.
    Y1 = clamp(y_low + h, 1, H)
    X1 = clamp(x_low + w, 1, W)

    # Low index in `template`.
    y0 = Y0 - y_low
    x0 = X0 - x_low

    # High index in `template`.
    y1 = Y1 - y_low
    x1 = X1 - x_low

    base[t, Y0:Y1, X0:X1] = (base[t, Y0:Y1, X0:X1] + template[y0:y1, x0:x1]).clip(0, 1)

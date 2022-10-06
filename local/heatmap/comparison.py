from typing import Callable, Optional, Tuple
import cv2
import numpy as np

import torch
from torch import Tensor


F64_EPSILON = torch.finfo(torch.float64).eps # 2.220446049250313e-16 => Same as MATLAB (according to Bylinskii)
Comparison = Callable[[Tensor, Tensor], float]


# NOTE: Much of this follows "What do different evaluation metrics tell us about saliency models?"
#       Arxiv: https://arxiv.org/abs/1604.03605
#       Code:  https://github.com/cvzoya/saliency/tree/master/code_forMetrics

# TODO: Consider exposing this in utils.
# TODO: Test!
def _make_probability(x: Tensor, move_zero: bool = False, epsilon: Optional[float] = F64_EPSILON) -> Tensor:
    """
    Turn `x` into a probability distribution (integral equals 1).

    * By default, asserts `x >= 0`. If `move_zero=True`, enforces `x.min() == 0` instead.
    * Asserts `x` has strictly positive elements (after `move_zero` is applied).
    * If `epsilon` is `None`, asserts that the sum is not too close to zero to normalize. Otherwise, adds `epsilon` to `x` before normalizing (avoids failures for constant maps).
    """

    if move_zero:
        x = x - x.min()
    else:
        assert 0 <= x.min() < torch.inf

    if epsilon is None:
        assert F64_EPSILON < x.sum() < torch.inf, f"Cannot normalize with given sum: {x.sum()}"
    else:
        x = x + epsilon # Allows us to map the zero tensor to a uniform distribution, while being continuous for small norm distributions.

    return x / x.sum()


# TODO: Consider exposing this in utils.
# TODO: Test!
def _get_ranks(x: Tensor) -> Tensor:
    tmp = x.reshape(-1).argsort()
    ranks = torch.zeros_like(tmp)
    ranks[tmp] = torch.arange(len(tmp))
    ranks = ranks.reshape(x.shape)
    return ranks


def ensure_shape(human: Tensor, machine: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Squeezes both tensors, asserts that they have the same shape, and returns them.

    * Output order same as input: `(human, machine)`.
    """

    human = human.squeeze()
    machine = machine.squeeze()

    assert human.shape == machine.shape, f"Expected matching sizes for `human` and `machine`; found human: {human.shape}; machine: {machine.shape}"

    return human, machine


def pearson_correlation(human: Tensor, machine: Tensor) -> float:
    """
    Calculates Pearson's Correlation Coefficient.

    * Similarity measure in range [-1, +1]: higher is better.
    * This is the normal way to calculate correlation.
    """

    human, machine = ensure_shape(human, machine)

    # PyTorch doesn't like taking means on integer types.
    human = human.float()
    machine = machine.float()

    h = human - human.mean()
    m = machine - machine.mean()

    cov = (h * m).mean()
    hv = (h * h).mean().sqrt()
    mv = (m * m).mean().sqrt()

    corr = cov / (hv * mv)

    return corr.item()


def spearman_correlation(human: Tensor, machine: Tensor) -> float:
    """
    Calculates Spearman's Rank Correlation Coefficient.

    * Similarity measure in range [-1, +1]: higher is better.
    """

    human, machine = ensure_shape(human, machine)

    h = _get_ranks(human)
    m = _get_ranks(machine)

    return pearson_correlation(h, m)


def _nd_tensor_to_signature(x: Tensor) -> np.ndarray:
    """
    Transform an ND input Tensor (grayscale image or video) into the format needed by cv2.EMD().

    * cv2.EMD() expects inputs to be shaped like (samples) x (dims + 1), with each row having the sample probability followed by its coordinates.
    * To transform 2D or 3D images into the correct format, we need to list every pixel as a row entry, starting by the pixel value and following by the pixel coordinates.
    """
    y: np.ndarray = x.numpy()
    flat_indices = np.arange(y.size) # 1D array: 0 ... (total pixel count - 1)
    nd_indices = np.unravel_index(flat_indices, y.shape) # (total pixel count) x 2 array: nd_indices[i] gives the coordinates of the i-th pixel.
    signature = np.stack((y.ravel(), *nd_indices), axis=-1) # Prepend the pixel values to the corresponding indices.
    signature = signature.astype(np.float32) # cv2.EMD() is picky about the format.
    return signature


# TODO: Test more.
def earth_movers_distance(human: Tensor, machine: Tensor) -> float:
    """
    Calculates the 1st Wasserstein Distance (a.k.a. the Earth Mover's Distance) for the joint distribution.

    * Positive distance measure: lower is better.
    * Expects `human` and `machine` to have the same shape (up to squeezing).
    * Computationally expensive; cost scales with tensor dimension.
    """
    # NOTE: See https://en.wikipedia.org/wiki/Earth_mover%27s_distance
    #       See https://en.wikipedia.org/wiki/Wasserstein_metric

    # NOTE: Implementations in the wild:
    #       SciPy 1D implementation (cannot use, Euclidean distance between coordinates is important) https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wasserstein_distance.html
    #       StackOverflow solution for n 2D samples (cannot use, different data format) https://stackoverflow.com/questions/57562613/python-earth-mover-distance-of-2d-arrays

    human, machine = ensure_shape(human, machine)

    human = _make_probability(human)
    machine = _make_probability(machine)

    human_signature = _nd_tensor_to_signature(human)
    machine_signature = _nd_tensor_to_signature(machine)

    emd, _, _ = cv2.EMD(human_signature, machine_signature, cv2.DIST_L2)
    return emd


# TODO: Test more.
def histogram_intersection(human: Tensor, machine: Tensor) -> float:
    """
    Calculates the Similarity metric (a.k.a. Histogram Intersection).

    * Similarity measure in range [0, 1]: higher is better.
    * Normalizes the input to min=0, sum=1 (will fail for constant maps).
    * Takes per-entry minimum and sums.
    """

    human, machine = ensure_shape(human, machine)

    # Probability distribution normalization required.
    human = _make_probability(human, move_zero=True)
    machine = _make_probability(machine, move_zero=True)

    # Calculations
    intersection = torch.minimum(human, machine)
    sim = intersection.sum()

    return sim.item()


# TODO: Test more.
def kullback_leibler(human: Tensor, machine: Tensor, epsilon: float = F64_EPSILON) -> float:
    """
    Calculates the Kullback-Leibler Divergence D_KL(human || machine)

    * Distance measure in range [0, inf): lower is better.
    * See Wikipedia: https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence
    * See paper: https://ieeexplore.ieee.org/abstract/document/8315047
    * Adapted quote: "expected excess surprise from using `machine` as a model when the actual distribution is `human`".
    """

    # human -> Wikipedia P; machine -> Wikipedia Q.
    # Wikipedia: D_KL(P || Q) = sum_x P(x) * log(P(x) / Q(x))
    # Paper: KL(human, machine) = sum_ij human * log(epsilon + human / (epsilon + machine))

    # NOTE: Behavior when Q has zeroes is not well defined in Kullback-Leibler. Paper solves it with the stabilization constant.
    # NOTE: The default value 1e-16 is adapted from the paper: "The MIT Saliency Benchmark uses MATLAB’s built-in eps = 2.2204e-16"

    human, machine = ensure_shape(human, machine)

    # Probability distribution normalization required.
    human = _make_probability(human)
    machine = _make_probability(machine)

    # Calculations
    scores = human * torch.log(epsilon + human / (epsilon + machine))
    value = scores.sum()

    return value.item()


# TODO: Test
def mean_per_frame(method: Comparison, human: Tensor, machine: Tensor) -> float:
    """
    Runs the given comparison `method` frame-by-frame, and averages the resulting scores over all remaining dimensions.

    * Expects `human` and `machine` to have the same shape (up to squeezing).
    * Expects shared shape to be (..., H, W).
    """

    human, machine = ensure_shape(human, machine)
    assert len(human.shape) >= 2
    height, width = human.shape[-2:]

    human   = human  .reshape(-1, height, width)
    machine = machine.reshape(-1, height, width)

    num_slices = human.shape[0]
    metric = torch.empty(num_slices)

    for idx in range(num_slices):
        metric[idx] = method(human[idx], machine[idx])

    return metric.mean().item()

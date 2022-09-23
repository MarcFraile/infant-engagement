import random
import math
from typing import Tuple, Optional

import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from local.types import TensorMap
from local.util import remap

class NormalNoise:
    """
    Transform for image / video pre-processing.
    Adds normal white noise to every pixel.
    """

    def __init__(self, std: Tuple[float, float]):
        self.std = std

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        std   = random.uniform(*self.std)
        noise = torch.randn_like(x).mul_(std)
        return x.add_(noise)


class AdjustValues:
    """
    Transform for image / video pre-processing.
    Chooses bias and scale randomly from the provided ranges, and returns `bias + x * scale`.
    """

    def __init__(self, bias: Tuple[float, float], scale: Tuple[float, float]):
        self.bias = bias
        self.scale = scale

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        bias  = random.uniform(*self.bias )
        scale = random.uniform(*self.scale)

        return x.mul_(scale).add_(bias)


class AdjustColorChannels:
    """
    Transform for video pre-processing.
    Chooses bias and scale randomly from the provided ranges, per color channel.

    * Input expected to be in shape ...CTHW
    """

    def __init__(self, bias: Tuple[float, float], scale: Tuple[float, float]):
        self.bias = bias
        self.scale = scale

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        bias  = remap(torch.rand(3), (0, 1), self.bias )
        scale = remap(torch.rand(3), (0, 1), self.scale)

        extra_dims = max(0, x.dim() - 4)
        for _ in range(extra_dims):
            bias .unsqueeze(dim=0)
            scale.unsqueeze(dim=0)

        x.mul_(scale[..., None, None, None])
        x.add_(bias [..., None, None, None])

        return x


class AdjustLengthCentral:
    """
    Transform for video pre-processing.
    Shortens the frame length of a video, keeping the middle part.

    * Keeps N frames, where N in range keep[0] ... keep[1] (inclusive).
    * Input expected to be in shape ...THW.
    """

    def __init__(self, keep: Tuple[int, int]):
        self.keep = keep

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        keep = random.randint(*self.keep)
        T = x.shape[-3]
        remove = max(T - keep, 0)
        back = remove // 2
        front = remove - back
        return x[..., front:-back, :, :]


class AdjustLength:
    """
    Transform for video pre-processing.
    Shortens the frame length of a video, keeping a random part.

    * Keeps N frames, where N in range keep[0] ... keep[1] (inclusive).
    * Input expected to be in shape ...THW.
    """

    def __init__(self, keep: Tuple[int, int]):
        self.keep = keep

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        keep = random.randint(*self.keep)
        T = x.shape[-3]
        remove = max(T - keep, 0)
        start = random.randint(0, remove)
        stop  = start + keep
        return x[..., start:stop, :, :]


class BlockRegion:
    """
    Transform for video pre-processing.
    Zeroes out a rectangle in the image through a time range.

    * The sides of the block take a random fraction of the total span, in range fraction[0] ... fraction[1] (inclusive).
    * Input expected to be in shape ...THW.
    """

    def __init__(self, fraction: Tuple[float, float]):
        self.fraction = fraction

    def range(self, span: int) -> Tuple[int, int]:
        m, M = self.fraction

        low  = int(m * span)
        high = int(M * span)

        size = random.randint(low, high)
        start = random.randint(0, span - size)
        stop = start + size

        return (start, stop)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        T, H, W = x.shape[-3:]

        t0, t1 = self.range(T)
        y0, y1 = self.range(H)
        x0, x1 = self.range(W)

        x[..., t0:t1, y0:y1, x0:x1] = 0

        return x


class TemporalDownsample:
    """
    Transform for video processing.
    Performs temporal downsampling with variable target framerate.
    Performs data augmentation by starting from a random offset.

    * Downsamples to (original fps) / N, where N in range skip_frames[0] ... skip_frames[1] (inclusive).
    * Input expected to be in shape ...THW.
    """

    def __init__(self, skip_frames : Tuple[int, int]):
        self.skip_frames = skip_frames

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        T = x.shape[-3]
        skip = random.randint(*self.skip_frames)
        offset = random.randint(0, skip - 1)

        return x[..., offset::skip, :, :]


class SaltPepperNoise:
    """
    Transform for video processing.
    Adds salt and pepper noise (isolated extreme high and low values).

    * Sets pixels to high (resp. low) value with probability `salt_prob` (resp. `pepper_prob`).
    * Expects both probabilities to be < 0.5 (uses this fact to avoid interference).
    * Input expected to be in shape ...CTHW.
    """

    salt_prob   : Tuple[float, float]
    pepper_prob : Tuple[float, float]
    dim         : int

    def __init__(self, salt_prob: Tuple[float, float], pepper_prob: Tuple[float, float], dim: Optional[int] = None):
        self.salt_prob = salt_prob
        self.pepper_prob = pepper_prob
        self.dim = -4 if (dim is None) else dim

    def exp_uniform(self, range: Tuple[float, float]) -> float:
        """
        Randomly chooses a < x < b, such that log(x) is uniformly distributed.
        """

        m = math.log(range[0])
        M = math.log(range[1])
        l = random.uniform(m, M)
        x = math.exp(l)

        return x

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        salt_prob   = self.exp_uniform(self.salt_prob  )
        pepper_prob = self.exp_uniform(self.pepper_prob)
        dim = self.dim % x.dim()

        shape = x.shape[:dim] + (1,) + x.shape[(dim+1):] # Change dim to 1.
        m, M = x.min(), x.max() # TODO: Should try -1, +1, if we're assuming data normalized.

        # How do you respect both P(salt) and P(pepper)?
        # If you just applied one and then the other, the second would sometimes override the first.
        # One way (assuming both probs < 0.5) is to use the high and low ends of X ~ U(0, 1).
        probs = torch.rand(shape)
        probs = torch.cat([probs, probs, probs], dim=dim)
        salt_idx   = (probs > (1 - salt_prob))
        pepper_idx = (probs < pepper_prob)

        x[salt_idx  ] = M
        x[pepper_idx] = m

        return x



class Contiguous:
    """
    Ensures the data is contiguous in memory.

    * Calls x.contiguous()
    """
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x.contiguous()


class AssertDims:
    """
    Checks if width and/or height match an exact size.

    * Input expected to be in format ...HW.
    """

    def __init__(self, width: Optional[int] = None, height: Optional[int] = None):
        self.width = width
        self.height = height

    def __call__(self, x) -> torch.Tensor:
        if self.width != None:
            w = x.shape[-1]
            assert w == self.width, f"Input width expected to be {self.width}, but found {w}."

        if self.height != None:
            h = x.shape[-2]
            assert h == self.height, f"Input height expected to be {self.height}, but found {h}."

        return x


def get_default_transforms() -> Tuple[TensorMap, TensorMap]:
    """
    Returns (train_transform, test_transform) with the standard augmentations.
    """

    train = transforms.Compose([
        AssertDims(height=160),
        AdjustLength(keep=(60, 60)),
        transforms.RandomApply([transforms.RandomRotation(degrees=(-8, +8), interpolation=InterpolationMode.BILINEAR)], p=0.35),
        transforms.RandomResizedCrop((112, 112), scale=(0.6, 1.5), ratio=(0.5, 1.5)),
        Contiguous(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([AdjustValues(bias=(-0.15, +0.15), scale=(0.75, 1.25))], p=0.35),
        transforms.RandomApply([AdjustColorChannels(bias=(-0.1, +0.1), scale=(0.8, 1.2))], p=0.35),
        transforms.RandomApply([transforms.GaussianBlur(5)], p=0.35),
        transforms.RandomApply([NormalNoise(std=(0.01, 0.03))], p=0.35),
    ])

    test = transforms.Compose([
        AssertDims(height=160),
        AdjustLength(keep=(60, 60)),
        transforms.CenterCrop(160),
        Contiguous(),
        transforms.RandomHorizontalFlip(),
    ])

    return (train, test)

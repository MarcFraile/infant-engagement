from typing import Callable

import torch
from torch import Tensor


Profiler    = torch.profiler.profile
TensorMap   = Callable[[Tensor], Tensor]
TensorBimap = Callable[[Tensor, Tensor], Tensor]

from typing import Iterator, Tuple, Protocol
from torch import Tensor


class Loader(Protocol):
    """
    Protocol for dataset loaders.

    * Iterating over a `Loader` returns pairs of tensors representing a batch: `(samples, labels)`.
    * `len(loader)` returns the number of batches in the loader.
    """

    def __iter__(self) -> Iterator[Tuple[Tensor, Tensor]]:
        pass

    def __len__(self) -> int:
        pass


class DummyLoader:
    """
    `Loader` used to return a whole dataset at once (stored as in-memory tensors `H` and `Y`).
    """

    def __init__(self, H: Tensor, Y: Tensor):
        self.H : Tensor = H
        self.Y : Tensor = Y

    def __len__(self) -> int:
        return 1

    def __iter__(self) -> "DummyIter":
        return DummyIter(self.H, self.Y)


class DummyIter:
    """
    `Iterator` used to return a whole dataset at once (stored as in-memory tensors `H` and `Y`).
    """

    def __init__(self, H: Tensor, Y: Tensor):
        self.H    : Tensor = H
        self.Y    : Tensor = Y
        self.flag : bool   = False

    def __len__(self) -> int:
        return 1

    def __iter__(self) -> "DummyIter":
        return self

    def __next__(self) -> Tuple[Tensor, Tensor]:
        if self.flag == False:
            self.flag = True
            return (self.H, self.Y)
        else:
            raise StopIteration


class LimitedLoader:
    """
    Wrapper around a `Loader`. Used to limit the number of batches to the desired length.
    """

    def __init__(self, loader: Loader, limit: int):
        self.loader = loader
        self.limit  = min(limit, len(loader))

    def __len__(self) -> int:
        return self.limit

    def __iter__(self) -> "LimitedIter":
        return LimitedIter(iter(self.loader), self.limit)


class LimitedIter:
    """
    `Iterator` produced by `LimitedLoader`.
    """

    def __init__(self, iter: Iterator[Tuple[Tensor, Tensor]], limit: int):
        self.iter  = iter
        self.limit = limit

        self.current : int = 0

    def __len__(self) -> int:
        return self.limit

    def __iter__(self) -> "LimitedIter":
        return self

    def __next__(self) -> Tuple[Tensor, Tensor]:
        if self.current < self.limit:
            self.current += 1
            return next(self.iter)
        else:
            raise StopIteration

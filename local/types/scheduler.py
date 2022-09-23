from typing import List, Any, Protocol


class Scheduler(Protocol):
    """
    Protocol for learning rate schedulers.

    * `scheduler.get_last_lr()` returns a list of learning rates corresponding to the last time `scheduler.step()` was called.
    * `scheduler.step()` updates the learning rate in the wrapped optimizer.
    """

    def get_last_lr(self) -> List[Any]: # TODO: Refine this further.
        pass

    def step(self) -> None:
        pass


class DummyScheduler:
    """
    Testing utility. Always returns the same learning rate (specified in constructor).
    """

    def __init__(self, lr: float):
        self.lr = lr

    def get_last_lr(self) -> List[Any]:
        return [self.lr]

    def step(self) -> None:
        pass

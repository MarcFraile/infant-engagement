from typing import Generic, List, TypeVar
import random


T = TypeVar("T")


class Sampler(Generic[T]):
    def __call__(self) -> T:
        raise NotImplementedError()

    def __repr__(self) -> str:
        raise NotImplementedError()


class ConstantSampler(Sampler[T]):
    def __init__(self, value: T):
        self.value = value

    def __call__(self) -> T:
        return self.value

    def __repr__(self) -> str:
        return f"ConstantSampler(value: {self.value})"


class LinearSampler(Sampler[float]):
    def __init__(self, min: float, max: float):
        self.min = min
        self.max = max

    def __call__(self) -> float:
        return random.uniform(self.min, self.max)

    def __repr__(self) -> str:
        return f"LinearSampler(min: {self.min}, max: {self.max})"


class ExponentialSampler(Sampler[float]):
    def __init__(self, base: float, min_exp: float, max_exp: float):
        self.base    = base
        self.min_exp = min_exp
        self.max_exp = max_exp

    def __call__(self) -> float:
        exp = random.uniform(self.min_exp, self.max_exp)
        return self.base ** exp

    def __repr__(self) -> str:
        return f"ExponentialSampler(base: {self.base}, min_exp: {self.min_exp}, max_exp: {self.max_exp})"


class CategoricalSampler(Sampler[T]):
    def __init__(self, categories: List[T]):
        self.categories = categories

    def __call__(self) -> T:
        return random.choice(self.categories)

    def __repr__(self) -> str:
        return f"CategoricalSampler(categories: {self.categories})"


class MixtureSampler(Sampler[T]):
    def __init__(self, samplers: List[Sampler[T]], weights: List[float]):
        self.samplers = samplers
        self.weights  = weights

    def __call__(self) -> T:
        sampler = random.choices(self.samplers, self.weights)[0]
        return sampler()

    def __repr__(self) -> str:
        return f"MixtureSampler(samplers: {self.samplers}, weights: {self.weights})"

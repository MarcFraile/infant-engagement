from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Tuple, List

import torch
from torch import Tensor

from local.training import KFoldsManager
from local.loader import DummyLoader


@dataclass
class TensorSampleData:
    """
    * Holds samples and labels as `Tensor` (fields `X` and `Y`).
    * Expected shapes: `(num_samples, ...)` (`X` and `Y` should match on the first dimension).
    """
    X : Tensor
    Y : Tensor


def detect_folds(feature_root: Path, task: str, variable: str, device: torch.device) -> Tuple[List[TensorSampleData], List[TensorSampleData]]:
    """
    Loads all folds from `feature_root` directly into GPU.

    * Returns `(train, test)`.
    * Expects samples to be saved as `H_(train|test)_<fold number>.pt`
    * Expects labels to be saved as `Y_(train|test)_<fold number>.pt`
    * Expects the test fold to be the last, and to only have test data: `(H|Y)_test_*.pt` should have one more entry than `(H|Y)_train_*.pt`
    """

    files = [ file for file in feature_root.iterdir() if file.is_file() and file.suffix == ".pt" ]
    files.sort()

    tensors   : Dict[str, Dict[str, List[Tensor]]] = dict()
    num_folds : int = -1

    for fold_type in ["train", "test"]:
        tensors[fold_type] = dict()
        for tensor_type in ["H", "Y"]:
            prefix = f"{tensor_type}_{task}_{variable}_{fold_type}"
            classified : List[Path] = [ file for file in files if file.stem.startswith(prefix) ]

            expected_num_folds = len(classified)
            assert expected_num_folds > 1
            if fold_type == "test":
                expected_num_folds -= 1

            if num_folds < 0:
                num_folds = expected_num_folds
            else:
                assert num_folds == expected_num_folds

            tensors[fold_type][tensor_type] = []

            for k in range(len(classified)):
                path = classified[k]
                assert path.stem == f"{prefix}_{k}"
                data : Tensor = torch.load(path).to(device)
                tensors[fold_type][tensor_type].append(data)

    # tensors = { prefix: [ torch.load(file).to(device) for file in files ] for (prefix, files) in classified.items() }

    train = [ TensorSampleData(H_train.float(), Y_train.long()) for (H_train, Y_train) in zip(tensors["train"]["H"], tensors["train"]["Y"]) ]
    test  = [ TensorSampleData(H_test .float(), Y_test .long()) for (H_test , Y_test ) in zip(tensors["test" ]["H"], tensors["test" ]["Y"]) ]

    return (train, test)


class TensorManager(KFoldsManager):
    """
    * Handles leave-one-out k-folds augmentation for `Tensor` collections.
    * Use `manager.leave_one_out(k)` to fuse all added folds into one training set, and use fold `k` as a validation set.
    """

    feature_root : Path
    task         : str
    variable     : str
    device       : torch.device

    _num_folds   : int
    _train_folds : List[TensorSampleData]
    _val_folds   : List[TensorSampleData]
    _test_fold   : TensorSampleData

    def __init__(self, feature_root: Path, task: str, variable: str, device: torch.device):
        assert feature_root.is_dir()
        assert task     in [ "people", "eggs", "drums" ]
        assert variable in [ "attending", "participating" ]

        self.feature_root = feature_root
        self.task         = task
        self.variable     = variable
        self.device       = device

        train_folds, test_folds = detect_folds(feature_root, task, variable, device)

        # We expect the test fold to be the last one, and to not have training data.
        assert len(train_folds) + 1 == len(test_folds), f"Expected to find one more test fold (the data used for testing only) than train folds (the data used for k-folds cross validation). Found train: {len(train_folds)}, test: {len(test_folds)}."

        self._train_folds = train_folds
        self._val_folds   = test_folds[:-1]
        self._test_fold   = test_folds[-1]

        self._num_folds = len(self._train_folds) # We ignore the test fold.

    def num_folds(self) -> int:
        """
        Number of training / validation folds (not counting the testing fold).
        """
        return self._num_folds

    def leave_one_out(self, k: int) -> Tuple[DummyLoader, DummyLoader]:
        """
        * Keeps fold `k` as a validation set.
        * Merges all other folds into a training set.
        * Returns `(train_set, val_set)`
        """
        assert self._num_folds > 0
        assert 0 <= k < self._num_folds

        train_data : List[TensorSampleData] = []
        val_data   : TensorSampleData

        for idx in range(self._num_folds):
            if idx == k:
                val_data = self._val_folds[idx]
            else:
                train_data.append(self._train_folds[idx])

        X_train : Tensor = torch.cat([ data.X for data in train_data ])
        Y_train : Tensor = torch.cat([ data.Y for data in train_data ])

        train = DummyLoader(X_train, Y_train)
        val   = DummyLoader(val_data.X, val_data.Y)

        return (train, val)

    def test_loader(self) -> DummyLoader:
        """
        Creates a new instance of the test `Loader`.
        """
        return DummyLoader(
            self._test_fold.X,
            self._test_fold.Y,
        )

    def full_train_loader(self) -> DummyLoader:
        """
        Joins all non-test folds into a single training fold.

        * Use only for final training! For validation tasks, use `leave_one_out(k)`.
        """
        X = torch.cat([ data.X for data in self._train_folds ])
        Y = torch.cat([ data.Y for data in self._train_folds ])

        return DummyLoader(X, Y)

    def leave_one_out_prob(self, k: int) -> float:
        """
        * Returns the empirical probability of the training set returned by `leave_one_out()`.
        """

        assert self._num_folds > 0
        assert 0 <= k < self._num_folds

        total_vids       : int   = 0
        positive_samples : float = 0

        for idx in range(self._num_folds):
            if idx == k:
                continue
            else:
                labels = self._train_folds[idx].Y
                total_vids += len(labels)
                positive_samples += labels.sum().item()

        return positive_samples / total_vids

    def full_train_prob(self) -> float:
        """
        * Returns the empirical probability of the full training set returned by `full_train_loader()`.
        """

        assert self._num_folds > 0

        total_vids       : int   = 0
        positive_samples : float = 0

        for idx in range(self._num_folds):
            labels = self._train_folds[idx].Y
            total_vids += len(labels)
            positive_samples += labels.sum().item()

        return positive_samples / total_vids

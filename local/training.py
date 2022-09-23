from typing import Dict, Optional, List, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import copy

from tqdm import tqdm, trange
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
from torch.optim import Optimizer

from local.cli import PrettyCli
from local.metrics import confusion as get_conf
from local.types import Profiler, TensorMap, TensorBimap, Loader, Scheduler


def freeze(model: nn.Module) -> None:
    """
    Freeze the `model` so it does not train.

    * Sets `requires_grad = False` for all the model's parameters.
    """
    for param in model.parameters():
        param.requires_grad = False


def thaw(model: nn.Module) -> None:
    """
    Un-freeze the `model` so it can train again.

    * Sets `requires_grad = True` for all the model's parameters.
    """
    for param in model.parameters():
        param.requires_grad = True


class KFoldsManager:
    """
    Base class for K-folds data managers.
    """
    def num_folds(self) -> int:
        """
        Number of training / validation folds (not counting the testing fold).
        """
        raise NotImplementedError()

    def one_fold_loader(self, k: int, is_train=True) -> Loader:
        """
        * Returns a loader for fold `k`.
        * Use `is_train` to determine the data augmentation pipeline.
        """
        raise NotImplementedError()

    def leave_one_out(self, k: int) -> Tuple[Loader, Loader]:
        """
        * Keeps fold `k` as a validation set.
        * Merges all other folds into a training set.
        * Returns `(train_loader, val_loader)`.
        """
        raise NotImplementedError()

    def test_loader(self) -> Loader:
        """
        Creates a new instance of the test `Loader`.
        """
        raise NotImplementedError()

    def full_train_loader(self) -> Loader:
        """
        Joins all non-test folds into a single training fold.

        * Use only for final training! For validation tasks, use `leave_one_out(k)`.
        """
        raise NotImplementedError()

    def leave_one_out_prob(self, k: int) -> float:
        """
        * Returns the empirical probability of the training set returned by `leave_one_out()`.
        """
        raise NotImplementedError()

    def full_train_prob(self) -> float:
        """
        * Returns the empirical probability of the full training set returned by `full_train_loader()`.
        """
        raise NotImplementedError()


@dataclass
class FoldMetrics:
    """
    Container dataclass for metrics.

    * Can be used both for history and for snapshots.
    * `loss` should have shape `()` for snapshot, `(epochs,)` for history.
    * `confusion` should have shape `(num_classes, num_classes)` for snapshot, `(epochs, num_classes, num_classes)` for history.
    * `metrics` entries should have shape `()` for snapshot, `(epochs,)` for history.
    * `FoldMetrics(loss, confusion, metrics)` should be prefered for snapshots.
    * `FoldMetrics.History(epochs, metric_funcs, num_classes?)` initializes a history-type `FoldMetrics` instance with zeros.
    """
    loss      : Tensor
    confusion : Tensor
    metrics   : Dict[str, Tensor]

    @staticmethod
    def History(epochs: int, metric_funcs: Dict[str, TensorMap], num_classes: int = 2) -> "FoldMetrics":
        """Initializes a history-type FoldMetrics instance with zeros."""

        assert epochs >= 0
        assert num_classes >= 2

        return FoldMetrics(
            loss = torch.zeros(epochs + 1),
            confusion = torch.zeros(epochs + 1, num_classes, num_classes),
            metrics = { name: torch.zeros(epochs + 1) for name in metric_funcs },
        )

    def set(self, idx: int, snapshot: "FoldMetrics") -> None:
        assert 0 <= idx < len(self.loss)
        assert len(self.confusion) == len(self.loss)

        assert snapshot.loss     .shape == (1,)  and len(self.loss.shape) == 1         , f"Incompatible loss shapes: snapshot={snapshot.loss.shape}, history={self.loss.shape}"
        assert snapshot.confusion.shape == (2,2) and self.confusion.shape[1:] == (2,2) , f"Incompatible confusion shapes: snapshot={snapshot.confusion.shape}, history={self.confusion.shape}"

        assert snapshot.metrics.keys() == self.metrics.keys(), f"Incompatible metric sets: snapshot={snapshot.metrics.keys()}, history={self.metrics.keys()}"

        self.loss     [idx] = snapshot.loss
        self.confusion[idx] = snapshot.confusion

        for (key, value) in snapshot.metrics.items():
            assert len(self.metrics[key]) == len(self.loss)
            self.metrics[key][idx] = value

    def __getitem__(self, idx: int) -> "FoldMetrics":
        assert -len(self.loss) < idx < len(self.loss)
        assert len(self.confusion) == len(self.loss)

        loss = self.loss[idx]
        confusion = self.confusion[idx]
        metrics = { key: value[idx] for (key, value) in self.metrics.items() }

        return FoldMetrics(loss, confusion, metrics)

    def __len__(self) -> int:
        return len(self.loss)

    def as_dict(self) -> Dict[str, Tensor]:
        """
        Returns a dict containing the `loss` and the entries in `metrics` (but not the `confusion` matrix).
        """
        assert "loss" not in self.metrics
        output = { "loss": self.loss.cpu().squeeze() }
        output.update(self.metrics)
        return output


@dataclass
class FoldPack:
    train : FoldMetrics
    val   : Optional[FoldMetrics]
    test  : Optional[FoldMetrics]

    def __len__(self) -> int:
        return len(self.train)

    def valid(self) -> bool:
        return (len(self.train) > 0) \
            and ((self.val  is None) or (len(self.val ) == len(self.train))) \
            and ((self.test is None) or (len(self.test) == len(self.train)))


@dataclass
class RunHistory:
    epochs           : int
    total_duration   : timedelta
    best_val_epoch   : Optional[int]
    best_net         : nn.Module
    durations        : List[timedelta]
    original_metrics : FoldPack
    smoothed_metrics : FoldPack


def _get_figure(name: str, train: Tensor, val: Optional[Tensor] = None, test: Optional[Tensor] = None, best_epoch: Optional[int] = None) -> Figure:
    assert len(train) > 0
    assert (val  is None) or (len(val ) == len(train))
    assert (test is None) or (len(test) == len(train))

    x = range(len(train))
    fig = plt.figure()
    legend = []

    if best_epoch is not None:
        plt.axvline(x=best_epoch, color="red")
        legend.append("best epoch")

    plt.plot(x, train)
    legend.append("train")

    if val is not None:
        plt.plot(x, val)
        legend.append("val")

    if test is not None:
        plt.plot(x, test)
        legend.append("test")

    plt.legend(legend)
    plt.xlabel("epochs")
    plt.ylabel(name)

    return fig


def plot_metric(pack: FoldPack, metric: str, best_epoch: Optional[int] = None) -> Figure:
    """
    Returns a figure showing the train, validation and test histories for the given metric.
    """
    assert metric in pack.train.metrics
    return _get_figure(
        metric,
        pack.train.metrics[metric],
        pack.val  .metrics[metric] if pack.val  else None,
        pack.test .metrics[metric] if pack.test else None,
        best_epoch,
    )


def plot_loss(pack: FoldPack) -> Figure:
    return _get_figure(
        "loss",
        pack.train.loss,
        pack.val  .loss if pack.val  else None,
        pack.test .loss if pack.test else None,
    )


def plot_pack_metrics(pack: FoldPack, best_epoch: Optional[int] = None) -> Dict[str, Figure]:
    """
    Returns a dict of plot figures corresponding to each metric (including the loss).
    """
    output = { metric: plot_metric(pack, metric, best_epoch) for metric in pack.train.metrics }
    output["loss"] = plot_loss(pack)
    return output


def plot_history_metrics(history: RunHistory) -> Dict[str, Figure]:
    """
    Returns a dict of plot figures corresponding to each metric (including the loss).
    """
    original = { "original_" + key: value for (key, value) in plot_pack_metrics(history.original_metrics, history.best_val_epoch).items() }
    smoothed = { "smoothed_" + key: value for (key, value) in plot_pack_metrics(history.smoothed_metrics, history.best_val_epoch).items() }
    return { **original, **smoothed }


def test(
    model         : nn.Module,
    loader        : Loader,
    loss_func     : TensorBimap,
    metric_funcs  : Dict[str, TensorMap],
    device        : torch.device,
    show_progress : bool,
    desc          : str,
) -> FoldMetrics:
    """
    Test `model` network with `loader` data on `device`.

    * Returns (loss, accuracy, confusion).
    * Confusion is a tensor containing [tp, fp, tn, fn].
    """

    loss      : Tensor = torch.zeros(1,    device=device)
    confusion : Tensor = torch.zeros(2, 2, device=device)

    size : int = len(loader)
    if show_progress:
        loader = tqdm(loader, total=size, desc=desc, leave=False)

    model.eval()
    with torch.no_grad():
        batch  : Tensor
        labels : Tensor
        for (batch, labels) in loader:
            batch, labels = batch.to(device), labels.to(device)

            scores: Tensor = model(batch)
            predictions = scores.argmax(dim=1)

            loss += loss_func(scores, labels) / size
            confusion += get_conf(predictions, labels)

        metrics = { name: func(confusion) for (name, func) in metric_funcs.items() }

    return FoldMetrics(loss, confusion, metrics)


def train_epoch(
    model        : nn.Module,
    optimizer    : Optimizer,
    loader       : Loader,
    loss_func    : TensorBimap,
    metric_funcs : Dict[str, TensorMap],
    device       : torch.device,
    use_tqdm     : bool,
) -> FoldMetrics:

    loss      : Tensor = torch.zeros(1)
    confusion : Tensor = torch.zeros(2, 2)

    size : int = len(loader)
    if use_tqdm:
        loader = tqdm(loader, total=size, desc="train", leave=False)

    model.train()

    batch  : Tensor
    labels : Tensor
    for (batch, labels) in loader:

        batch  = batch .to(device)
        labels = labels.to(device)

        _loss  : List[Tensor] = [None]
        _preds : List[Tensor] = [None]
        def closure():
            optimizer.zero_grad()

            scores : Tensor = model(batch)
            predictions = scores.argmax(dim=1).float()

            closure_loss = loss_func(scores, labels)
            closure_loss.backward()

            _loss[0]  = closure_loss.detach().cpu()
            _preds[0] = predictions.detach().cpu()

            return closure_loss

        optimizer.step(closure)
        optimizer.zero_grad()

        loss += _loss[0] / size
        confusion += get_conf(_preds[0], labels.detach().cpu())

    metrics = { name: func(confusion) for (name, func) in metric_funcs.items() }

    return FoldMetrics(loss, confusion, metrics)


def _get_smoothed(metrics: FoldMetrics, current_epoch: int, smoothing_window: int) -> FoldMetrics:
    assert len(metrics) > 1
    assert 0 <= current_epoch < len(metrics)
    assert smoothing_window > 0

    smooth_start = max(0, current_epoch - smoothing_window)
    smooth_end   = min(current_epoch + 1, len(metrics))

    return FoldMetrics(
        loss      = metrics.loss     [smooth_start:smooth_end].mean(dim=0).reshape((1,)),
        confusion = metrics.confusion[smooth_start:smooth_end].mean(dim=0),
        metrics   = { key: value     [smooth_start:smooth_end].mean(dim=0) for (key, value) in metrics.metrics.items() },
    )


def train(
    model            : nn.Module,
    optimizer        : Optimizer,
    scheduler        : Scheduler, # TODO: This should be Optional.
    train_loader     : Loader,
    val_loader       : Optional[Loader],
    test_loader      : Optional[Loader],
    loss_func        : TensorBimap,
    metric_funcs     : Dict[str, TensorMap],
    target_metric    : Optional[str],
    device           : torch.device,
    cli              : PrettyCli,
    epochs           : int,
    verbose          : bool = True,
    show_progress    : str = "permanent", # no | temporary | permanent
    smoothing_window : int = 5,
    warmup_window    : int = 5,
    # backprop_batch_size : Optional[int] = None,
    profiler         : Optional[Profiler] = None,
) -> RunHistory:
    """
    Train the `model`.

    Inputs:
    * `model`:               Trained in-place for the given number of `epochs`.
    * `metric_funcs`:        Dictionary where each `(name, func)` entry represents one metric to be tracked (accuracy, F1 score, balanced accuracy...). Functions take confusion matrix as input, shape `(label, predicted)`.
    * `target_metric`:       Name of the metric that will be used to deteremine the best iteration (based on validation data).
    * `backprop_batch_size`: If `None`, ignored. If `int`, number of samples to process before one backpropagation pass. Used to aggregate several batches in weaker hardware.
    * `verbose`:             Controls if metric reports are printed (does not handle progress bars).
    * `show_progress`:       Set to `"no"` to skip progress bars; `"temporary"` to remove top-level progress bars after finishing; `"permanent"` to persist top-level progress bars.

    Outputs: `(best_net, hist, best_val_epoch)`
    * `best_net`: copy of the network as it was in the best epoch, as judged by the given `target_metric`.
    * `hist`:
    """

    assert len(metric_funcs) > 0
    assert (target_metric is None) or (target_metric in metric_funcs)
    assert epochs >= 0
    assert show_progress in ["no", "temporary", "permanent"]
    # assert smoothing_window > 0
    # assert warmup_window >= 0

    use_tqdm   : bool = (show_progress != "no")
    leave_tqdm : bool = (show_progress == "permanent")

    train_hist     : FoldMetrics = FoldMetrics.History(epochs, metric_funcs)
    train_smoothed : FoldMetrics = FoldMetrics.History(epochs, metric_funcs)

    val_hist     : Optional[FoldMetrics] = None
    val_smoothed : Optional[FoldMetrics] = None
    if val_loader:
        val_hist     = FoldMetrics.History(epochs, metric_funcs)
        val_smoothed = FoldMetrics.History(epochs, metric_funcs)

    test_hist     : Optional[FoldMetrics] = None
    test_smoothed : Optional[FoldMetrics] = None
    if test_loader:
        test_hist     = FoldMetrics.History(epochs, metric_funcs)
        test_smoothed = FoldMetrics.History(epochs, metric_funcs)

    durations  : List[timedelta] = [None] * (epochs + 1)

    best_val_target : float         = float("-inf")
    best_val_epoch  : Optional[int] = None
    best_net        : nn.Module

    start_time = datetime.now()

    should_checkpoint : bool = (val_loader is not None) and (target_metric is not None)
    train_stats       : FoldMetrics
    val_stats         : Optional[FoldMetrics]
    test_stats        : Optional[FoldMetrics]

    untrained_iter = trange(1, desc="Untrained (epoch 0)", leave=leave_tqdm) if use_tqdm else range(1)
    for _ in untrained_iter:
        train_stats = test(model, train_loader, loss_func, metric_funcs, device, use_tqdm, desc="train")
        train_hist    .set(0, train_stats)
        train_smoothed.set(0, train_stats)

        if val_loader:
            assert (val_hist is not None) and (val_smoothed is not None)
            val_stats = test(model, val_loader, loss_func, metric_funcs, device, use_tqdm, desc="val" )
            val_hist    .set(0, val_stats)
            val_smoothed.set(0, val_stats)

            # Start early: if training only makes things worse (e.g. if we're fine-tuning after head training), preserve the original.
            if should_checkpoint:
                assert target_metric is not None
                best_val_target = val_stats.metrics[target_metric].item()
                best_val_epoch  = 0
                best_net        = copy.deepcopy(model).cpu()

        if test_loader:
            assert (test_hist is not None) and (test_smoothed is not None)
            test_stats = test(model, test_loader, loss_func, metric_funcs, device, use_tqdm, desc="test" )
            test_hist    .set(0, test_stats)
            test_smoothed.set(0, test_stats)

    curr_time = datetime.now()
    durations[0] = curr_time - start_time
    prev_time = curr_time

    if verbose:
        cli.section(f"EPOCH:   0 / {epochs} (  0.0%)")
        display_data = {
            "Duration"  : durations[0],
            "Train"     : train_stats.as_dict(),
        }
        if (val_stats is not None):
            display_data["Validation"] = val_stats.as_dict()
        if (test_stats is not None):
            display_data["Test"] = test_stats.as_dict(),
        cli.print(display_data)

    epoch_iter = trange(1, epochs + 1, desc="Training", leave=leave_tqdm) if use_tqdm else range(1, epochs + 1)
    for current_epoch in epoch_iter:

        train_stats = train_epoch(model, optimizer, train_loader, loss_func, metric_funcs, device, use_tqdm)#, backprop_batch_size)
        train_hist.set(current_epoch, train_stats)
        train_smoothed.set(current_epoch, _get_smoothed(train_hist, current_epoch, smoothing_window))

        if val_loader:
            assert (val_hist is not None) and (val_smoothed is not None)
            val_stats = test(model, val_loader, loss_func, metric_funcs, device, use_tqdm, desc="val")
            val_hist.set(current_epoch, val_stats)
            val_smoothed.set(current_epoch, _get_smoothed(val_hist, current_epoch, smoothing_window))

        if test_loader:
            assert (test_hist is not None) and (test_smoothed is not None)
            test_stats = test(model, test_loader, loss_func, metric_funcs, device, use_tqdm, desc="test")
            test_hist.set(current_epoch, test_stats)
            test_smoothed.set(current_epoch, _get_smoothed(test_hist, current_epoch, smoothing_window))

        curr_time = datetime.now()
        durations[current_epoch] = curr_time - prev_time
        prev_time = curr_time

        if verbose:
            percent = ((1000 * current_epoch) // epochs) / 10 # Rounded to 1 decimal place.
            cli.section(f"EPOCH: {current_epoch:3} / {epochs} ({percent:03.01f}%)")
            display_data = {
                "Duration"      : f"{durations[current_epoch]} ({curr_time - start_time})",
                "Learning Rate" : scheduler.get_last_lr(),
                "Train"         : train_stats.as_dict(),
            }
            if (val_stats is not None):
                display_data["Validation"] = val_stats.as_dict()
            if (test_stats is not None):
                display_data["Test"] = test_stats.as_dict(),
            cli.print(display_data)

        if should_checkpoint and (current_epoch > warmup_window):
            assert (val_smoothed is not None) and (target_metric is not None)
            val_target : float = val_smoothed[current_epoch].metrics[target_metric].item()
            if val_target > best_val_target:
                best_val_target = val_target
                best_val_epoch  = current_epoch
                best_net = copy.deepcopy(model).cpu()

        scheduler.step()

        if profiler is not None:
            profiler.step()

    if not should_checkpoint:
        best_net = copy.deepcopy(model).cpu()

    total_duration = datetime.now() - start_time

    model.eval()
    return RunHistory(
        epochs           = epochs,
        total_duration   = total_duration,
        best_val_epoch   = best_val_epoch,
        best_net         = best_net,
        durations        = durations,
        original_metrics = FoldPack(train_hist, val_hist, test_hist),
        smoothed_metrics = FoldPack(train_smoothed, val_smoothed, test_smoothed),
    )


def l2_loss(model: nn.Module, device: Optional[torch.device] = None) -> Tensor:
    loss : Tensor = torch.zeros(1, device=device)

    for p in model.parameters():
        loss += (p ** 2).sum()

    return loss


class BCELL2Loss:
    """
    Binary Cross Entropy with Logits and L2 Penalization Loss.
    """

    def __init__(self, model: nn.Module, parameter_decay: float, positive_weight: Optional[Tensor] = None, device: Optional[torch.device] = None):
        self.model = model
        self.parameter_decay = parameter_decay
        self.positive_weight = positive_weight
        self.device = device

    def __call__(self, input: Tensor, target: Tensor) -> Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(input, target, pos_weight=self.positive_weight)
        l2_penalty = self.parameter_decay * l2_loss(self.model, self.device)
        return bce_loss + l2_penalty


class MCELL2Loss:
    """
    Multi-Class Cross Entropy with Logits and L2 Penalization Loss.
    """

    def __init__(self, model: nn.Module, parameter_decay: float, class_weights: Optional[Tensor] = None, device: Optional[torch.device] = None):
        self.model = model
        self.parameter_decay = parameter_decay
        self.class_weights = class_weights
        self.device = device

    def __call__(self, input: Tensor, target: Tensor) -> Tensor:
        ce_loss = F.cross_entropy(input, target, weight=self.class_weights)
        l2_penalty = self.parameter_decay * l2_loss(self.model, self.device)
        return ce_loss + l2_penalty

from dataclasses import dataclass, asdict
from typing import Any, Optional, Dict, List

import torch
from torch import Tensor, optim, nn

import pandas as pd

from local.cli import PrettyCli
from local.types import Loader
from local.types import Scheduler, TensorMap, TensorBimap
from local.training import MCELL2Loss, RunHistory, train
from local.metrics import acc_from_conf, f1, balanced_accuracy


class TrainingHelper:
    """
    Helper class. Takes care of initializing the needed materials for training; training; and compiling statistics.

    * Call `get_materials()` to translate script parameters to actual optimizers, losses, weights...
    * Call `train()` as a managed wrapper to `e2e.babylab.folds.training.train()`.
    * Call `get_stats()` to obtain a `DataFrame` containing the metrics for the best (or last) run, among other data.
    """

    @dataclass
    class Meta:
        """Data needed by the training process that is not considered a hyper-parameter."""
        epochs         : int
        empirical_prob : float
        fold_number    : Optional[int] = None
        repetition     : Optional[int] = None

    @dataclass
    class Params:
        """Hyper-parameter settings needed by the training process."""
        lr_init           : float
        lr_decay          : float
        weight_decay      : float
        use_class_weights : bool
        optimizer_name    : Optional[str] = None

    @dataclass
    class Materials:
        """Actual objects used for training, derived from the metadata and hyper-parameters."""
        class_weights : Optional[Tensor]
        loss_func     : TensorBimap
        optimizer     : optim.Optimizer
        scheduler     : Scheduler
        metric_funcs  : Dict[str, TensorMap]
        target_metric : str

    @dataclass
    class Loaders:
        """
        Train (mandatory), validation (optional) and test (optional) data loaders.

        * A validation loader is needed if checkpointing is desired.
        """
        train : Loader
        val   : Optional[Loader] = None
        test  : Optional[Loader] = None

    @staticmethod
    def param_names() -> List[str]:
        """Return the names of all parameters in a `TrainingHelper.Params` pack, including optional parameters."""
        return [ key for key in TrainingHelper.Params.__dataclass_fields__ ]

    net    : nn.Module
    meta   : Meta
    params : Params
    gpu    : torch.device
    cli    : PrettyCli
    hist   : Optional[RunHistory]

    def __init__(self, net: nn.Module, meta: Meta, params: Params, gpu: torch.device, cli: PrettyCli):
        self.net    = net
        self.meta   = meta
        self.params = params
        self.gpu    = gpu
        self.cli    = cli


    def get_materials(self) -> Materials:
        """Translate the metadata and hyper-parameters passed in the constructor call to actual losses, optimizers..."""
        class_weights : Optional[Tensor]
        if self.params.use_class_weights:
            # TODO: TEST THAT THE ORDER IS CORRECT!
            class_weights = torch.tensor([ 1 / (1 - self.meta.empirical_prob), 1 / self.meta.empirical_prob ], device=self.gpu) # TODO: Should we add an epsilon to avoid division by 0?
        else:
            class_weights = None

        loss_func : TensorBimap
        optimizer : optim.Optimizer
        if (self.params.optimizer_name is None) or (self.params.optimizer_name == "ADAM"):
            # loss_func = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            loss_func = nn.CrossEntropyLoss(class_weights)
            optimizer = optim.Adam(self.net.parameters(), lr=self.params.lr_init, weight_decay=self.params.weight_decay)
        elif self.params.optimizer_name == "LBFGS":
            loss_func = MCELL2Loss(self.net, self.params.weight_decay, class_weights, self.gpu)
            optimizer = optim.LBFGS(self.net.parameters(), lr=self.params.lr_init)
        else:
            raise Exception(f"Unknown optimizer_name: {self.params.optimizer_name}")

        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=(1.0 - self.params.lr_decay))

        metric_funcs  : Dict[str, TensorMap] = { "accuracy": acc_from_conf, "f1": f1, "balanced_accuracy": balanced_accuracy, }
        target_metric : str                  = "f1"

        return TrainingHelper.Materials(class_weights, loss_func, optimizer, scheduler, metric_funcs, target_metric)

    def train(self, materials: Materials, loaders: Loaders, **kwargs) -> RunHistory:
        """Calls the standard `train()` procedure with the necessary parameters."""
        self.hist = train(
            self.net,
            materials.optimizer, materials.scheduler,
            loaders.train, loaders.val, loaders.test,
            materials.loss_func, materials.metric_funcs, materials.target_metric,
            self.gpu, self.cli, self.meta.epochs,
            **kwargs,
        )
        return self.hist

    def get_stats(self) -> pd.DataFrame:
        """
        Get the stats entry for this repetition in `DataFrame` format.

        * Only use this if you have already ran `train()`! Needs the history data.
        * If validation data was provided, reports the best performing epoch. Otherwise, reports the last epoch.
        * Train data is always reported. Validation and test data are reported if available.
        """

        params = asdict(self.params)
        assert "optimizer_name" in params
        if params["optimizer_name"] == None:
            params.pop("optimizer_name")

        assert self.hist is not None

        idx : int
        if self.hist.best_val_epoch is not None:
            idx = self.hist.best_val_epoch
        else:
            idx = -1

        rep_data : Dict[str, Any] = {}
        if self.meta.fold_number is not None:
            rep_data["fold_num"] = self.meta.fold_number
        if self.meta.repetition is not None:
            rep_data["repetition"] = self.meta.repetition

        train_report : Dict[str, Any]
        val_report   : Dict[str, Any] = {}
        test_report  : Dict[str, Any] = {}

        train_stats  = self.hist.original_metrics.train[idx].as_dict()
        train_report = { "train_" + key : value.numpy() for (key, value) in train_stats.items() }

        if self.hist.original_metrics.val:
            val_stats  = self.hist.original_metrics.val[idx].as_dict()
            val_report = { "validation_" + key : value.numpy() for (key, value) in val_stats.items() }

        if self.hist.original_metrics.test:
            test_stats  = self.hist.original_metrics.test[idx].as_dict()
            test_report = { "test_" + key : value.numpy() for (key, value) in test_stats.items() }

        return pd.DataFrame({
            **params,
            **rep_data,
            **train_report,
            **val_report,
            **test_report,
            "duration": self.hist.total_duration,
        }, index=[0])

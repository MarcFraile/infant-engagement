#!/bin/env -S python3 -u


# ================ IMPORTS ================ #


from dataclasses import asdict
import os
from pathlib import Path
from typing import Tuple

import torch
from torch import nn

import pandas as pd
from tqdm import trange

from local.cli import PrettyCli, CliHelper
from local.hyperparam_sampling import Sampler, ConstantSampler, ExponentialSampler, CategoricalSampler, MixtureSampler
from local.network_models import Classifier
from local.training_helper import TrainingHelper
from local.types import Loader
from local.datasets import TensorManager
from local import search
from local.search import SearchResults


# ================ SETTINGS ================ #


# ---- Hand-Chosen Parameters ---- #

# The parameters are defined in the dict RUN_PARAMS and injected using locals().update(...)
# We add the following lines so linters don't get sad.

# Script Params
TASK        : str
VARIABLE    : str
EPOCHS      : int
SAMPLES     : int
REPETITIONS : int

# Search Space
LR_INIT           : Sampler[float]
LR_DECAY          : Sampler[float]
WEIGHT_DECAY      : Sampler[float]
USE_CLASS_WEIGHTS : Sampler[bool]

RUN_PARAMS = {
    "SCRIPT_PARAMS" : {
        "TASK"        : "eggs", # people | eggs | drums
        "VARIABLE"    : "participating",
        "EPOCHS"      : 20,
        "SAMPLES"     : 250,
        "REPETITIONS" : 4,
    },
    "SEARCH_SPACE" : {
        "LR_INIT"           : ExponentialSampler(base=10, min_exp=-5, max_exp=0),
        "LR_DECAY"          : MixtureSampler(samplers=[ConstantSampler(value=0.0), ExponentialSampler(base=10, min_exp=-3, max_exp=-1)], weights=[1, 4]),
        "WEIGHT_DECAY"      : MixtureSampler(samplers=[ConstantSampler(value=0.0), ExponentialSampler(base=10, min_exp=-4, max_exp=+1)], weights=[1, 4]),
        "USE_CLASS_WEIGHTS" : CategoricalSampler(categories=[True, False]),
    },
}

locals().update(RUN_PARAMS["SCRIPT_PARAMS"]) # Make params into variables.
locals().update(RUN_PARAMS["SEARCH_SPACE"])

# ---- Input sources ---- #

# Same deal as above.

CWD             : Path = Path(os.getcwd()).resolve() # Book-keeping.
FEATURE_ROOT    : Path # Load pre-computed encoded samples from this directory.
CURRENT_SCRIPT  : Path # Book-keeping.

INPUT_DIRS = {
    "CWD"          : CWD,
    "FEATURE_ROOT" : Path("data/processed/baked_samples/"),
}

INPUT_FILES = {
    "CURRENT_SCRIPT" : Path(__file__).resolve().relative_to(CWD),
}

locals().update(INPUT_DIRS)
locals().update(INPUT_FILES)

OUTPUT_ROOT : Path = Path("artifacts/head_search/")

# CLI pretty-printing and script helper functions

cli = PrettyCli()
helper = CliHelper(cli)


# ================ FUNCTIONS ================ #


def main():
    cli.main_title(f'HEAD TRAINING: "{TASK}"\nHYPERPARAMETER SEARCH')

    env = helper.report_environment()
    helper.report_input_sources(INPUT_DIRS, INPUT_FILES)
    gpu = helper.report_gpu()
    start_time, out_path = helper.setup_output_dir(OUTPUT_ROOT, RUN_PARAMS)
    manager = helper.report_tensor_manager(FEATURE_ROOT, TASK, VARIABLE, gpu)

    results = train_head(gpu, manager)

    torch.save(results.best_net, out_path / "best_net.pt")
    search.report_results(helper, out_path, start_time, results)
    search.stats_and_plots(out_path, results.stats)


def train_head(gpu: torch.device, manager: TensorManager) -> search.SearchResults:
    cli.subchapter("Training")

    stats           : pd.DataFrame = pd.DataFrame()
    best_avg_val_f1 : float        = -1.0
    best_net        : nn.Module
    best_params     : TrainingHelper.Params

    for _ in trange(SAMPLES, desc="Sample"):
        params = TrainingHelper.Params(LR_INIT(), LR_DECAY(), WEIGHT_DECAY(), USE_CLASS_WEIGHTS())
        best_param_net, param_stats = train_with_params(gpu, manager, params)

        avg_param_val_f1 = param_stats["validation_f1"].mean()
        param_stats["average_validation_f1"] = avg_param_val_f1
        stats = pd.concat([stats, param_stats], ignore_index=True)

        if avg_param_val_f1 > best_avg_val_f1:
            best_avg_val_f1 = avg_param_val_f1
            best_net = best_param_net
            best_params = params

    return SearchResults(best_avg_val_f1, best_params, best_net, stats)


def train_with_params(gpu: torch.device, manager: TensorManager, params: TrainingHelper.Params) -> Tuple[nn.Module, pd.DataFrame]:
    param_stats : pd.DataFrame = pd.DataFrame()
    best_val    : float        = float("-inf")
    best_net    : nn.Module

    train_loader : Loader
    val_loader   : Loader
    test_loader  : Loader = manager.test_loader()

    for k in trange(manager.num_folds(), desc="Fold", leave=False):

        train_loader, val_loader = manager.leave_one_out(k)
        empirical_prob = manager.leave_one_out_prob(k)

        for repetition in trange(REPETITIONS, desc="Repetition", leave=False):

            net         = Classifier().to(gpu)
            meta        = TrainingHelper.Meta(EPOCHS, empirical_prob, k, repetition)
            head_helper = TrainingHelper(net, meta, params, gpu, cli)
            materials   = head_helper.get_materials()
            loaders     = TrainingHelper.Loaders(train_loader, val_loader, test_loader)

            hist = head_helper.train(materials, loaders, verbose=False, show_progress="no")

            # MyPy can be annoying.
            assert (hist.smoothed_metrics.val is not None) and (hist.best_val_epoch is not None)
            val_best_stats = hist.smoothed_metrics.val[hist.best_val_epoch].as_dict()

            rep_stats = head_helper.get_stats()
            param_stats = pd.concat([param_stats, rep_stats], ignore_index=True)

            best_rep_val : float = val_best_stats[materials.target_metric].item()
            if best_rep_val > best_val:
                best_val = best_rep_val
                best_net = hist.best_net

    return best_net, param_stats


# ================ KICKSTART ================ #


if __name__ == "__main__":
    main()

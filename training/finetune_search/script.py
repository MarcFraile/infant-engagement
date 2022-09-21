#!/bin/env -S python3 -u


# ================ IMPORTS ================ #


from dataclasses import dataclass, asdict
import sys, os
from pathlib import Path
from datetime import datetime
import copy
from typing import Tuple, Optional
import warnings

import torch
from torch import nn

import pandas as pd
from tqdm import trange

from local.cli.pretty_cli import PrettyCli
from local.cli.cli_helpers import RelabeledCliHelper, Environment
from local.hyperparam_sampling import Sampler, ConstantSampler, ExponentialSampler, CategoricalSampler, MixtureSampler
from local.network_models import Net
from local.training_helper import TrainingHelper
from local.training import Profiler, Loader
from local.loader import LimitedLoader
from local.video_dataset import VideoManager


# ================ SETTINGS ================ #


warnings.filterwarnings("ignore", category=UserWarning)


# ---- Hand-Chosen Parameters ---- #

# The parameters are defined in the dict RUN_PARAMS and injected using locals().update(...)
# We add the following lines so linters don't get sad.

# Script Params
TASK     : str # people | eggs | drums
VARIABLE : str
HEAD_RUN : str # ISO-8601 timestamp / folder name for the output of a successful head hyperparameter search. Used to pre-load the head.

EPOCHS              : int   # Number of epochs per training run.
SAMPLES             : int   # Number of hyperparameters combinations to test (one combination results in `num_folds` training runs).
SNIPPET_DURATION    : float # Duration (in seconds) of each video snippet sampled for training / testing.
SNIPPET_SUBDIVISION : int   # Number of snippets to sample from each session, per epoch.
REPETITIONS         : int   # Number of times each training run should be repeated (to account for random initialization factors).

BATCH_SIZE  : int           # Parameter taken by the loader. Number of samples per batch.
NUM_WORKERS : int           # Parameter taken by the loader. Number of parallel processes loading data.
MAX_BATCHES : Optional[int] # Debug helper. Ignored if `None`. Otherwise, limits the maximum number of batches per epoch to `MAX_BATCHES`.
PROFILE     : bool          # Debug helper. If `True`, profile the current run.
VERBOSE     : bool          # Should we output more stuff in the training loop?

# Search Space
LR_INIT           : Sampler[float]
LR_DECAY          : Sampler[float]
WEIGHT_DECAY      : Sampler[float]
USE_CLASS_WEIGHTS : Sampler[bool]

# RUN_PARAMS = { # "Test" params
#     "SCRIPT_PARAMS" : {
#         "TASK"     : "people",
#         "VARIABLE" : "participating",
#         "HEAD_RUN" : "20220326T124239",

#         "EPOCHS"              : 4,
#         "SAMPLES"             : 10,
#         "SNIPPET_DURATION"    : 5.0,
#         "SNIPPET_SUBDIVISION" : 10,

#         "BATCH_SIZE"  : 16,
#         "NUM_WORKERS" : 8,
#         "MAX_BATCHES" : None,
#         "PROFILE"     : False,
#         "VERBOSE"     : True,
#     },
#     "SEARCH_SPACE" : {
#         "LR_INIT"           : ExponentialSampler(base=10, min_exp=-7, max_exp=-1),
#         "LR_DECAY"          : MixtureSampler(samplers=[ConstantSampler(value=0.0), ExponentialSampler(base=10, min_exp=-4, max_exp=-1)], weights=[1, 4]),
#         "WEIGHT_DECAY"      : MixtureSampler(samplers=[ConstantSampler(value=0.0), ExponentialSampler(base=10, min_exp=-4, max_exp=-1)], weights=[1, 4]),
#         "USE_CLASS_WEIGHTS" : CategoricalSampler(categories=[True, False]),
#     },
# }

RUN_PARAMS = { # "Prod" params
    "SCRIPT_PARAMS" : {
        "TASK"     : "people",
        "VARIABLE" : "participating",
        "HEAD_RUN" : "20220326T124239",

        "EPOCHS"              : 100,
        "SAMPLES"             : 150,
        "SNIPPET_DURATION"    : 5.0,
        "SNIPPET_SUBDIVISION" : 10,
        "REPETITIONS"         : 5,

        "BATCH_SIZE"  : 16,
        "NUM_WORKERS" : 8,
        "MAX_BATCHES" : None,
        "PROFILE"     : False,
        "VERBOSE"     : False,
    },
    "SEARCH_SPACE" : {
        "LR_INIT"           : ExponentialSampler(base=10, min_exp=-8, max_exp=-2),
        "LR_DECAY"          : MixtureSampler(samplers=[ConstantSampler(value=0.0), ExponentialSampler(base=10, min_exp=-4, max_exp=0)], weights=[1, 5]),
        "WEIGHT_DECAY"      : MixtureSampler(samplers=[ConstantSampler(value=0.0), ExponentialSampler(base=10, min_exp=-4, max_exp=0)], weights=[1, 5]),
        "USE_CLASS_WEIGHTS" : CategoricalSampler(categories=[True, False]),
    },
}

locals().update(RUN_PARAMS["SCRIPT_PARAMS"]) # Make params into variables.
locals().update(RUN_PARAMS["SEARCH_SPACE"])

# ---- Filesystem ---- #

# Same deal as above.

CWD             : Path = Path(os.getcwd()).resolve() # Book-keeping.
CURRENT_SCRIPT  : Path # Book-keeping.

VIDEO_ROOT      : Path # Load vids from this directory.
HEAD_ROOT       : Path # Output folder from a successful head training run.
ANNOTATION_FILE : Path # Load annotations from this file.
STATS_FILE      : Path # Load fold sample statistics from this file.

INPUT_DIRS = {
    "CWD"        : CWD,
    "VIDEO_ROOT" : Path("video/relabeled/"),
    "HEAD_ROOT"  : Path(f"output/relabeled/head/{HEAD_RUN}"),
}

INPUT_FILES = {
    "CURRENT_SCRIPT"  : Path(__file__).resolve().relative_to(CWD),
    "ANNOTATION_FILE" : Path("annotation/elan/folds.csv"),
    "STATS_FILE"      : Path("annotation/elan/stats.json")
}

locals().update(INPUT_DIRS)
locals().update(INPUT_FILES)

DEFAULT_OUTPUT_ROOT = Path("output/relabeled/finetune/")

# ---- Helpers ---- #

cli = PrettyCli()
helper = RelabeledCliHelper(cli)


# ================ METHODS ================ #


def main(output_root: Optional[Path] = None, profiler: Optional[Profiler] = None):
    cli.main_title(F'FINE-TUNING: "{TASK}"\nHYPERPARAMETER SEARCH')

    add_timestamp : bool
    if output_root:
        add_timestamp = False
    else:
        add_timestamp = True
        output_root = DEFAULT_OUTPUT_ROOT

    env = helper.report_environment()
    helper.report_input_sources(INPUT_DIRS, INPUT_FILES)
    gpu = helper.report_gpu()
    start_time, out_path = helper.setup_output_dir(output_root, RUN_PARAMS, add_timestamp)

    manager = helper.report_video_manager(VIDEO_ROOT, ANNOTATION_FILE, STATS_FILE, TASK, VARIABLE, SNIPPET_DURATION, SNIPPET_SUBDIVISION, BATCH_SIZE, NUM_WORKERS, VERBOSE)
    head = helper.load_pickled_net(HEAD_ROOT / "best_net.pt")
    # head_params = report_head_params() # TODO: Check if needed

    results = search_params(gpu, head, manager, env, profiler)

    report_results(out_path, start_time, results)
    torch.save(results.best_net, out_path / "best_net.pt")


@dataclass
class SearchResults:
    best_avg_val_f1 : float
    best_params     : TrainingHelper.Params
    best_net        : nn.Module
    stats           : pd.DataFrame


def search_params(gpu: torch.device, head: nn.Module, manager: VideoManager, env: Environment, profiler: Optional[Profiler]) -> SearchResults:
    """
    * For every combination of hyperparameters, run k-folds validation.
    * The k-folds validation step train_with_params() returns the best net for that set of params, based on (individual fold) validation F1 score.
    * Each returned net is evaluated based on the average validation F1 score. The best one is returned as the winner.
    """
    cli.subchapter("TRAINING")

    stats           : pd.DataFrame = pd.DataFrame()
    best_avg_val_f1 : float        = -1.0
    best_net        : nn.Module
    best_params     : TrainingHelper.Params

    attempts : int
    if env.slurm:
        # TODO: We are assuming that task_id is sequential, and 0 <= task_id < num_tasks. None of these are guaranteed by SLURM.
        #       We can ensure a full run uses IDs in range 0..num_tasks, but partial runs (recovery from partial failure) would not comply.
        #       Another option is to pass the theoretical max ID separately to the script.
        #       I need to figure out how to coordinate that with the values passed to SLURM.

        # Do approximately (SAMPLES / slurm.num_tasks) runs, distributing the remainder.
        attempts  = SAMPLES // env.slurm.num_tasks
        remainder = SAMPLES %  env.slurm.num_tasks
        if env.slurm.task_id < remainder:
            attempts += 1
    else:
        attempts = SAMPLES

    for sample in trange(attempts, desc="Hyper-Param"):
        params = TrainingHelper.Params(LR_INIT(), LR_DECAY(), WEIGHT_DECAY(), USE_CLASS_WEIGHTS())
        best_param_net, param_stats = train_with_params(gpu, head, manager, params, profiler)

        avg_param_val_f1 = param_stats["validation_f1"].mean()
        stats = pd.concat([stats, param_stats])

        if avg_param_val_f1 > best_avg_val_f1:
            best_avg_val_f1 = avg_param_val_f1
            best_net = best_param_net
            best_params = params

    return SearchResults(best_avg_val_f1, best_params, best_net, stats)


def train_with_params(gpu: torch.device, head: nn.Module, manager: VideoManager, params: TrainingHelper.Params, profiler: Optional[Profiler]) -> Tuple[nn.Module, pd.DataFrame]:

    param_stats : pd.DataFrame = pd.DataFrame()
    best_val    : float        = -1.0
    best_net    : nn.Module

    train_loader : Loader
    val_loader   : Loader
    test_loader  : Loader = manager.test_loader()

    if MAX_BATCHES is not None:
        test_loader = LimitedLoader(test_loader, MAX_BATCHES)

    for k in trange(manager.num_folds(), desc="Fold", leave=False):
        train_loader, val_loader = manager.leave_one_out(k)
        empirical_prob = manager.leave_one_out_prob(k)

        if MAX_BATCHES is not None:
            train_loader = LimitedLoader(train_loader, MAX_BATCHES)
            val_loader   = LimitedLoader(val_loader  , MAX_BATCHES)

        for repetition in trange(REPETITIONS, desc="Repetition", leave=False):

            net : nn.Module = Net()
            net.classifier = copy.deepcopy(head)
            net = net.to(gpu)

            meta         = TrainingHelper.Meta(EPOCHS, empirical_prob, k, repetition)
            train_helper = TrainingHelper(net, meta, params, gpu, cli)
            materials    = train_helper.get_materials()
            loaders      = TrainingHelper.Loaders(train_loader, val_loader, test_loader)

            show_progress = "temporary" if VERBOSE else "no"
            hist = train_helper.train(materials, loaders, profiler=profiler, verbose=False, show_progress=show_progress)

            # MyPy can be annoying.
            assert (hist.smoothed_metrics.val is not None) and (hist.best_val_epoch is not None)
            val_best_stats = hist.smoothed_metrics.val[hist.best_val_epoch].as_dict()

            rep_stats = train_helper.get_stats()
            param_stats = pd.concat([param_stats, rep_stats], ignore_index=True)

            best_rep_val : float = val_best_stats[materials.target_metric].item()
            if best_rep_val > best_val:
                best_val = best_rep_val
                best_net = hist.best_net

    return best_net, param_stats


def report_results(out_path: Path, start_time: datetime, search_results: SearchResults) -> None:
    """
    Report durations and basic training statistics.
    """

    stats = search_results.stats

    end_time = datetime.now()
    elapsed = end_time - start_time

    stats.to_csv(out_path / "stats.csv", index=False)

    summary = {
        "General": {
            "Started"      : start_time,
            "Ended"        : end_time,
            "Total_Time"   : str(elapsed),
            "Time_per_Rep" : stats["duration"].describe().to_dict(),
            "F1" : {
                "Train"      : stats["train_f1"     ].describe().to_dict(),
                "Validation" : stats["validation_f1"].describe().to_dict(),
                "Test"       : stats["test_f1"      ].describe().to_dict(),
            },
        },
        "Best_Avg_Val_F1" : search_results.best_avg_val_f1,
        "Best_Params"     : asdict(search_results.best_params),
    }

    cli.section("Final Results")
    cli.print(summary)
    helper.json_write(summary, out_path / "results.json")


# ================ KICKSTART ================ #


def kickstart():
    out_root : Optional[Path] = None
    if len(sys.argv) > 1:
        out_root = Path(sys.argv[1])

    if PROFILE == True:
        with torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=2, warmup=2, active=1, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler("./runs"),
            profile_memory=True,
            with_stack=True,
        ) as prof:
            main(out_root, prof)
    else:
        main(out_root)


if __name__ == "__main__":
    kickstart()

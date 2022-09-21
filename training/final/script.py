#!/bin/env -S python3 -u


# ================ IMPORTS ================ #


import os
from pathlib import Path
from datetime import datetime
import copy
from typing import Optional, Tuple, Dict, Any

import torch
import pandas as pd
from matplotlib import pyplot as plt

from local.cli.pretty_cli import PrettyCli
from local.cli.cli_helpers import Environment, RelabeledCliHelper
from local.training import FoldMetrics, FoldPack, RunHistory, plot_history_metrics
from local.network_models import Classifier, Net
from local.training_helper import TrainingHelper


# ================ SETTINGS ================ #

# ---- Hand-Chosen Parameters ---- #

# The parameters are defined in the dict RUN_PARAMS and injected using locals().update(...)
# We add the following lines so linters don't get sad.

# Script Params
TASK     : str
VARIABLE : str

# Head training
HEAD_EPOCHS           : int
HEAD_SMOOTHING_WINDOW : int
HEAD_WARMUP_WINDOW    : int
HEAD_VERBOSE          : bool

HEAD_LR_INIT           : float
HEAD_LR_DECAY          : float
HEAD_WEIGHT_DECAY      : float
HEAD_USE_CLASS_WEIGHTS : bool

# Fine-tuning
FINETUNE_EPOCHS              : int
FINETUNE_BATCH_SIZE          : int
FINETUNE_NUM_WORKERS         : int
FINETUNE_SMOOTHING_WINDOW    : int
FINETUNE_WARMUP_WINDOW       : int
FINETUNE_SNIPPET_DURATION    : float
FINETUNE_SNIPPET_SUBDIVISION : int
FINETUNE_VERBOSE             : bool

FINETUNE_LR_INIT           : float
FINETUNE_LR_DECAY          : float
FINETUNE_WEIGHT_DECAY      : float
FINETUNE_USE_CLASS_WEIGHTS : bool

RUN_PARAMS : Dict[str, Dict[str, Any]] = {
    "GENERAL": {
        "TASK"     : "people", # people | eggs | drums
        "VARIABLE" : "attending", # participating | attending
    },
    "HEAD": {
        "EPOCHS"           : 1_500,
        "SMOOTHING_WINDOW" : 8,
        "WARMUP_WINDOW"    : 8,
        "VERBOSE"          : True,

        "LR_INIT"           : 0.01,
        "LR_DECAY"          : 0.005,
        "WEIGHT_DECAY"      : 0.0011629532351435,
        "USE_CLASS_WEIGHTS" : True,
    },
    "FINETUNE": {
        "EPOCHS"              : 5_000,
        "BATCH_SIZE"          : 64,
        "NUM_WORKERS"         : 8,
        "SMOOTHING_WINDOW"    : 8,
        "WARMUP_WINDOW"       : 0,
        "SNIPPET_DURATION"    : 5.0,
        "SNIPPET_SUBDIVISION" : 20,
        "VERBOSE"             : True,

        "LR_INIT"           : 1e-06,
        "LR_DECAY"          : 0.001,
        "WEIGHT_DECAY"      : 0.0024318658150018,
        "USE_CLASS_WEIGHTS" : False,
    },
}

# Make params into variables.
locals().update(RUN_PARAMS["GENERAL"])
# Prefix HEAD and FINETUNE entries with the corresponding name before making into vars.
locals().update({ "HEAD_" + key: value for (key, value) in RUN_PARAMS["HEAD"].items() })
locals().update({ "FINETUNE_" + key: value for (key, value) in RUN_PARAMS["FINETUNE"].items() })

# ---- Filesystem ---- #

# Same deal as above.

CWD             : Path = Path(os.getcwd()).resolve() # Book-keeping.
CURRENT_SCRIPT  : Path # Book-keeping.

VIDEO_ROOT      : Path # Load vids from this directory.
FEATURE_ROOT    : Path # Load pre-computed encoded samples from this directory.
ANNOTATION_FILE : Path # Load annotations from this file.
STATS_FILE      : Path # Load fold sample statistics from this file.

INPUT_DIRS = {
    "CWD"          : CWD,
    "VIDEO_ROOT"   : Path("video/relabeled/"),
    "FEATURE_ROOT" : Path("preprocessed/relabeled/"),
}

INPUT_FILES = {
    "CURRENT_SCRIPT"  : Path(__file__).resolve().relative_to(CWD),
    "ANNOTATION_FILE" : Path("annotation/elan/folds.csv"),
    "STATS_FILE"      : Path("annotation/elan/stats.json")
}

locals().update(INPUT_DIRS)
locals().update(INPUT_FILES)

OUTPUT_ROOT = Path("output/relabeled/main")

# ---- Helpers ---- #

cli = PrettyCli()
helper = RelabeledCliHelper(cli)


# ================ FUNCTIONS ================ #


def main():
    cli.main_title(f"FULL TRAINING: \"{TASK}\"\nHEAD TRAINING + FINETUNE")

    env = helper.report_environment()
    helper.report_input_sources(INPUT_DIRS, INPUT_FILES)
    gpu = helper.report_gpu()
    start_time, out_path = helper.setup_output_dir(OUTPUT_ROOT, RUN_PARAMS)

    head_hist, head_stats = train_head(gpu, env)
    net_hist , net_stats  = finetune(head_hist.best_net, gpu, env)

    report_params(start_time, out_path, head_hist, net_hist, head_stats, net_stats)
    torch.save(net_hist.best_net, out_path / "net.pt")


def train_head(gpu: torch.device, env: Environment) -> Tuple[RunHistory, pd.DataFrame]:
    cli.subchapter("Head Training")

    manager = helper.report_tensor_manager(FEATURE_ROOT, TASK, VARIABLE, gpu)
    last_fold = manager.num_folds() - 1
    train_loader, val_loader = manager.leave_one_out(last_fold)
    empirical_prob = manager.leave_one_out_prob(last_fold)
    test_loader = manager.test_loader()

    loaders = TrainingHelper.Loaders(train_loader, val_loader, test_loader)
    params  = TrainingHelper.Params(HEAD_LR_INIT, HEAD_LR_DECAY, HEAD_WEIGHT_DECAY, HEAD_USE_CLASS_WEIGHTS)

    head         = Classifier().to(gpu)
    meta         = TrainingHelper.Meta(HEAD_EPOCHS, empirical_prob)
    train_helper = TrainingHelper(head, meta, params, gpu, cli)
    materials    = train_helper.get_materials()

    cli.section("Train Head")

    show_progress = "temporary" if (env.slurm is None) and (HEAD_VERBOSE is True) else "no"
    hist          = train_helper.train(materials, loaders, verbose=False, show_progress=show_progress, smoothing_window=HEAD_SMOOTHING_WINDOW, warmup_window=HEAD_WARMUP_WINDOW)
    stats         = train_helper.get_stats()

    cli.print({ key: stats[key].item() for key in stats })

    return hist, stats


def finetune(head: Classifier, gpu: torch.device, env: Environment) -> Tuple[RunHistory, pd.DataFrame]:
    cli.subchapter("Fine-Tuning")

    net = Net()
    net.classifier = copy.deepcopy(head)
    net = net.to(gpu)

    manager = helper.report_video_manager(
        VIDEO_ROOT, ANNOTATION_FILE, STATS_FILE,
        TASK, VARIABLE, FINETUNE_SNIPPET_DURATION, FINETUNE_SNIPPET_SUBDIVISION,
        FINETUNE_BATCH_SIZE, FINETUNE_NUM_WORKERS, verbose=True
    )
    last_fold = manager.num_folds() - 1
    train_loader, val_loader = manager.leave_one_out(last_fold)
    empirical_prob = manager.leave_one_out_prob(last_fold)
    test_loader = manager.test_loader()

    meta         = TrainingHelper.Meta(FINETUNE_EPOCHS, empirical_prob)
    params       = TrainingHelper.Params(FINETUNE_LR_INIT, FINETUNE_LR_DECAY, FINETUNE_WEIGHT_DECAY, FINETUNE_USE_CLASS_WEIGHTS)
    train_helper = TrainingHelper(net, meta, params, gpu, cli)
    materials    = train_helper.get_materials()
    loaders      = TrainingHelper.Loaders(train_loader, val_loader, test_loader)

    cli.section("Fine-Tune Network")

    show_progress = "temporary" if (env.slurm is None) and (FINETUNE_VERBOSE is True) else "no"
    hist          = train_helper.train(materials, loaders, verbose=False, show_progress=show_progress, smoothing_window=FINETUNE_SMOOTHING_WINDOW, warmup_window=FINETUNE_WARMUP_WINDOW)
    stats         = train_helper.get_stats()

    cli.print({ key: stats[key].item() for key in stats })

    return hist, stats


def dict_helper(fold: FoldMetrics, idx: int) -> Dict[str, float]:
    return { key: value.item() for (key, value) in fold[idx].as_dict().items() }


def fold_dicts(pack: FoldPack, idx: Optional[int]) -> Dict[str, Dict[str, float]]:
    output = {}

    if idx is None:
        idx = -1

    if pack.train is not None:
        output["Train"] = dict_helper(pack.train, idx)
    if pack.val is not None:
        output["Val"] = dict_helper(pack.val, idx)
    if pack.test is not None:
        output["Test"] = dict_helper(pack.test, idx)

    return output


def get_hist_report(hist: RunHistory) -> dict:
    assert (hist.original_metrics.train is not None) and (hist.original_metrics.val is not None) and (hist.original_metrics.test is not None)
    assert (hist.smoothed_metrics.train is not None) and (hist.smoothed_metrics.val is not None) and (hist.smoothed_metrics.test is not None)

    time_per_rep = pd.Series(hist.durations) \
        .describe()[["mean", "std", "min", "max"]] \
        .map(lambda dt: str(dt.to_pytimedelta())) \
        .to_dict()

    return {
        "Total_Time"               : str(hist.total_duration),
        "Time_per_Rep"             : time_per_rep,
        "Initial_Metrics"          : fold_dicts(hist.original_metrics, 0),
        "Final_Metrics (Original)" : fold_dicts(hist.original_metrics, hist.best_val_epoch),
        "Final_Metrics (Smoothed)" : fold_dicts(hist.smoothed_metrics, hist.best_val_epoch),
    }


def save_figs(fig_path: Path, hist: RunHistory, prefix: str) -> None:
    fig_path.mkdir(parents=False, exist_ok=True)
    for (name, fig) in plot_history_metrics(hist).items():
        fig.savefig(fig_path / f"{prefix}_{name}")
        plt.close(fig)


def report_params(start_time: datetime, out_path: Path, head_hist: RunHistory, net_hist: RunHistory, head_stats: pd.DataFrame, net_stats: pd.DataFrame) -> None:
    end_time = datetime.now()
    elapsed = end_time - start_time

    head_stats.to_csv(out_path / "head_stats.csv", index=False)
    net_stats .to_csv(out_path / "net_stats.csv" , index=False)

    assert net_hist.original_metrics.test is not None
    assert net_hist.smoothed_metrics.test is not None

    summary = {
        "General" : {
            "Started"                : start_time,
            "Ended"                  : end_time,
            "Total_Time"             : str(elapsed),
            "Final_Test_F1"          : net_hist.original_metrics.test[-1].metrics["f1"],
            "Final_Test_F1_Smoothed" : net_hist.smoothed_metrics.test[-1].metrics["f1"],
        },
        "Head"     : get_hist_report(head_hist),
        "Finetune" : get_hist_report(net_hist),
    }

    cli.section("Final Results")
    cli.print(summary)
    helper.json_write(summary, out_path / "results.json")

    fig_path = out_path / "fig"
    save_figs(fig_path, head_hist, "head")
    save_figs(fig_path, net_hist , "finetune")


# ================ KICKSTART ================ #


if __name__ == "__main__":
    main()

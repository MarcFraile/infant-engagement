#!/bin/env -S python3 -u


# ==================================================================================================== #
# =============================== HUMAN-MACHINE RELIABILITY CALCULATOR =============================== #
# ==================================================================================================== #
#
# * Calculates pairwise reliability statistics (Cohen's Kappa, raw agreement) between each rater pair,
#   including the human annotators and the network.
# * Does so by sampling the test set "back to back": taking as many samples without overlap per video
#   as possible.
# * The pairs then need to be filtered to remove the videos that each human annotator didn't cover.
# * Finally, the statistics are computed (1) globally, (2) per task, (3) per variable, and (4) per
#   (task, variable) pair.
# * Both the pair data, the human-human average scores, and the human-machine average scores are saved
#   to the `artifacts/` directory.
# * Seaborn is used to generate some nice graphics for the pair data.
#
# ==================================================================================================== #
# ==================================================================================================== #


# ================ IMPORTS ================ #


import shutil
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List

import pandas as pd

import torch
from torch import nn
from torchvision import transforms

from local.cli import PrettyCli, CliHelper
from local.transforms import AssertDims, AdjustLength, Contiguous
from local.datasets import VideoDataset
from local.reliability import calculate_reliability


# ================ SETTINGS ================ #


# ---- Constants ---- #

TASKS      : List[str] = [ "people", "eggs", "drums" ]
VARIABLES  : List[str] = [ "attending", "participating" ]
ANNOTATORS : List[str] = [ "ew", "mf", "myz" ]

# ---- Net Dirs ---- #

NET_DIRS : Dict[str, Dict[str, Path]] = {
    "people": {
        "attending"     : Path("artifacts/final/20221006T102353"),
        # "participating" : Path("artifacts/final/20220405T211630"),
    },
    # "eggs"  : {
    #     "attending"     : Path("artifacts/final/20220405T222620"),
    #     "participating" : Path("artifacts/final/20220407T110032"),
    # },
    # "drums" : {
    #     "attending"     : Path("artifacts/final/20220406T155349"),
    #     "participating" : Path("artifacts/final/20220407T105856"),
    # },
}

FLATTENED_NET_DIRS = { f"NET_ROOT_{task.upper()}_{variable.upper()}": path for (task, inner) in NET_DIRS.items() for (variable, path) in inner.items() }

# ---- Filesystem ---- #

CWD             : Path = Path(os.getcwd()).resolve() # Book-keeping.
CURRENT_SCRIPT  : Path # Book-keeping.

VIDEO_ROOT      : Path # Load vids from this directory.
ANNOTATION_FILE : Path # Load annotations from this file.
STATS_FILE      : Path # Load fold sample statistics from this file.
SNIPPETS_FILE   : Path # Load the choice of snippets to process from here.

INPUT_DIRS = {
    "CWD"          : CWD,
    "VIDEO_ROOT"   : Path("data/processed/video/"),
    **FLATTENED_NET_DIRS
}

INPUT_FILES = {
    "CURRENT_SCRIPT"  : Path(__file__).resolve().relative_to(CWD),
    "ANNOTATION_FILE" : Path("data/processed/engagement/stratified_annotation_spans.csv"),
    "STATS_FILE"      : Path("data/processed/fold_statistics.json"),
    "SNIPPETS_FILE"   : Path("data/processed/human_attention/candidate_snippets.csv"),
}

locals().update(INPUT_DIRS)
locals().update(INPUT_FILES)

OUTPUT_ROOT = Path("artifacts/human_machine_reliability")

# ---- Script Param Reporting ---- #

SCRIPT_PARAMS = {
    "NET_DIRS"     : NET_DIRS,
}

# ---- CLI pretty-printing and script helper functions ---- #

cli = PrettyCli()
helper = CliHelper(cli)


# ================ FUNCTIONS ================ #


def main() -> None:
    """
    Main function.

    * Calculates pairwise reliability statistics (Cohen's Kappa, raw agreement) between each rater pair, including the human annotators and the network.
    * Does so by sampling the test set "back to back": taking as many samples without overlap per video as possible.
    * The pairs then need to be filtered to remove the videos that each human annotator didn't cover.
    * Finally, the statistics are computed (1) globally, (2) per task, (3) per variable, and (4) per (task, variable) pair.
    * Both the pair data, the human-human average scores, and the human-machine average scores are saved to the `artifacts/` directory.
    * Seaborn is used to generate some nice graphics for the pair data.
    """

    cli.main_title("Human vs. Machine: Reliability Comparison")

    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)

    env = helper.report_environment()
    helper.report_input_sources(INPUT_DIRS, INPUT_FILES)
    gpu = helper.report_gpu()

    labels = calculate_samples(gpu)
    calculate_reliability(labels, OUTPUT_ROOT, TASKS, VARIABLES)


def calculate_samples(gpu: torch.device) -> pd.DataFrame:
    """
    * For each (task, variable) pair:
        * Loads the corresponding network and test dataset.
        * Queries the dataset for all the relevant human labels.
        * Computes the network label.
    * Compiles the data into a DataFrame, indexed by (task, variable, session, start_ms).
    * Prints DataFrame summary, saves it to disk, and returns it.
    """

    samples = pd.DataFrame()

    for task in TASKS:
        for variable in VARIABLES:
            cli.chapter(f"Task: {task}; Variable: {variable}")

            net_dir = NET_DIRS[task][variable]
            net = helper.load_pickled_net(net_dir / "net.pt", gpu)
            net_params = report_net_params(net_dir, task, variable)
            dataset = load_dataset(task, variable, net_params)

            partial = calculate_sample_subset(dataset, net, gpu, net_params)
            partial["task"    ] = task
            partial["variable"] = variable

            samples = pd.concat([samples, partial], ignore_index=True)

    samples = samples.set_index(["task", "variable", "session", "start_ms"]).sort_index()

    cli.print(samples)
    cli.small_divisor()
    cli.print(samples.info())

    samples.to_csv(OUTPUT_ROOT / "labels.csv")
    return samples


@dataclass
class NetParams:
    task                : str
    variable            : str
    snippet_duration    : float
    snippet_subdivision : int


def report_net_params(net_dir: Path, expected_task: str, expected_variable: str) -> NetParams:
    """
    Load basic information about a classifier training run, and report it.
    """

    cli.section("Net Params")

    param_object = helper.json_read(net_dir / "script_params.json")
    cli.print(param_object)

    task                = param_object["GENERAL" ]["TASK"]
    variable            = param_object["GENERAL" ]["VARIABLE"]
    snippet_duration    = param_object["FINETUNE"]["SNIPPET_DURATION"]
    snippet_subdivision = param_object["FINETUNE"]["SNIPPET_SUBDIVISION"]

    assert type(task)                == str   , f"Invalid type for 'task'. Expected 'str', found '{type(task)}'."
    assert type(variable)            == str   , f"Invalid type for 'variable'. Expected 'str', found '{type(variable)}'."
    assert type(snippet_duration)    == float , f"Invalid type for 'snippet_duration'. Expected 'float', found '{type(snippet_duration)}'."
    assert type(snippet_subdivision) == int   , f"Invalid type for 'snippet_subdivision'. Expected 'int', found '{type(snippet_subdivision)}'."

    assert task in TASKS         , f"Invalid value for 'task'. Found '{task}', expected one of {TASKS}."
    assert variable in VARIABLES , f"Invalid value for 'variable'. Found '{variable}', expected one of {VARIABLES}."

    assert task == expected_task          , f"Invalid value for 'task'. Found '{task}', expected '{expected_task}'."
    assert variable == expected_variable  , f"Invalid value for 'variable'. Found '{variable}', expected {expected_variable}."

    return NetParams(task, variable, snippet_duration, snippet_subdivision)


def load_dataset(task: str, variable: str, net_params: NetParams) -> VideoDataset:
    """
    Load the "relabeled" test set, with no test-time augmentation.
    """

    test_transform = transforms.Compose([
        AssertDims(height=160),
        AdjustLength(keep=(60, 60)),
        transforms.CenterCrop(160),
        Contiguous(),
    ])
    manager = helper.report_video_manager(
        VIDEO_ROOT, ANNOTATION_FILE, STATS_FILE,
        task, variable, net_params.snippet_duration, net_params.snippet_subdivision,
        test_transform=test_transform,
    )
    dataset = manager.test_set()

    return dataset


def calculate_sample_subset(dataset: VideoDataset, net: nn.Module, gpu: torch.device, net_params: NetParams) -> pd.DataFrame:
    """
    * For each session in the dataset:
        * Takes contiguous samples (i.e. the stride is the same as the sample length).
            * Calculates all the available annotator labels for that sample.
            * Calculates the network's prediction for that sample.
    * Accumulates all the data in one DataFrame and returns it.
    """

    data = pd.DataFrame(columns=["session", "start_ms", "ew", "mf", "myz", "machine"])
    test_sessions = [ sample.session for sample in dataset.samples ]

    for session in test_sessions:
        session_start_ms, session_end_ms = dataset.get_common_times(session)
        session_annotators = dataset.get_annotators(session)

        assert 0 <= session_start_ms < 1_000_000
        assert 0 <= session_end_ms   < 1_000_000
        assert session_start_ms < session_end_ms
        assert len(session_annotators) >= 2

        labels : Dict[str, List] = { annotator: [] for annotator in session_annotators }
        labels["machine"] = []
        labels["start_ms"] = []

        snippet_duration_ms = int(1_000 * net_params.snippet_duration)
        num_samples = int((session_end_ms - session_start_ms) / snippet_duration_ms)

        for i in range(num_samples):
            sample_start_ms = session_start_ms + i * snippet_duration_ms
            machine_label = None
            labels["start_ms"].append(sample_start_ms)
            for annotator in session_annotators:
                frames, annotator_label = dataset.get_sample(session, annotator, sample_start_ms)
                labels[annotator].append(int(annotator_label.item()))
                if machine_label is None:
                    with torch.no_grad():
                        machine_score = net(frames.to(gpu).unsqueeze(dim=0))
                        machine_label = (machine_score > 0).float()
                    labels["machine"].append(int(machine_label.item()))

        partial = pd.DataFrame(labels)
        partial["session"] = session
        data = pd.concat([data, partial])

    return data


# ================ KICKSTART ================ #


if __name__ == "__main__":
    main()

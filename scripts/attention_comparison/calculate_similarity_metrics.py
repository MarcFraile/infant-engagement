#!/bin/env -S python3 -u


# ==================================================================================================== #
# ========================= HUMAN VS. MACHINE HEATMAPS: SIMILARITY CALCULATOR ======================== #
# ==================================================================================================== #
#
# For each sample in the comparison set:
#   * Loads the hand-painted human attention map.
#   * Loads the pre-calculated machine attention maps.
#     * Methods: Backprop, guided backprop, Grad-CAM, guided Grad-CAM, occlusion.
#     * Loads both the *matching condition* (same target label as the consensus label) and the
#       *opposite condition* (opposite label as the consensus label).
#   * Resizes to a low-resolution, homogeneous size.
#     * Uses Gaussian downsampling for larger images.
#     * Uses bicubic upsampling for Grad-CAM.
#   * Calculates a selection of distribution similarity measures.
#     * Current choices: linear correlation, rank correlation, earth mover's distance, histogram intersection, Kullback-Leibler divergence.
#     * Measures calculated both (1) over the whole volume, and (2) averaged frame-by-frame.
#       * Exception: Earth mover's distance only calculated frame-by-frame because it takes an unkown long time to finish.
#
# ==================================================================================================== #
# ==================================================================================================== #


# ================ IMPORTS ================ #


import os
from pathlib import Path
from dataclasses import dataclass, asdict
import random
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

import torch
from torch import Tensor, nn
from torchvision import transforms

from local import img, util
from local.cli import PrettyCli, CliHelper
from local.transforms import AssertDims, AdjustLength, Contiguous
from local.datasets import VideoDataset
from local.heatmap import comparison


# ================ SETTINGS ================ #


# ---- Constants ---- #

TASKS     : List[str] = [ "people", "eggs", "drums" ]
VARIABLES : List[str] = [ "attending", "participating" ]

# ---- Parameters ---- #

# NOTE: Target is 32x32.
#       Full-size maps | high frequency detail | 160x160 | downsampled with stride 5.
#       Small maps     | low frequency detail  | 10x10   | upsampled using bicubic interpolation.

# NOTE: I have not found a clear answer online on what constitutes a good standard deviation for the Gaussian smoothing.
#       sigma = stride = 5 seems to work well for our use case.

# NOTE: Similarly, upsampling Grad-CAM and Occlusion is done with bicubic interpolation because it seemed to have the least directional artifacts from the available options.

TARGET_SIZE       : Tuple[int, int] = (32, 32)
DOWNSAMPLE_STRIDE : int             = 5
DOWNSAMPLE_SIGMA  : float           = 5.0

# ---- Net Dirs ---- #

NET_DIRS : Dict[str, Dict[str, Path]] = {
    "people": {
        "attending"     : Path("output/relabeled/main/20220405T212235"),
        "participating" : Path("output/relabeled/main/20220405T211630"),
    },
    "eggs"  : {
        "attending"     : Path("output/relabeled/main/20220405T222620"),
        "participating" : Path("output/relabeled/main/20220407T110032"),
    },
    "drums" : {
        "attending"     : Path("output/relabeled/main/20220406T155349"),
        "participating" : Path("output/relabeled/main/20220407T105856"),
    },
}

FLATTENED_NET_DIRS = { f"NET_ROOT_{task.upper()}_{variable.upper()}": path for (task, inner) in NET_DIRS.items() for (variable, path) in inner.items() }

# ---- Filesystem ---- #

CWD             : Path = Path(os.getcwd()).resolve() # Book-keeping.
CURRENT_SCRIPT  : Path # Book-keeping.

VIDEO_ROOT      : Path # Load vids from this directory.
ANNOTATION_FILE : Path # Load annotations from this file.
STATS_FILE      : Path # Load fold sample statistics from this file.
SNIPPETS_FILE   : Path # Load the choice of snippets to process from here.

MACHINE_ROOT : Path # Load machine heatmaps from folders in this root.
HUMAN_ROOT   : Path # Load human heatmaps from folders in this root.

INPUT_DIRS = {
    "CWD"          : CWD,
    "VIDEO_ROOT"   : Path("data/processed/video/"),
    "MACHINE_ROOT" : Path("artifacts/machine_attention/20221007T105429"),
    "HUMAN_ROOT"   : Path("data/processed/human_attention/"),
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

OUTPUT_ROOT = Path("artifacts/human_machine_similarity")

# ---- Script Param Reporting ---- #

SCRIPT_PARAMS = {
    "SIZE_CONFIG" : {
        "TARGET_SIZE"       : TARGET_SIZE,
        "DOWNSAMPLE_STRIDE" : DOWNSAMPLE_STRIDE,
        "DOWNSAMPLE_SIGMA"  : DOWNSAMPLE_SIGMA,
    },
    "INPUTS" : {
        "NET_DIRS"     : NET_DIRS,
        "MACHINE_ROOT" : MACHINE_ROOT,
    },
}

# ---- CLI pretty-printing and script helper functions ---- #

cli = PrettyCli()
helper = CliHelper(cli)


# ================ FUNCTIONS ================ #


def main() -> None:
    """
    Main function.

    * Loads all hand-painted human attention maps.
    * Loads all pre-calculated machine attention maps.
    * For each sample in the comparison set: calculates 5 frame-by-frame similarity measures, and 4 whole-video similarity measures.
    * Saves results as a CSV file.
    """

    cli.main_title("Human vs. Machine Attention")

    env = helper.report_environment()
    gpu = helper.report_gpu()
    helper.report_input_sources(INPUT_DIRS, INPUT_FILES)
    start_time, out_dir = helper.setup_output_dir(OUTPUT_ROOT, SCRIPT_PARAMS)
    snippets = helper.report_snippet_info(SNIPPETS_FILE)

    scores = human_machine_comparison(snippets, gpu)
    cli.print(scores)
    scores.to_csv(out_dir / "scores.csv")


def human_machine_comparison(snippets: pd.DataFrame, gpu: torch.device) -> pd.DataFrame:
    """
    Calculate attention map similarity metrics between the manually annotated human attention map and the pre-calculated machine attention maps.
    """

    scores = pd.DataFrame()

    for task in TASKS:
        for variable in VARIABLES:
            cli.chapter(f"Task: {task}; Variable: {variable}")

            net_dir = NET_DIRS[task][variable]
            net = helper.load_pickled_net(net_dir / "net.pt", gpu)
            net_params = report_net_params(net_dir)

            assert net_params.task == task
            assert net_params.variable == variable

            key = (task, variable)
            assert key in snippets.index
            relevant_snippets = snippets.loc[key]

            dataset = load_dataset(task, variable, net_params)
            samples = load_samples(variable, relevant_snippets, dataset, net, gpu)

            partial = compare_attention(samples, task, variable)
            scores = pd.concat([scores, partial], ignore_index=True)

    scores = scores.set_index(["task", "variable", "annotator", "session", "start_ms", "condition", "consensus_label", "annotator_label", "machine_prediction", "target_label", "method"]).sort_index()
    return scores


@dataclass
class NetParams:
    task                : str
    variable            : str
    snippet_duration    : float
    snippet_subdivision : int


def report_net_params(net_dir: Path) -> NetParams:
    """
    Load basic information about a classifier training run.
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

    return NetParams(task, variable, snippet_duration, snippet_subdivision)


def load_dataset(task: str, variable: str, net_params: NetParams) -> VideoDataset:
    """
    Load the test set, with no test-time augmentation.
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


@dataclass
class SampleMeta:
    session   : str
    annotator : str
    start_ms  : int


@dataclass
class Heatmaps:
    backprop        : Tensor
    guided_backprop : Tensor
    gradcam         : Tensor
    guided_gradcam  : Tensor
    occlusion       : Tensor


@dataclass
class Sample:
    meta               : SampleMeta
    video              : Tensor
    consensus_label    : Tensor
    annotator_label    : Tensor
    machine_prediction : Tensor
    human              : Tensor
    matching           : Heatmaps
    opposite           : Heatmaps


def load_samples(variable: str, snippets: pd.DataFrame, dataset: VideoDataset, net: nn.Module, gpu: torch.device) -> List[Sample]:
    """
    Loads metadata, video frames, consensus label, human attention, and both matching and opposite machine attention sets.

    * Expects `snippets` to already be filtered by task and variable, so only relevant samples are present.
    """

    cli.section("Load Samples")

    all_annotators = snippets["annotator"].unique().tolist()
    all_sessions   = snippets["session"  ].unique().tolist()

    machine_dirs = filter_machine_dirs()
    human_dirs = filter_human_dirs(variable, all_annotators, all_sessions)

    samples : List[Sample] = []

    for (_, entry) in snippets.iterrows():
        cli.small_divisor()

        session = entry["session"]
        snippet_annotator = entry["annotator"]
        start_ms = int(entry["start_ms"])

        assert type(session) == str
        assert type(snippet_annotator) == str

        present_annotators = [ annotator for annotator in all_annotators if annotator != snippet_annotator ]
        consensus_annotator = random.choice(present_annotators)

        cli.print({
            "Session"             : session,
            "Snippet Annotator"   : snippet_annotator,
            "Consensus Annotator" : consensus_annotator,
            "Start (ms)"          : start_ms,
        })

        video, consensus_label = dataset.get_sample(session, consensus_annotator, start_ms)

        with torch.no_grad():
            machine_score = net(video.unsqueeze(dim=0).to(gpu)).cpu().squeeze()
            machine_prediction = (machine_score > 0).float()

        machine_dir = machine_dirs[(session, start_ms)]
        matching, opposite = load_machine_attention(machine_dir)

        human_dir = human_dirs[(snippet_annotator, session, start_ms // 1000)]
        human_data = load_human_attention(human_dir)

        assert human_data.annotator       == snippet_annotator
        assert human_data.session         == session
        assert human_data.variable        == variable
        assert human_data.consensus_label == consensus_label

        human_map = human_data.map
        annotator_label = torch.tensor(human_data.annotator_label)

        cli.print({
            "Consensus Label"        : consensus_label.item(),
            "Annotator Label"        : annotator_label.item(),
            "Machine Prediction"     : machine_prediction.item(),
            "Video Shape"            : video.shape,
            "Guided Backprop Shape"  : matching.guided_backprop.shape,
            "Grad-CAM Shape"         : matching.gradcam.shape,
            "Human Annotation Shape" : human_map.shape,
        })

        meta = SampleMeta(session, snippet_annotator, start_ms)
        sample = Sample(meta, video, consensus_label, annotator_label, machine_prediction, human_map, matching, opposite)
        samples.append(sample)

    return samples


def filter_machine_dirs() -> Dict[Tuple[str, int], Path]:
    """
    Detects available MACHINE attention dirs and classifies them by `(session, timestamp_ms)`.

    * Notice MILLISECOND precision on timestamp.
    """

    heatmap_dirs = [ item for item in MACHINE_ROOT.iterdir() if item.is_dir() ]
    heatmap_dirs.sort()

    dir_map = dict()

    for d in heatmap_dirs:
        session, _timestamp = d.stem.split("_")
        assert _timestamp[-2:] == "ms"
        timestamp = int(_timestamp[:-2])
        dir_map[(session, timestamp)] = d

    return dir_map


def filter_human_dirs(variable: str, all_annotators: List[str], all_sessions: List[str]) -> Dict[Tuple[str, str, int], Path]:
    """
    Detects available HUMAN attention dirs and classifies them by `(annotator, session, timestamp_s)`.

    * Notice SECOND precision on timestamp.
    """

    output = dict()

    for d in HUMAN_ROOT.iterdir():
        if not d.is_dir():
            continue
        parts = d.stem.split("_")

        assert parts[0] == "snippet"
        assert len(parts) == 7

        annotator = parts[3]
        session = parts[4]
        seconds = int(parts[5][:-1])
        snippet_variable = parts[6]

        assert annotator in all_annotators
        assert session in all_sessions
        assert snippet_variable in VARIABLES

        if snippet_variable != variable:
            continue

        output[(annotator, session, seconds)] = d

    return output


def load_machine_attention(path: Path) -> Tuple[Heatmaps, Heatmaps]:
    """
    Loads the machine attention data from a single sample.
    """

    def _load_folder_heatmaps(path: Path) -> Heatmaps:
        backprop = torch.load(path / "gradient_saliency.pt").squeeze()
        backprop = img.gauss_downsample(backprop, DOWNSAMPLE_STRIDE, DOWNSAMPLE_SIGMA)
        backprop = util.remap(backprop)

        guided_backprop = torch.load(path / "guided_backprop_saliency.pt").squeeze()
        guided_backprop = img.gauss_downsample(guided_backprop, DOWNSAMPLE_STRIDE, DOWNSAMPLE_SIGMA)
        guided_backprop = util.remap(guided_backprop)

        gradcam = torch.load(path / "gradcam.pt").squeeze()
        gradcam = img.scale_spatial_dimensions(gradcam, TARGET_SIZE, mode="cubic")
        gradcam = util.remap(gradcam)

        guided_gradcam = torch.load(path / "guided_gradcam_saliency.pt").squeeze()
        guided_gradcam = img.gauss_downsample(guided_gradcam, DOWNSAMPLE_STRIDE, DOWNSAMPLE_SIGMA)
        guided_gradcam = util.remap(guided_gradcam)

        occlusion = torch.load(path / "occlusion.pt").squeeze()
        occlusion = img.scale_spatial_dimensions(occlusion, TARGET_SIZE, mode="cubic")
        occlusion = util.remap(occlusion)

        return Heatmaps(backprop, guided_backprop, gradcam, guided_gradcam, occlusion)

    matching : Optional[Heatmaps] = None
    opposite : Optional[Heatmaps] = None

    for d in path.iterdir():
        if not d.is_dir():
            continue
        if d.stem.startswith("matching"):
            matching = _load_folder_heatmaps(d)
        elif d.stem.startswith("opposite"):
            opposite = _load_folder_heatmaps(d)

    assert matching is not None
    assert opposite is not None

    return matching, opposite


@dataclass
class HumanAttention:
    map: Tensor
    session: str
    annotator: str
    variable: str
    consensus_label: int
    annotator_label: int


def load_human_attention(path: Path) -> HumanAttention:
    """
    Loads the human attention data from a single sample.
    """

    crop = transforms.CenterCrop(160)

    map = np.load(path / "heatmap.npy")
    map = torch.tensor(map)
    map = crop(map) # NOTE: Annotations done on whole-size video.
    map = img.gauss_downsample(map, DOWNSAMPLE_STRIDE, DOWNSAMPLE_SIGMA)
    map = util.remap(map)

    info = helper.json_read(path / "info.json")

    session   = info["sample"]["session"        ]
    annotator = info["sample"]["annotator"      ]
    variable  = info["sample"]["target_variable"]

    consensus_label = int(info["sample"]["consensus_label"])
    annotator_label = int(info["labels"][variable         ] != "no")

    return HumanAttention(map, session, annotator, variable, consensus_label, annotator_label)


def compare_attention(samples: List[Sample], task: str, variable: str) -> pd.DataFrame:
    """
    Calculate map similarity scores for each sample.
    """
    data = pd.DataFrame()

    for sample in samples:
        matching = compare_condition(sample.human, sample.matching, "matching", sample.consensus_label)
        opposite = compare_condition(sample.human, sample.opposite, "opposite", 1 - sample.consensus_label)

        partial = pd.concat([matching, opposite], ignore_index=True)

        partial["annotator"         ] = sample.meta.annotator
        partial["session"           ] = sample.meta.session
        partial["start_ms"          ] = sample.meta.start_ms
        partial["consensus_label"   ] = int(sample.consensus_label.item())
        partial["annotator_label"   ] = int(sample.annotator_label.item())
        partial["machine_prediction"] = int(sample.machine_prediction.item())

        data = pd.concat([data, partial], ignore_index=True)

    data["task"    ] = task
    data["variable"] = variable

    return data


def compare_condition(human: Tensor, machine: Heatmaps, condition: str, target_label: Tensor) -> pd.DataFrame:
    """
    Calculate the similarity scores for a given condition (part of a sample).
    """
    cli.subchapter(condition)
    data = pd.DataFrame()

    for name, map in asdict(machine).items():
        cli.section(name)
        entry = compare_maps(human, map)
        cli.print(entry)
        entry["method"] = name
        data = pd.concat([data, entry], ignore_index=True)

    data["condition"   ] = condition
    data["target_label"] = int(target_label.item())

    return data


def compare_maps(human: Tensor, machine: Tensor) -> pd.DataFrame:
    """
    Calculate linear correlation, rank correlation, earth mover's distance, \
    histogram intersection, and Kullback-Leibler divergence; both per-frame \
    and globally, for the given human and machine samples.

    * Samples must have the same shape (up to squeezing).
    """

    methods: Dict[str, comparison.Comparison] = {
        "linear_correlation"     : comparison.pearson_correlation   ,
        "rank_correlation"       : comparison.spearman_correlation  ,
        "earth_mover"            : comparison.earth_movers_distance ,
        "histogram_intersection" : comparison.histogram_intersection,
        "kullback_leibler"       : comparison.kullback_leibler      ,
    }

    scores: Dict[str, float] = {}
    for (name, method) in methods.items():
        if name != "earth_mover": # HACK: Earth Mover's Distance in 3D takes weeks (possibly more) in our available hardware.
            scores[f"{name}_global"] = method(human, machine)
        scores[f"{name}_frame" ] = comparison.mean_per_frame(method, human, machine)

    return pd.DataFrame({ key: [value] for (key, value) in scores.items() })


# ================ KICKSTART ================ #


if __name__ == "__main__":
    main()

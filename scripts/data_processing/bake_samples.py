#!/bin/env -S python3 -u


# ===================================================================================== #
# ==================================== SAMPLE BAKER =================================== #
# ===================================================================================== #
#
# Bakes train and test samples for all applicable folds, allowing much faster and less
# resource-hungry classifier head training.
#
# * For each train / validation fold (all but last), bakes train and test samples
#   (different augmentation).
# * For the test set (last fold), bakes test samples only.
# * Bakes a set number of epochs for each sample pack.
#
# ===================================================================================== #
# ===================================================================================== #


# ================ IMPORTS ================ #


import os
from pathlib import Path
import shutil
from dataclasses import dataclass
from typing import List

import pandas as pd
from tqdm import tqdm, trange

import torch
from torch import nn, Tensor

from local.cli import PrettyCli, CliHelper
from local.transforms import get_default_transforms
from local.network_models import Encoder
from local.types import Loader
from local.datasets import VideoManager


# ================ SETTINGS ================ #


EPOCHS          : int   = 100
SAMPLE_DURATION : float = 5.0
SUBDIVISION     : int   = 10

BATCH_SIZE  : int = 16
NUM_WORKERS : int = 8

CWD             : Path
VIDEO_ROOT      : Path
CURRENT_SCRIPT  : Path
ANNOTATION_FILE : Path
STATS_FILE      : Path

INPUT_DIRS = {
    "CWD"        : Path(os.getcwd()).resolve(),
    "VIDEO_ROOT" : Path("data/processed/video/"),
}

INPUT_FILES = {
    "CURRENT_SCRIPT"  : Path(__file__).resolve().relative_to(INPUT_DIRS["CWD"]),
    "ANNOTATION_FILE" : Path("data/processed/engagement/stratified_annotation_spans.csv"),
    "STATS_FILE"      : Path("data/processed/fold_statistics.json")
}

locals().update(INPUT_DIRS)
locals().update(INPUT_FILES)

OUTPUT_ROOT = Path("data/processed/baked_samples")

cli = PrettyCli()
helper = CliHelper(cli)


# ================ FUNCTIONS ================ #


def report_output() -> None:
    cli.section("Output")

    if OUTPUT_ROOT.is_dir():
        shutil.rmtree(OUTPUT_ROOT)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)

    cli.print(OUTPUT_ROOT)


@dataclass
class AnnotationData:
    annotations : pd.DataFrame
    sessions    : List[str]
    annotators  : List[str]
    tasks       : List[str]
    variables   : List[str]
    folds       : List[str]


def report_annotations() -> AnnotationData:
    cli.section("Annotations")

    annotations = pd.read_csv(ANNOTATION_FILE)

    sessions   = annotations["session"  ].sort_values().unique().tolist()
    annotators = annotations["annotator"].sort_values().unique().tolist()
    tasks      = annotations["task"     ].sort_values().unique().tolist()
    variables  = annotations["variable" ].sort_values().unique().tolist()
    folds      = annotations["fold"     ].sort_values().unique().tolist()

    cli.print(annotations)
    cli.small_divisor()
    cli.print({ "Sessions": sessions, "Annotators": annotators, "Tasks": tasks, "Folds": folds })
    return AnnotationData(annotations, sessions, annotators, tasks, variables, folds)


def report_encoder(gpu: torch.device) -> nn.Module:
    cli.section("Encoder")

    encoder = Encoder()
    encoder = encoder.to(gpu)
    encoder.eval()

    helper.report_net(encoder)

    return encoder


def bake_fold(loader: Loader, encoder: nn.Module, gpu: torch.device, out_path: Path, id: str) -> None:
    H : List[Tensor] = []
    Y : List[Tensor] = []

    for _epoch in trange(EPOCHS, desc="Epoch", leave=False):
        X : Tensor
        y : Tensor
        for (X, y) in tqdm(loader, total=len(loader), desc="Batch", leave=False):
            with torch.no_grad():
                X = X.to(gpu)
                h : Tensor = encoder(X)
                H.append(h.detach().cpu())
                Y.append(y.detach().cpu())

    H_tensor : Tensor = torch.cat(H, dim=0)
    Y_tensor : Tensor = torch.cat(Y, dim=0)

    torch.save(H_tensor, str(out_path / f"H_{id}.pt"))
    torch.save(Y_tensor, str(out_path / f"Y_{id}.pt"))


def bake_data() -> None:
    cli.main_title("Bake Relabeled Data")

    _env = helper.report_environment()
    helper.report_input_sources(INPUT_DIRS, INPUT_FILES)
    report_output()

    gpu = helper.report_gpu()
    encoder = report_encoder(gpu)
    data = report_annotations()
    train_transform, test_transform = get_default_transforms()

    cli.section("Baking Samples")
    for task in tqdm(data.tasks, desc="Task"):
        for variable in tqdm(data.variables, desc="Variable"):
            manager = VideoManager(
                VIDEO_ROOT, ANNOTATION_FILE, STATS_FILE,
                task, variable, SAMPLE_DURATION, SUBDIVISION,
                train_transform, test_transform,
                BATCH_SIZE, NUM_WORKERS, verbose=True,
            )
            for k in trange(manager.num_folds(), desc="Fold", leave=False):
                train_loader = manager.one_fold_loader(k, is_train=True)
                test_loader  = manager.one_fold_loader(k, is_train=False)

                bake_fold(train_loader, encoder, gpu, OUTPUT_ROOT, f"{task}_{variable}_train_{k}")
                bake_fold(test_loader , encoder, gpu, OUTPUT_ROOT, f"{task}_{variable}_test_{k}" )

            test_loader = manager.test_loader()
            bake_fold(test_loader, encoder, gpu, OUTPUT_ROOT, f"{task}_{variable}_test_{manager.num_folds()}")


# ================ KICKSTART ================ #


if __name__ == "__main__":
    bake_data()

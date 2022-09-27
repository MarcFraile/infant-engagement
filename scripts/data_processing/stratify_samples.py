#!/bin/env -S python3 -u


# ===================================================================================== #
# ================================= SAMPLE STRATIFIER ================================= #
# ===================================================================================== #
#
# Script in charge of partitioning the available sessions into well-balanced folds.
#
# 1. Finds the sessions that have both annotation data and available processed videos.
# 2. Uses rejection sampling to stratify the sessions into folds.
# 3. Calculates relevant statistics per-fold.
# 4. Saves updated annotations with (1) the fold and (2) a binarized target label.
#
# ===================================================================================== #
# ===================================================================================== #


# ================ IMPORTS ================ #


import json
from typing import Dict, List, Set, Tuple
from pathlib import Path
import random

import numpy as np
import pandas as pd
import imageio.v3 as iio
from tqdm import tqdm, trange

from local.cli import PrettyCli


# ================ SETTINGS ================ #


NUM_FOLDS       : int = 5
SPLIT_NUM_TRIES : int = 100_000

VARIABLES            : List[str] = [ "attending", "participating" ]
TASKS                : List[str] = [ "people", "eggs", "drums" ]
E2_EXPECTED_SESSIONS : Set[str]  = { "fp29", "fp30", "fp31", "fp38", "fp39", "fp40" } # Sessions done by experimenter 2.

VIDEO_DIR              = Path("data/processed/video")
ANNOTATION_INPUT_FILE  = Path("data/processed/engagement/filled_annotation_spans.csv")
ANNOTATION_OUTPUT_FILE = Path("data/processed/engagement/stratified_annotation_spans.csv")
STATS_OUTPUT_FILE      = Path("data/processed/fold_statistics.json")

cli = PrettyCli()


# ================ FUNCTIONS ================ #


def stratify_samples() -> None:
    """
    Main function.

    1. Loads `ANNOTATION_INPUT_FILE` and binarizes the labels.
    2. Deduces the common sessions between the annotation data and the videos in `VIDEO_DIR`.
    3. Uses rejection sampling to calculate a nicely stratified partition into `NUM_FOLDS`.
    4. Saves updated annotations to `ANNOTATION_OUTPUT_FILE`.
    5. Calculates fold statistics, including color average and std, and saves them to `STATS_OUTPUT_FILE`.
    """

    cli.main_title("STRATIFY SAMPLES")

    random.seed(2064781408)

    annotations = get_annotations()
    sessions = get_sessions(annotations)
    folds, fold_stats = split_sessions(sessions, annotations)
    save_updated_annotations(annotations, folds)
    get_and_save_color_stats(fold_stats)


def get_annotations() -> pd.DataFrame:
    """
    * Loads `ANNOTATION_INPUT_FILE`.
    * Binarizes labels: `value` (categorical) -> `engaged` (int).
    * Indexes by `(task, variable, session, annotator)`.
    * Prints header and relevant information.
    """

    cli.section("Get Annotations")
    cli.print({ "Annotation File" : ANNOTATION_INPUT_FILE })

    assert ANNOTATION_INPUT_FILE.is_file()
    annotations : pd.DataFrame = pd.read_csv(ANNOTATION_INPUT_FILE)

    # Binarize the labels.
    annotations["value"] = (annotations["value"] != "no").astype(int)
    annotations = annotations.rename(columns={"value":"engaged"})

    annotations = annotations.set_index([ "task", "variable", "session", "annotator" ]).sort_index()

    cli.print(annotations)
    return annotations


def get_empirical_prob(annotations: pd.DataFrame) -> float:
    """
    Returns the overall empirical probability for the positive binary class (child is engaged).

    * Calculated as (length of positive annotations) / (total length of annotations).
    """

    durations = (annotations["end_ms"] - annotations["start_ms"])
    engaged_durations = annotations["engaged"] * durations
    empirical_probability = engaged_durations.sum() / durations.sum()

    return empirical_probability


def get_sessions(annotations: pd.DataFrame) -> List[str]:
    """
    Returns the sessions that have both a video and an annotation.

    * Checks annotation sessions from provided dataframe.
    * Checks video sessions from processed video dir.
    * Asserts that all expected Experimenter 2 sessions are present (important for stratification).
    * Prints headers and relevant information.
    """

    cli.section("Get Sessions")

    cli.subchapter("Annotated Sessions")

    csv_sessions : List[str] = annotations.index.get_level_values("session").unique().tolist()
    csv_sessions.sort()

    cli.print(csv_sessions)

    cli.subchapter("Video Sessions")

    assert VIDEO_DIR.is_dir()

    dir_sessions : List[str] = [f.stem for f in VIDEO_DIR.iterdir() if f.suffix == ".mp4"]
    dir_sessions.sort()

    cli.print(dir_sessions)

    cli.subchapter("Joint Sessions")

    sessions : List[str] = [s for s in csv_sessions if s in dir_sessions]

    e2_observed_sessions = E2_EXPECTED_SESSIONS.intersection(sessions)
    assert len(e2_observed_sessions) == len(E2_EXPECTED_SESSIONS), f"Expected e2_observed_sessions (len {len(e2_observed_sessions)}) to be the same as E2_EXPECTED_SESSIONS (len {len(E2_EXPECTED_SESSIONS)}). Found: {e2_observed_sessions}"

    cli.print({
        "All Found Sessions"      : sessions,
        "Experimenter 2 Sessions" : E2_EXPECTED_SESSIONS,
    })

    return sessions


def split_sessions(sessions: List[str], annotations: pd.DataFrame) -> Tuple[List[List[str]], dict]:
    """
    Use rejection sampling to find a good stratification into `NUM_FOLDS`.

    * Enforces even distribution of Experimenter 2 sessions.
    * Tries to match empirical probabilities between each fold and the total set. Each `(task, variable)` pair is considered. MSE is used to penalize divergence.
    """

    cli.section("Session Split Search")
    cli.print({
        "Num Folds" : NUM_FOLDS,
        "Num Tries" : SPLIT_NUM_TRIES,
    })

    best_score : float = float("inf")
    best_split : List[List[str]]

    expected_probabilities  = { task: { variable: get_empirical_prob(annotations.loc[task, variable]) for variable in VARIABLES } for task in TASKS }
    expected_video_count : float = len(sessions) / NUM_FOLDS

    best_probabilities : List[Dict[str, Dict[str, float]]] # fold -> task -> var -> prob.
    best_video_counts  : List[int] # fold -> count

    for _ in trange(SPLIT_NUM_TRIES):
        # Start with a blank slate.
        score        : float           = 0.0
        folds        : List[List[str]] = [ [] for _ in range(NUM_FOLDS) ]
        video_counts : List[int]       = [ 0  for _ in range(NUM_FOLDS) ]
        probabilities : List[Dict[str, Dict[str, float]]] = [ { task: { variable: 0.0 for variable in VARIABLES } for task in TASKS } for _ in range(NUM_FOLDS) ]

        # First, randomly distribute the Experimenter 2 sessions between the folds, as evenly as possible.
        e2_sessions = list(E2_EXPECTED_SESSIONS)
        random.shuffle(e2_sessions)
        shuffled_folds = folds.copy() # SHALLOW copy.
        random.shuffle(shuffled_folds)
        for idx, session in enumerate(e2_sessions):
            shuffled_folds[idx % NUM_FOLDS].append(session)

        # Then, get all the sessions that don't have Experimenter 2, and distribute them among the folds.
        rest = list(set(sessions).difference(E2_EXPECTED_SESSIONS))
        random.shuffle(rest)

        for k in range(NUM_FOLDS):
            start_idx = ((k + 0) * len(rest)) // NUM_FOLDS
            end_idx   = ((k + 1) * len(rest)) // NUM_FOLDS

            folds[k] += rest[start_idx:end_idx]
            folds[k].sort()

            fold_index = annotations.index.get_level_values("session").isin(folds[k])
            fold_annotations = annotations[fold_index]

            fold_probabilities = { task: {variable: get_empirical_prob(fold_annotations.loc[task, variable]) for variable in VARIABLES } for task in TASKS }
            fold_video_count = len(fold_annotations.index.get_level_values("session").unique())

            probabilities[k] = fold_probabilities
            video_counts [k] = fold_video_count

            prob_score = 0.0
            for task in TASKS:
                for variable in VARIABLES:
                    observed = fold_probabilities[task][variable]
                    expected = expected_probabilities[task][variable]
                    prob_score += ((observed - expected) / expected) ** 2

            video_count_score = ((fold_video_count - expected_video_count) / expected_video_count) ** 2

            score += 2.0 * prob_score + 1.0 * video_count_score

            no_e2 = E2_EXPECTED_SESSIONS.isdisjoint(folds[k])
            if no_e2:
                score += 100

        if score < best_score:
            best_split = folds
            best_score = score
            best_probabilities = probabilities
            best_video_counts  = video_counts

    cli.section("Best Split Data")
    cli.print({
        "Expected Values": {
            "Empirical Probabilities" : expected_probabilities,
            "Video Count"             : expected_video_count,
        },
        "Results": {
            "Final Score"                 : best_score,
            "Empirical Probabilities"     : { k: best_probabilities[k] for k in range(len(best_probabilities)) },
            "Experimenter 2 Video Counts" : [ len(E2_EXPECTED_SESSIONS.intersection(fold)) for fold in best_split ],
            "Video Counts"                : best_video_counts,
        },
        "Final Folds": { k: best_split[k] for k in range(len(best_split)) },
    })

    fold_stats = {
        "num_folds" : NUM_FOLDS,
        "stats" : [
            { "empirical_probability": best_probabilities[k], "sessions": best_split[k] }
            for k in range(NUM_FOLDS)
        ]
    }

    return best_split, fold_stats


def get_and_save_color_stats(fold_stats: dict) -> None:
    """
    For each fold:
    1. Loads all videos.
    2. Calculates overall mean and standard deviation per color channel.
    3. Saves the data to `STATS_OUTPUT_FILE`.
    """

    cli.section("Color Stats")

    for stats in tqdm(fold_stats["stats"], desc="Fold"):
        sessions = stats["sessions"]
        videos = [ VIDEO_DIR / f"{session}.mp4" for session in sessions ]
        all_frames = np.concatenate([ iio.imread(vid) for vid in videos ], axis=0)
        mean = all_frames.mean(axis=(0, 1, 2)).tolist()
        std  = all_frames.std (axis=(0, 1, 2)).tolist()
        pixel_stats = { "mean": mean, "std": std }
        stats["pixel_values"] = pixel_stats
        cli.print(pixel_stats)

    with open(STATS_OUTPUT_FILE, "w") as file:
        json.dump(fold_stats, file, indent=4)


def save_updated_annotations(annotations: pd.DataFrame, folds: List[List[str]]) -> None:
    """
    Augment the annotation entries with their fold number, and save to `ANNOTATION_OUTPUT_FILE`.
    """

    cli.section("Save Updated Annotations")

    assert len(folds) == NUM_FOLDS

    for k in range(NUM_FOLDS):
        key = annotations.index.get_level_values("session").isin(folds[k])
        annotations.loc[key, "fold"] = k

    annotations["fold"] = annotations["fold"].astype(int)
    annotations = annotations[["fold", "engaged", "start_ms", "end_ms"]]
    annotations = annotations.reorder_levels(["session", "annotator", "task", "variable"])

    cli.print(annotations)
    annotations.to_csv(ANNOTATION_OUTPUT_FILE)


# ================ KICKSTART ================ #


if __name__ == "__main__":
    stratify_samples()

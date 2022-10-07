#!/bin/env -S python3 -u


# ===================================================================================== #
# ============================ METRIC BASELINE CALCULATOR ============================= #
# ===================================================================================== #
#
# Uses `ANNOTATION_INPUT_FILE` to calculate the empirical probability for the positive
# class, separated by `(task, variable)` pair, and calculated as the total length of
# positive annotations over the total length of annotations. Saves the results as JSON
# in `METRICS_OUTPUT_FILE`.
#
# ===================================================================================== #
# ===================================================================================== #


# ================ IMPORTS ================ #


import json
from typing import Dict, List
from pathlib import Path

import pandas as pd

from local.cli import PrettyCli
from local import baseline


# ================ SETTINGS ================ #


TASKS     : List[str] = [ "people", "eggs", "drums" ]
VARIABLES : List[str] = [ "attending", "participating" ]

ANNOTATION_INPUT_FILE = Path("data/processed/engagement/stratified_annotation_spans.csv")
METRICS_OUTPUT_FILE   = Path("data/processed/metric_baselines.json")

cli = PrettyCli()


# ================ FUNCTIONS ================ #


def calculate_metric_baselines() -> None:
    """
    Main function.

    Uses `ANNOTATION_INPUT_FILE` to calculate the empirical probability for the positive class, \
    separated by `(task, variable)` pair, and calculated as the total length of positive \
    annotations over the total length of annotations. Saves the results as JSON in `METRICS_OUTPUT_FILE`.
    """

    cli.main_title("CALCULATE METRIC BASELINES")

    annotations = get_annotations()
    metrics = get_metrics(annotations)

    with open(METRICS_OUTPUT_FILE, "w") as file:
        json.dump(metrics, file, indent=4)


def get_annotations() -> pd.DataFrame:
    """
    * Loads `ANNOTATION_INPUT_FILE`.
    * Indexes by `(task, variable)`.
    * Prints header and relevant information.
    """
    cli.section("Get Annotations")
    cli.print({ "Annotation File" : ANNOTATION_INPUT_FILE })

    assert ANNOTATION_INPUT_FILE.is_file()
    annotations : pd.DataFrame = pd.read_csv(ANNOTATION_INPUT_FILE)

    annotations = annotations.set_index([ "task", "variable" ]).sort_index()

    cli.print(annotations)
    return annotations


def get_metrics(annotations: pd.DataFrame) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Returns a mapping (task -> variable -> metric -> value), representing:

    * Empirical measures (e.g., empirical probability for the positive class)
    * The best theoretical value that a random classifier could achieve (e.g., accuracy baseline).
    """
    cli.section("Get Metrics")

    metrics = {}

    for task in TASKS:
        metrics[task] = {}
        for variable in VARIABLES:
            metrics[task][variable] = metrics_case(annotations, task, variable)

    cli.print(metrics)
    return metrics


def metrics_case(annotations: pd.DataFrame, task: str, variable: str) -> Dict[str, float]:
    """
    Returns a mapping (metric -> value) for a single (task, variable) pair, representing:

    * Positive class empirical probability.
    * Best random accuracy.
    * Best random F1 score.
    """

    case = {}

    annotations = annotations.loc[(task, variable)]
    empirical_prob = get_empirical_prob(annotations)
    case["positive_class_probability"] = empirical_prob
    case["accuracy_baseline"] = baseline.binary_accuracy(empirical_prob)
    case["f1_baseline"] = baseline.binary_f1(empirical_prob)

    return case


def get_empirical_prob(annotations: pd.DataFrame) -> float:
    """
    Returns the overall empirical probability for the positive binary class (child is engaged).

    * Calculated as (length of positive annotations) / (total length of annotations).
    """

    durations = (annotations["end_ms"] - annotations["start_ms"])
    engaged_durations = annotations["engaged"] * durations
    empirical_probability = engaged_durations.sum() / durations.sum()

    return empirical_probability


# ================ KICKSTART ================ #


if __name__ == "__main__":
    calculate_metric_baselines()

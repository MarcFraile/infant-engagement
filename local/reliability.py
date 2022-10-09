#!/bin/env -S python3 -u


# ================ IMPORTS ================ #


import itertools
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats import inter_rater

from local import util


# ================ FUNCTIONS ================ #


def calculate_reliability(labels: pd.DataFrame, output_root: Path, tasks: List[str], variables: List[str], prefix: str = "") -> None:
    """
    * Calculates reliability scores globally, per task, per variable, and per (task, variable) pair.
    * Scores calculated: Cohen's Kappa, raw agreement.
    * `labels` expected to have a row per observation, and a column per rater.
    * `labels` expected to have index (task, variable, ...).
    * To compare with the network, add it as a rater called `'machine'`.
    """

    if (len(prefix) > 0) and (prefix[-1] != "_"):
        prefix += "_"

    # ---------------- [1] Global Scores ---------------- #

    global_scores, global_averages = calculate_reliability_case(labels)
    plot_reliability(global_scores, output_root, "global", prefix)

    global_scores  .to_csv(output_root / f"{prefix}pairwise_reliability_global.csv")
    global_averages.to_csv(output_root / f"{prefix}average_reliability_global.csv")

    # ---------------- [2] Per-Task Scores ---------------- #

    task_scores   = pd.DataFrame()
    task_averages = pd.DataFrame()

    for task in tasks:
        task_labels = labels.loc[task]
        task_score_partial, task_average_partial = calculate_reliability_case(task_labels)
        plot_reliability(task_score_partial, output_root, f"task_{task}", prefix)

        task_score_partial["task"] = task
        task_score_partial = task_score_partial.reset_index().set_index("task").sort_index()
        task_scores = pd.concat([task_scores, task_score_partial])

        task_average_partial["task"] = task
        task_average_partial = task_average_partial.reset_index().set_index("task").sort_index()
        task_averages = pd.concat([task_averages, task_average_partial])


    task_scores  .to_csv(output_root / f"{prefix}pairwise_reliability_task.csv")
    task_averages.to_csv(output_root / f"{prefix}average_reliability_task.csv")

    # ---------------- [3] Per-Variable Scores ---------------- #

    variable_scores   = pd.DataFrame()
    variable_averages = pd.DataFrame()

    for variable in variables:
        variable_labels = labels.loc[pd.IndexSlice[:, variable, :, :]]
        variable_score_partial, variable_average_partial = calculate_reliability_case(variable_labels)
        plot_reliability(variable_score_partial, output_root, f"variable_{variable}", prefix)

        variable_score_partial["variable"] = variable
        variable_score_partial = variable_score_partial.reset_index().set_index("variable").sort_index()
        variable_scores = pd.concat([variable_scores, variable_score_partial])

        variable_average_partial["variable"] = variable
        variable_average_partial = variable_average_partial.reset_index().set_index("variable").sort_index()
        variable_averages = pd.concat([variable_averages, variable_average_partial])

    variable_scores  .to_csv(output_root / f"{prefix}pairwise_reliability_variable.csv")
    variable_averages.to_csv(output_root / f"{prefix}average_reliability_variable.csv")

    # ---------------- [4] Combo Scores (task, variable) ---------------- #

    combo_scores   = pd.DataFrame()
    combo_averages = pd.DataFrame()

    for task in tasks:
        for variable in variables:
            combo_labels = labels.loc[task, variable]
            combo_score_partial, combo_average_partial = calculate_reliability_case(combo_labels)
            plot_reliability(combo_score_partial, output_root, f"combo_{task}_{variable}", prefix)

            combo_score_partial["task"    ] = task
            combo_score_partial["variable"] = variable
            combo_score_partial = combo_score_partial.reset_index().set_index(["task", "variable"]).sort_index()
            combo_scores = pd.concat([combo_scores, combo_score_partial])

            combo_average_partial["task"    ] = task
            combo_average_partial["variable"] = variable
            combo_average_partial = combo_average_partial.reset_index().set_index(["task", "variable"]).sort_index()
            combo_averages = pd.concat([combo_averages, combo_average_partial])

    combo_scores  .to_csv(output_root / f"{prefix}pairwise_reliability_combo.csv")
    combo_averages.to_csv(output_root / f"{prefix}average_reliability_combo.csv")


def calculate_reliability_case(labels: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Handles one case in calculate_reliability().

    * Calculates Cohen's Kappa and the raw agreement for each rater pair.
    * Calculates the human-human average and the human-machine average.
    * Returns `(pair_data, average_data)`.
    * Expects `labels` to have a row for each observation, and a column for each annotator.
    * For each rater pair, rows with missing data will be skipped in the calculation.
    """

    pair_dict: Dict[str, List] = { "rater_1": [], "rater_2": [], "cohen_kappa": [], "raw_agreement": [], }
    annotators = labels.columns

    for i in range(len(annotators) - 1):
        for j in range(i + 1, len(annotators)):

            first = annotators[i]
            second = annotators[j]

            index = labels[first].notnull() & labels[second].notnull()
            selected_labels = labels[index][[first, second]]

            frequency_table, _ = inter_rater.to_table(selected_labels, bins=2)
            raw_agreement      = frequency_table.diagonal().sum() / frequency_table.sum()
            cohen_kappa        = inter_rater.cohens_kappa(frequency_table, return_results=False)

            pair_dict["rater_1"      ].append(first)
            pair_dict["rater_2"      ].append(second)
            pair_dict["cohen_kappa"  ].append(cohen_kappa)
            pair_dict["raw_agreement"].append(raw_agreement)


    pairs = pd.DataFrame(pair_dict)

    human_mean   = pairs[(pairs.rater_1 != "machine") & (pairs.rater_2 != "machine")].mean(numeric_only=True)
    machine_mean = pairs[(pairs.rater_1 == "machine") | (pairs.rater_2 == "machine")].mean(numeric_only=True)

    pairs = pairs.set_index(["rater_1", "rater_2"]).sort_index()

    averages = pd.concat([human_mean, machine_mean], axis=1)
    averages.columns = ["human", "machine"]
    averages = averages.T

    return pairs, averages


def plot_reliability(scores: pd.DataFrame, output_root: Path, suffix: str, prefix: str) -> None:
    """
    Plot Cohen's Kappa and the raw agreement (empirical agreement probability) for each rater pair.

    * Expects `scores` to have index `(rater_1, rater_2)` and one column per agreement metric (`cohen_kappa`, `raw_agreement`).
    * Expands each metric into a symmetric 2D table with shape `(raters, raters)`.
    * Plots two versions of each metric: one with the original initials, and one anonymized.
    * When anonymizing, "network" or "machine" (case insensitive) are mapped to "network"; anything else is mapped to a sequential letter ("A", "B", "C"...)
    """

    raters = list(dict.fromkeys(itertools.chain(scores.index.get_level_values(0), scores.index.get_level_values(1))))

    anonymized_raters = []
    current_letter = "A"
    for rater in raters:
        if rater.strip().lower() in ["machine", "network"]:
            anonymized_raters.append("network")
        else:
            anonymized_raters.append(current_letter)
            current_letter = util.next_letter(current_letter)

    agreement_matrix = np.empty((len(raters), len(raters)))
    kappa_matrix     = np.empty((len(raters), len(raters)))

    for i1, r1 in enumerate(raters):
        for i2, r2 in enumerate(raters):
            kappa     : float = 1.0
            agreement : float = 1.0

            if i1 != i2:
                if (r1, r2) in scores.index:
                    kappa     = scores.loc[(r1, r2), "cohen_kappa"  ]
                    agreement = scores.loc[(r1, r2), "raw_agreement"]
                else: # Flip order to (r2, r1)
                    kappa     = scores.loc[(r2, r1), "cohen_kappa"  ]
                    agreement = scores.loc[(r2, r1), "raw_agreement"]

            kappa_matrix    [i1, i2] = kappa
            agreement_matrix[i1, i2] = agreement

    kappa_df     = pd.DataFrame(kappa_matrix    , index=raters, columns=raters)
    agreement_df = pd.DataFrame(agreement_matrix, index=raters, columns=raters)

    def _plot(df: pd.DataFrame, filename: str) -> None:
        sns.set(font_scale=1.5)
        fig = plt.figure()
        sns.heatmap(df, vmin=0, vmax=1, annot=True)
        plt.yticks(rotation=0)
        fig.savefig(output_root / f"{prefix}{filename}_{suffix}.png", bbox_inches="tight")
        plt.close(fig)

    def _anonymize(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out.columns = anonymized_raters
        out.index   = anonymized_raters
        return out

    _plot(kappa_df    , "cohen_kappa_comparison"  )
    _plot(agreement_df, "raw_agreement_comparison")

    _plot(_anonymize(kappa_df    ), "anonymized_cohen_kappa_comparison"  )
    _plot(_anonymize(agreement_df), "anonymized_raw_agreement_comparison")

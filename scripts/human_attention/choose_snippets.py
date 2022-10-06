#!/bin/env -S python3 -u


# ===================================================================================== #
# ================================== SNIPPET CHOOSER ================================== #
# ===================================================================================== #
#
# Chooses snippets for human attention painting.
#
# 1. For each annotator, choose a test session that they did not annotate.
# 2. For each (annotator, task, variable) triple, try to extract `EXPECTED_SAMPLES`
#    agreed positive samples (all other annotators labeled it as 1), and the same
#    number of agreed negative samples (all other annotators labeled it as 0) from the
#    chosen session.
#
# * All extracted samples lie in the *common span* annotated by all available
#   annotators.
# * All extracted samples are guaranteed to be non-overlapping.
# * The target `EXPECTED_SAMPLES` is not guaranteed to be reached, since it depends on
#   the available annotations.
#
# ===================================================================================== #
# ===================================================================================== #


# ================ IMPORTS ================ #


from pathlib import Path
import random
from typing import Dict, List, Optional

import pandas as pd
from torch import Tensor

from local.cli import PrettyCli
from local.transforms import get_default_transforms
from local.datasets import VideoManager


# ================ SETTINGS ================ #


VIDEO_ROOT      = Path("data/processed/video/")
ANNOTATION_FILE = Path("data/processed/engagement/stratified_annotation_spans.csv")
STATS_FILE      = Path("data/processed/fold_statistics.json")

OUTPUT_ROOT = Path("data/processed/human_attention/")
OUTPUT_FILE = OUTPUT_ROOT / "candidate_snippets.csv"

SAMPLE_DURATION_S = 5.0
SUBDIVISION       = 10

TRIALS           = 10_000
EXPECTED_SAMPLES = 1

cli = PrettyCli()


# ================ FUNCTIONS ================ #


def choose_snippets() -> None:
    """
    Main function.

    1. For each annotator, choose a test session that they did not annotate.
    2. For each (annotator, task, variable) triple, try to extract `EXPECTED_SAMPLES` agreed \
       positive samples (all other annotators labeled it as 1), and the same number of agreed \
       negative samples (all other annotators labeled it as 0) from the chosen session.

    * All extracted samples lie in the *common span* annotated by all available annotators.
    * All extracted samples are guaranteed to be non-overlapping.
    * The target `EXPECTED_SAMPLES` is not guaranteed to be reached, since it depends on the available annotations.
    """

    cli.main_title("Choose Snippets")
    annotator_session_pairs = get_annotator_session_pairs()
    candidate_snippets = pairs_to_snippets(annotator_session_pairs)

    if not OUTPUT_ROOT.is_dir():
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
    candidate_snippets.to_csv(OUTPUT_FILE, index=False)


def get_annotator_session_pairs() -> Dict[str, str]:
    """
    Returns a mapping from annotator to session, indicating which session the snippets should be extracted from.

    1. Extract available annotators and sessions from the test set.
    2. For each available annotator, choose a session that they did not annotate.
    """
    cli.chapter("Get Annotator-Session Pairs")

    test_data = pd.read_csv(ANNOTATION_FILE, index_col="fold").loc[4]

    test_sessions = test_data["session"].unique()
    test_annotators = test_data["annotator"].unique()
    random.shuffle(test_sessions)
    random.shuffle(test_annotators)
    cli.print({
        "Test Sessions": test_sessions,
        "Test Annotators" : test_annotators,
    })

    all_pairs: pd.DataFrame = test_data[["session", "annotator"]].drop_duplicates().set_index("session")

    chosen_pairs: Dict[str, str] = {}
    for annotator in test_annotators:
        for session in test_sessions:
            a_in_s = all_pairs.loc[session, "annotator"].eq(annotator).any()
            if not a_in_s:
                chosen_pairs[annotator] = session
                break
        assert annotator in chosen_pairs # None skipped

    cli.print(chosen_pairs)
    return chosen_pairs



def pairs_to_snippets(annotator_session_pairs: Dict[str, str]) -> pd.DataFrame:
    """
    Chooses the specific samples that should be painted for attention by each annotator.

    * Returns a `DataFrame` containing (annotator, session, task, variable, sample start (ms), label).
    * All samples for the same annotator come from the session in `annotation_session_pairs`, and are non-overlapping.
    * All samples come from the common span of time covered by all task annotations.
    * Only chooses "agreement samples": if the sample is for A, then it is a sample that was originally labelled for engagement by B and C. We only choose samples such that label(B) == label(C).
    * For each (annotator, task, variable) triple: if possible, returns a total of `EXPECTED_SAMPLES` positive samples (label == 1), and the same number of negative samples (label == 0).
    """
    cli.chapter("Get Snippets From Pairs")

    train_transform, test_transform = get_default_transforms()
    results = pd.DataFrame()

    for (annotator, session) in annotator_session_pairs.items():

        other_annotators : List[str  ] = [ other for other in annotator_session_pairs.keys() if other != annotator ]
        all_samples      : List[float] = []

        assert len(other_annotators) == 2

        for task in ["people", "eggs", "drums"]:
            for variable in ["attending", "participating"]:

                manager = VideoManager(VIDEO_ROOT, ANNOTATION_FILE, STATS_FILE, task, variable, SAMPLE_DURATION_S, SUBDIVISION, train_transform, test_transform)
                dataset = manager.test_set()

                task_start_ms, task_end_ms = dataset.get_common_times(session)

                task_duration_ms = task_end_ms - task_start_ms
                adjusted_duration = task_duration_ms - (SAMPLE_DURATION_S * 1000)

                positive_samples : List[float] = []
                negative_samples : List[float] = []

                for trial in range(TRIALS):
                    sample_start : int = int(task_start_ms + random.uniform(0, adjusted_duration))
                    label : Optional[Tensor] = None
                    should_skip : bool = False

                    for other in other_annotators:
                        _, l = dataset.get_sample(session, other, sample_start)
                        if label is None:
                            label = l
                        elif l != label:
                            should_skip = True
                            break

                    if should_skip:
                        continue

                    if overlaps(all_samples, sample_start):
                        continue
                    all_samples.append(sample_start)

                    if label == 0:
                        negative_samples.append(sample_start)
                    elif label == 1:
                        positive_samples.append(sample_start)
                    else:
                        assert False, f"Invalid label value: {label}"

                    if (len(positive_samples) >= EXPECTED_SAMPLES) and (len(negative_samples) >= EXPECTED_SAMPLES):
                        break

                positive_results: pd.DataFrame
                if len(positive_samples) < EXPECTED_SAMPLES:
                    cli.print(f"[ {session} | {annotator} | {task} | {variable} ] Too few positive samples collected: {len(positive_samples)} (expected at least {EXPECTED_SAMPLES}).")
                    positive_results = pd.DataFrame()
                else:
                    random.shuffle(positive_samples)
                    positive_samples = positive_samples[:EXPECTED_SAMPLES]
                    positive_results = pd.DataFrame({ "start_ms": positive_samples })
                    positive_results["label"] = 1

                negative_results: pd.DataFrame
                if len(negative_samples) < EXPECTED_SAMPLES:
                    cli.print(f"[ {session} | {annotator} | {task} | {variable} ] Too few negative samples collected: {len(negative_samples)} (expected at least {EXPECTED_SAMPLES}).")
                    negative_results = pd.DataFrame()
                else:
                    random.shuffle(negative_samples)
                    negative_samples = negative_samples[:EXPECTED_SAMPLES]
                    negative_results = pd.DataFrame({ "start_ms": negative_samples })
                    negative_results["label"] = 0

                partial = pd.concat([positive_results, negative_results])
                partial["session"] = session
                partial["annotator"] = annotator
                partial["task"] = task
                partial["variable"] = variable

                results = pd.concat([results, partial])


    results = results.set_index(["session", "start_ms"]).sort_index().reset_index()

    cli.section("Results")
    cli.print(results)
    return results


def overlaps(samples_ms: List[float], candidate_ms: float) -> bool:
    """Return true if any sample in `samples_ms` overlaps `candidate_ms` (both represented as the start time; length given by `SAMPLE_DURATION_S`)."""
    for sample in samples_ms:
        if (sample <= candidate_ms < sample + 1000 * SAMPLE_DURATION_S) or (candidate_ms <= sample < candidate_ms + 1000 * SAMPLE_DURATION_S):
            return True
    return False


# ================ KICKSTART ================ #


if __name__ == "__main__":
    choose_snippets()

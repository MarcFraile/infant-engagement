#!/bin/env -S python3 -u


# ==================================================================================================== #
# ================================ HUMAN-HUMAN RELIABILITY CALCULATOR ================================ #
# ==================================================================================================== #
#
# Calculates reliability scores for the continuous-form annotations over the whole dataset, by
# sampling every 100ms (1/10th of a second). Saves CSV reports and PNG representations to
# artifacts/human_human_reliability
#
# ==================================================================================================== #
# ==================================================================================================== #


# ================ IMPORTS ================ #


from pathlib import Path
import shutil
from typing import Dict, List

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype

from local.cli import PrettyCli
from local.reliability import calculate_reliability


# ================ SETTINGS ================ #


def cat(*args: str) -> CategoricalDtype:
    """Convert `args` to an ordered `CategoricalDType` variable."""
    return CategoricalDtype(args, ordered=True)

EXPECTED_ANNOTATORS = cat("ew", "mf", "myz") # Lexic.

FIRST_PILOT  = cat("fp19", "fp28", "fp29", "fp35")                 # 1st pilot  -> lexicographical order.
SECOND_PILOT = cat("fp32")                                         # 2nd pilot  -> lexicographical order.
BATCH_EW     = cat("fp15", "fp27", "fp20", "fp14", "fp21", "fp16") # EW  => MYZ -> order in Coding Guide (randomized).
BATCH_MF     = cat("fp33", "fp24", "fp18", "fp39", "fp25", "fp40") # MF  => EW  -> order in Coding Guide (randomized).
BATCH_MYZ    = cat("fp38", "fp30", "fp31", "fp13", "fp17", "fp34") # MYZ => MF  -> order in Coding Guide (randomized).

ANNOTATOR_TO_SESSION = {
    "ew"  : cat(*FIRST_PILOT.categories, *SECOND_PILOT.categories, *BATCH_EW .categories, *BATCH_MF .categories),
    "mf"  : cat(*FIRST_PILOT.categories, *SECOND_PILOT.categories, *BATCH_MF .categories, *BATCH_MYZ.categories),
    "myz" : cat(*FIRST_PILOT.categories, *SECOND_PILOT.categories, *BATCH_MYZ.categories, *BATCH_EW .categories),
}

for annotator in EXPECTED_ANNOTATORS.categories:
    assert annotator in ANNOTATOR_TO_SESSION

ALL_EXPECTED_SESSIONS = cat(
    *FIRST_PILOT .categories,
    *SECOND_PILOT.categories,
    *BATCH_EW    .categories,
    *BATCH_MF    .categories,
    *BATCH_MYZ   .categories,
)

SESSION_TO_ANNOTATOR : Dict[str, CategoricalDtype] = {}
for session in ALL_EXPECTED_SESSIONS.categories:
    annotators = []

    for annotator in EXPECTED_ANNOTATORS.categories:
        if session in ANNOTATOR_TO_SESSION[annotator].categories:
            annotators.append(annotator)

    assert len(annotators) >= 2
    SESSION_TO_ANNOTATOR[session] = cat(*annotators)

ANNOTATION_VARIABLES = cat("attending", "participating")
REQUIRED_VARIABLES   = cat("attending", "participating", "task") # Lexic. <=> annotation vars. first.
OPTIONAL_VARIABLES   = cat("ignore")
ALLOWED_VARIABLES    = cat(*REQUIRED_VARIABLES.categories, *OPTIONAL_VARIABLES.categories) # required -> optional.

EXPECTED_VALUES = {
    "participating" : cat("no", "self", "joint"),
    "attending"     : cat("no", "attending", "excited"),
    "task"          : cat("people", "eggs", "drums"),
}
ALL_EXPECTED_VALUES = cat(*list(dict.fromkeys(value for variable in EXPECTED_VALUES.values() for value in variable.categories)))

for key in EXPECTED_VALUES:
    assert key in ALLOWED_VARIABLES.categories

for KEY in REQUIRED_VARIABLES.categories:
    assert KEY in EXPECTED_VALUES


for var in ANNOTATION_VARIABLES.categories:
    assert var in REQUIRED_VARIABLES.categories

OUT_ROOT = Path("artifacts/human_human_reliability/")
ORIGINAL_ANNOTATIONS = Path("data/processed/engagement/original_annotation_spans.csv")

# ---- Helpers ---- #

cli = PrettyCli()


# ================ FUNCTIONS ================ #


def main():
    """
    * Validates correctness and completeness of extracted ELAN annotations.
    * Unrolls interval-based annotations into frame-based annotations.
        * Takes samples every 0.1s, only in the common intervals that have annotations for all available annotators in a session.
    * Calculates inter-rater reliability for the unrolled data.
    """

    cli.main_title("ELAN EXPLORATION")

    if OUT_ROOT.is_dir():
        shutil.rmtree(OUT_ROOT)
    OUT_ROOT.mkdir(exist_ok=False, parents=True)

    data = load_data()
    validate_data(data)

    task_timespans = get_task_spans(data)
    unrolled = unroll_annotations(data, task_timespans)
    calculate_all_reliabilities(unrolled)


def load_data() -> pd.DataFrame:
    """
    Read `CSV_FILE` from disk, print an excerpt, and return it as a `DataFrame`.
    """

    data = pd.read_csv(ORIGINAL_ANNOTATIONS)

    data["session"  ] = data["session"  ].astype(ALL_EXPECTED_SESSIONS)
    data["annotator"] = data["annotator"].astype(EXPECTED_ANNOTATORS)
    data["variable" ] = data["variable" ].astype(ALLOWED_VARIABLES)
    data["value"    ] = data["value"    ].astype(ALL_EXPECTED_VALUES)

    # Drop optional variables ("ignore") and re-encode.
    required_idx = data["variable"].isin(REQUIRED_VARIABLES.categories)
    data = data[required_idx]
    data["variable"] = data["variable"].astype(REQUIRED_VARIABLES)
    data.reset_index()

    cli.section("Raw Data")
    cli.print(data)
    cli.small_divisor()
    cli.print(data.info())

    return data


def validate_data(data: pd.DataFrame) -> None:
    """
    Performs assertions on the data to ensure its correctness.

    * Ensure all expected sessions, annotators and variables are present.
    * Ensure all `(variable, value)` pairs are legal.
    * Ensure all `(session, annotator, variable)` combinations are present.
    * Ensure `task` has exactly three entries for each (session, annotator) pair: `[people, eggs, drums]` (in this order).
    """

    cli.section("Validation")

    # ---- [1] assert session, annotator and variable each have ONLY correct values, and ALL correct values are present ---- #

    # NOTE: When casting data to a CategoricalDtype, values not in the category will be cast to NaN.
    #       Series.isnull() returns a boolean series. I don't know the exact rules of what counts as "null", but NaN does.

    # NOTE: Series.value_counts() uses the CategoricalDtype entries as the index, including missing categories (they will have a count of 0).
    #       Therefore, if value_counts() has any zeroes, there are missing categories.

    assert not data["session"].isnull().any(), "Invalid session values found."
    assert data["session"].value_counts().min() > 0, "Missing expected sessions."
    cli.print("Sessions   OK")

    assert not data["annotator"].isnull().any(), "Invalid annotator values found."
    assert data["annotator"].value_counts().min() > 0, "Missing expected annotators."
    cli.print("Annotators OK")

    assert not data["variable"].isnull().any(), "Invalid variable values found."
    assert data["variable"].value_counts().min() > 0, "Missing expected variables."
    cli.print("Variables  OK")

    # ---- [2] Assert that all present (variable, value) pairs are legal combinations ---- #

    # NOTE: DataFrameGroupBy.count() returns an index entry per group, and a count column for each original column. The individual counts only count non-null entries.
    #       We take iloc[:,0] assuming that we don't have nulls, to reduce this to a single count per group.

    var_val_count = data.groupby(["variable", "value"]).count().iloc[:,0]
    for (var, val), count in var_val_count.items():
        if count < 1:
            continue

        assert val in ALL_EXPECTED_VALUES.categories
        assert var in EXPECTED_VALUES

        expected_vals = EXPECTED_VALUES[var].categories
        assert val in expected_vals, f"Bad value in variable '{var}': '{val}' (expected one of: {expected_vals})."

    cli.print("Values     OK")

    # ---- [3] Assert that all expected (session, annotator, variable) combinations are present ---- #

    missing_data = []
    indexed = data.set_index(["session", "annotator", "variable"]).sort_index()
    for annotator in EXPECTED_ANNOTATORS.categories:
        for session in ANNOTATOR_TO_SESSION[annotator].categories:
            for variable in REQUIRED_VARIABLES.categories:
                try:
                    combo = indexed.loc[(session, annotator, variable)]
                    assert not combo.empty
                except KeyError:
                    missing_data.append((session, annotator, variable))

    assert len(missing_data) == 0, f"Missing data:\n{pd.DataFrame(missing_data, columns=['session', 'annotator', 'variable'])}"
    cli.print("All (session, annotator, variable) combinations found.")

    # ---- [4] Ensure `task` has exactly three entries for each (session, annotator) pair: `[people, eggs, drums]` (in this order) ---- #

    for session in ALL_EXPECTED_SESSIONS.categories:
        for annotator in SESSION_TO_ANNOTATOR[session].categories:
            entries = indexed.loc[(session, annotator, "task"), :]
            assert len(entries) == 3, f"Session '{session}', annotator '{annotator}' has incorrect number of task entries. Expected 3, found {len(entries)}. Entries:\n{entries}"
            assert entries.iloc[0]["value"] == "people"
            assert entries.iloc[1]["value"] == "eggs"
            assert entries.iloc[2]["value"] == "drums"

    cli.print("All 'task' variables contain ['people', 'eggs', 'drums'] (in this order, no repeats).")


def get_task_spans(data: pd.DataFrame) -> pd.DataFrame:
    """
    Returns (min, max) values for (start_ms, end_ms) per (session, task) combination.
    """

    cli.section("Task Timespans")

    task_data   : pd.DataFrame = data.set_index("variable").loc["task", ["session", "value", "start_ms", "end_ms"]]
    time_counts : pd.DataFrame = task_data.groupby(["session", "value"]).describe().loc[:, pd.IndexSlice[:, ["min", "max"]]]
    time_counts.index = time_counts.index.rename("task", level=1)

    cli.print(time_counts)
    cli.small_divisor()
    cli.print(time_counts.info())

    return time_counts


def unroll_annotations(data: pd.DataFrame, task_timespans: pd.DataFrame) -> pd.DataFrame:
    """
    Convert `data` from the span-based format `(..., variable, value, start_ms, end_ms)` to the "unrolled" frame-based format `(..., task, timestamp, attending, attending_binary, participating, participating_binary)`.

    * Only takes samples in the time interval during each task that all available annotators covered (from `task_timespans[(session, task), ("start_ms", "max")]` to `task_timespans[(session, task), ("end_ms", "min")]`).
    * Saves the unrolled version to the artifacts folder, prints an excerpt to screen, and returns it.
    * Timestamps are tenths of seconds ("deciseconds") since the beginning of the video.
    * `*_binary` versions of the annotation variables take value 0 when the original variable is 0, 1 otherwise.
    """

    cli.section("Unroll Annotations")

    def _millis_to_decis(millis: int) -> int:
        return int(np.round(millis / 100))

    # ---------------- [1] Create an empty array (pre-filled with 0 / "no") ---------------- #

    cli.print("Initializing...")

    unrolled_dict: Dict[str, List] = { "session": [], "annotator": [], "task": [], "time": [], }
    for variable in ANNOTATION_VARIABLES.categories:
        unrolled_dict[variable] = []
        unrolled_dict[f"{variable}_binary"] = []

    for session in ALL_EXPECTED_SESSIONS.categories:
        for task in EXPECTED_VALUES["task"].categories:

            # NOTE: [task_start, task_end) is the span covered by all available annotators.
            #       We take samples every 0.1s during this interval.
            entry      = task_timespans.loc[(session, task)]
            task_start = _millis_to_decis(entry[("start_ms", "max")])
            task_end   = _millis_to_decis(entry[("end_ms"  , "min")])

            for annotator in SESSION_TO_ANNOTATOR[session].categories:
                for time in range(task_start, task_end):

                    unrolled_dict["session"  ].append(session)
                    unrolled_dict["annotator"].append(annotator)
                    unrolled_dict["task"     ].append(task)
                    unrolled_dict["time"     ].append(time)

                    for variable in ANNOTATION_VARIABLES.categories:
                        unrolled_dict[variable].append(0)
                        unrolled_dict[f"{variable}_binary"].append(0)

    unrolled = pd.DataFrame(unrolled_dict)
    unrolled["session"  ] = unrolled["session"  ].astype(ALL_EXPECTED_SESSIONS)
    unrolled["annotator"] = unrolled["annotator"].astype(EXPECTED_ANNOTATORS)
    unrolled["task"     ] = unrolled["task"     ].astype(EXPECTED_VALUES["task"])
    unrolled = unrolled.set_index([ "session", "annotator", "task", "time" ]).sort_index()

    # ---------------- [2] Fill the array with the span information ---------------- #

    cli.print("Filling...")

    for (_, entry) in data.iterrows():
        session   = entry["session"]
        annotator = entry["annotator"]
        variable  = entry["variable"]
        value     = entry["value"]

        if variable == "task":
            continue

        entry_start = _millis_to_decis(entry["start_ms"])
        entry_end   = _millis_to_decis(entry["end_ms"  ])

        encoded_value = pd.Series(value, dtype=EXPECTED_VALUES[variable]).cat.codes.item()

        # NOTE: We only sample during the tasks and not in between.
        #       This means that timespans can e.g. start during one task and end during another.
        #       To account for this, we need to compare each entry with all 3 task intervals.
        for task in EXPECTED_VALUES["task"].categories:
            task_row = task_timespans.loc[(session, task)]
            task_start = _millis_to_decis(task_row[("start_ms", "max")])
            task_end   = _millis_to_decis(task_row[("end_ms"  , "min")])

            m = max(entry_start, task_start)
            M = min(entry_end, task_end)

            if m < M:
                unrolled.loc[pd.IndexSlice[session, annotator, task, m:M], variable] = encoded_value
                unrolled.loc[pd.IndexSlice[session, annotator, task, m:M], f"{variable}_binary"] = int(encoded_value > 0)

    unrolled = unrolled.reset_index()

    cli.print("Done.")

    path = OUT_ROOT / "unrolled_annotations.csv"
    unrolled.to_csv(path, index=False)

    cli.print(unrolled)
    cli.small_divisor()
    cli.print(unrolled.info())
    cli.small_divisor()
    cli.print(f"Saved in {path}")

    return unrolled


def calculate_all_reliabilities(unrolled: pd.DataFrame) -> None:
    """
    Calculates inter-rater agreement scores.
    """

    cli.section("Calculating Reliability")

    unrolled = unrolled.set_index(["session", "annotator", "task", "time"])

    originals = unrolled[["attending", "participating"]]
    originals = originals.melt(ignore_index=False).reset_index().set_index(["task", "variable", "session", "annotator", "time"]).unstack("annotator")
    originals.columns = originals.columns.get_level_values(1)

    calculate_reliability(originals, OUT_ROOT, ["people", "eggs", "drums"], ["attending", "participating"], prefix="original")

    binaries = unrolled[["attending_binary", "participating_binary"]]
    binaries = binaries.melt(ignore_index=False).reset_index().set_index(["task", "variable", "session", "annotator", "time"]).unstack("annotator")
    binaries.columns = binaries.columns.get_level_values(1)

    calculate_reliability(binaries, OUT_ROOT, ["people", "eggs", "drums"], ["attending_binary", "participating_binary"], prefix="binary")


# ================ KICKSTART ================ #


if __name__ == "__main__":
    main()

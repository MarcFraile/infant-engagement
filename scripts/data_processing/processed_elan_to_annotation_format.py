#!/bin/env -S python3 -u


# ===================================================================================== #
# ========================== ENGAGEMENT ANNOTATION PROCESSOR ========================== #
# ===================================================================================== #
#
# Transforms the extracted engagement annotations into a training-ready format.
#
# 1. Loads `original_annotation_spans.csv`, dropping the variable `ignore`.
# 2. Extracts task duration data.
# 3. Limits annotations to the task spans, marking each annotation with its
#    corresponding task (drops `task` entries).
# 4. Adds spans labeled `no`, so that each variable covers its task duration.
# 5. Saves the tasks spans to `task_spans.csv` and the filled variable spans to
#    `filled_annotation_spans.csv`.
#
# (Second script in engagement annotation processing; preceded by process_elan.py)
#
# ===================================================================================== #
# ===================================================================================== #


# ================ IMPORTS ================ #


from pathlib import Path
import pandas as pd
from pandas.api.types import CategoricalDtype

from local.cli import PrettyCli


# ================ SETTINGS ================ #


ANNOTATORS = CategoricalDtype([ "ew", "mf", "myz" ], ordered=True)
SESSIONS   = CategoricalDtype([
    "fp19", "fp28", "fp29", "fp35",                 # 1st pilot  (lexicographical)
    "fp32",                                         # 2nd pilot  (-)
    "fp15", "fp27", "fp20", "fp14", "fp21", "fp16", # EW  => MYZ (Coding Guide order)
    "fp33", "fp24", "fp18", "fp39", "fp25", "fp40", # MF  => EW  (Coding Guide order)
    "fp38", "fp30", "fp31", "fp13", "fp17", "fp34", # MYZ => MF  (Coding Guide order)
], ordered=True)
VARIABLES = CategoricalDtype([ "participating", "attending", "task" ], ordered=True)
VALUES    = CategoricalDtype([ "no", "self", "joint", "attending", "excited", "people", "eggs", "drums" ], ordered=True)
TASKS     = CategoricalDtype([ "people", "eggs", "drums" ], ordered=True)

ANNOTATION_ROOT = Path("data/processed/engagement/")
INPUT_FILE      = ANNOTATION_ROOT / "original_annotation_spans.csv"
FILLED_FILE     = ANNOTATION_ROOT / "filled_annotation_spans.csv"
TASK_SPAN_FILE  = ANNOTATION_ROOT / "task_spans.csv"

cli = PrettyCli()


# ================ FUNCTIONS ================ #


def reformat_elan() -> None:
    """
    Main function.

    Converts the aggregated engagement timespans into a format consumable by the data loaders.
    """

    cli.main_title("Processed ELAN to Wide Format")

    long = load_data()
    task_spans = get_task_spans(long)
    cut = cut_annotations(long, task_spans)
    filled = fill_blanks(cut, task_spans)

    task_spans.to_csv(TASK_SPAN_FILE)
    filled.to_csv(FILLED_FILE, index=False)


def load_data() -> pd.DataFrame:
    """
    Loads `INPUT_FILE` as a dataframe, using ordered categorical types.

    * Drops `ignore` variable.
    * Prints chapter header and dataframe info.
    """

    cli.chapter("Load Data")

    assert INPUT_FILE.is_file(), f"Aggregated timespan CSV not found: {INPUT_FILE}"
    long = pd.read_csv(INPUT_FILE)

    long["session"  ] = long["session"  ].astype(SESSIONS)
    long["annotator"] = long["annotator"].astype(ANNOTATORS)
    long["variable" ] = long["variable" ].astype(VARIABLES)
    long["value"    ] = long["value"    ].astype(VALUES)

    long = long.dropna() # Remove "ignore"

    cli.print(long.info())
    return long


def get_task_spans(data: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts task duration data into `(session, annotator, task) -> (start_ms, end_ms)`.

    * Prints chapter header and dataframe.
    """

    cli.chapter("Task Spans")

    task_spans = data[data["variable"] == "task"].drop(columns="variable")
    task_spans = task_spans.rename({"value": "task"}, axis=1)
    task_spans = task_spans.set_index(["session", "annotator", "task"]).sort_index()

    cli.print(task_spans)
    return task_spans


def cut_annotations(data: pd.DataFrame, task_spans: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a copy of `data`, cutting entries to the lengths of the tasks spans.

    * Drops `task` entries.
    * If needed, educes the duration of each entry so it fits within a task.
    * Splits entries that overlap several tasks into disjoint entries.
    * Marks each entry with its corresponding task.
    * Drops entries that would become zero-length.
    """

    cli.chapter("Cut Annotations")

    cut = []
    for (_, row) in data.iterrows():
        if row["variable"] == "task":
            continue

        rm, rM = row[["start_ms", "end_ms"]]
        for task in TASKS.categories:
            tm, tM = task_spans.loc[(row["session"], row["annotator"], task)]
            m = max(rm, tm)
            M = min(rM, tM)

            if m < M:
                cut_row = row.copy()
                cut_row["task"] = task
                cut_row[["start_ms", "end_ms"]] = [m, M]
                cut.append(cut_row)

    output = pd.DataFrame(cut, columns=[ "session", "annotator", "task", "variable", "value", "start_ms", "end_ms" ])
    output["task"     ] = output["task"     ].astype(TASKS)
    output["session"  ] = output["session"  ].astype(SESSIONS)
    output["annotator"] = output["annotator"].astype(ANNOTATORS)
    output["variable" ] = output["variable" ].astype(VARIABLES)
    output["value"    ] = output["value"    ].astype(VALUES)
    output = output.sort_values(by=[ "session", "annotator", "task", "variable", "start_ms", "end_ms"])

    cli.print(output)
    return output


def fill_blanks(cut: pd.DataFrame, task_spans: pd.DataFrame) -> pd.DataFrame:
    """
    Adds spans labeled `no` so that each variable densely covers the duration of each task.
    """

    cli.chapter("Fill Blanks")

    indexed = cut.set_index([ "task", "session", "annotator", "variable" ]).sort_index()
    rows = []

    for task in TASKS.categories:
        for session in SESSIONS.categories:
            for annotator in ANNOTATORS.categories:
                span_key = (session, annotator, task)
                if not span_key in task_spans.index:
                    continue
                task_start, task_end = task_spans.loc[span_key]
                for variable in VARIABLES.categories:
                    current_time = task_start
                    key = (task, session, annotator, variable)
                    if key in indexed.index:
                        entries = indexed.loc[key]
                        for _, entry in entries.iterrows():
                            value, entry_start, entry_end = entry
                            if entry_start > current_time:
                                rows.append((session, annotator, task, variable, "no", current_time, entry_start))
                                current_time = entry_start
                            rows.append((session, annotator, task, variable, value, current_time, entry_end))
                            current_time = entry_end
                    if current_time < task_end:
                        rows.append((session, annotator, task, variable, "no", current_time, task_end))

    filled = pd.DataFrame(rows, columns=cut.columns)
    filled["task"     ] = filled["task"     ].astype(TASKS)
    filled["session"  ] = filled["session"  ].astype(SESSIONS)
    filled["annotator"] = filled["annotator"].astype(ANNOTATORS)
    filled["variable" ] = filled["variable" ].astype(VARIABLES)
    filled["value"    ] = filled["value"    ].astype(VALUES)
    filled = filled.sort_values(by=[ "session", "annotator", "task", "variable", "start_ms", "end_ms" ])

    cli.print(filled)
    return filled


# ================ KICKSTART ================ #


if __name__ == "__main__":
    reformat_elan()

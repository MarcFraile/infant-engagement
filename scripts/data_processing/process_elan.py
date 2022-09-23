#!/bin/env -S python3 -u


# ===================================================================================== #
# ================================ ELAN DATA CONVERTER ================================ #
# ===================================================================================== #
#
# Reads in EAF files from the raw engagement annotation folder, validates them, and
# saves them in formats that can be consumed by other scripts and tools.
#
# * Aggregates all timespans into elan/converted.csv for use with Python + Pandas.
# * Creates 2-rater comparison tab-delimited files for use with EasyDIAg.
#
# ===================================================================================== #
# ===================================================================================== #


# ================ IMPORTS ================ #


from pathlib import Path
from typing import Any, Dict, List
import xml.etree.ElementTree as ET
import pandas as pd

from local.cli import PrettyCli


# ================ SETTINGS ================ #


EAF_ROOT     = Path("data/raw/engagement/")
CSV_FILE     = Path("data/processed/engagement/annotation_spans.csv")
TABFILE_ROOT = Path("data/processed/engagement/tabfiles")

PARTICIPATING = "participating"
ATTENDING     = "attending"
TASK          = "task"
IGNORE        = "ignore"

PEOPLE = "people"
EGGS   = "eggs"
DRUMS  = "drums"

EXCITED = "excited"
DROPPED_TOY = "dropped toy"
CHANGE_OF_TEMPO = "change of tempo"

EW  = "ew"
MF  = "mf"
MYZ = "myz"
EXPECTED_ANNOTATORS = [ EW, MF, MYZ ]

FIRST_PILOT  = [ "fp19", "fp28", "fp29", "fp35" ]                 # 1st pilot
SECOND_PILOT = [ "fp32" ]                                         # 2nd pilot
BATCH_EW     = [ "fp15", "fp27", "fp20", "fp14", "fp21", "fp16" ] # EW  => MYZ
BATCH_MF     = [ "fp33", "fp24", "fp18", "fp39", "fp25", "fp40" ] # MF  => EW
BATCH_MYZ    = [ "fp38", "fp30", "fp31", "fp13", "fp17", "fp34" ] # MYZ => MF

ALL_EXPECTED_SESSIONS = [ *FIRST_PILOT, *SECOND_PILOT, *BATCH_EW, *BATCH_MF, *BATCH_MYZ ]
ANNOTATOR_TO_SESSION = {
    EW  : [ *FIRST_PILOT, *SECOND_PILOT, *BATCH_EW , *BATCH_MF  ],
    MF  : [ *FIRST_PILOT, *SECOND_PILOT, *BATCH_MF , *BATCH_MYZ ],
    MYZ : [ *FIRST_PILOT, *SECOND_PILOT, *BATCH_MYZ, *BATCH_EW  ],
}

for annotator in EXPECTED_ANNOTATORS:
    assert annotator in ANNOTATOR_TO_SESSION

SESSION_TO_ANNOTATOR : Dict[str, List[str]] = {}
for session in ALL_EXPECTED_SESSIONS:
    SESSION_TO_ANNOTATOR[session] = []

    for annotator in EXPECTED_ANNOTATORS:
        if session in ANNOTATOR_TO_SESSION[annotator]:
            SESSION_TO_ANNOTATOR[session].append(annotator)

    assert len(SESSION_TO_ANNOTATOR[session]) >= 2

REQUIRED_VARIABLES = [ PARTICIPATING, ATTENDING, TASK ]
OPTIONAL_VARIABLES = [ IGNORE ]
ALLOWED_VARIABLES  = [ *REQUIRED_VARIABLES, *OPTIONAL_VARIABLES ]
EXPECTED_VALUES    = {
    PARTICIPATING : [ "self", "joint" ],
    ATTENDING     : [ ATTENDING, EXCITED ],
    TASK          : [ PEOPLE, EGGS, DRUMS ],
}

for key in EXPECTED_VALUES: # Sanity check
    assert key in ALLOWED_VARIABLES

# Some typos, some annotation differences in "ignore".
KNOWN_VALUE_CORRECTIONS : Dict[str, Dict[str, str]] = {
    PARTICIPATING : {},
    ATTENDING : {
        "attednding" : ATTENDING,
        "attendinh"  : ATTENDING,
        "excitied"   : EXCITED,
    },
    TASK : {
        "egg" : EGGS,
    },
    IGNORE : {
        "egg fell"        : DROPPED_TOY,
        "pieces fell"     : DROPPED_TOY,
        "stick fell"      : DROPPED_TOY,
        "drop the toy"    : DROPPED_TOY,
        "dropped the toy" : DROPPED_TOY,
        "flip the drum"   : CHANGE_OF_TEMPO,
    },
}

assert list(KNOWN_VALUE_CORRECTIONS.keys()) == ALLOWED_VARIABLES

# ---- Helpers ---- #

cli = PrettyCli()


# ================ FUNCTIONS ================ #


def main():
    """
    Main function.

    1. Orchestrates reading, validation and aggregation of all EAF files into a single Pandas dataframe.
    2. Saves dataframe as CSV file.
    3. Saves pairwise rater annotations as tab-delimited files for use with EasyDIAg.
    """

    cli.main_title("ELAN TO CSV")

    assert EAF_ROOT.is_dir()

    cli.section("Paths")
    cli.print({
        "EAF Root": EAF_ROOT,
        "CSV File": CSV_FILE,
        "Tabfile Root": TABFILE_ROOT,
    })

    cli.section("Load Data")
    data = load_data()
    cli.print("Loaded and validated.")

    cli.section("Check Missing Data")
    validate_missing_data(data)
    cli.print("No data is missing.")

    cli.section("Validate Task Spans")
    validate_task_spans(data)
    cli.print("All tasks found in every session annotation.")

    cli.section("Tier Info")
    cli.print(data)

    cli.section("Save Data")
    data.to_csv(CSV_FILE, index=False)
    save_tabfiles(data)


def load_data() -> pd.DataFrame:
    """
    Iterates over all EAF files in EAF_ROOT, loading and validating the file names and file contents. Aggregates contents into the returned DataFrame.
    """

    eaf_files = [ file for file in EAF_ROOT.iterdir() if file.is_file() and file.suffix == ".eaf" ]
    eaf_files.sort()

    data = pd.DataFrame()

    for eaf_file in eaf_files:
        cli.subchapter(eaf_file.name)

        name_parts = eaf_file.stem.split("_")
        assert len(name_parts) == 3, f"Expected naming pattern <session>_<initials>_<ISO date>.eaf, found '{eaf_file.stem}'"

        [ session, initials, iso_date ] = name_parts

        assert session  in ALL_EXPECTED_SESSIONS    , f"Expected session to be in {ALL_EXPECTED_SESSIONS}, found '{session}'"
        assert initials in EXPECTED_ANNOTATORS      , f"Expected initials to be in {EXPECTED_ANNOTATORS}, found '{initials}'"
        assert 20220207 <= int(iso_date) < 20220401 , f"Expected ISO date to be an int in range [20220207, 20220401), found '{iso_date}'"

        file_data = load_eaf(eaf_file)

        file_data["session"  ] = session
        file_data["annotator"] = initials

        data = pd.concat([data, file_data], ignore_index=True)

    data = data[["session", "annotator", "variable", "value", "start_ms", "end_ms"]] # Force nice order of columns.
    return data


def load_eaf(eaf_path: Path) -> pd.DataFrame:
    """
    Loads a single EAF file, reads and validates its contents, and produces a Pandas dataframe.
    """

    def _find_or_panic(element: ET.Element, tag: str) -> ET.Element:
        child = element.find(tag)
        assert child is not None
        return child

    def _get_or_panic(element: ET.Element, key: str) -> str:
        value = element.get(key)
        assert value is not None
        return value

    tree = ET.parse(eaf_path)
    root = tree.getroot()

    header           : ET.Element = _find_or_panic(root  , "HEADER")
    media_descriptor : ET.Element = _find_or_panic(header, "MEDIA_DESCRIPTOR")

    time_units         : str = _get_or_panic(header, "TIME_UNITS")
    media_url          : str = _get_or_panic(media_descriptor, "MEDIA_URL")
    relative_media_url : str = _get_or_panic(media_descriptor, "RELATIVE_MEDIA_URL")

    assert time_units == "milliseconds", f"[{eaf_path.stem}] Expected milliseconds, found: {time_units}"

    cli.section("Media Info")
    cli.print({
        "Absolute Media URL" : media_url,
        "Relative Media URL" : relative_media_url
    })

    time_order = _find_or_panic(root, "TIME_ORDER")
    timeslots : Dict[str, int] = {}
    for child in time_order:
        assert child.tag == "TIME_SLOT"
        id     = str(_get_or_panic(child, "TIME_SLOT_ID"))
        millis = int(_get_or_panic(child, "TIME_VALUE"  ))
        timeslots[id] = millis

    cli.section("Timeslots (s)")
    ts = [ v / 1000 for v in timeslots.values() ]
    if len(ts) <= 7:
        cli.print(ts)
    else:
        cli.print(f"[{ts[0]}, {ts[1]}, {ts[2]}, ... , {ts[-3]}, {ts[-2]}, {ts[-1]}]")

    data : Dict[str, List[Any]] = { "variable": [], "value": [], "start_ms": [], "end_ms": [] }

    for tier in root:
        if tier.tag != "TIER":
            continue

        tier_id = _get_or_panic(tier, "TIER_ID")
        assert tier_id in ALLOWED_VARIABLES, f"Unexpected variable name: {tier_id}"

        for big_matrioska in tier:
            assert big_matrioska.tag == "ANNOTATION"

            middle_matrioska = _find_or_panic(big_matrioska, "ALIGNABLE_ANNOTATION")
            little_matrioska = _find_or_panic(middle_matrioska, "ANNOTATION_VALUE")

            start_id = _get_or_panic(middle_matrioska, "TIME_SLOT_REF1")
            end_id   = _get_or_panic(middle_matrioska, "TIME_SLOT_REF2")

            start = timeslots[start_id]
            end   = timeslots[end_id]
            value = little_matrioska.text

            assert value is not None, f"Annotation entry has no value! (start: {start}ms, end: {end}ms)."

            value = value.strip().lower()

            value_corrections = KNOWN_VALUE_CORRECTIONS[tier_id]
            if value in value_corrections:
                value = value_corrections[value]

            if tier_id in EXPECTED_VALUES:
                valid_values = EXPECTED_VALUES[tier_id]
                assert value in valid_values, f"[{eaf_path.stem}] Invalid value for tier '{tier_id}': '{value}' (valid values: {valid_values})"

            data["variable"].append(tier_id)
            data["value"   ].append(value)
            data["start_ms"].append(start)
            data["end_ms"  ].append(end)

    for variable in REQUIRED_VARIABLES:
        assert variable in data["variable"]

    return pd.DataFrame(data)


def validate_missing_data(data: pd.DataFrame) -> None:
    """
    Checks if all (session, annotator, variable) triplets are present. Panics if any are missing, listing all the missing pairs.
    """

    missing_data = []
    indexed = data.set_index(["session", "annotator", "variable"]).sort_index()

    for annotator in EXPECTED_ANNOTATORS:
        for session in ANNOTATOR_TO_SESSION[annotator]:
            for variable in REQUIRED_VARIABLES:
                try:
                    combo : pd.DataFrame = indexed.loc[(session, annotator, variable)]
                    assert not combo.empty
                except KeyError:
                    missing_data.append((session, annotator, variable))

    assert len(missing_data) == 0, f"Missing data:\n{pd.DataFrame(missing_data, columns=['session', 'annotator', 'variable'])}"


def validate_task_spans(data: pd.DataFrame) -> None:
    """
    Validates that each valid (session, annotator) pair contains exactly 3 entries in `"task"`: `["people", "eggs", "drums"]` (in this order).
    """

    validation_errors : List[str] = []
    indexed = data.set_index(["session", "annotator", "variable"]).sort_index()

    for session in ALL_EXPECTED_SESSIONS:
        for annotator in SESSION_TO_ANNOTATOR[session]:
            entries = indexed.loc[(session, annotator, "task"), :]

            if len(entries) != 3:
                validation_errors.append(f"Session '{session}', annotator '{annotator}' has incorrect number of task entries. Expected 3, found {len(entries)}. Entries:\n{entries}")
                continue

            for (idx, expected_value) in enumerate(EXPECTED_VALUES[TASK]):
                observed_value = entries.iloc[idx]["value"]
                if observed_value != expected_value:
                    validation_errors.append(f"Session '{session}', annotator '{annotator}': Task entry [{idx}] has incorrect value '{observed_value}' (expected '{expected_value}').")

    error_message = f"Found {len(validation_errors)} error(s) when validating the contents of the variable 'task':"
    for idx, msg in enumerate(validation_errors):
        error_message += f"\n[{idx}] {msg}"
    assert len(validation_errors) == 0, error_message


def save_tabfiles(data: pd.DataFrame) -> None:
    """
    Generates tabfiles to be consumed by EasyDIAg, as an alternative way to generate the Kappa values.
    """

    if not TABFILE_ROOT.exists():
        TABFILE_ROOT.mkdir(parents=False, exist_ok=False)

    indexed = data.set_index("annotator").sort_index()

    for i in range(len(EXPECTED_ANNOTATORS) - 1):
        for j in range(i + 1, len(EXPECTED_ANNOTATORS)):
            first  = EXPECTED_ANNOTATORS[i]
            second = EXPECTED_ANNOTATORS[j]

            path = TABFILE_ROOT / f"{first}_{second}.txt"

            entries : pd.DataFrame = indexed.loc[[first, second], :].sort_index()
            entries["file"] = entries["session"] + f"_{first}_{second}.eaf"
            entries.loc[first , "variable"] = entries.loc[first , "variable"] + "_R1"
            entries.loc[second, "variable"] = entries.loc[second, "variable"] + "_R2"
            entries = entries.reset_index(drop=True).loc[:, ["variable", "start_ms", "end_ms", "value", "file"]]

            cli.print(entries)
            entries.to_csv(path, sep="\t", header=False, index=False)


# ================ KICKSTART ================ #


if __name__ == "__main__":
    main()

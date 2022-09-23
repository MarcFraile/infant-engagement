#!/bin/env -S python3 -u


# ===================================================================================== #
# ================================== VIDEO PROCESSOR ================================== #
# ===================================================================================== #
#
# Script in charge of preparing the raw videos for ML use.
#
# 1. Checks that all raw videos follow the correct naming scheme <session>.mp4.
# 2. Makes downsampled copies to the processed folder, ready for training.
#
# ===================================================================================== #
# ===================================================================================== #


# ================ IMPORTS ================ #


import os
import re
from pathlib import Path
import shutil

from tqdm import tqdm

from local.cli import PrettyCli


# ================ SETTINGS ================ #


VIDEO_INPUT_DIR  = Path("data/raw/video")
VIDEO_OUTPUT_DIR = Path("data/processed/video")

SESSION_NAME = re.compile("fp\d{2}") # Matches fp00, fp01...

cli = PrettyCli()


# ================ FUNCTIONS ================ #


def process_videos() -> None:
    """
    Main function.

    * Ensures `VIDEO_INPUT_DIR` contains multiple MP4 files named `fpXX.mp4`, where `XX` is a two-digit integer.
    * Uses the `ffmpeg` CLI to copy each video to `VIDEO_OUTPUT_DIR`, with the following processing:
        1. Remove audio.
        2. Reduce spatial resolution to `height=160px`; `width=proportional` (rounded to the nearest multiple of `16px`).
        3. Reduce the framerate to 25/8 (3.125) frames per second.
    """

    cli.main_title("PROCESS VIDEOS")
    cli.print({
        "Input Dir"  : VIDEO_INPUT_DIR,
        "Output DIr" : VIDEO_OUTPUT_DIR,
    })

    assert VIDEO_INPUT_DIR.is_dir()

    sessions = [ file.stem for file in VIDEO_INPUT_DIR.iterdir() if file.suffix == ".mp4" ]
    sessions.sort()

    assert len(sessions) > 1
    for session in sessions:
        assert SESSION_NAME.fullmatch(session)

    if VIDEO_OUTPUT_DIR.exists():
        shutil.rmtree(VIDEO_OUTPUT_DIR)
    VIDEO_OUTPUT_DIR.mkdir(parents=True, exist_ok=False)

    succeeded = []
    failed    = []

    for session in tqdm(sessions):
        in_file  = VIDEO_INPUT_DIR  / f"{session}.mp4"
        out_file = VIDEO_OUTPUT_DIR / f"{session}.mp4"
        # NOTE: If we had ffmpeg 5+, we could say 'fps=source_fps/8' instead of hardcoding the 25 (checked using ffprobe).
        #       Sadly, it's not in pip yet.
        command = f"ffmpeg -hide_banner -loglevel error -y -i {in_file} -an -vf scale=-16:160,fps=fps=25/8 -preset slow -crf 18 {out_file}"

        cli.print(f"\033[31m{command}\033[0m")
        r = os.system(command)
        cli.print(f"\033[33mreturned: {r}\033[0m")

        if r == 0:
            succeeded.append(session)
        else:
            failed.append(session)

    cli.subchapter("Conversion Results")
    cli.print({
        "Sessions"  : len(sessions),
        "Succeeded" : len(succeeded),
        "Failed"    : failed,
    })


# ================ KICKSTART ================ #


if __name__ == "__main__":
    process_videos()

#!/bin/bash

# Run this from the root folder!

set -o pipefail # Keep the failure status if something fails in a pipe chain.
POINTER_PATH="artifacts/final/current.txt"

function save_log() {
    # If the script lived long enough to set up the pointer, copy the run log to the new output folder.
    if [ -f $POINTER_PATH ]; then
        mv artifacts/temp.log $(cat $POINTER_PATH)/run.log
        mv artifacts/temp.err $(cat $POINTER_PATH)/run.err
        rm $POINTER_PATH
    fi

    exit
}

trap save_log SIGINT

# Delete the pointer file if it exists (avoids overwriting of previous run logs).
rm -f $POINTER_PATH

# Main script
scripts/training/final/script.py $@ > >(tee -a artifacts/temp.log) 2> >(tee -a artifacts/temp.err >&2)

save_log

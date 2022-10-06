#!/bin/bash -e


# ================ jobfile.sh ================ #
# Job script, called from distributed.sh
# Exactly one argument should be passed to this script: the path to the timestamped output folder.
# Each individual task will be saved in its own subfolder: $1/task_$SLURM_ARRAY_TASK_ID


#SBATCH -A SNIC2021-7-145
#SBATCH -p alvis
#SBATCH -J infant_engagement_finetune_worker

#SBATCH -e log/worker-%A_%a.err
#SBATCH -o log/worker-%A_%a.log

#SBATCH -t 1-00:00:00
#SBATCH --gpus-per-node=A40:1
#SBATCH --array=0-74


module purge

POETRY=~/.local/bin/poetry
BASE_IMG=/apps/containers/PyTorch/PyTorch-1.12-NGC-22.05.sif
OVERLAY=img/singularity.img
SCRIPT=scripts/training/finetune_search/script.py

singularity exec --overlay $OVERLAY:ro $BASE_IMG $POETRY run $SCRIPT $1

#!/bin/bash -e


# ================ distributed.sh ================ #
# 1. Creates a unique timestamped output folder.
# 2. Runs the hyperparameter search as a job array.
# 3. Unifies the results.


#SBATCH -A SNIC2021-7-145
#SBATCH -p alvis
#SBATCH -J infant_engagement_finetune

#SBATCH -e log/master-%A.err
#SBATCH -o log/master-%A.log

#SBATCH -t 0-12:15:00
#SBATCH --gpus-per-node=T4:1


module purge

BASE_IMG=/apps/containers/PyTorch/PyTorch-1.11-NGC-21.12.sif
OVERLAY=img/singularity.img
SCRIPT_FOLDER=scripts/training/finetune_search

echo "Script folder: $SCRIPT_FOLDER"

# Create the output folder as an ISO-8601 timestamp.
TIMESTAMP=$(date +"%Y%m%dT%k%M%S")
OUTPUT_ROOT=artifacts/finetune_search/$TIMESTAMP
LOGDIR=$OUTPUT_ROOT/log

echo "Output dir: $OUTPUT_ROOT"

mkdir -p $OUTPUT_ROOT
mkdir -p $LOGDIR

echo "Running job array..."
CHILD_ID=$(sbatch --wait $SCRIPT_FOLDER/jobfile.sh $OUTPUT_ROOT | tr -dc '0-9')

echo "Unifying..."
singularity exec --overlay $OVERLAY:ro $BASE_IMG python3 $SCRIPT_FOLDER/unify.py $OUTPUT_ROOT

echo "Moving child logs..."
mv log/worker-${CHILD_ID}_* $LOGDIR

echo "Moving self log..."
mv log/master-${SLURM_JOB_ID}* $LOGDIR

echo "Done."

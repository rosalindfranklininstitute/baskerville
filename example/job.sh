#!/bin/bash
#SBATCH --account ffnr0871-rfi-test
#SBATCH --qos rfi
#SBATCH --mail-user joss.whittle@rfi.ac.uk
#SBATCH --mail-type ALL
#SBATCH --time 5
#SBATCH --nodes 2
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 36
#SBATCH --gpus-per-node 2

# Load a bare-bones environment that will support CUDA enabled MPI within singularity containers
module purge
module load baskerville
module load bask-apps/live
module load OpenMPI/4.0.5-gcccuda-2020b

# Enable verbose logging of job script commands
set -x

# Root directory for project storage
export PROJECT_DIR="/bask/projects/f/ffnr0871-rfi-test/pje39613"

# Place singularity cache dir in project storage since /home/ is limited to 20GB per user
# Singularity can use up a LOT of cache space when converti/bask/projects/f/ffnr0871-rfi-test/pje39613ng OCI images to singularity images!
export SINGULARITY_CACHEDIR="$PROJECT_DIR/.singularity-cache"

# Use container uri with hash digest so that the correct container version gets used even if the job queues
# for a long time and you have pushed a newer version of the container in the meantime for future experiments
export CONTAINER="docker://quay.io/rosalindfranklininstitute/jax@sha256:5011eed822e7af340c0a4120d2a03f383e8894e7dfd83d3e5219702514883349"

# Point to activeloop hub datasets
export TRAIN_DATASET_DIR="$PROJECT_DIR/mnist-train"
export VAL_DATASET_DIR="$PROJECT_DIR/mnist-test"

# Execute the parallel job
mpirun singularity run --nv $CONTAINER python example/job.py --log_nvsmi --train_dataset "$TRAIN_DATASET_DIR" --val_dataset "$VAL_DATASET_DIR"

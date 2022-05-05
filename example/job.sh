#!/bin/bash
#SBATCH --account ffnr0871-rfi-test
#SBATCH --qos rfi
#SBATCH --mail-user joss.whittle@rfi.ac.uk
#SBATCH --mail-type ALL
#SBATCH --time 2
#SBATCH --nodes 2
#SBATCH --ntasks-per-node 2
#SBATCH --cpus-per-task 1
#SBATCH --gpus-per-node 0

# Load a bare-bones environment that will support CUDA enabled MPI
module purge
module load baskerville
module load bask-apps/live
module load OpenMPI/4.0.5-gcccuda-2020b

# Enable verbose logging of job script commands
set -x

# Compile the job
mpicc example/job.c -o example/job

# Execute the parallel job
srun --mpi=cray_shasta example/job

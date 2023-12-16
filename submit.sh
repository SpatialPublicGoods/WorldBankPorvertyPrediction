#!/bin/bash

# This is a comment

#---------------------------------------------------------------------------------
# Account information

#SBATCH --account=phd              # basic (default), staff, phd, faculty

#---------------------------------------------------------------------------------
# Resources requested

#SBATCH --partition=standard       # standard (default), long, gpu, mpi, highmem
#SBATCH --cpus-per-task=1          # number of CPUs requested (for parallel tasks)
#SBATCH --mem=240G                   # requested memory
#SBATCH --time=0-00:05:00          # wall clock limit (d-hh:mm:ss)

# We requested 1 CPU, 240GB RAM, and 5 minutes of run time

#---------------------------------------------------------------------------------
# Job specific name (helps organize and track progress of jobs)

#SBATCH --job-name=sample_slurm_job    # user-defined job name

# Error and Output file names

#SBATCH --output=slurm-test-%j.out
#SBATCH --error=slurm-test-%j.err

#---------------------------------------------------------------------------------
# Print some useful variables

echo "Job ID: $SLURM_JOB_ID"
echo "Job User: $SLURM_JOB_USER"
echo "Num Cores: $SLURM_JOB_CPUS_PER_NODE"

#---------------------------------------------------------------------------------
# Load necessary modules for the job

module load python/booth/3.10   # Assuming you want Python 3.10

#---------------------------------------------------------------------------------
# Commands to execute below...

cd code
python3 "04_generate_prediction_report.py"
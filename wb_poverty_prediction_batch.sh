#!/bin/bash

# This is a comment

#---------------------------------------------------------------------------------
# Account information

#SBATCH --account=phd              # basic (default), staff, phd, faculty

#---------------------------------------------------------------------------------
# Resources requested

#SBATCH --partition=standard       # standard (default), long, gpu, mpi, highmem
#SBATCH --cpus-per-task=10          # number of CPUs requested (for parallel tasks)
#SBATCH --mem=240G                   # requested memory
#SBATCH --time=0-05:00:00          # wall clock limit (d-hh:mm:ss)

# We requested 1 CPU, 240GB RAM, and 5 minutes of run time

#---------------------------------------------------------------------------------
# Job specific name (helps organize and track progress of jobs)

#SBATCH --job-name=wb_pred    # user-defined job name

# Error and Output file names

#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

#---------------------------------------------------------------------------------
# Specify email notifications

#SBATCH --mail-user=francocalle93@gmail.com   # Replace with your email address
#SBATCH --mail-type=END                      # Receive an email at the end of the job

#---------------------------------------------------------------------------------
# Print some useful variables

echo "Job ID: $SLURM_JOB_ID"
echo "Job User: $SLURM_JOB_USER"
echo "Num Cores: $SLURM_JOB_CPUS_PER_NODE"

#---------------------------------------------------------------------------------
# Load necessary modules for the job

module load python/booth/3.10    # load python module

#---------------------------------------------------------------------------------
# Commands to execute below...

cd code

ulimit -n 4096

echo "Run Python Script: 01_describe_ml_consolidated_dataset.py"
python3 "01_describe_ml_consolidated_dataset.py"

echo "Run Python Script: 03_run_income_prediction_lasso_weighted.py"
python3 "03_run_income_prediction_lasso_weighted.py"

echo "Run Python Script: 04_generate_prediction_report.py"
python3 "04_generate_prediction_report.py"

#---------------------------------------------------------------------------------
# Send an email with the slurm-test-%j.out file as an attachment

echo "Send email to Franco Calle"

cd '..'

if [ -f "slurm-$SLURM_JOB_ID.out" ]; then
    echo -e "Subject: Slurm Job Completed\n\n$(cat "slurm-$SLURM_JOB_ID.out")" | sendmail francocalle93@gmail.com
else
    echo -e "Subject: Slurm Job Completed\n\nThe slurm-$SLURM_JOB_ID.out file was not found." | sendmail francocalle93@gmail.com
fi

echo "Done"

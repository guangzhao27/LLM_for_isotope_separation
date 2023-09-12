#!/bin/bash
#SBATCH --qos=regular
#SBATCH --time=3:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --constraint=gpu
#SBATCH --gpus-per-node=1
#SBATCH --output=my_job_output.txt
#SBATCH --account=m4138_g

conda activate
python fine_tune_scibert.py